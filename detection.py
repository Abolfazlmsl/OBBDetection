#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 09:47:33 2025

@author: abolfazl
"""

import os
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
from mmdet.apis import init_detector, inference_detector
from concurrent.futures import ThreadPoolExecutor

# ===== User Settings =====
CALCULATE_METRICS = True

TILE_MODELS = [
    {"config": "configs/obb/oriented_rcnn/faster_rcnn_orpn_r101_fpn_1x_dota10.py",
      "checkpoint": "Checkpoints/best128.pth", "tile_size": 128, "overlap": 50}
    # {"config": "configs/obb/retinanet_obb/retinanet_obb_r101_fpn_2x_dota10.py",
    #   "checkpoint": "Checkpoints/best128-ret.pth", "tile_size": 128, "overlap": 50}
    # {"config": "configs/obb/faster_rcnn_obb/faster_rcnn_obb_r101_fpn_1x_dota10.py",
    #   "checkpoint": "Checkpoints/best128-cnn.pth", "tile_size": 128, "overlap": 50}
    # {"config": "configs/obb/oriented_rcnn/faster_rcnn_orpn_r101_fpn_1x_dota10.py",
    #  "checkpoint": "Checkpoints/best416.pth", "tile_size": 416, "overlap": 150}
    # {"config": "configs/obb/retinanet_obb/retinanet_obb_r101_fpn_2x_dota10.py",
    #  "checkpoint": "Checkpoints/best416-ret.pth", "tile_size": 416, "overlap": 150}
    # {"config": "configs/obb/faster_rcnn_obb/faster_rcnn_obb_r101_fpn_1x_dota10.py",
    #  "checkpoint": "Checkpoints/best416-cnn.pth", "tile_size": 416, "overlap": 150}
]


IOU_THRESHOLD = 0.2# Representation
iou_thr = 0.2 # Metric
EVAL_GEOM = "poly"
IOU_LIST_FOR_MAP = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MAP_MIN_SCORE = 0.001  # keep all detections (mAP will sweep scores)

CLASS_NAMES = {
    0: "Landslide 1",
    1: "Strike",
    2: "Spring 1",
    3: "Minepit 1",
    4: "Hillside",
    5: "Feuchte",
    6: "Torf",
    7: "Bergsturz",
    8: "Landslide 2",
    9: "Spring 2",
    10: "Spring 3",
    11: "Minepit 2",
    12: "Spring B2",
    13: "Hillside B2",
}

# Define colors for different classes
CLASS_COLORS = {
    0: (255, 0, 0),  # Landslide
    1: (0, 255, 0),  # Strike
    2: (0, 0, 255),  # Spring
    3: (255, 255, 0),  # Minepit
    4: (255, 0, 255),  # Hillside
    5: (0, 255, 255),  # Feuchte
    6: (0, 0, 0),  # Torf
    7: (240, 34, 0),  # Bergsturz
    8: (50, 20, 60), # Landslide 2  
    9: (60, 50, 20), # Spring 2  
    10: (200, 150, 80), # Spring 3  
    11: (100, 200, 150), # Minepit 2 
    12: (12, 52, 83), # Spring B2
    13: (123, 232, 23), # Spring B2 
}

if CALCULATE_METRICS:
    CLASS_THRESHOLDS = {i: 0.25 for i in CLASS_NAMES.keys()}
else:
    CLASS_THRESHOLDS = {
        0: 0.6,  # Landslide 1
        1: 0.8,  # Strike
        2: 0.8,  # Spring 1
        3: 0.8,  # Minepit 1
        4: 0.8,  # Hillside
        5: 0.7,  # Feuchte
        6: 0.7,  # Torf
        7: 0.92,  # Bergsturz
        8: 0.8,  # Landslide 2
        9: 0.7,  # Spring 2
        10: 0.7,  # Spring 3
        11: 0.6,  # Minepit 2
        12: 0.05,  # Spring B2
        13: 0.05,  # Hillside B2
    }

EXCLUDED_CLASSES = {12, 13}
all_dets_per_image = {}

INPUT_DIR = "Input"
OUTPUT_DIR = "Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_DUAL_GPU = False  # True for dual GPU, False for single GPU

# ===== Helper Functions =====
def _poly_to_aabb_xyxy(box8):
    """box8 = [x1,y1,...,x4,y4]  ->  [xmin, ymin, xmax, ymax]"""
    xs = box8[0::2]; ys = box8[1::2]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

def _iou_aabb(b1, b2):
    x1,y1,x2,y2 = b1; X1,Y1,X2,Y2 = b2
    ix1 = max(x1, X1); iy1 = max(y1, Y1)
    ix2 = min(x2, X2); iy2 = min(y2, Y2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    a1 = max(0.0, (x2-x1)) * max(0.0, (y2-y1))
    a2 = max(0.0, (X2-X1)) * max(0.0, (Y2-Y1))
    return float(inter / (a1 + a2 - inter + 1e-9))

def polygon_iou(box1, box2):
    poly1 = Polygon([(box1[i], box1[i+1]) for i in range(0, 8, 2)])
    poly2 = Polygon([(box2[i], box2[i+1]) for i in range(0, 8, 2)])
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - inter
    return inter / union if union > 0 else 0.0

def eval_iou(box8_pred, box8_gt):
    if EVAL_GEOM == "aabb":
        return _iou_aabb(_poly_to_aabb_xyxy(box8_pred),
                         _poly_to_aabb_xyxy(box8_gt))
    else:
        return polygon_iou(box8_pred, box8_gt)

def _center_of_poly8(poly8):
    # poly8: [x1,y1,x2,y2,x3,y3,x4,y4]
    xs = poly8[0::2]; ys = poly8[1::2]
    return (sum(xs) / 4.0, sum(ys) / 4.0)

def merge_detections(detections, iou_threshold=0.5, excluse_check=True):
    """
    Merge overlapping detections while considering confidence and class types.
    detections: [(x1..y4, cls, conf), ...]
    """
    if not detections:
        return []

    detections.sort(key=lambda x: x[9], reverse=True)
    merged = []

    excluded_boxes = [det[:11] for det in detections if det[8] in EXCLUDED_CLASSES]

    for det1 in detections:
        box1, cls1, conf1 = det1[:8], det1[8], det1[9]
        if cls1 in EXCLUDED_CLASSES:
            continue

        keep = True
 
        if excluse_check:
            for det_excl in excluded_boxes:
                excl_box, excl_cls, excl_conf = det_excl[:8], det_excl[8], det_excl[9]
                iou = eval_iou(box1, excl_box)
                if iou > 0.3:
                    if conf1 > 0.85 or excl_conf < 0.5:
                        continue
                    else:
                        keep = False
                        break

        for det2 in merged:
            box2, cls2 = det2[:8], det2[8]
            if cls1 == cls2 and eval_iou(box1, box2) >= iou_threshold:
                keep = False
                break

        if keep:
            merged.append(det1)

    return merged

def process_image(image_path, models, gpu_id):
    image = cv2.imread(image_path)
    all_detections = []
    for model_info, model in zip(TILE_MODELS, models):
        dets = run_tiled_detection(model, image, model_info["tile_size"], model_info["overlap"])
        all_detections.extend(dets)

    merged = merge_detections(all_detections, IOU_THRESHOLD, False)
    result_img = image.copy()
    rows = []
    for x1,y1,x2,y2,x3,y3,x4,y4,cls,conf in merged:
        color = CLASS_COLORS.get(cls, (0,255,255))
        label = CLASS_NAMES.get(cls, f"Class{cls}")
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
        cv2.polylines(result_img, [pts], True, color, 2)
        text_x = int(round(min(x1, x2, x3, x4)))
        text_y = int(round(min(y1, y2, y3, y4) - 5))     
        h, w = result_img.shape[:2]
        text_x = max(0, min(text_x, w - 1))
        text_y = max(0, min(text_y, h - 1))
        label_str = str(label)
        conf_val = float(conf)
        
        cv2.putText(
            result_img,
            f"{label_str} {conf_val:.2f}",
            (text_x, text_y),                
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

        rows.append([label,x1,y1,x2,y2,x3,y3,x4,y4,conf])

    out_img_path = os.path.join(OUTPUT_DIR, f"gpu{gpu_id}_" + os.path.basename(image_path))
    excel_path = os.path.splitext(out_img_path)[0] + ".xlsx"
    cv2.imwrite(out_img_path, result_img)
    pd.DataFrame(rows, columns=["Class","X1","Y1","X2","Y2","X3","Y3","X4","Y4","Conf"]).to_excel(excel_path, index=False)
    print(f"[GPU{gpu_id}] Saved: {out_img_path} and {excel_path}")
    
    all_dets_per_image[image_path] = merged
    
# ===== Evaluation utils =====

def _label_path_for_image(image_path):
    base = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    cand1 = os.path.join(os.path.dirname(image_path), base)
    if os.path.exists(cand1):
        return cand1
    labels_dir = os.path.join(os.path.dirname(image_path), "Labels")
    cand2 = os.path.join(labels_dir, base)
    if os.path.exists(cand2):
        return cand2
    return None  # same logic as YOLO script

def _load_gt_as_pixels(image_path):
    lp = _label_path_for_image(image_path)
    gts = []
    if lp is None or not os.path.exists(lp):
        return gts
    img = cv2.imread(image_path)
    if img is None:
        return gts
    h, w = img.shape[:2]
    with open(lp, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            cls_id = int(parts[0])
            vals = list(map(float, parts[1:]))
            pts_pix = [(vals[i]*w, vals[i+1]*h) for i in range(0, 8, 2)]
            gts.append({"cls": cls_id, "pts": pts_pix})
    return gts  # :contentReference[oaicite:2]{index=2}

def _match_dets_to_gts_pixel(dets, gts, iou_thr=0.5):
    used = [False]*len(gts)
    tp = 0
    for det in dets:
        box1 = det[:8]
        cls1 = int(det[8])
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if used[j] or cls1 != g["cls"]:
                continue
            box2 = [coord for pt in g["pts"] for coord in pt]
            iou = eval_iou(box1, box2)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            used[best_j] = True
            tp += 1
    fp = len(dets) - tp
    fn = used.count(False)
    return tp, fp, fn  # :contentReference[oaicite:3]{index=3}

def _prec_rec_f1(tp, fp, fn):
    P = tp / (tp + fp + 1e-9)
    R = tp / (tp + fn + 1e-9)
    F1 = 2 * P * R / (P + R + 1e-9)
    return P, R, F1  # :contentReference[oaicite:4]{index=4}

def _evaluate_dataset(all_images, conf_thr, iou_thr):
    tot_tp = tot_fp = tot_fn = 0
    for img_path in all_images:
        dets_all = all_dets_per_image.get(img_path, [])
        filtered = [d for d in dets_all if (d[9] >= conf_thr and int(d[8]) not in EXCLUDED_CLASSES)]
        gts = [g for g in _load_gt_as_pixels(img_path) if g["cls"] not in EXCLUDED_CLASSES]
        tp, fp, fn = _match_dets_to_gts_pixel(filtered, gts, iou_thr=iou_thr)
        tot_tp += tp; tot_fp += fp; tot_fn += fn
    return _prec_rec_f1(tot_tp, tot_fp, tot_fn)

def evaluate_center_hit(all_images, conf_thr=0.5):
    """
    Center-Hit: a detection is TP if its center lies inside a GT polygon of the same class.
    EXCLUDED_CLASSES are ignored (both for GT and dets), just like in your other metrics.
    """
    tp = fp = fn = 0

    for img_path in all_images:
        # Detections: apply conf threshold & exclude classes
        dets = [d for d in all_dets_per_image.get(img_path, [])
                if (d[9] >= conf_thr and int(d[8]) not in EXCLUDED_CLASSES)]

        # Ground truths: exclude classes
        gts = [g for g in _load_gt_as_pixels(img_path) if g["cls"] not in EXCLUDED_CLASSES]

        used = [False] * len(gts)  # each GT can be matched at most once

        for d in dets:
            cls = int(d[8])
            cx, cy = _center_of_poly8(d[:8])
            p_center = Point(cx, cy)

            matched = False
            for j, g in enumerate(gts):
                if used[j] or g["cls"] != cls:
                    continue
                poly = Polygon(g["pts"])
                if not poly.is_valid:
                    continue
                if poly.contains(p_center):
                    tp += 1
                    used[j] = True
                    matched = True
                    break

            if not matched:
                fp += 1

        # any GTs left unmatched are FNs
        fn += sum(1 for u in used if not u)

    P, R, F1 = _prec_rec_f1(tp, fp, fn)
    print(f"[Center-Hit @ conf≥{conf_thr:.2f}] P={P:.3f} R={R:.3f} F1={F1:.3f} (TP={tp}, FP={fp}, FN={fn})")
    return P, R, F1

def _find_best_conf_threshold(all_images, iou_thr=0.5):
    best = {"thr": 0.25, "P": 0.0, "R": 0.0, "F1": -1.0}
    for thr in np.linspace(0.05, 0.95, 19):
        P, R, F1 = _evaluate_dataset(all_images, conf_thr=float(thr), iou_thr=iou_thr)
        if F1 > best["F1"]:
            best = {"thr": float(thr), "P": float(P), "R": float(R), "F1": float(F1)}
    return best  # :contentReference[oaicite:6]{index=6}

def _classwise_report(all_images, conf_thr, iou_thr):
    """
    Build a per-class table (TP/FP/FN/P/R/F1) while ignoring EXCLUDED_CLASSES.
    Saves to fusion_classwise_metrics.xlsx
    """
    rows = []
    all_cids = set()

    # collect class ids from GT (preferred, since mAP and fairness rely on GT presence)
    for img_path in all_images:
        for g in _load_gt_as_pixels(img_path):
            if g["cls"] not in EXCLUDED_CLASSES:
                all_cids.add(int(g["cls"]))

    # also add any classes that appear only in detections (and are not excluded)
    for dets in all_dets_per_image.values():
        for d in dets:
            cid = int(d[8])
            if cid not in EXCLUDED_CLASSES:
                all_cids.add(cid)

    all_cids = sorted(all_cids)

    # compute per-class metrics
    for cid in all_cids:
        tp = fp = fn = 0
        for img_path in all_images:
            dets_all = all_dets_per_image.get(img_path, [])
            dets_c = [d for d in dets_all if (int(d[8]) == cid and d[9] >= conf_thr)]
            gts_c = [g for g in _load_gt_as_pixels(img_path) if g["cls"] == cid]
            tpp, fpp, fnn = _match_dets_to_gts_pixel(dets_c, gts_c, iou_thr=iou_thr)
            tp += tpp; fp += fpp; fn += fnn

        P, R, F1 = _prec_rec_f1(tp, fp, fn)
        cname = CLASS_NAMES.get(cid, str(cid))
        rows.append([cid, cname, tp, fp, fn, P, R, F1])

    if pd is None:
        print("\n[Classwise metrics]")
        for r in rows:
            print(f"cls={r[0]:>2} {r[1]:<20} TP={r[2]:<5} FP={r[3]:<5} FN={r[4]:<5} "
                  f"P={r[5]:.3f} R={r[6]:.3f} F1={r[7]:.3f}")
        return None
    else:
        df = pd.DataFrame(rows, columns=["cls_id","class","TP","FP","FN","Precision","Recall","F1"])
        try:
            save_dir = output_dir
        except NameError:
            save_dir = "."
        out_path = os.path.join(save_dir, "fusion_classwise_metrics.xlsx")
        df.to_excel(out_path, index=False)
        print(f"[Saved] {out_path}")
        return df

def _gt_class_ids(all_images):
    """Return sorted list of class ids present in GT, excluding EXCLUDED_CLASSES."""
    cids = set()
    for img_path in all_images:
        for g in _load_gt_as_pixels(img_path):
            if g["cls"] not in EXCLUDED_CLASSES:
                cids.add(int(g["cls"]))
    return sorted(cids)

def gather_detections_and_gts(all_images, cls_id):
    """
    Build detection list and GT dict for a given class id.
    - dets: [{'image_id', 'score', 'bbox(8)'}], sorted later by score
    - gts : {image_id: [bbox8, bbox8, ...]}
    """
    if cls_id in EXCLUDED_CLASSES:
        return [], {}

    dets, gts = [], {}
    for img_path in all_images:
        # detections of this class (DO NOT filter by report threshold)
        img_dets_all = all_dets_per_image.get(img_path, [])
        for d in img_dets_all:
            if int(d[8]) == cls_id:
                dets.append({"image_id": img_path, "score": float(d[9]), "bbox": d[:8]})
        # GT of this class
        gt_boxes = [g for g in _load_gt_as_pixels(img_path) if g["cls"] == cls_id]
        gts[img_path] = [[c for pt in g["pts"] for c in pt] for g in gt_boxes]
    return dets, gts

def compute_pr_for_class(dets, gts, iou_thr=0.5):
    """
    dets: list of {'image_id','score','bbox'}
    gts : dict img -> [bbox8,...]
    returns: precision array, recall array, ap
    """
    npos = sum(len(v) for v in gts.values())
    if npos == 0:
        return np.array([0.0]), np.array([0.0]), None

    # sort by score desc
    dets_sorted = sorted(dets, key=lambda x: -x["score"])
    tp = np.zeros(len(dets_sorted), dtype=np.float32)
    fp = np.zeros(len(dets_sorted), dtype=np.float32)

    used = {img: np.zeros(len(gts_img), dtype=bool) for img, gts_img in gts.items()}

    for i, d in enumerate(dets_sorted):
        img = d["image_id"]
        gts_img = gts.get(img, [])
        used_flags = used.get(img, np.zeros(0, dtype=bool))

        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gts_img):
            if used_flags[j]:
                continue
            iou = eval_iou(d["bbox"], gb)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_j >= 0 and best_iou >= iou_thr:
            tp[i] = 1.0
            used_flags[best_j] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / (npos + 1e-9)
    prec = tp_cum / (tp_cum + fp_cum + 1e-9)

    # precision envelope (VOC/COCO style)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for k in range(len(mpre) - 1, 0, -1):
        mpre[k - 1] = max(mpre[k - 1], mpre[k])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return prec, rec, ap

def evaluate_map(all_images, iou_list=None):
    """
    Compute mAP@0.5 and mAP@[0.5:0.95] over NON-excluded classes only.
    Returns: dict with keys 'mAP@0.5', 'mAP@[0.5:0.95]', 'per_iou'
    """
    if iou_list is None:
        iou_list = IOU_LIST_FOR_MAP

    class_ids = _gt_class_ids(all_images)  # uses GT, already excludes
    per_iou_map = {}

    for iou in iou_list:
        ap_list = []
        for cid in class_ids:
            dets, gts = gather_detections_and_gts(all_images, cid)
            _, _, ap = compute_pr_for_class(dets, gts, iou_thr=iou)
            if ap is not None:
                ap_list.append(ap)
        per_iou_map[iou] = float(np.mean(ap_list)) if ap_list else 0.0

    map50 = per_iou_map.get(0.5, 0.0)
    map5095 = float(np.mean([per_iou_map[i] for i in iou_list])) if iou_list else 0.0
    return {"mAP@0.5": map50, "mAP@[0.5:0.95]": map5095, "per_iou": per_iou_map}

def run_fusion_eval(input_dir, iou_thr=0.5):
    all_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))]
    if not all_images:
        print("[Eval] No images found for evaluation.")
        return
    print(f"Config: {TILE_MODELS[0]['config']}")
    print(f"Tile models: {[m['tile_size'] for m in TILE_MODELS]}")
    print(f"Overlap: {[m['overlap'] for m in TILE_MODELS]}")
    best = _find_best_conf_threshold(all_images, iou_thr=iou_thr)
    print(f"[Fusion] Best confidence threshold (by F1): {best['thr']:.2f} | P={best['P']:.3f} R={best['R']:.3f} F1={best['F1']:.3f}")
    P, R, F1 = _evaluate_dataset(all_images, conf_thr=best['thr'], iou_thr=iou_thr)
    print(f"[Fusion @ {best['thr']:.2f}] Precision={P:.3f} | Recall={R:.3f} | F1={F1:.3f}")
    _classwise_report(all_images, conf_thr=best['thr'], iou_thr=iou_thr)
    # Center-Hit at the same chosen confidence threshold
    evaluate_center_hit(all_images, conf_thr=best['thr'])

    # compute and save mAPs
    maps = evaluate_map(all_images, iou_list=list(np.arange(0.5, 0.96, 0.05)))
    print("[mAP Results]")
    print(f"mAP@0.5 = {maps['mAP@0.5']:.4f}")
    print(f"mAP@[0.5:0.95] = {maps['mAP@[0.5:0.95]']:.4f}")
    
    maps_soft = evaluate_map(all_images, iou_list=[0.30, 0.40, 0.50, 0.60, 0.70])
    print("[mAP (soft) Results]")
    print(f"mAP@0.3 = {maps_soft['per_iou'][0.30]:.4f}")
    soft_avg = float(np.mean([maps_soft['per_iou'][i] for i in [0.30,0.40,0.50,0.60,0.70]]))
    print(f"mAP@[0.3:0.7] = {soft_avg:.4f}")

    try:
        save_dir = output_dir
    except NameError:
        save_dir = "."
    per_iou_rows = [{"iou": k, "mAP": v} for k, v in maps["per_iou"].items()]
    pd.DataFrame(per_iou_rows).to_excel(os.path.join(save_dir, "fusion_map_per_iou.xlsx"), index=False)
    print(f"[Saved] mAP per IoU table to {os.path.join(save_dir, 'fusion_map_per_iou.xlsx')}")
    
    soft_rows = [{"iou": k, "mAP": v} for k, v in maps_soft["per_iou"].items()]
    pd.DataFrame(soft_rows).to_excel(os.path.join(save_dir, "fusion_map_per_iou_soft.xlsx"), index=False)
    print(f"[Saved] soft mAP per IoU table to {os.path.join(save_dir, 'fusion_map_per_iou_soft.xlsx')}")

        
def run_tiled_detection(model, image, tile_size, overlap):
    """
    Runs tiled inference and always returns detections as:
    [x1,y1,x2,y2,x3,y3,x4,y4, class_id, score]
    Supports:
      - OBB: (cx, cy, w, h, angle, score)
    """
    h, w, _ = image.shape
    step = tile_size - overlap
    detections = []

    def obb_to_poly(cx, cy, ww, hh, theta):
        if abs(theta) > 2 * np.pi:
            theta = np.deg2rad(theta)
    
        c, s = np.cos(theta), np.sin(theta)
    
        dx, dy = ww / 2.0, hh / 2.0
        corners = np.array(
            [[-dx, -dy],
             [ dx, -dy],
             [ dx,  dy],
             [ -dx,  dy]], dtype=np.float32
        )

        R = np.array([[ c,  s],
                      [-s,  c]], dtype=np.float32)
    
        pts = (R @ corners.T).T + np.array([cx, cy], dtype=np.float32)
        return pts

    for y in range(0, h, step):
        for x in range(0, w, step):
            crop_detections = []
            crop = image[y:y + tile_size, x:x + tile_size]
            # if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
            #     continue

            results = inference_detector(model, crop)

            for cls_id, bboxes in enumerate(results):
                for bbox in bboxes:
                    score = float(bbox[-1])
                    if score < CLASS_THRESHOLDS.get(cls_id, 0.05):
                        continue

                    # Normalize to Python list
                    bb = bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)

                    # OBB: (cx, cy, w, h, angle, score)
                    cx, cy, ww, hh, theta = map(float, bb[:5])
                    pts = obb_to_poly(cx, cy, ww, hh, theta)
                    pts[:, 0] += x
                    pts[:, 1] += y
                    poly = pts.reshape(-1).tolist()
                    crop_detections.append(poly + [cls_id, score])
            detections.extend(merge_detections(crop_detections, IOU_THRESHOLD))

    return detections

# ===== Main Run =====
if __name__ == "__main__":
    image_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if USE_DUAL_GPU:
        half = len(image_files) // 2
        images_gpu0 = image_files[:half]
        images_gpu1 = image_files[half:]

        models_gpu0 = [init_detector(m["config"], m["checkpoint"], device='cuda:0') for m in TILE_MODELS]
        models_gpu1 = [init_detector(m["config"], m["checkpoint"], device='cuda:1') for m in TILE_MODELS]

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(lambda: [process_image(img, models_gpu0, 0) for img in images_gpu0])
            executor.submit(lambda: [process_image(img, models_gpu1, 1) for img in images_gpu1])

    else:
        try:
            models_gpu0 = [init_detector(m["config"], m["checkpoint"], device='cuda:0') for m in TILE_MODELS]
            for img in image_files:
                process_image(img, models_gpu0, 0)
        except:
            models_gpu0 = [init_detector(m["config"], m["checkpoint"], device='cpu') for m in TILE_MODELS]
            for img in image_files:
                process_image(img, models_gpu0, 0)
            
    if CALCULATE_METRICS:
        try:
            run_fusion_eval(INPUT_DIR, iou_thr=IOU_THRESHOLD)
        except Exception as e:
            print(f"[Eval] Skipped due to error: {e}")
