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
from shapely.geometry import Polygon
from mmdet.apis import init_detector, inference_detector
from concurrent.futures import ThreadPoolExecutor

# ===== User Settings =====
CALCULATE_METRICS = True

TILE_MODELS = [
    {"config": "configs/obb/oriented_rcnn/faster_rcnn_orpn_r101_fpn_1x_dota10.py",
      "checkpoint": "Checkpoints/best128.pth", "tile_size": 128, "overlap": 50}
    # {"config": "configs/obb/oriented_rcnn/faster_rcnn_orpn_r101_fpn_1x_dota10.py",
    #  "checkpoint": "Checkpoints/best416.pth", "tile_size": 416, "overlap": 150}
    # {"config": "configs/obb/retinanet_obb/retinanet_obb_r101_fpn_2x_dota10.py",
    #  "checkpoint": "Checkpoints/best416-ret.pth", "tile_size": 416, "overlap": 150}
    # {"config": "configs/obb/faster_rcnn_obb/faster_rcnn_obb_r101_fpn_1x_dota10.py",
    #  "checkpoint": "Checkpoints/best416-cnn.pth", "tile_size": 416, "overlap": 150}
]


iou_thr = 0.25
IOU_THRESHOLD = 0.2

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
    CLASS_THRESHOLDS = {
        0: 0.0,  # Landslide 1
        1: 0.0,  # Strike
        2: 0.0,  # Spring 1
        3: 0.0,  # Minepit 1
        4: 0.0,  # Hillside
        5: 0.0,  # Feuchte
        6: 0.0,  # Torf
        7: 0.0,  # Bergsturz
        8: 0.0,  # Landslide 2
        9: 0.0,  # Spring 2
        10: 0.0,  # Spring 3
        11: 0.0,  # Minepit 2
        12: 0.0,  # Spring B2
        13: 0.0,  # Hillside B2
    }
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

EXCLUDED_CLASSES = {} if CALCULATE_METRICS else {12, 13}
all_dets_per_image = {}

INPUT_DIR = "Input"
OUTPUT_DIR = "Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_DUAL_GPU = False  # True for dual GPU, False for single GPU

# ===== Helper Functions =====
def polygon_iou(box1, box2):
    poly1 = Polygon([(box1[i], box1[i+1]) for i in range(0, 8, 2)])
    poly2 = Polygon([(box2[i], box2[i+1]) for i in range(0, 8, 2)])
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - inter
    return inter / union if union > 0 else 0.0


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
                iou = polygon_iou(box1, excl_box)
                if iou > 0.3:
                    if conf1 > 0.85 or excl_conf < 0.5:
                        continue
                    else:
                        keep = False
                        break

        for det2 in merged:
            box2, cls2 = det2[:8], det2[8]
            if cls1 == cls2 and polygon_iou(box1, box2) >= iou_threshold:
                keep = False
                break

        if keep:
            merged.append(det1)

    return merged

# ===== Evaluation utils (YOLO-like) =====

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
            iou = polygon_iou(box1, box2)
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

def _evaluate_dataset(all_images, conf_thr=0.25, iou_thr=0.5):
    tot_tp = tot_fp = tot_fn = 0
    for img_path in all_images:
        dets_all = all_dets_per_image.get(img_path, [])
        filtered = [d for d in dets_all if d[9] >= conf_thr]
        gts = _load_gt_as_pixels(img_path)
        tp, fp, fn = _match_dets_to_gts_pixel(filtered, gts, iou_thr=iou_thr)
        tot_tp += tp; tot_fp += fp; tot_fn += fn
    return _prec_rec_f1(tot_tp, tot_fp, tot_fn)  # :contentReference[oaicite:5]{index=5}

def _find_best_conf_threshold(all_images, iou_thr=0.5):
    best = {"thr": 0.25, "P": 0.0, "R": 0.0, "F1": -1.0}
    for thr in np.linspace(0.05, 0.95, 19):
        P, R, F1 = _evaluate_dataset(all_images, conf_thr=float(thr), iou_thr=iou_thr)
        if F1 > best["F1"]:
            best = {"thr": float(thr), "P": float(P), "R": float(R), "F1": float(F1)}
    return best  # :contentReference[oaicite:6]{index=6}

def _classwise_report(all_images, conf_thr=0.25, iou_thr=0.5):
    rows = []
    all_cids = set()
    for dets in all_dets_per_image.values():
        for d in dets:
            all_cids.add(int(d[8]))
    all_cids = sorted(all_cids)

    for cid in all_cids:
        tp=fp=fn=0
        for img_path in all_images:
            dets_all = all_dets_per_image.get(img_path, [])
            dets_c = [d for d in dets_all if (int(d[8])==cid and d[9]>=conf_thr)]
            gts = _load_gt_as_pixels(img_path)
            gts_c = [g for g in gts if g["cls"]==cid]
            tpp,fpp,fnn = _match_dets_to_gts_pixel(dets_c, gts_c, iou_thr=iou_thr)
            tp += tpp; fp += fpp; fn += fnn
        P,R,F1 = _prec_rec_f1(tp,fp,fn)
        cname = CLASS_NAMES.get(cid, str(cid))
        rows.append([cid, cname, tp, fp, fn, P, R, F1])

    df = pd.DataFrame(rows, columns=["cls_id","class","TP","FP","FN","Precision","Recall","F1"])
    out_path = os.path.join(OUTPUT_DIR, "fusion_classwise_metrics.xlsx")
    df.to_excel(out_path, index=False)
    print(f"[Saved] {out_path}")
    return df  # :contentReference[oaicite:7]{index=7}

def run_fusion_eval(input_dir, iou_thr=0.5):
    all_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))]
    if not all_images:
        print("[Eval] No images found for evaluation.")
        return
    print(f"Tile models: {[m['tile_size'] for m in TILE_MODELS]}")
    best = _find_best_conf_threshold(all_images, iou_thr=iou_thr)
    print(f"[Fusion] Best confidence threshold (by F1): {best['thr']:.2f} | P={best['P']:.3f} R={best['R']:.3f} F1={best['F1']:.3f}")
    P, R, F1 = _evaluate_dataset(all_images, conf_thr=best['thr'], iou_thr=iou_thr)
    print(f"[Fusion @ {best['thr']:.2f}] Precision={P:.3f} | Recall={R:.3f} | F1={F1:.3f}")
    _classwise_report(all_images, conf_thr=best['thr'], iou_thr=iou_thr)  # :contentReference[oaicite:8]{index=8}


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
            crop = image[y:y + tile_size, x:x + tile_size]
            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                continue

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
                    detections.append(poly + [cls_id, score])


    return detections

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
        models_gpu0 = [init_detector(m["config"], m["checkpoint"], device='cuda:0') for m in TILE_MODELS]
        for img in image_files:
            process_image(img, models_gpu0, 0)
            
    if CALCULATE_METRICS:
        try:
            run_fusion_eval(INPUT_DIR, iou_thr=iou_thr)
        except Exception as e:
            print(f"[Eval] Skipped due to error: {e}")
