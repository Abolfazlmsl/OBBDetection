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
TILE_MODELS = [
    # {"config": "configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_tile128.py",
    #  "checkpoint": "best128.pth", "tile_size": 128, "overlap": 20},
    {"config": "configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_dota10.py",
     "checkpoint": "Checkpoints/best416.pth", "tile_size": 416, "overlap": 150}
]

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

def merge_detections(dets, iou_thr=0.5):
    if not dets:
        return []
    dets.sort(key=lambda x: x[9], reverse=True)
    merged = []
    for det1 in dets:
        b1, cls1 = det1[:8], det1[8]
        keep = True
        for det2 in merged:
            b2, cls2 = det2[:8], det2[8]
            if cls1 == cls2 and polygon_iou(b1, b2) >= iou_thr:
                keep = False
                break
        if keep:
            merged.append(det1)
    return merged

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

    merged = merge_detections(all_detections, IOU_THRESHOLD)
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
