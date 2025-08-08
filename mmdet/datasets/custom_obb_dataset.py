from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

import numpy as np

def bbox_obb_to_hbb(obbs):
    # obbs: [N, 8]
    x = obbs[:, 0::2]
    y = obbs[:, 1::2]
    x_min = np.min(x, axis=1)
    y_min = np.min(y, axis=1)
    x_max = np.max(x, axis=1)
    y_max = np.max(y, axis=1)
    return np.stack([x_min, y_min, x_max, y_max], axis=1).astype(np.float32)


@DATASETS.register_module()
class CustomOBBDataset(CustomDataset):
    CLASSES = [
        "Landslide1", "Strike", "Spring1", "Minepit1", "Hillside", "Feuchte", "Torf",
        "Bergsturz", "Landslide2", "Spring2", "Spring3", "Minepit2", "SpringB2", "HillsideB2"
    ]

    def load_annotations(self, ann_file):
        import os
        data_infos = []
        for label_path in os.listdir(ann_file):
            if not label_path.endswith(".txt"):
                continue
            name = os.path.splitext(label_path)[0]
            img_path = os.path.join(self.img_prefix, name + ".jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.img_prefix, name + ".png")
            height, width = self._get_img_size(img_path)
            data_infos.append(dict(
                filename=os.path.basename(img_path),
                width=width,
                height=height,
                ann=dict(ann_file=os.path.join(ann_file, label_path))
            ))
        print(f"[CustomOBBDataset] Loaded {len(data_infos)} image-label pairs from: {ann_file}")

        return data_infos

    def get_ann_info(self, idx):
        import numpy as np
        ann_path = self.data_infos[idx]['ann']['ann_file']
        #print(f"[CustomOBBDataset] Loading annotation: {ann_path}")
    
        bboxes = []
        labels = []
    
        with open(ann_path) as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 9:
                    print(f"[Warning] Skipping invalid line {line_num} in {ann_path}: {line.strip()}")
                    continue
    
                try:
                    coords = list(map(float, parts[:8]))
                    class_name = parts[8]
                except Exception as e:
                    print(f"[Error] Failed to parse line {line_num} in {ann_path}: {e}")
                    continue
    
                if class_name not in self.CLASSES:
                    print(f"[Warning] Unknown class '{class_name}' in {ann_path}, skipping.")
                    continue
    
                bboxes.append(coords)
                labels.append(self.CLASSES.index(class_name))
    
        # If no valid bboxes found
        if len(bboxes) == 0:
            print(f"[CustomOBBDataset] No valid bboxes found in: {ann_path}")
            return dict(
                bboxes=np.zeros((0, 8), dtype=np.float32),
                labels=np.zeros((0,), dtype=np.int64)
            )

        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.ndim == 1:
            bboxes = bboxes[None, :]  # convert shape (8,) → (1, 8)
        elif bboxes.shape[1] != 8:
            print(f"[ERROR] Invalid bbox shape: {bboxes.shape} in {ann_path}")  
            
        return dict(
            gt_obboxes=bboxes,
            gt_labels=np.array(labels, dtype=np.int64),
            gt_bboxes=bbox_obb_to_hbb(bboxes),     
            bboxes=bbox_obb_to_hbb(bboxes),        
            labels=np.array(labels, dtype=np.int64)
        )


    def _get_img_size(self, img_path):
        import cv2
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        return h, w
