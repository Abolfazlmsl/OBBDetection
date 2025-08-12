#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 11:03:00 2025

@author: abolfazl
"""

import os
import subprocess
import sys
from datetime import datetime

# ===== GPU Mode Toggle =====
USE_DUAL_GPU = False  # Set to True for dual-GPU training, False for single-GPU

# ===== Resume Settings =====
RESUME_TRAINING = False  # If False, always start from scratch
RESUME_FOLDER = "faster_rcnn_orpn_r50_fpn_1x_dota10_20250811_111928"  # Folder name under work_dirs

# ===== Stability & Safety =====
if USE_DUAL_GPU:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

os.environ["PYTHONFAULTHANDLER"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# ===== Config =====
CFG = sys.argv[1] if len(sys.argv) > 1 else \
    "configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_dota10.py"

# ===== Workdir =====
WORKDIR_ROOT = "/home/DATA/RAMDISK/OBBDetection/work_dirs"
os.makedirs(WORKDIR_ROOT, exist_ok=True)

resume_from = None
if RESUME_TRAINING:
    target_dir = os.path.join(WORKDIR_ROOT, RESUME_FOLDER)
    if os.path.isdir(target_dir):
        ckpts = sorted(
            [f for f in os.listdir(target_dir) if f.endswith(".pth")],
            reverse=True
        )
        if ckpts:
            resume_from = os.path.join(target_dir, ckpts[0])
            WORKDIR = target_dir
            print(f"[INFO] Resuming from checkpoint: {resume_from}")
        else:
            print(f"[WARNING] No checkpoints found in {target_dir}. Starting fresh.")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            WORKDIR = os.path.join(WORKDIR_ROOT, f"{os.path.basename(CFG).split('.py')[0]}_{timestamp}")
            os.makedirs(WORKDIR, exist_ok=True)
    else:
        raise FileNotFoundError(f"[ERROR] Resume folder {RESUME_FOLDER} does not exist.")
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    WORKDIR = os.path.join(WORKDIR_ROOT, f"{os.path.basename(CFG).split('.py')[0]}_{timestamp}")
    os.makedirs(WORKDIR, exist_ok=True)

# ===== Command =====
if USE_DUAL_GPU:
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "tools/train.py", CFG,
        "--launcher", "pytorch",
        "--work-dir", WORKDIR,
        "--seed", "42", "--deterministic"
    ]
else:
    cmd = [
        sys.executable, "tools/train.py", CFG,
        "--launcher", "none",
        "--gpu-ids", "0",
        "--work-dir", WORKDIR,
        "--seed", "42", "--deterministic"
    ]

if resume_from:
    cmd.extend(["--resume-from", resume_from])

print("[INFO] Running:", " ".join(cmd))

try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"[ERROR] Training process failed with code {e.returncode}")
    sys.exit(1)
except RuntimeError as e:
    if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
        print("[FATAL] CUDA Out Of Memory detected! Stopping process...")
        sys.exit(1)
    else:
        raise