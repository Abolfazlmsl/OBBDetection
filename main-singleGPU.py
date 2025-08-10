#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 11:03:00 2025

@author: abolfazl
"""

# main.py (single-GPU, no DDP)
import os
import subprocess
import sys
from datetime import datetime

# --- Stability & safety settings ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["PYTHONFAULTHANDLER"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# --- Config and work directory ---
CFG = "configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_dota10.py"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

WORKDIR_ROOT = "/home/DATA/RAMDISK/OBBDetection/work_dirs"
WORKDIR = os.path.join(WORKDIR_ROOT, f"{os.path.basename(CFG).split('.py')[0]}_{timestamp}")
os.makedirs(WORKDIR, exist_ok=True)

# --- Command to run train.py on a single GPU ---
cmd = [
    sys.executable, "tools/train.py", CFG,
    "--launcher", "none",         # Important: disable DDP
    "--gpu-ids", "0",              
    "--work-dir", WORKDIR,
    "--seed", "42", "--deterministic"
]
print("[INFO] Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
