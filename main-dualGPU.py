import os
import subprocess
import sys
from datetime import datetime

# ===== Stability & safety =====
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["PYTHONFAULTHANDLER"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

# ===== Config & workdir =====
CFG = sys.argv[1] if len(sys.argv) > 1 else "configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_dota10.py"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

WORKDIR_ROOT = "/home/DATA/RAMDISK/OBBDetection/work_dirs"
os.makedirs(WORKDIR_ROOT, exist_ok=True)
WORKDIR = os.path.join(WORKDIR_ROOT, f"{os.path.basename(CFG).split('.py')[0]}_{timestamp}")
os.makedirs(WORKDIR, exist_ok=True)

# ===== torchrun (2 GPUs) =====
cmd = [
    "torchrun",
    "--standalone",
    "--nproc_per_node=2",
    "tools/train.py", CFG,
    "--launcher", "pytorch",
    "--work-dir", WORKDIR,
    "--seed", "42", "--deterministic",
]

print("[INFO] Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
