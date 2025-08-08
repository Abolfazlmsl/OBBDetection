import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import subprocess
subprocess.run([
    "python", "-m", "torch.distributed.launch",
    "--nproc_per_node=2",
    "tools/train.py",
    "configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_dota10.py",
    "--launcher", "pytorch"
])
