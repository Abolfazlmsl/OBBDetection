#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 13:23:33 2025

@author: abolfazl
"""

import subprocess

split_name = "test" # train or val or test
sizes = 416         
gaps = 150 
save_dir = f"/home/DATA/RAMDISK/OBBDetection/mmdata/GeoMap/{split_name}_split"
# os.makedirs(save_dir, exist_ok=True)

cmd = [
    "python", "img_split_dota.py",
    "--load_type", "dota",
    "--img_dirs", f"mmdata/GeoMap/{split_name}/images",
    "--ann_dirs", f"mmdata/GeoMap/{split_name}/labelTxt",
    "--classes", "Landslide1|Strike|Spring1|Minepit1|Hillside|Feuchte|Torf|Bergsturz|Landslide2|Spring2|Spring3|Minepit2|SpringB2|HillsideB2",
    "--save_dir", save_dir,
    "--sizes", str(sizes),
    "--gaps", str(gaps),
    "--img_rate_thr", "0.6",
    "--iof_thr", "0.3",
    "--save_ext", ".png",
    "--nproc", "8",
    "--generate_txt"
]

subprocess.run(cmd, check=True)

