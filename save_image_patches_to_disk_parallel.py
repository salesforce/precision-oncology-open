#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read SVS files from disk and save patches.
Patches per image are saved in separate folder.

Example usage:
    python save_image_patches_to_disk_parallel.py \
            --svs_dir=/export/home/data/ucsf/svs_test/ \
            --write_dir=/export/medical_ai/ucsf/tmp \
            --resize_size=224 \
            --patch_size=256 \
            --overlap=0 \
            --level=3
"""

#! /home/ubuntu/anaconda3/bin/python

import numpy as np
import os
import py_wsi
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate patches from svs files.')

    parser.add_argument("--svs_dir", type=str, default="/export/home/data/ucsf/svs_test", help="Directory of .svs files.")
    parser.add_argument("--write_dir", type=str, default="/export/medical_ai/ucsf/tmp", help="Directory to output patches.")
    parser.add_argument("--resize_size", type=int, default=224, help="Resize the patches to N x N pixels, for model training.")
    parser.add_argument("--patch_size", type=int, default=256, help="Crop patches of size N x N pixels, from the original svs")
    parser.add_argument("--overlap", type=int, default=0, help="The pixel overlap between patches.")
    parser.add_argument("--level", type=int, default=3, choices=[1,2,3], help="The zoom level. 3 is 10x zoom, 2 is 20x zoom, 1 is 40x zoom.")

    args = parser.parse_args()
    svs_dir = args.svs_dir
    write_dir = args.write_dir
    resize_size = args.resize_size
    patch_size = args.patch_size
    overlap = args.overlap
    level = args.level

    print("Parsing .svs files into patches \nsvs_dir={}\nwrite_dir={}\nresize_size={}\npatch_size={}\noverlap={}\nlevel={}".format(
        svs_dir, write_dir, resize_size, patch_size, overlap, level
    ))

    db_location = os.path.join(write_dir, "patches_" + str(patch_size) + "_patchsize_" + str(resize_size) + "_resize_" + str(overlap) + "_overlap_" + str(level) + "_level") + "/"
    db_name = "patches_" + str(patch_size) + "_patchsize_" + str(resize_size) + "_resize_" + str(overlap) + "_overlap_" + str(level) + "_level"

#    turtle = py_wsi.Turtle(svs_dir, db_location, db_name, label_map={}, storage_type='disk')
    turtle = py_wsi.TurtleParallel(svs_dir, db_location, db_name, label_map={}, storage_type='disk', n_jobs=-1)
    turtle.sample_and_store_patches(patch_size, level, overlap, load_xml=False, limit_bounds=False, normalize=False)

    print("Parsed .svs files into patches \nsvs_dir={}\nwrite_dir={}\nresize_size={}\npatch_size={}\noverlap={}\nlevel={}".format(
        svs_dir, write_dir, resize_size, patch_size, overlap, level
    ))

# ----
#    file_dir = '/export/home/data/ucsf/svs_test/'
#    file_dir = '/export/home/code/metamind/precision_oncology/box_navigator/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202/9.14.2020'
#    file_dir = '/export/home/data/ucsf/RTOG-9202/9.14.2020/'
#    #file_dir = '/export/home/data/ucsf/RTOG-9413/8.25.2020/'
#    #file_dir = '/export/home/nnaik/data/usc/svs/'
#    #file_dir = '/home/nnaik/research/2019_usc_data/TCGA_BREAST_TEMP_SAMPLE/svs/'
#    resize_size = 224
#    patch_size = 256
#    overlap = 0
#    #overlap = np.round(patch_size/4)
#    #overlap = overlap.astype(int)
#    level = 2
#
#    db_location = os.path.join(file_dir, "patches_" + str(patch_size) + "_patchsize_" + str(resize_size) + "_resize_" + str(overlap) + "_overlap_" + str(level) + "_level") + "/"
#    db_name = "patches_" + str(patch_size) + "_patchsize_" + str(resize_size) + "_resize_" + str(overlap) + "_overlap_" + str(level) + "_level"
#
#    #turtle = py_wsi.Turtle(file_dir, db_location, db_name, label_map={}, storage_type='disk')
#    turtle = py_wsi.TurtleParallel(file_dir, db_location, db_name, label_map={}, storage_type='disk', n_jobs=-1)
#    turtle.sample_and_store_patches(patch_size, level, overlap, load_xml=False, limit_bounds=False, normalize=True)
#
