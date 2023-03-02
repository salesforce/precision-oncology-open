#!/bin/sh
python save_image_patches_to_disk_parallel.py \
        --svs_dir=/export/medical_ai/ucsf/RTOG-9202/svs \
        --write_dir=/export/medical_ai/ucsf/RTOG-9202/patches \
        --resize_size=224 \
        --patch_size=256 \
        --overlap=0 \
        --level=2
