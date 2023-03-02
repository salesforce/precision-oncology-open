#!/bin/sh
python save_image_patches_to_disk_parallel.py \
        --svs_dir=/export/medical_ai/ucsf/svs_test/ \
        --write_dir=/export/medical_ai/ucsf/svs_test/ \
        --resize_size=224 \
        --patch_size=256 \
        --overlap=0 \
        --level=2
