#!/bin/sh
#
# Note that the folders below need the '/' appended.

python save_image_patches_to_disk_parallel.py \
        --svs_dir=/export/medical_ai/ucsf/RTOG-9910/svs/ \
        --write_dir=/export/medical_ai/ucsf/RTOG-9910/patches/ \
        --resize_size=224 \
        --patch_size=256 \
        --overlap=0 \
        --level=3
