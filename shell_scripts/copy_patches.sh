#!/bin/sh
source_dir="/export/home/data/ucsf/RTOG-9202/9.14.2020/95cpu_patches_256_patchsize_224_resize_0_overlap_3_level"
target_dir="/export/share/aesteva/data/ucsf/RTOG-9202/9.14.2020/95cpu_patches_256_patchsize_224_resize_0_overlap_3_level"

#ls ${source_dir}
ls ${source_dir} | parallel -v -j 95 rsync -raz --progress ${source_dir}/{} ${target_dir}/{}
