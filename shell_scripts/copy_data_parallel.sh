SOURCE_DIR="/export/home/data/ucsf/RTOG-9413/8.25.2020/95cpu_patches_256_patchsize_224_resize_0_overlap_2_level"
TARGET_DIR="/export/share/aesteva/data/ucsf/RTOG-9413/8.25.2020"

SOURCE_DIR="/export/home/code/metamind/precision_oncology/box_navigator/box_data"
TARGET_DIR="/export/medical_ai/ucsf"

patch_dir=$(basename ${SOURCE_DIR})
target_dir="${TARGET_DIR}/${patch_dir}"

echo "run: mkdir $target_dir"
mkdir -p ${target_dir}

echo "copying patches: \nfrom ${SOURCE_DIR} \nto   ${target_dir}"
ls $SOURCE_DIR | parallel -v -j95 rsync -raz --progress ${SOURCE_DIR}/{} ${target_dir}/{}
