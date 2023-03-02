# Script to create symbolic links in a single folder that point to all the .svs. files
# scattered across all the sub-directories of an RTOG-XXXX study
# E.g.
# RTOG/9.14.2020/*.svs -> /export/medical_ai/ucsf/RTOG-9202/svs-links/*.svs
# RTOG/10.5.2020/*.svs -> /export/medical_ai/ucsf/RTOG-9202/svs-links/*.svs
#
# Usage: Select from the source-target pairs, below, then run:
#   ./links_box_svs_auto.sh
# 
# Assumes each RTOG folder contains a set of directories (typically named by date of upload)
# that contain the .svs files.

# RTOG-9202
rtog_dir="/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/RTOG-9202"
target_dir="/export/medical_ai/ucsf/RTOG-9202/svs"

for source_dir in "$rtog_dir"/*;
do
  for i in "$source_dir"/*.svs;
  do
    echo "Linking $i to $target_dir/$(basename "$i")";
    ln -s "$i" "$target_dir"/$(basename "$i");
  done
done
