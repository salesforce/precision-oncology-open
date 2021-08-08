# Script to create symbolic links in a single folder that point to all the .svs. files
# scattered across all the sub-directories of an RTOG-XXXX study
# E.g.
# RTOG/9.14.2020/*.svs -> /export/medical_ai/ucsf/RTOG-9202/svs-links/*.svs
# RTOG/10.5.2020/*.svs -> /export/medical_ai/ucsf/RTOG-9202/svs-links/*.svs
#
# Usage:
#   ./links_box_svs.sh

/bin/bash ./link_box_svs_9202.sh
/bin/bash ./link_box_svs_9413.sh
/bin/bash ./link_box_svs_9408.sh
/bin/bash ./link_box_svs_9910.sh
/bin/bash ./link_box_svs_0126.sh
