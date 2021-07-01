"""Script to download the entire Box directory structure.

Skips anything that has been downloaded before.
Syncs to LOCAL_BOX_DIR
To obtain a developer token, navigate to 
https://salesforcecorp.app.box.com/developers/console/app/1366340/configuration
and select "Generate Developer Token", then copy-paste it below.

Exampe Usage:
    python download_box_data.py
"""

import box_auth
from box_auth import BoxNavigator


DEVELOPER_TOKEN_60MINS="uu4OyqV78GydCvVLAvzZvXh1kpkHeGnL"
LOCAL_BOX_DIR="/export/medical_ai/ucsf/box_data"

if __name__ == "__main__":
    bn = BoxNavigator(token=DEVELOPER_TOKEN_60MINS)
    bn.locally_recreate_filesystem_directory_structure(root_path=LOCAL_BOX_DIR)
    bn.maybe_download_filesystem(root_path=LOCAL_BOX_DIR)
