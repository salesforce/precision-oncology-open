# Box Navigator
Repo to download Box data from UCSF to local.

## Usage:
Install dependencies
```
pip install boxsdk
pip install tqdm
```
Download data. Note that you'll need to refresh the developer token in
the script below, every hour. The script will pick up from wherever it
left off.
```
python download_box_data.py
```
