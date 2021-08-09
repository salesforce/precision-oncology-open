import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from natsort import natsorted
import shutil
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--study', type=str, default=None, required=True,
                    help='RTOG-XXXX - study to process tiles of')
parser.add_argument('--v3', action='store_true', default=False, 
                    help="If a patch is classified as non tissue, but all neighbours are classified as tissue, this patch will also be tissue.")
parser.add_argument('--input_dir', type=str, default=None, required=False,
                    help='folder containing folders with tiles. When not provided, /export/medical_ai/ucsf/[study] will be used')
parser.add_argument('--output_dir', type=str, default=None, required=False,
                    help="output folder, when not provided /export/medical_ai/ucsf/tissue_vs_non_pkl/v2-or-v3/[study] will be used")

def get_neighbours(x, y):
    """Get neighbours for an x, y location
    Args:
        x (int): x coord
        y (int): y coord
    Returns:
        List: list of tuples with x,y coords
    """
    neighbours = []
    neighbours += [(x-1, y)]
    neighbours += [(x, y-1)]
    neighbours += [(x, y+1)]
    neighbours += [(x+1, y)]
    
    neighbours += [(x+1, y+1)]
    neighbours += [(x+1, y-1)]
    neighbours += [(x-1, y+1)]
    neighbours += [(x-1, y-1)]

    return neighbours

def check_label(x, y, label_df):
    neighbours = get_neighbours(x,y)
    neighbour_labels = [label_df[x] for x in neighbours]
    if all(neighbour_labels):
        return True
    return False

def fix_false(df):
    # df = pd.read_pickle(tissue_df)
    df["x"] = df.path.apply(lambda x: int(x.split("/")[-1].split("_")[1]))
    df["y"] = df.path.apply(lambda x: int(x.split("/")[-1].split("_")[2]))    
    label_df = defaultdict(bool)
    for x, y, label in zip(df.x, df.y, df.tissue_vs_non):
        label_df[(x,y)] = label
        
    fixed_labels = []
    for x, y, curr_label in zip(df.x, df.y, df.tissue_vs_non):
        if curr_label:
            fixed_labels += [True]
        else:
            fixed_labels += [check_label(x, y, label_df)]
    df["tissue_vs_non"] = fixed_labels
    return df

class CustomDataSetLoader(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, self.total_imgs[idx]


def create_folder(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

args = parser.parse_args()
study_name = args.study
version = "v3" if args.v3 else "v2"
if args.output_dir == None:
    output_path = "/export/medical_ai/ucsf/tissue_vs_non_pkl/{}/{}".format(version, study_name)
else:
    output_path = args.output_dir

if args.input_dir == None:
    path = '/export/medical_ai/ucsf/{}/patches/patches_256_patchsize_224_resize_0_overlap_3_level'.format(study_name)
else:
    path = args.input_dir
# output_path = os.path.join('tissue_vs_non_pickles', study_name)
model_path = '/export/medical_ai/ucsf/whitespace_classification/models/tissue_vs_non_2.pth'
# path = '/export/medical_ai/ucsf/{}/patches/patches_256_patchsize_224_resize_0_overlap_3_level'.format(study_name)
folders = os.listdir(path)
print("Total slides found: ", len(folders))
folders = list(set(folders) - set([f.split('.')[0] for f in os.listdir(output_path)]))
print("Unprocessed slides: ", len(folders))

empty_wsi = []
# create_folder(output_path)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)

for folder in tqdm(folders):
    folder_path = os.path.join(path, folder)
    files = os.listdir(folder_path)
    if not files:
        empty_wsi.append(folder)
        continue

    image_datasets = CustomDataSetLoader(folder_path, data_transforms['val'])
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=256, shuffle=True, num_workers=8)

    pred_dict = {}

    with torch.no_grad():
        for i, (inputs, image_name) in enumerate(dataloaders):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                image_path = os.path.join(path, folder, image_name[j])
                pred_dict[image_path] = int(preds[j]) == 1
        df = pd.DataFrame(list(pred_dict.items()), columns=['path', 'tissue_vs_non'])
        if args.v3:
            df = fix_false(df)
        df.to_pickle(os.path.join(output_path, '{}.pkl'.format(folder)))
    tqdm._instances.clear()

print('Empty Folders in {}'.format(study_name), empty_wsi)
