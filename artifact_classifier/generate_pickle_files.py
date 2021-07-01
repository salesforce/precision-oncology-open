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


study_name = 'RTOG-9408'
output_path = os.path.join('tissue_vs_non_pickles', study_name)
model_path = '/export/medical_ai/ucsf/whitespace_classification/models/tissue_vs_non_2.pth'
path = '/export/medical_ai/ucsf/{}/patches/patches_256_patchsize_224_resize_0_overlap_3_level'.format(study_name)
folders = os.listdir(path)
print(len(folders))
folders = list(set(folders) - set([f.split('.')[0] for f in os.listdir(output_path)]))
print(len(folders))

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
        df.to_pickle(os.path.join(output_path, '{}.pkl'.format(folder)))
    tqdm._instances.clear()

print('Empty Folders in {}'.format(study_name), empty_wsi)

