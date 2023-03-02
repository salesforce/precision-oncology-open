"""
dataset.py : Script / Library for tiling up histopathological datasets.
Consists of 2 classes :
    CutOut - Randomly mask out one or more patches from an image
    PatchDataset - load square patches of a specifc size

Example usage :
1. Tile up histopathology slides (tissue data)
    Edit root_dir to enter the path of histopathology slides in the main function
    This will output csv files for training and validation with IDs of tiled up images

    python dataset.py

2. Load tiled data to train/test model
    Used as a library in run.py to load the tiled tissue images

"""


from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import csv
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from skimage import color
from scipy.ndimage.interpolation import rotate
import pandas as pd

MEAN_color = [168.29286065, 151.07900252, 184.56781278]
STD_color = [49.64751763, 50.96645999, 39.5275019]


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class PatchDataset(Dataset):
    """
    PatchDataset : load square patches of a specifc size
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """

    def __init__(self,  root_dir, csv_file, image_size, num_samples, transform=True, grayscale=False, resize=True,
                 mean_regularize=-1, cutout=0, otsu=False):
        df_labels = pd.read_csv(os.path.join(root_dir, csv_file), delimiter=',')
        self.labels = df_labels['label'].tolist()
        self.root_dir = root_dir
        self.transform = transform
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self.dir_names = df_labels['image'].tolist()
        self.grayscale = grayscale
        self.num_samples = num_samples
        self.image_size = image_size
        self._resize = transforms.Resize(image_size)
        self.resize = resize
        self.mean_regularize = mean_regularize
        self.cutout = cutout
        self.otsu = otsu
        self.hard_mine_patches = [[] for i in range(len(self.dir_names))]
        self.otsu = otsu

    def calc_lab_mean_sd(self, targetim):
        """ Calculates mean standard deviation and mean
        Args:
            targetim (PIL Image) : target image
        Returns:
            d (dict) :
                'mean' : mean of image pixels
                'sd' : standard deviation of image pixels
                'im_lab' : RGB to lab color space conversion.
        """
        lab = color.rgb2lab(targetim[:, :, :3])
        m = np.mean(lab, axis=(0, 1))
        sd = np.sqrt(np.var(lab - m, axis=(0, 1)))

        return {"mean": m, "sd": sd, "im_lab": lab}

    def remove_hue(self, image):
        """ Function to remove hue
        Args:
            image (PIL image) : input image
        Returns:
            image_h (PIL image) : output image with hue correction
        """
        image = image.astype('float32')
        if image.shape[-1] != 3:
            image = image.transpose(1,2,0)
        assert image.shape[-1] == 3
        hsv = rgb_to_hsv(image)
        hsv[..., 2] = 0
        image_h = hsv_to_rgb(hsv)
        return image_h

    def color_transform_image(self, sourceim, target_vector):
        """ Calculates mean standard deviation and mean
        Args:
            targetim (PIL Image) : target image
        Returns:
            d (dict) :
                'mean' : mean of image pixels
                'sd' : standard deviation of image pixels
                'im_lab' : RGB to lab color space conversion.
        """
        source = self.calc_lab_mean_sd(sourceim)

        source["im_lab"] -= source["mean"]
        source["im_lab"] /= source["sd"]

        source["im_lab"] *= target_vector["sd"]
        source["im_lab"] += target_vector["mean"]

        return np.clip(color.lab2rgb(source["im_lab"]), 0, 1)

    def __len__(self):
        """ Returns length of labels in PatchDataset"""
        return len(self.labels)
 
    def __getitem__(self, index):
        """
        this function will load all patches from given batch.
        Args :
            index (int) : index of train/test patches
        Returns :
            patch_bag (list(tensor)) : list of tile image tensors
            label_bag (list) : list of bag labels for each corresponding patch
            files (list) : Returns list of file names corresponding to the histopathology slides
        """
        #dir_name = self.dir_names[index]
        #current_dir = os.path.join(self.root_dir, str(dir_name))

        current_dir = self.dir_names[index]
        #print('current_dir', current_dir)

        if self.otsu:
            with open(os.path.join(self.root_dir, 'otsu_random_2000', dir_name+'.txt_top_2000.txt')) as f:
                files = f.read().splitlines()
        else:
            files = os.listdir(current_dir)
        files = os.listdir(current_dir)

        valid_files = []
        for f in files:
            file_name = f.split('.')[0]
            if file_name.endswith('ws'):
                continue
            else:
                valid_files.append(f)

        files = valid_files

        #print('Number of valid patches : ', len(valid_files))

        # Filter all files for whitespace
        np.random.seed()
        n_samples = np.min([self.num_samples, len(files)])
        files = np.random.choice(files, n_samples, replace=False).tolist()

        for i in range(len(self.hard_mine_patches[index])):
            files[i] = self.hard_mine_patches[i][0]

        patch_bag = []
        label_bag = [1]*len(files)
        current_label = self.labels[index]
        label_bag = [x*current_label for x in label_bag]

        for file in files:
            if np.random.rand() > self.mean_regularize:
                try:
                    img = Image.open(os.path.join(current_dir,file))
                    if self.resize:
                        img = self._resize(img)
                    if self.transform:
                        # basic color jitter and rotations, flips. Work best
                        if not self.grayscale:
                           img = self._color_jitter(img)
                        if np.random.rand() > 0.5:
                            img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        num_rotate = np.random.randint(0, 4)
                        img = img.rotate(90 * num_rotate)

                    if self.grayscale:
                        img = np.expand_dims(np.asarray(img.convert('L'), dtype='float32'), 0)
                    else:
                        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

                    img = (img - 128.0)/128.0
                    img = torch.from_numpy(np.asarray(img))  # create the image tensor
                except:
                    img = torch.ones((3, self.image_size, self.image_size))
                    for c, ch in enumerate([0.87316266, 0.79902739, 0.84941472]):
                        img[c, :, :] = img[c, :, :] * ch
                    all_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04), transforms.ToTensor()])
                    img = all_transforms(img)
            else:
                img = torch.ones((3, self.image_size, self.image_size))
                for c, ch in enumerate([0.87316266, 0.79902739, 0.84941472]):
                    img[c, :, :] = img[c, :, :] * ch
                all_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04), transforms.ToTensor()])
                img = all_transforms(img)
            if self.cutout:
                cutout_transform = transforms.Compose([Cutout(n_holes=1, length=self.cutout)])
                img = cutout_transform(img)
            patch_bag.append(img)
        if len(patch_bag) == 0:
            print('current_dir', current_dir)
        patch_bag = torch.stack(patch_bag, dim=0)
        return patch_bag, label_bag, files


if __name__ == "__main__":

    # tile up dataset from root_dir and save tile IDs to train and test csv file
#   TRAIN_ROOT_DIR = '/home/nnaik/research/2019_usc_data/patches_64_patchsize_16_overlap_9_level_normalized/split/train'
#   VAL_ROOT_DIR = '/home/nnaik/research/2019_usc_data/patches_64_patchsize_16_overlap_9_level_normalized/split/val'
    TRAIN_ROOT_DIR = '/export/home/data/ucsf/RTOG-9202/9.14.2020/95cpu_patches_256_patchsize_224_resize_0_overlap_2_level'
    VAL_ROOT_DIR = '/export/home/data/ucsf/RTOG-9202/9.14.2020/95cpu_patches_256_patchsize_224_resize_0_overlap_2_level'
    TRAIN_CSV = "ssl_train_patient_deID.csv"
    VAL_CSV = "ssl_train_patient_deID.csv"
    train_loader = PatchDataset(
        root_dir=TRAIN_ROOT_DIR,
        csv_file=TRAIN_CSV,
        image_size=64,
        grayscale=False,
        num_samples=20)
    val_loader = PatchDataset(
        root_dir=VAL_ROOT_DIR,
        image_size=64,
        csv_file=VAL_CSV,
        grayscale=False,
        num_samples=20)

    len_bag_list_val = []
    usc_bags_val = 0
    n_val = len(val_loader)
    np.random.seed()
    choices = np.random.permutation(n_val)
    for i in range(n_val):
        for j in range(5):
            bag, label, files = val_loader[choices[i]]
            print(val_loader.dir_names[choices[i]], bag.shape, len(label))
        print("="*80)
