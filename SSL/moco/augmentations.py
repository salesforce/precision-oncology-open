import sys
import os
import argparse
import logging
import shutil
import re
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torchvision.datasets import VisionDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class TranslateRotate(object):

    def __init__(self, translation, rotation, size, window=1, allow_missing=True):

        if isinstance(size, int):
            size = (size, size)
        size = tuple(size)
        if len(size) == 2:
            size += (3,)
        assert translation < min(size[0], size[1])
        self.window = window
        self.size = size
        if len(self.size) == 2:
            self.size += (3,)
        self.translation = translation
        self.allow_missing = allow_missing
        if rotation:
            self.rotation = transforms.RandomRotation([-rotation, rotation], Image.BICUBIC)
        else:
            self.rotation = False

    def _return_location(self, x):
        w, h = x.split('_')[-2], x.split('_')[-1].split('.')[0]
        return int(w), int(h)

    def _return_neighbors(self, x, window=1):

        neighbors = []
        w, h = self._return_location(x)
        for d_w in range(-window, window+1):
            for d_h in range(-window, window+1):
                new_neighbor = x.replace(
                        str(w).zfill(3), str(w+d_w).zfill(3)).replace(
                        str(h).zfill(3), str(h+d_h).zfill(3))
                neighbor_cls = x.split('/')[-2]
                exists = False
                for cls in range(6):
                    alt = new_neighbor.replace('/{}/'.format(neighbor_cls), '/{}/'.format(cls))
                    if os.path.exists(alt):
                        new_neighbor, exists = alt, True
                        break
                if exists:
                    neighbors.append(new_neighbor)
                else:
                    neighbors.append('missing')
        return neighbors

    def load_image(self, name, size):

        if name == 'missing':
            return (255 * np.ones(size)).astype(np.uint8)
        else:
            return np.array(Image.open(name))


    def combine_neighbors(self, x, size=(224, 224, 3), window=1):

        output = []
        neighbors = self._return_neighbors(x, window)
        if not neighbors:
            return None
        neighbors_images = [self.load_image(n, size) for n in neighbors]
        width = 2 * window + 1
        for i in range(width):
            output.append(np.concatenate(neighbors_images[i * width: (i+1) * width], axis=1))
        return Image.fromarray(np.concatenate(output, axis=0))

    def __call__(self, image_name):

        combined = self.combine_neighbors(image_name, self.size, self.window)
        if self.rotation:
            combined = self.rotation(combined)
        t_w, t_h = np.random.choice(np.arange(-self.translation, self.translation), 2)
        tile = np.array(combined)[self.size[0]+t_w: 2 * self.size[0] + t_w,
                       self.size[1] + t_h: 2 * self.size[1] + t_h]
        return Image.fromarray(tile)
