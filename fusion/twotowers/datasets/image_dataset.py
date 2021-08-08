import torch
import numpy as np
import pandas as pd
import pickle
import os
import sys

from ..constants import *
from omegaconf import OmegaConf
from torchvision.datasets import VisionDataset

class ImageDataset(VisionDataset):
    """
    Adapted from: https://github.com/MetaMind/precision_oncology/blob/master/SSL/simclr/train_rtog_cnn_on_representations_outcome.py
    """

    def __init__(self, cfg, split='train', transform=None, use_cache=True, max_cache_size=0):
        super(ImageDataset, self).__init__('', transform=transform)

        if OmegaConf.select(cfg, "data.use_parsed_img") is True:
            self.use_parsed_img = True
        else: 
            self.use_parsed_img = False

        # TODO
        #self.parsed_emr = pickle.load(open(PARSED_EMR, 'rb'))
        self.parsed_emr = pickle.load(open(PARSED_EMR_2, 'rb'))

        img_feature_cols = [f'f{i}' for i in range(128)]
        if split == 'test': 
            self.df = self.parsed_emr['df_test'] 
            #self.df = self.df[self.df[img_feature_cols].all(axis=1)]
        else: 
            self.df = self.parsed_emr['df_train'] 
            if split == 'train': 
                self.df = self.df[self.df[f"Split{cfg.data.cv_split}"] == 0]
            else:
                self.df = self.df[self.df[f"Split{cfg.data.cv_split}"] == 1]
        self.df = self.df[~self.df.quilt_path.isna()]

        # define features
        self.ids = self.df['id'].tolist()
        self.X = self.df['quilt_path'].tolist()
        self.img_features = self.df[img_feature_cols] 
        self.y = self.df['distant_met_5year'].tolist()

        self.use_cache = use_cache
        self._cache = {}
        self.transform = transform
        if max_cache_size <= 0:
            self.max_cache_size = np.inf
        else:
            self.max_cache_size = max_cache_size
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        target = self.y[index]
        idx = self.ids[index]

        if self.use_parsed_img:
            img_feat = self.img_features.iloc[index].to_numpy()
            img_feat = img_feat.astype(np.float32)
            return img_feat, target, idx
        else: 
            feature_quilt_path= self.X[index]
            if self.use_cache and len(self._cache) < self.max_cache_size:
                if feature_quilt_path in self._cache:
                    feature_quilt = self._cache[feature_quilt_path]
                else:
                    feature_quilt = pickle.load(open(feature_quilt_path, 'rb'))
                    self._cache[feature_quilt_path] = feature_quilt
            else:
                feature_quilt = pickle.load(open(feature_quilt_path, 'rb'))

            if self.transform is not None:
                feature_quilt = self.transform(feature_quilt)

            feature_quilt = torch.Tensor(feature_quilt.transpose((2,0,1)))
            return feature_quilt, target, idx

    def get_sampler(self): 
        neg_class_count = (np.array(self.y) == 0).sum()
        pos_class_count = (np.array(self.y) == 1).sum()
        class_weight = [1/neg_class_count, 1/pos_class_count]
        weights = [class_weight[i] for i in self.y]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, 
            num_samples=len(weights),
            replacement=True)
        
        return sampler


# for debugging
if __name__ == "__main__": 

    cls = ImageDataset(None, 'train') 