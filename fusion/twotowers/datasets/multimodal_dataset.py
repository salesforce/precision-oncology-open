import torch
import numpy as np
import pandas as pd
import pickle
import os
import sys

from omegaconf import OmegaConf
from ..constants import *
from torchvision.datasets import VisionDataset

class MultimodalDataset(VisionDataset):
    """
    Adapted from: https://github.com/MetaMind/precision_oncology/blob/master/SSL/simclr/train_rtog_cnn_on_representations_outcome.py
    """

    def __init__(self, cfg, split='train', transform=None, use_cache=True, max_cache_size=0):
        super(MultimodalDataset, self).__init__('', transform=transform)

        self.use_parsed_img = OmegaConf.select(cfg, 'data.use_parsed_img') is True
        # TODO: 
        self.parsed_emr = pickle.load(open(PARSED_EMR, 'rb'))
        feature_cols = self.parsed_emr['features']
        img_feature_cols = [f'f{i}' for i in range(128)]

        if split == 'test': 
            self.df = self.parsed_emr['df_test']
            self.df = self.df[self.df[img_feature_cols].all(axis=1)]
        else: 
            self.df = self.parsed_emr['df_train']
            if split == 'train':
                self.df = self.df[self.df[f"Split{cfg.data.cv_split}"] == 0]
            else:
                self.df = self.df[self.df[f"Split{cfg.data.cv_split}"] == 1]

        # TODO: double check
        #if self.use_parsed_img:
        #    self.df = self.df[self.df[img_feature_cols].all(axis=1)]
        #else: 
        #    self.df = self.df[~self.df.quilt_path.isna()]
        self.df = self.df[self.df[img_feature_cols].all(axis=1)]

        # define features
        self.ids = self.df['id'].tolist()
        self.y = self.df['distant_met_5year'].tolist()
        self.quilt_paths = self.df['quilt_path'].tolist()
        self.img_features = self.df[img_feature_cols] 
        self.df_X = self.df[feature_cols] 

        # configs for loading quilts
        self.use_cache = use_cache
        self._cache = {}
        self.transform = transform
        if max_cache_size <= 0:
            self.max_cache_size = np.inf
        else:
            self.max_cache_size = max_cache_size
        assert len(self.df_X) == len(self.y)

        self.cat_idxs = self.parsed_emr['cat_idxs']
        self.cat_dims = self.parsed_emr['cat_dims'] 
        self.cat_emb_dims = self.parsed_emr['cat_emb_dims'] 
        self.input_dim = self.parsed_emr['input_dim'] 

    def get_category_info(self): 
        return list(self.cat_idxs), list(self.cat_dims), list(self.cat_emb_dims)
        
    def get_input_dim(self): 
        return self.input_dim 

    def __len__(self):
        return len(self.df_X)

    def __getitem__(self, index):
        x = np.array(self.df_X.iloc[index]).astype(np.float32)
        target = self.y[index]
        idx = self.ids[index]

        if self.use_parsed_img:
            img_feat = self.img_features.iloc[index].to_numpy()
            img_feat = img_feat.astype(np.float32)
            return x, img_feat, target, idx
        else: 
            feature_quilt_path = self.quilt_paths[index]

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
            return x, feature_quilt, target, idx

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

    cls = MultimodalDataset(None, 'train') 