import torch
import numpy as np
import pandas as pd
import pickle
import sys

from ..constants import *
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class EMRDataset(Dataset):

    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split

        # TODO: 
        #self.parsed_emr = pickle.load(open(PARSED_EMR, 'rb'))
        self.parsed_emr = pickle.load(open(PARSED_EMR_2, 'rb'))

        if split == 'test': 
            self.df = self.parsed_emr['df_test'] 
            img_feature_cols = [f'f{i}' for i in range(128)]
            # TODO: remove for all sample
            #self.df = self.df[self.df[img_feature_cols].all(axis=1)]
        else: 
            self.df = self.parsed_emr['df_train'] 
            if split == 'train': 
                self.df = self.df[self.df[f"Split{cfg.data.cv_split}"] == 0]
            else:
                self.df = self.df[self.df[f"Split{cfg.data.cv_split}"] == 1]
        
        # get X,y  
        feature_cols = self.parsed_emr['features']
        self.df_X = self.df[feature_cols]
        self.df_y = self.df[self.cfg.data.target]
        self.df_y.columns = [self.cfg.data.target]
        self.ids = self.df['id'].tolist()

        # tabnet params 
        self.cat_idxs = self.parsed_emr['cat_idxs']
        self.cat_dims = self.parsed_emr['cat_dims'] 
        self.cat_emb_dims = self.parsed_emr['cat_emb_dims'] 
        self.input_dim = self.parsed_emr['input_dim'] 

    def get_category_info(self): 
        return list(self.cat_idxs), list(self.cat_dims), list(self.cat_emb_dims)
        
    def get_input_dim(self): 
        return self.input_dim 

    def __getitem__(self, index):

        x = np.array(self.df_X.iloc[index])
        y = np.array(self.df_y.iloc[index])
        idx = self.ids[index]
        return x, y, idx

    def __len__(self):
        return len(self.df_y)
    
    def get_sampler(self): 

        neg_class_count = (np.array(self.df_y) == 0).sum()
        pos_class_count = (np.array(self.df_y) == 1).sum()
        class_weight = [1/neg_class_count, 1/pos_class_count]
        weights = [class_weight[i] for i in self.df_y]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, 
            num_samples=len(weights),
            replacement=True)
        
        return sampler


# for debugging
if __name__ == "__main__": 

    cls = EMRDataset(None, 'train') 