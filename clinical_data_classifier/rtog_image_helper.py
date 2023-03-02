import numpy as np
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
import pickle


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):

        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):

        h = self.conv(x)
        h = self.bn(h)
        h = self.relu(h)
        return h

class NanoModel(nn.Module):

    def __init__(self, num_classes, n_input_features=128):

        super().__init__()
        self.cl1_1 = ConvLayer(n_input_features, 128, 3, 1)
        self.cl1_2 = ConvLayer(128, 128, 3, 1)
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, x):

        h = self.cl1_1(x)
        h = self.cl1_2(h)
        h = torch.nn.MaxPool2d(3, 3)(h)
        h = torch.mean(h, axis=(-1, -2))
        return self.fc(h)

    def features(self, x):

        h = self.cl1_1(x)
        h = self.cl1_2(h)
        h = torch.nn.MaxPool2d(3, 3)(h)
        h = torch.mean(h, axis=(-1, -2))
        return h


class ImageLoader(object):

    def __init__(self):
        self._image_cache = {}

    def _meanpool(quilt):
        v = np.mean(quilt, axis=(0,1))
        v -= np.mean(v)
        v /= np.max(np.abs(v))
        return v

    def compute_feature_matrix(self, df, func_reduce):
        feature_matrix = []
        for i, row in tqdm(df.iterrows()):
            # MeanPool the feature quilt and whiten the resultant vector
            path = row['featquilt']
            if path in self._image_cache:
                feature_matrix.append(self._image_cache[path])
                continue

            if path and type(path) == str:
                feature_quilt = pickle.load(open(path, 'rb'))
                feature_vec = func_reduce(feature_quilt)
            else:
                feature_vec = []
            feature_matrix.append(feature_vec)
            self._image_cache[path] = feature_vec

        m = max([len(r) for r in feature_matrix])
        for i,r in enumerate(feature_matrix):
            if len(r) == 0:
                feature_matrix[i] = np.zeros(m)
        feature_matrix = np.vstack(feature_matrix)
        return feature_matrix


    def _add_feature_matrix(self, mat, df):
        for i in range(mat.shape[1]):
            df['f{}'.format(i)] = mat[:, i]


    def load_feature_matrix(self, df, func_reduce=None):
        assert 'featquilt' in df.keys()
        if func_reduce is None:
            print("Adding meanPool image features to dataframe, from feature quilts!")
            mat = self.compute_feature_matrix(df, ImageLoader._meanpool)
        else:
            print("Adding custom image features to dataframe, from feature quilts")
            mat = self.compute_feature_matrix(df, func_reduce)
        self._add_feature_matrix(mat, df)
