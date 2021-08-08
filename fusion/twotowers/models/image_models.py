"""
Adapted from: https://github.com/MetaMind/precision_oncology/blob/master/SSL/simclr/train_rtog_cnn_on_representations_outcome.py
"""

import torch
import torch.nn as nn


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


class MiniModel(nn.Module):

    def __init__(self, num_classes, n_input_features=128):

        super().__init__()
        self.cl1_1 = ConvLayer(n_input_features, 128, 3, 1)
        self.cl1_2 = ConvLayer(128, 128, 3, 1)
        self.cl2_1 = ConvLayer(128, 128, 3, 1)
        self.cl2_2 = ConvLayer(128, 128, 3, 1)
        self.cl3_1 = ConvLayer(128, 128, 3, 1)
        self.cl3_2 = ConvLayer(128, 128, 3, 1)
        self.fc = torch.nn.Linear(128, num_classes)


    def forward(self, x):

        h = self.cl1_1(x)
        h = self.cl1_2(h)
        h = torch.nn.MaxPool2d(3, 3)(h)
        h = self.cl2_1(h)
        h = self.cl2_2(h)
        h = torch.nn.MaxPool2d(3, 3)(h)
        h = self.cl3_1(h)
        h = self.cl3_2(h)
        h = torch.nn.MaxPool2d(3, 3)(h)
        h = torch.mean(h, axis=(-1, -2))
        return self.fc(h)


class MicroModel(nn.Module):

    def __init__(self, num_classes, n_input_features=128):

        super().__init__()
        self.cl1_1 = ConvLayer(n_input_features, 128, 3, 1)
        self.cl1_2 = ConvLayer(128, 128, 3, 1)
        self.cl2_1 = ConvLayer(128, 128, 3, 1)
        self.cl2_2 = ConvLayer(128, 128, 3, 1)
        self.fc = torch.nn.Linear(128, num_classes)


    def forward(self, x):

        h = self.cl1_1(x)
        h = self.cl1_2(h)
        h = nn.MaxPool2d(3, 3)(h)
        h = self.cl2_1(h)
        h = self.cl2_2(h)
        h = nn.MaxPool2d(3, 3)(h)
        h = torch.mean(h, axis=(-1, -2))
        return self.fc(h)


class NanoModel(nn.Module):

    def __init__(
            self, 
            num_classes, 
            n_kernels=128, 
            n_input_features=128, 
            use_batchnorm=False,
            use_hidden_layer=False, 
            dropout_prob=0.0
        ):

        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.use_hidden_layer = use_hidden_layer
        self.dropout_prob = dropout_prob
        self.n_kernels = n_kernels

        self.cl1_1 = ConvLayer(n_input_features, n_kernels, 3, 1)
        self.cl1_2 = ConvLayer(n_kernels, n_kernels, 3, 1)
        self.batchnorm = nn.BatchNorm1d(n_kernels)

        # TODO: use num_hidden and sequential instead
        self.fc_in = nn.Linear(n_kernels, n_kernels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.fc_out = nn.Linear(n_kernels, num_classes)

    def forward(self, x, get_features=False):

        h = self.cl1_1(x)
        h = self.cl1_2(h)
        h = nn.MaxPool2d(3, 3)(h)
        h = torch.mean(h, axis=(-1, -2))

        if self.use_batchnorm:
            h = self.batchnorm(h)

        if self.use_hidden_layer:
            h = self.fc_in(h)
            h = self.activation(h)
            if self.dropout_prob > 0.0: 
                h = self.dropout(h)
        
        if get_features:
            return h

        return self.fc_out(h)

class LinearModel(nn.Module):

    def __init__(
            self, 
            num_classes, 
            n_input_features=128, 
            n_neurons=128, 
            n_hidden=1,
            use_batchnorm=False,
            dropout_prob=0.0
        ):

        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob
        self.n_kernels = n_neurons
        self.n_hidden = n_hidden

        self.batchnorm = nn.BatchNorm1d(n_input_features)


        # output layers 
        fc_layers = [
            nn.Linear(n_input_features, n_neurons),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob)]
        for _ in range(self.n_hidden):
            fc_layers.append(nn.Linear(n_neurons, n_neurons))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=self.dropout_prob))

        """
        self.fc_layers = nn.Sequential(*fc_layers)
        self.fc_output = nn.Linear(n_neurons, num_classes)

        # TODO: for old ckpts 
        """
        fc_layers.append(nn.Linear(n_neurons, num_classes))
        self.fc_layers = nn.Sequential(*fc_layers)





    def forward(self, x, get_features=False):


        """

        if self.use_batchnorm:
            x = self.batchnorm(x)

        x = self.fc_layers(x)
        if get_features: 
            return x 
        else:
            return self.fc_output(x)
        
 
        # TODO: For old ckpt 
        """
        if self.use_batchnorm:
            x = self.batchnorm(x)
        return self.fc_layers(x)

