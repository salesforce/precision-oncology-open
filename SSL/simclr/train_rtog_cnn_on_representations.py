"""Fine-tunes a small CNN on existing simclr features.

Example Call

"""

import sys
import os
import argparse
import logging
import shutil
import re
import pickle
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torchvision.datasets import VisionDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from utils import *
from auto_tqdm import tqdm
import augmentations
from sklearn.model_selection import train_test_split
from utils import balanced_dataset

import sys
sys.path.append("../clinical_data_classifier")
from rtog_helper import rtog_from_study_number

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(
    description='NRG Feature-Block Gleason Finetuning.')
parser.add_argument('--optimizer', default='Adam', type=str,
                    help='Options: Adam, SGD.')
parser.add_argument('--num_epochs', default=50, type=int,
                    help='The number of training epochs.')
parser.add_argument('--weight_decay', default=5e-3, type=float,
                    help='The l2 penalty of the optimizer.')
parser.add_argument('--learning_rate', default=0.0003, type=float,
                    help='The learning rate. Uses cosine decay.')
parser.add_argument('--n_input_features', default=128, type=int,
                    help='Size of features from the SimCLR model. 128 for R34, 4096 for R50, etc.')
parser.add_argument('--use_cache', default=True, type=str2bool,
                    help='Caches data in memory for faster training.')
parser.add_argument('--max_cache_size', default=0, type=int,
                    help='Max number of data points to cache. If 0, cache all the data..')
parser.add_argument('--batch_size', default=8, type=int,
                    help='If using feature-size of 4096 (ResNet50), 1-2 data points will fit on each GPU.')
parser.add_argument('--study_number_path', type=str, default='/export/medical_ai/ucsf/simclr_rtog/model_simclr/checkpoint_110/RTOG_{}_simclr',
                    help='E.g. /export/medical_ai/ucsf/simclr_rtog/model_simclr/checkpoint_110/RTOG_{}_simclr')
parser.add_argument('--output_dir', type=str, default='/export/medical_ai/ucsf/tmp',
                    help='Directory to save data and model')
parser.add_argument('--base_model', type=str, default='NanoModel',
                    help='Options: NanoModel, MicroModel, MiniModel')
args = parser.parse_args()


def gleason_isup(primary, secondary):
    if primary + secondary in {9,10}:
        return 5
    elif primary + secondary == 8:
        return 4
    elif primary == 4.0 and secondary == 3.0:
        return 3
    elif primary == 3.0 and secondary == 4.0:
        return 2
    elif primary + secondary <= 6.0:
        return 1
    else: # catchall for unknown
        return 0


def load_quilts(directory, df):
    """Loads features, images, and isup score

    Args:
        directory(str): folder contained entries of the form '[caseid]_quilt.tiff' and '[caseid]_quilt_feature.pkl'
        df(Dataframe): Reference dataframe with gleason primary/secondary for a given case id.
            Load using rtog_from_study_number(rtog_num)

    Returns tuple of the form:
        (cn_deid, feature_quilt_paths, image_quilt_paths, isup_score)
    """
    feature_quilt_paths = sorted([os.path.join(directory,i) for i in os.listdir(directory) if '.pkl' in i])
    image_quilt_paths = sorted([os.path.join(directory,i) for i in os.listdir(directory) if '.tiff' in i])
    cn_deids = np.array([int(i.split("/")[-1].split('_')[0]) for i in feature_quilt_paths])

    primary_from_cndeid = lambda x: df.loc[df['cn_deidentified'] == x, 'gleason_primary'].values[0]
    primary = [primary_from_cndeid(i) for i in cn_deids]

    if 'gleason_secondary' in df.keys():
        secondary_from_cndeid = lambda x: df.loc[df['cn_deidentified'] == x, 'gleason_secondary'].values[0]
        secondary = [secondary_from_cndeid(i) for i in cn_deids]
    else: #have gleason combined
        secondary_from_cndeid = lambda x: df.loc[df['cn_deidentified'] == x, 'gleason_combined'].values[0] - df.loc[df['cn_deidentified'] == x, 'gleason_primary'].values[0]
        secondary = [secondary_from_cndeid(i) for i in cn_deids]

    isup = [gleason_isup(p, s) for p, s in zip(primary, secondary)]
    df['isup'] = np.nan
    for c, i in zip(cn_deids, isup):
        df.loc[df['cn_deidentified'] == c, 'isup'] = i

    return cn_deids, feature_quilt_paths, image_quilt_paths, isup, df


class CustomDataset(VisionDataset):

    def __init__(self, X, y, transform=None, use_cache=True, max_cache_size=0):

        super(CustomDataset, self).__init__('', transform=transform)
        self.X = X
        self.y = y
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
        feature_quilt_path, target = self.X[index], self.y[index]
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
        return feature_quilt, target

    def extra_repr(self):
        return "CustomDataset: use_cache={}, max_cache_size={}.".format(
                        self.use_cache,
                        self.max_cache_size,
                    )


def train(device, model, train_loader, optimizer, epoch):

    model.train()
    train_metrics = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        train_metrics.append(dict(
            epoch=epoch,
            loss=loss.item()))
        if batch_idx % 10 == 0:
            logging.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader) * args.batch_size,
                           100. * batch_idx / len(train_loader), loss.item()))
    return train_metrics


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
        logging.info("Initializing MiniModel(6 layers)")

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
        logging.info("Initializing MicroModel(4 layers)")

        super().__init__()
        self.cl1_1 = ConvLayer(n_input_features, 128, 3, 1)
        self.cl1_2 = ConvLayer(128, 128, 3, 1)
        self.cl2_1 = ConvLayer(128, 128, 3, 1)
        self.cl2_2 = ConvLayer(128, 128, 3, 1)
        self.fc = torch.nn.Linear(128, num_classes)


    def forward(self, x):

        h = self.cl1_1(x)
        h = self.cl1_2(h)
        h = torch.nn.MaxPool2d(3, 3)(h)
        h = self.cl2_1(h)
        h = self.cl2_2(h)
        h = torch.nn.MaxPool2d(3, 3)(h)
        h = torch.mean(h, axis=(-1, -2))
        return self.fc(h)


class NanoModel(nn.Module):

    def __init__(self, num_classes, n_input_features=128):
        logging.info("Initializing NanoModel(2 layers)")

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


class MLP(nn.Module):

    def __init__(self, num_classes):

        super().__init__()
        self.fc1 = torch.nn.Linear(128 * 200 * 200, 4000)
        self.fc2 = torch.nn.Linear(4000, 200)
        self.fc3 = torch.nn.Linear(200, num_classes)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(x):
        h = self.fc1(x)
        h = self.relu(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.fc3(h)


def plot_confusion_matrix(true, predicted, title_prefix="", epoch="", savedir="./plot_cm"):
    fig = plt.figure(figsize=(8, 8))
    cm = confusion_matrix(np.concatenate(true), np.concatenate(predicted))
    normalized_cm = cm/np.sum(cm, -1, keepdims=True)
    plt.imshow(normalized_cm, vmin=0., vmax=1)
    plt.imshow(normalized_cm, vmin=0., vmax=1)
    plt.colorbar()
    plt.xticks(np.arange(6), ['ISUP {}'.format(i) for i in range(6)], fontsize=12)
    plt.yticks(np.arange(6), ['ISUP {}'.format(i) for i in range(6)], fontsize=12)
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('True', fontsize=15)
    true = np.concatenate(true)
    predicted = np.concatenate(predicted)
    sensitivities = []
    for val in np.unique(true):
        s = 100 * np.mean( true[true == val] == predicted[true == val] )
        sensitivities.append(s)
    savename = "{}_epoch={}_AvgSens={}".format(title_prefix, epoch, np.mean(sensitivities))
    if title_prefix:
        title_prefix += ": "
    plt.title('{0} Average Sensitivity = {1:.1f}%'.format(
            title_prefix + "(Epoch {})".format(epoch),
            np.mean(sensitivities)
            ),
            fontsize=15)
    for i in range(6):
        for j in range(6):
            plt.annotate('{0:.2f}%'.format(100 * normalized_cm[i, j]), (j-0.3, i+0.1), color='white', fontsize=12)

    plt.show()
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.savefig('{}/{}.png'.format(savedir, savename))
    return np.mean(sensitivities)


def predict_all(model, dataloader):
    model.eval()
    true, predicted = [], []
    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        predicted.append(logits.argmax(-1).detach().cpu().numpy())
        true.append(target.detach().cpu().numpy())
    return predicted, true


# Directory to save model checkpoints and results plots
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()
logging.info('CNN finetuning on Feature-Quilts')
logging.info('Args: %s', args)

# Select SimCLR Features and load quilts for training.
X = []
X_images = []
y = []
new_dfs = {}
for sn in ['9202', '9413', '9408']:
    cn_deids, feature_quilt_paths, image_quilt_paths, isup, df = load_quilts(
        args.study_number_path.format(sn), rtog_from_study_number(sn).df)
    new_dfs[sn] = df
    check_name = lambda u, v: u.split('/')[-1].split('_')[0] == v.split('/')[-1].split('_')[0]
    for f, i in zip(feature_quilt_paths, image_quilt_paths):
        assert check_name(f,i), 'mismatch {} {}'.format(f,i)
    logging.info("Loaded {} from {}".format(len(feature_quilt_paths), args.study_number_path))
    X.extend(feature_quilt_paths)
    X_images.extend(image_quilt_paths)
    y.extend(isup)
X = np.array(X)
y = np.array(y)

for i in range(6):
    logging.info("Class {} has {} data points".format(i, np.sum(y == i)))

# Use GPU
cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else 'cpu'

# Train/Validation Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
logging.info("Train:")
for i in range(6):
    logging.info("Class {} has {} data points".format(i, np.sum(y_train == i)))
logging.info("Test:")
for i in range(6):
    logging.info("Class {} has {} data points".format(i, np.sum(y_test == i)))
X_train, y_train = balanced_dataset(X_train, y_train, min_size=np.max(np.unique(y_train, return_counts=True)) * 6)
#X_test, y_test = balanced_dataset(X_test, y_test, min_size=np.max(np.unique(y_test, return_counts=True)) * 6)
train_dataset = CustomDataset(X_train, y_train, use_cache=args.use_cache, max_cache_size=args.max_cache_size)
test_dataset = CustomDataset(X_test, y_test, use_cache=args.use_cache, max_cache_size=args.max_cache_size)
logging.info(train_dataset)
logging.info(test_dataset)

# Data settings
kwargs = {'num_workers': 12, 'pin_memory': False} if use_cuda else {}
kwargs = {}
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
test_loader =  DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

# Select Model
if args.base_model.lower() == "nanomodel":
    model = NanoModel(num_classes=6, n_input_features=args.n_input_features)
elif args.base_model.lower() == "micromodel":
    model = MicroModel(num_classes=6, n_input_features=args.n_input_features)
elif args.base_model.lower() == "minimodel":
    model = MiniModel(num_classes=6, n_input_features=args.n_input_features)
else:
    raise ValueError("Base model not supported: {}".format(args.base_model))

if use_cuda:
    model = torch.nn.DataParallel(model).cuda()

# Optimizers on Torch: https://pytorch.org/docs/stable/optim.html
lr = args.learning_rate
if args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(),
                       lr=lr,
                       weight_decay=args.weight_decay,
                       amsgrad=False)
elif args.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9,
                          weight_decay=args.weight_decay,
                          nesterov=True)
else:
    raise ValueError("Optimization method not supported: {}".format(args.optimizer))

# Train
logging.info("Training")
for epoch in range(0, args.num_epochs):
    if epoch % 5 == 0:
        logging.info("Computing Confusion Matrices. Epoch={}".format(epoch))
        # Validation Accuracy and CM
        predicted, true = predict_all(model, test_loader)
        average_sensitivity = plot_confusion_matrix(true, predicted, title_prefix="Testing", epoch=epoch, savedir=args.output_dir)
        logging.info("Validation: Average Sensitivity {}".format(average_sensitivity))

        # Training Accuracy and CM
        predicted, true = predict_all(model, train_loader)
        average_sensitivity = plot_confusion_matrix(true, predicted, title_prefix="Training", epoch=epoch, savedir=args.output_dir)
        logging.info("Training: Average Sensitivity {}".format(average_sensitivity))

    lr = lr  * 0.5 * (1 + np.cos((epoch - 1) / 200 * np.pi))
    logging.info('Setting learning rate to %g' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    train(device, model, train_loader, optimizer, epoch)


