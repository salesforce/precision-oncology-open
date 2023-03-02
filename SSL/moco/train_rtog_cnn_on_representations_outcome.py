"""Fine-tunes a small CNN on existing simclr features. Predicts outcome (e.g. 5-year DM)

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
from sklearn.metrics import auc, roc_curve

from utils import *
from auto_tqdm import tqdm
import augmentations
from sklearn.model_selection import train_test_split
from utils import balanced_dataset

import sys
sys.path.append("../../clinical_data_classifier")
from rtog_helper import rtog_from_study_number

# Currently only supports binary outcomes
NUM_CLASSES = 2

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
parser.add_argument('--schedule', default='constant', type=str,
                    help='The learning rate schedule. constant, cos, or [E]_[gamma] s.t. every E epochs the lr becomes lr * gamma.')
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
#TODO: propagate the addition of args.outcome
parser.add_argument('--outcome', type=str, default='distant_met_5year',
                    help='Options: distant_met_5year')
args = parser.parse_args()


def load_quilts(directory, rtog):
    """Loads features, images, and outcomes

    Args:
        directory(str): folder contained entries of the form '[caseid]_quilt.tiff' and '[caseid]_quilt_feature.pkl'
        df(Dataframe): Reference dataframe with outcome for a given case id.
            Load using rtog_from_study_number(rtog_num)

    Returns tuple of the form:
        (cn_deid, feature_quilt_paths, image_quilt_paths, isup_score)
    """
    feature_quilt_paths = sorted([os.path.join(directory,i) for i in os.listdir(directory) if '.pkl' in i])
    image_quilt_paths = sorted([os.path.join(directory,i) for i in os.listdir(directory) if '.tiff' in i])
    cn_deids = np.array([int(i.split("/")[-1].split('_')[0]) for i in feature_quilt_paths])

    rx, ry = rtog.get_Xy(y_var=args.outcome, make_binary=True)
    outcome_from_cndeid = lambda x: ry.df.loc[rx.df['cn_deidentified'] == x, args.outcome].values[0]
    outcomes = [outcome_from_cndeid(i) for i in cn_deids]

    return cn_deids, feature_quilt_paths, image_quilt_paths, outcomes


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


def predict_all_proba(model, dataloader):
    model.eval()
    true, probs = [], []
    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        probs.append(torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy())
        true.append(target.detach().cpu().numpy())
    return probs, true


def plot_auc(true, proba, title_prefix="", epoch=None, savedir="/tmp/cnn_auc"):
    fig = plt.figure(figsize=(8, 8))
    fpr, tpr, thresholds = roc_curve(true, proba)
    auc_val = auc(fpr, tpr)
    tnr = 1 - fpr
    plt.plot(tpr, tnr, color='blue', label='ROC Curve (area = {})'.format(auc_val))
    plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Sensitivity")
    plt.ylabel("Specificity")
    savename = "{}_epoch={}_auc={}".format(title_prefix, epoch, auc_val)
    if title_prefix:
        title_prefix += ": "
    plt.title("{}AUC = {} (Epoch {})".format(title_prefix, epoch, auc_val), fontsize=15)
    plt.show()
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.savefig('{}/{}.png'.format(savedir, savename))
    return auc_val


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
for sn in ['9202', '9413', '9408']:
    cn_deids, feature_quilt_paths, image_quilt_paths, outcomes = load_quilts(
        args.study_number_path.format(sn), rtog_from_study_number(sn))
    check_name = lambda u, v: u.split('/')[-1].split('_')[0] == v.split('/')[-1].split('_')[0]
    for f, i in zip(feature_quilt_paths, image_quilt_paths):
        assert check_name(f,i), 'mismatch {} {}'.format(f,i)
    logging.info("Loaded {} from {}".format(len(feature_quilt_paths), args.study_number_path))
    X.extend(feature_quilt_paths)
    X_images.extend(image_quilt_paths)
    y.extend(outcomes)
    assert len(np.unique(outcomes)) == NUM_CLASSES, "Does not support num_outcomes != {}. Currently {}".format(
        NUM_CLASSES, len(np.unique(outcomes)))
X = np.array(X)
y = np.array(y)

for i in range(NUM_CLASSES):
    logging.info("Class {} has {} data points".format(i, np.sum(y == i)))

# Use GPU
cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else 'cpu'

# Train/Validation Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
logging.info("Train:")
for i in range(NUM_CLASSES):
    logging.info("Class {} has {} data points".format(i, np.sum(y_train == i)))
logging.info("Test:")
for i in range(NUM_CLASSES):
    logging.info("Class {} has {} data points".format(i, np.sum(y_test == i)))
X_train, y_train = balanced_dataset(X_train, y_train, min_size=np.max(np.unique(y_train, return_counts=True)) * NUM_CLASSES)
train_dataset = CustomDataset(X_train, y_train, use_cache=args.use_cache, max_cache_size=args.max_cache_size)
test_dataset = CustomDataset(X_test, y_test, use_cache=args.use_cache, max_cache_size=args.max_cache_size)
logging.info(train_dataset)
logging.info(test_dataset)
logging.info("Balanced Train:")
for i in range(NUM_CLASSES):
    logging.info("Class {} has {} data points".format(i, np.sum(y_train == i)))

# Data settings
kwargs = {'num_workers': 12, 'pin_memory': False} if use_cuda else {}
kwargs = {}
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
test_loader =  DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

# Select Model
if args.base_model.lower() == "nanomodel":
    model = NanoModel(num_classes=NUM_CLASSES, n_input_features=args.n_input_features)
elif args.base_model.lower() == "micromodel":
    model = MicroModel(num_classes=NUM_CLASSES, n_input_features=args.n_input_features)
elif args.base_model.lower() == "minimodel":
    model = MiniModel(num_classes=NUM_CLASSES, n_input_features=args.n_input_features)
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

    if args.schedule == "cos":
        lr = lr  * 0.5 * (1 + np.cos((epoch - 1) / 200 * np.pi))
    elif args.schedule == "constant":
        pass
    else:
        E, gamma = args.schedule.split('_')
        E = int(E)
        gamma = float(gamma)
        if epoch % E == 0:
            lr *= gamma
    logging.info('Setting learning rate to %g' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    train(device, model, train_loader, optimizer, epoch)

    if epoch % 5 == 0:
        logging.info("Computing AUC Plots. Epoch={}".format(epoch))
        # Validation Accuracy and CM
        probabilities, true = predict_all_proba(model, test_loader)
        true = np.concatenate(true)
        probabilities = np.concatenate(probabilities)[:, 1]
        auc_val = plot_auc(true, probabilities, title_prefix="Testing", epoch=epoch, savedir=args.output_dir)
        logging.info("Validation: AUC {}".format(auc_val))

        # Training Accuracy and CM
        probabilities, true = predict_all_proba(model, train_loader)
        true = np.concatenate(true)
        probabilities = np.concatenate(probabilities)[:, 1]
        auc_val = plot_auc(true, probabilities, title_prefix="Training", epoch=epoch, savedir=args.output_dir)
        logging.info("Training: AUC {}".format(auc_val))

        # Save model checkpoint
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'auc_val' : auc_val,
        }, os.path.join(args.output_dir, 'model.pt'))








