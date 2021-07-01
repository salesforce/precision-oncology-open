"""Fine-tunes a small CNN + simple tabular model on existing SSL features. Predicts outcome (e.g. 5-year DM)

Example Call

"""

import IPython
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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from utils import *
from auto_tqdm import tqdm
import augmentations
from sklearn.model_selection import train_test_split
from utils import balanced_dataset

# For tabular learning
from fastai.tabular import TabularModel
from image_tabular.model import CNNTabularModel

import sys
sys.path.append("../../clinical_data_classifier")
from rtog_helper import rtog_from_study_number
from rtog_constants import is_categorical, drop_confounding_variables

NUM_CLASSES = 2 # only binary for now.
RANDOM_SEED = 3 # Clinical-Only gives 0.745 (no ablation)

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
                    help='Size of features from the SSL model. 128 for R34, 4096 for R50, etc.')
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
parser.add_argument('--outcome', type=str, default='distant_met_5year',
                    help='Options: distant_met_5year')
args = parser.parse_args()

def load_data(directory, rtog):
    """Loads features, images, and outcomes

    Args:
        directory(str): folder contained entries of the form '[caseid]_quilt.tiff' and '[caseid]_quilt_feature.pkl'
        df(Dataframe): Reference dataframe with outcome for a given case id.
            Load using rtog_from_study_number(rtog_num)

    Returns tuple of the form:
        (cn_deid, feature_quilt_paths, image_quilt_paths, isup_score)
    """
    #Image Data
    feature_quilt_paths = sorted([os.path.join(directory,i) for i in os.listdir(directory) if '.pkl' in i])
    image_quilt_paths = sorted([os.path.join(directory,i) for i in os.listdir(directory) if '.tiff' in i])

    # Case IDs - these filter X and y, since not all cases have image data (yet).
    cn_deids = np.array([int(i.split("/")[-1].split('_')[0]) for i in feature_quilt_paths])

#   # Labels (Outcomes)
#   rx, ry = rtog.get_Xy(y_var=args.outcome, make_binary=True)
#   outcome_from_cndeid = lambda x: ry.df.loc[rx.df['cn_deidentified'] == x]#.values[0]
#   outcomes = pd.concat([outcome_from_cndeid(i) for i in cn_deids])

#   # Tabular Data
#   clinical_data_from_cndeid = lambda x: rx.df.loc[rx.df['cn_deidentified'] == x]#.values
#   clinical_data = pd.concat([clinical_data_from_cndeid(i) for i in cn_deids])
#   cat_idxs = np.where(is_categorical(rx.df.columns))[0]
#   numerical_idxs = list(set(range(len(rx.df.columns))) - set(cat_idxs))
#   assert len(set(cat_idxs).intersection(set(numerical_idxs))) == 0
#   categorical_dims = {col : len(rx.df[col].unique()) for col in rx.df.columns[cat_idxs]}
#   assert all(outcomes.index == clinical_data.index), "The rows from the outcomes and clinical_data tables don't match! SAD"

#   return cn_deids, feature_quilt_paths, image_quilt_paths, clinical_data, cat_idxs, categorical_dims, numerical_idxs, outcomes

    return cn_deids, feature_quilt_paths, image_quilt_paths


class CustomDataset(VisionDataset):

    def __init__(self, X, y, categorical_names, continuous_names, transform=None, use_cache=True, max_cache_size=0):

        super(CustomDataset, self).__init__('', transform=transform)
        self.X = X
        self.y = y
        self.categorical_names = categorical_names
        self.continuous_names = continuous_names
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
        feature_quilt_path = self.X['featquilt'].values[index]
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

        feature_quilt = torch.Tensor(feature_quilt.transpose((2,0,1))).double()
        continuous_data = torch.Tensor(self.X[self.continuous_names].values[index]).double()
        target = torch.Tensor(self.y.values[index]).long()
        categorical_data = torch.Tensor(self.X[self.categorical_names].values[index]).long()
        #image_tabular object CNNTabularModel expects x to be of the form:
        # x[0] - image batch (e.g. 64 x 3 x 200 x 200)
        # x[1][0] - categorical variables (e.g. 64 x 3) - 3 categorical vars
        # x[1][1] - continuous variables (e.g. 64 x 1) - 1 categorical var
        return (feature_quilt, categorical_data, continuous_data), target

    def extra_repr(self):
        return "CustomDataset: use_cache={}, max_cache_size={}.".format(
                        self.use_cache,
                        self.max_cache_size,
                    )


def train(device, model, train_loader, optimizer, epoch):

    model.train()
    train_metrics = []
    for batch_idx, (data, target) in enumerate(train_loader):
        for i, d in enumerate(data):
            data[i] = d.to(device)
        target = target.to(device).reshape(-1,)
        optimizer.zero_grad()
        logits = model(data[0], data[1:])
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
        for i, d in enumerate(data):
            data[i] = d.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        logits = model(data[0], data[1:])
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

# Load Training Data: Feature quilts (images with 128 channels), and clinical data (tables)
X_rtogs = []
y_rtogs = []
for sn in ['9202', '9413', '9408']:
    directory = args.study_number_path.format(sn)
    feature_quilt_paths = sorted([os.path.join(directory,i) for i in os.listdir(directory) if '.pkl' in i])
    cn_deids = np.array([int(i.split("/")[-1].split('_')[0]) for i in feature_quilt_paths])
    logging.info("Loaded {} from {}".format(len(feature_quilt_paths), args.study_number_path))

    rtog = rtog_from_study_number(sn, standardize=True)
    rtog.df['featquilt'] = ''
    for id, path in zip(cn_deids, feature_quilt_paths):
        rtog.df.loc[rtog.df['cn_deidentified'] == id, 'featquilt'] = path
    rx, ry = rtog.get_Xy(y_var=args.outcome, make_binary=True)
    X_rtogs.append(rx.df)
    y_rtogs.append(ry.df)
X = pd.concat(X_rtogs, sort=True)
X = drop_confounding_variables(X)
y = pd.concat(y_rtogs, sort=True)
y = y[X['featquilt'] != '']
X = X[X['featquilt'] != '']

# Categorical vs continuous variable names
categorical = is_categorical(X.columns)
categorical = np.where(categorical)[0]
categorical_names = X.columns[categorical]
print("Categorical variables: {}".format(categorical_names))
continuous = np.array(list(set(range(len(X.columns))) - set(categorical)))
continuous_names = X.columns.values[continuous]
print("Continuous variables: {}".format(continuous_names))
for nn in continuous_names:
    assert nn not in categorical_names
assert len(continuous) + len(categorical) == len(X.columns)
categorical_dims = {}
for col in categorical_names:
    # Catboost requires categorical variables to be string or integer. Float and nans must be converted to strings.
    X[col] = X[col].astype(str)
    categorical_dims[col] = len(X[col].unique())
#Remove featquilt - it gets lumped into continous
continuous = continuous[continuous_names != 'featquilt']
continuous_names = continuous_names[continuous_names != 'featquilt']

# Label-encode categorical vars
label_encoder = LabelEncoder()
for n in categorical_names:
    print("Label-encoding {}".format(n))
    label_encoder.fit(X[n])
    X[n] = label_encoder.transform(X[n])

# Impute missing data
for n in continuous_names:
    print("Imputing {}".format(n))
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    v = X[n].values.reshape(-1,1)
    imp.fit(v)
    new_vals = imp.transform(v).reshape(-1,)
    if len(new_vals) > 0:
        X[n] = new_vals
    else:
        print("WARNING: variable {} holds no data. Setting to 0.".format(n))
        X[n] = 0

# Normalize continuous variables
scaler = StandardScaler()
for n in continuous_names:
    print("Whitening {}".format(n))
    v = X[n].values.reshape(-1,1)
    scaler.fit(v)
    X[n] = scaler.transform(v)

# Use GPU
cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else 'cpu'

# Train/Validation Split
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=RANDOM_SEED)
logging.info("Train:")
for i in range(NUM_CLASSES):
    logging.info("Class {} has {} data points".format(i, np.sum(y_train == i)))
logging.info("Test:")
for i in range(NUM_CLASSES):
    logging.info("Class {} has {} data points".format(i, np.sum(y_test == i)))
X_train, y_train = balanced_dataset(X_train, y_train,
                                    min_size=np.max(np.unique(y_train, return_counts=True)[1]) * NUM_CLASSES,
                                    random_seed=RANDOM_SEED,
                                    )
train_dataset = CustomDataset(X_train, y_train, categorical_names, continuous_names,
                              use_cache=args.use_cache, max_cache_size=args.max_cache_size)
test_dataset = CustomDataset(X_test, y_test, categorical_names, continuous_names,
                             use_cache=args.use_cache, max_cache_size=args.max_cache_size)
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

# Select Image Model
cnn_out_sz = 128
if args.base_model.lower() == "nanomodel":
    cnn_model = NanoModel(num_classes=cnn_out_sz, n_input_features=args.n_input_features)
elif args.base_model.lower() == "micromodel":
    cnn_model = MicroModel(num_classes=cnn_out_sz, n_input_features=args.n_input_features)
elif args.base_model.lower() == "minimodel":
    cnn_model = MiniModel(num_classes=cnn_out_sz, n_input_features=args.n_input_features)
else:
    raise ValueError("Base model not supported: {}".format(args.base_model))

#Load a tabular model, then let model = function(cnn_model, tabularModel)
# Careful to extract *features* from cnn_model and tabularModel, not logits.
# Can use image_tabular model. Or TabNet. The former is easier.
# Convention that for a categorical variable with N classes, the embedding dimension is of size 1.6 * N ** 0.56 (per fastai)
embedding_sizes = list(zip(categorical_dims.values(),
                           [int(1.6 * dim ** 0.56) for dim in categorical_dims.values()]
                           ))
tab_out_sz = 64 # output size of the tabular model that will be concatenated with cnn model output
tabular_model = TabularModel(embedding_sizes, len(continuous), out_sz=tab_out_sz, layers=[8], ps=0.2)
print(tabular_model)
model = CNNTabularModel(cnn_model, tabular_model, layers=[cnn_out_sz + tab_out_sz, 32], ps=0.2, out_sz=NUM_CLASSES)
model.double()

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


