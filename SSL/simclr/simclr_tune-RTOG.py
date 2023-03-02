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
from sklearn.metrics import confusion_matrix
from functools import partial
import torch.nn as nn
from sklearn.model_selection import train_test_split
from utils import balanced_dataset

from utils import *
from auto_tqdm import tqdm
import augmentations

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import sys
# For PyCharm:
# for some reason, this does not work in pycharm and so I followed this link to add the sys.path :
# https://stackoverflow.com/questions/30924664/how-to-manage-sys-path-globally-in-pycharm
#
# For Command-line:
# note: may need to add absolute path to pythonpath.
#    export PYTHONPATH=$PYTHONPATH:/full/path/to/precision_oncology/clinical_data_classifier
sys.path.append(os.path.abspath("../clinical_data_classifier/"))
from rtog_helper import rtog_from_study_number

# Set random seed
RANDOM_STATE = 1


parser = argparse.ArgumentParser(
    description='NRG Feature-Block Gleason Finetuning.')
parser.add_argument('--quilt_path', default='/export/medical_ai/ucsf/ssl_rtog/simclr/model_resnet34/checkpoint_110/', type=str,
                    help='Must contain directories of the form RTOG_{}_quilts.')
parser.add_argument('--output_dir', default='/export/home/medical_ai/exp1_ray', type=str,
                    help='By convention, make this the quilt_path dir.')
parser.add_argument('--num_epochs', default=50, type=int,
                    help='The number of training epochs.')
parser.add_argument('--lr_max', default=1e-1, type=float,
                    help='Will be loguniform(max, min)')
parser.add_argument('--lr_min', default=1e-6, type=float,
                    help='Will be loguniform(max, min)')
parser.add_argument('--num_trials', default=3, type=int,
                    help='The number of trials (1 trial is 1 model trained with a set of hyperparams)')
parser.add_argument('--gpus_per_trial', default=2, type=int,
                    help='The number of trials (1 trial is 1 model trained with a set of hyperparams)')
parser.add_argument('--cpus_per_trial', default=5, type=int,
                    help='The number of trials (1 trial is 1 model trained with a set of hyperparams)')
args=parser.parse_args()


def gleason_isup(primary, secondary):
    if primary + secondary in {9, 10}:
        return 5
    elif primary + secondary == 8:
        return 4
    elif primary == 4.0 and secondary == 3.0:
        return 3
    elif primary == 3.0 and secondary == 4.0:
        return 2
    elif primary + secondary <= 6.0:
        return 1
    else:  # catchall for unknown
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
    feature_quilt_paths = sorted([os.path.join(directory, i) for i in os.listdir(directory) if '.pkl' in i])
    image_quilt_paths = sorted([os.path.join(directory, i) for i in os.listdir(directory) if '.tiff' in i])
    cn_deids = np.array([int(i.split("/")[-1].split('_')[0]) for i in feature_quilt_paths])

    primary_from_cndeid = lambda x: df.loc[df['cn_deidentified'] == x, 'gleason_primary'].values[0]
    primary = [primary_from_cndeid(i) for i in cn_deids]

    if 'gleason_secondary' in df.keys():
        secondary_from_cndeid = lambda x: df.loc[df['cn_deidentified'] == x, 'gleason_secondary'].values[0]
        secondary = [secondary_from_cndeid(i) for i in cn_deids]
    else:  # have gleason combined
        secondary_from_cndeid = lambda x: df.loc[df['cn_deidentified'] == x, 'gleason_combined'].values[0] - \
                                          df.loc[df['cn_deidentified'] == x, 'gleason_primary'].values[0]
        secondary = [secondary_from_cndeid(i) for i in cn_deids]

    isup = [gleason_isup(p, s) for p, s in zip(primary, secondary)]
    df['isup'] = np.nan
    for c, i in zip(cn_deids, isup):
        df.loc[df['cn_deidentified'] == c, 'isup'] = i

    return cn_deids, feature_quilt_paths, image_quilt_paths, isup, df


def load_data():
    # Why are we not using any transforms ???
    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #])

    X = []
    X_images = []
    y = []
    new_dfs = {}
    for sn in ['9202', '9413', '9408']:
#       sn_path = "/export/medical_ai/ucsf/simclr_rtog/model_resnet34/checkpoint_110/RTOG_{}_simclr/".format(sn)
        sn_path = os.path.join(args.quilt_path, "RTOG_{}_simclr/".format(sn))
        cn_deids, feature_quilt_paths, image_quilt_paths, isup, df = load_quilts(sn_path, rtog_from_study_number(sn).df)
        new_dfs[sn] = df
        check_name = lambda u, v: u.split('/')[-1].split('_')[0] == v.split('/')[-1].split('_')[0]
#       for f, i in zip(feature_quilt_paths, image_quilt_paths):
#           assert check_name(f, i), 'mismatch {} {}'.format(f, i)
        # print("Loaded {} from {}".format(len(feature_quilt_paths), sn_path))
        X.extend(feature_quilt_paths)
#       X_images.extend(image_quilt_paths)
        y.extend(isup)
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=RANDOM_STATE)

    train_set = (X_train, y_train)
    test_set = (X_test, y_test)

    return train_set, test_set


class CustomDataset(VisionDataset):

    def __init__(self, X, y):
        super(CustomDataset, self).__init__('', transform=None,
                                            target_transform=None)
        self.X = X
        self.y = y
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature_quilt_path, target = self.X[index], self.y[index]
        feature_quilt = pickle.load(open(feature_quilt_path, 'rb'))
        feature_quilt = torch.Tensor(feature_quilt.transpose((2, 0, 1)))
        return feature_quilt, target

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


def train_and_val(config):
    trainset, testset = load_data()

    X, y = trainset

    # for i in range(6):
    #    print("Class {} has {} data points".format(i, np.sum(y == i)))

    # Comment this line to make function serializable, which is required by ray tune library function tune.run()
    # tune.utils.diagnose_serialization(train_and_val)
    # cudnn.benchmark = True
    use_cuda = torch.cuda.is_available()
    print("Use CUDA is :", use_cuda)
    device = torch.device('cuda') if use_cuda else 'cpu'

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.15, random_state=RANDOM_STATE)
    X_train, y_train = balanced_dataset(X_train, y_train, min_size=np.max(np.unique(y_train, return_counts=True)) * 6)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    kwargs = {'num_workers': 6, 'pin_memory': False} if use_cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True, drop_last=True,
                              **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False, **kwargs)

    # Can modify this to include more config params
    model = MiniModel(num_classes=6)
    # model = MiniModel(num_classes=6, config["l1"], config["l2"])
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.Adam(model.parameters(),
                           lr=config["lr"],
                           weight_decay=config["weight_decay"])

    model.train()
    train_metrics = []
    for epoch in range(0, 50):

        for batch_idx, (data, target) in enumerate(train_loader):

            if config["cosine_scheduler"]:
                config["lr"] = config["lr"] * 0.5 * (1 + np.cos((epoch - 1) / 200 * np.pi))

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
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader) * config["batch_size"],
                               100. * batch_idx / len(train_loader), loss.item()))

        if epoch % 5 == 0:
            model.eval()
            true, predicted = [], []
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                logits = model(data)
                loss = F.cross_entropy(logits, target)
                val_loss += loss.detach().cpu().numpy()
                val_steps += 1
                predicted.append(logits.argmax(-1).detach().cpu().numpy())
                true.append(target.detach().cpu().numpy())
                total += len(true[0])
                correct += (predicted[0] == true[0]).sum().item()

            true = np.concatenate(true)
            predicted = np.concatenate(predicted)

            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

            average_sensitivity = plot_confusion_matrix(true, predicted, title_prefix="Testing", epoch=str(epoch),
                                                        savedir=checkpoint_dir)

            tune.report(loss=(val_loss / val_steps), accuracy=correct / total, sensitivity=average_sensitivity)
    print("Finished Training")


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

    def __init__(self, num_classes):
        super().__init__()
        self.cl1_1 = ConvLayer(128, 128, 3, 1)
        self.cl1_2 = ConvLayer(128, 128, 3, 1)
        self.cl2_1 = ConvLayer(128, 128, 3, 1)
        self.cl2_2 = ConvLayer(128, 128, 3, 1)
        self.cl3_1 = ConvLayer(128, 128, 3, 1)
        self.cl3_2 = ConvLayer(128, 128, 3, 1)
        self.dropout = nn.Dropout(0.2)
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.cl1_1(x)
        h = self.cl1_2(h)
        h = torch.nn.MaxPool2d(3, 3)(h)
        h = torch.mean(h, axis=(-1, -2))
        return self.fc(h)


def plot_confusion_matrix(true, predicted, title_prefix="", epoch="", savedir="./plot_cm"):
    fig = plt.figure(figsize=(8, 8))
    #cm = confusion_matrix(np.concatenate(true), np.concatenate(predicted))
    cm = confusion_matrix(true, predicted)
    normalized_cm = cm/np.sum(cm, -1, keepdims=True)
    plt.imshow(normalized_cm, vmin=0., vmax=1)
    plt.imshow(normalized_cm, vmin=0., vmax=1)
    plt.colorbar()
    plt.xticks(np.arange(6), ['ISUP {}'.format(i) for i in range(6)], fontsize=12)
    plt.yticks(np.arange(6), ['ISUP {}'.format(i) for i in range(6)], fontsize=12)
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('True', fontsize=15)
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


def test_accuracy(model, device="cpu"):
    trainset, testset = load_data()

    X_test, y_test = testset

    test_dataset = CustomDataset(X_test, y_test)
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 12, 'pin_memory': False} if use_cuda else {}
    # Do we have to hardcode batchsize in test set ???
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, **kwargs)

    model.eval()
    true, predicted = [], []
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        logits = model(data)
        predicted.append(logits.argmax(-1).detach().cpu().numpy())
        true.append(target.detach().cpu().numpy())
        total += len(true[0])
        correct += (predicted[0] == true[0]).sum().item()

    true = np.concatenate(true)
    predicted = np.concatenate(predicted)
    plot_confusion_matrix(true, predicted, title_prefix="Testing", epoch='test_plot', savedir=args.output_dir)

    return correct / total


def main(num_samples=6, max_num_epochs=50, gpus_per_trial=1, cpus_per_trial=5):
    use_cuda = torch.cuda.is_available()
    load_data()

    config = {
        # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(args.lr_min, args.lr_max),
        "batch_size": tune.choice([32]),
        "weight_decay": tune.choice([1e-4]),
        "cosine_scheduler": tune.choice([True, False])
    }
    torch.backends.cudnn.enabled = True
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "sensitivity", "training_iteration"])

    # tune.utils.diagnose_serialization(train_and_val)

    result = tune.run(
        partial(train_and_val),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        local_dir=os.path.join(args.output_dir, "ray_results"),
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # best_trained_model = MiniModel(best_trial.config["l1"], best_trial.config["l2"])
    best_trained_model = MiniModel(num_classes=6)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    sys.path.append(os.path.abspath("../clinical_data_classifier/"))
    main(num_samples=args.num_trials, max_num_epochs=args.num_epochs, gpus_per_trial=args.gpus_per_trial, cpus_per_trial=args.cpus_per_trial)
