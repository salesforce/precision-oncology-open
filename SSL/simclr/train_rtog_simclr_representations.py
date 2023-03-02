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
from auto_tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import VisionDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from multiprocessing import Pool


from utils import *
import augmentations
import augmentations_rtog

import sys
sys.path.append('../../clinical_data_classifier')
from histopathology_image_helper import CaseManager

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
    description='PyTorch Train SimCLR Representation')

parser.add_argument('--data_dir', default='/export/medical_ai/ucsf',
                    help='Directory of the dataset')
parser.add_argument('--tensorboard_dir', default="./runs/", help="tensorboard log dir")
parser.add_argument('--model_dir', default='./simclr_rtog/script',
                    help='Directory of model for saving checkpoint')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Cancels the run if an appropriate checkpoint is found')

# Logging and checkpointing
parser.add_argument('--log_interval', type=int, default=10,
                    help='Number of batches between logging of training status')
parser.add_argument('--save_freq', default=1, type=int,
                    help='Checkpoint save frequency (in epochs)')

# Generic training configs
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed. '
                         'Note: fixing the random seed does not give complete '
                         'reproducibility. See '
                         'https://pytorch.org/docs/stable/notes/randomness.html')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
# parser.add_argument('--epoch_size', type=int, default=1000, metavar='N',
#                     help='Input batch size for training (default: 1000)')
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='Number of epochs to train. '
                         'Note: we arbitrarily define an epoch as a pass '
                         'through 50K datapoints. This is convenient for '
                         'comparison with standard CIFAR-10 training '
                         'configurations.')
parser.add_argument('--eval_freq', default=5, type=int,
                    help='Eval frequency (in epochs)')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float)
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='Learning rate')
parser.add_argument('--lr_schedule', type=str, default='cosine',
                    choices=('trades', 'trades_fixed', 'cosine', 'wrn'),
                    help='Learning rate schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='Use extragrdient steps')
parser.add_argument('--balanced', action='store_true', default=False,
                    help='Use balanced dataset')
parser.add_argument('--n_views', type=int, default=2,
                    help='Number of views in the contrastive learning')
parser.add_argument('--base_model', type=str, default='resnet34',
                    help='Basemodel ')
parser.add_argument('--pretrained', type=str2bool, default=False,
                    help='Are imagenet pretrained weights loaded in the basemodel?')
parser.add_argument('--num_workers', type=int, default=32,
                    help='num preprocessing/dataloader workers')
parser.add_argument('--out_dim', type=int, default=128,
                    help='Feature dimensionality')
parser.add_argument('--temperature', type=float, default=0.07,
                    help='Softmax temperature')
parser.add_argument('--minimum_primary_gleason_score', default=0, type=int,
                    help='If >0, only tiles from cases with this minimum primary gleason score are used.')

#args = parser.parse_args('')
args = parser.parse_args()


def eval(args, model, eval_loader, epoch, writer, num_eval_batches=None):

    if num_eval_batches is None:
        num_eval_batches = len(eval_loader)
    model.eval()
    counter = 0
    loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, images in enumerate(eval_loader):
            images = torch.cat(images, dim=0)
            images = images.to(args.device)
            features = model(images)
            logits, labels = simclr_criterion(args, features)
            loss += torch.nn.CrossEntropyLoss().to(args.device)(logits, labels).item()
            total += 1
            counter += 1
            if counter >= num_eval_batches:
                break
    loss /= total
    eval_data = dict(loss=loss)
    writer.add_scalar("eval_loss", loss, epoch)
    return eval_data


def simclr_criterion(args, features):

    labels = torch.cat([torch.arange(args.batch_size) for i in range(args.n_views)],
                   dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(args.device)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)
    logits = logits / args.temperature

    return logits, labels


def train(args, model, train_loader, optimizer, epoch, writer, epoch_size=None):
    if epoch_size is None:
        epoch_size = len(train_loader)
    model.train()
    train_metrics = []
    counter = 0
    for batch_idx, images in enumerate(train_loader):
        images = torch.cat(images, dim=0)
        images = images.to(args.device)
        features = model(images)
        if len(features) != 2*args.batch_size:
            logging.info("Feature ({}) batch_size ({}) mismatch. Continuing. Likely end of epoch.".format(features.shape, args.batch_size))
            continue
        logits, labels = simclr_criterion(args, features)
        loss = torch.nn.CrossEntropyLoss().to(args.device)(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("train_loss", loss.item(), epoch)
        train_metrics.append(dict(
            epoch=epoch,
            loss=loss.item()))
        # logging.info progress
        if batch_idx % args.log_interval == 0:
            logging.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), epoch_size * args.batch_size,
                           100. * batch_idx / epoch_size, loss.item()))
        counter += 1
        if counter >= epoch_size:
            break
    return train_metrics

def process_gleason_filtering_paths(paths, cm):
    paths_kept=[]
    for path in tqdm(paths):
        study_number = path.split('RTOG-')[1].split('/')[0]
        slide_id = int(path.split('/')[-2])
        gleason_primary = cm.get_feature_by_slideid('gleason_primary', slide_id, study_number)
        if gleason_primary is None or np.isnan(gleason_primary):
            continue
        if int(gleason_primary) >= int(args.minimum_primary_gleason_score):
            paths_kept.append(path)
    return paths_kept

def equal_chunks(l, n):
    """ Yield n successive chunks from l."""

    newn = int(1.0 * len(l) / n + 0.5)
    for i in range(0, n-1):
        yield l[i*newn:i*newn+newn]
    yield l[n*newn-newn:]


# Directory to save model checkpoints
if args.overwrite:
    shutil.rmtree(args.model_dir)
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.model_dir, 'training.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()
logging.info('Robust self-training')
logging.info('Args: %s', args)


# Load tiles containing tissue (as determined by an artifact/whitespace classifier)
rtog_tile_masks = [
    "tissue_vs_non_pkl/v2/RTOG-9202",
    "tissue_vs_non_pkl/v2/RTOG-9413",
    "tissue_vs_non_pkl/v2/RTOG-9408",
]
dataframes = []
for d in rtog_tile_masks:
    logging.info("Loading data from {}".format(os.path.join(args.data_dir, d)))
    for p in tqdm(os.listdir(os.path.join(args.data_dir, d))):
        df = pickle.load(open(os.path.join(args.data_dir, d, p), 'rb'))
        dataframes.append(df)

df = pd.concat(dataframes)
df_tissue = df[df['tissue_vs_non']]

# Filter for gleason, if specified.
paths = np.array(df_tissue['path'])
if args.minimum_primary_gleason_score:
    logging.info("Filtering tiles: minimum primary gleason of {}".format(args.minimum_primary_gleason_score))
    cm = CaseManager()
    p = Pool(args.num_workers)
    chunks = list(equal_chunks(paths, args.num_workers))
    results = p.starmap(process_gleason_filtering_paths, zip(chunks, [cm] * args.num_workers ))
    p.close()

    paths = []
    for result in results:
        paths += result
    paths = np.array(paths)

# Collect train/val dataset X
X_train, X_val = train_test_split(paths, test_size=0.1, random_state=0)
logging.info("Found {} patches across these dirs: {}".format(len(paths), rtog_tile_masks))

# Use GPU if available
cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
args.device = torch.device('cuda') if use_cuda else 'cpu'
# Input data augmentation
s = 1
color_jitter = transforms.ColorJitter(0.2, 0.2, 0.1, 0.2)
simclr_transforms = transforms.Compose([
        augmentations_rtog.TranslateRotateRTOG(30, 30, size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1.5), ratio=(0.9, 1.1)),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.05),
        augmentations.GaussianBlur(kernel_size=int(0.1 * 224), sigma_min=0.01, sigma_max=1.0),
        transforms.ToTensor()
    ])
# tensorboard writer
writer = SummaryWriter(log_dir=args.tensorboard_dir + args.model_dir.split("/")[-1])

# Contrastive Dataset Loaders
# X_train = X_train[:500] #TODO: remove. line.
# print("I just clipped X_train to 500 entries. you sure about that?")
dataset = ContrastiveDataset(X_train, args.n_views, simclr_transforms)
val_dataset = ContrastiveDataset(X_val, args.n_views, simclr_transforms)
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

# Load Trained Model and set Optimizer
logging.info("Using model {}, pre-trained={}, out_dim={}".format(
    args.base_model, args.pretrained, args.out_dim,
))
model = ResNetSimCLR(args.base_model, args.out_dim, args.pretrained)
if use_cuda:
    model = torch.nn.DataParallel(model).cuda()
#   model = torch.nn.parallel.DistributedDataParallel(model).cuda() # uses multi-processing, version DataParallel which uses multi-threading.
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      nesterov=args.nesterov)

# Load the latest checkpoint and logs
init_epoch = 0
train_df = pd.DataFrame()
eval_df = pd.DataFrame()
checkpoints = [i for i in os.listdir(args.model_dir) if '.pt' in i]
checkpoints_epoch_num = [int(re.search('epoch(\d+)', c).group(1)) for c in checkpoints]
if len(checkpoints_epoch_num):
    init_epoch = np.max(checkpoints_epoch_num)
    checkpoint = torch.load(os.path.join(args.model_dir, 'checkpoint-epoch{}.pt'.format(init_epoch)))
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    opt_checkpoint = torch.load(os.path.join(args.model_dir, 'opt-checkpoint_epoch{}.tar'.format(init_epoch)))
    optimizer.load_state_dict(opt_checkpoint)
    logger.info('Loading checkpoint from epoch #{}'.format(init_epoch))
    train_df = pd.read_csv(os.path.join(args.model_dir, 'stats_train.csv'), index_col=0)
    train_df.drop(train_df.index[np.arange(init_epoch, len(eval_df))], inplace=True)
    eval_df = pd.read_csv(os.path.join(args.model_dir, 'stats_eval.csv'), index_col=0)
    eval_df.drop(eval_df.index[np.arange(init_epoch, len(eval_df))], inplace=True)
else:
    logging.info('No checkpoint found. Random initialization!')

# Train
for epoch in range(init_epoch+1, args.epochs + 1):

    lr = adjust_learning_rate(args, optimizer, epoch)
    logger.info('Setting learning rate to %g' % lr)
    train_data = train(args, model, train_loader, optimizer, epoch, writer, epoch_size=None)
    train_df = train_df.append(pd.DataFrame(train_data), ignore_index=True)

    # Self-evaluation
    logging.info(120 * '=')
    if epoch % args.eval_freq == 0 or epoch == args.epochs:
        eval_data = {'epoch': int(epoch)}
        eval_data.update(
            eval(args, model, val_loader, epoch, writer, num_eval_batches=100))
        eval_df = eval_df.append(pd.Series(eval_data), ignore_index=True)
        logging.info(120 * '=')

    # Save stats
    train_df.to_csv(os.path.join(args.model_dir, 'stats_train.csv'))
    eval_df.to_csv(os.path.join(args.model_dir, 'stats_eval.csv'))

    if epoch % args.save_freq == 0 or epoch == args.epochs:
        torch.save(dict(
                        state_dict=model.state_dict()),
                   os.path.join(args.model_dir,
                                'checkpoint-epoch{}.pt'.format(epoch)))
        torch.save(optimizer.state_dict(),
                   os.path.join(args.model_dir,
                                'opt-checkpoint_epoch{}.tar'.format(epoch)))

#   delete_old_ckpts(args.model_dir)


