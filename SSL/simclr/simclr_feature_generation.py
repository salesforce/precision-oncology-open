# python3 simclr_feature_generation.py 
# --output_dir /export/medical_ai/ucsf/simclr_rtog/model_resnet50_gp4plus_pretrained_lr=0.05_b=256/features/RTOG-9413_features/ 
# --input_dir /export/medical_ai/ucsf/RTOG-9413/tissue_pickles_v2/ 
# --checkpoint_path /export/medical_ai/ucsf/simclr_rtog/model_resnet50_gp4plus_pretrained_lr=0.05_b=256/checkpoint-epoch18.pt 
# --base_model resnet50

import tqdm
import torch
import os
import pickle
import numpy as np
import argparse 
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import models
from torch.autograd import Function
from multiprocessing import Process, Queue
from PIL import Image
import multiprocessing
from torch import nn
import time

cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
torch.manual_seed(1)

parser = argparse.ArgumentParser(description='Classifies tissue patches and saves the result per slide in a pickle.')
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--input_dir', type=str)
parser.add_argument('--num_loaders', type=int, default=2)
parser.add_argument('--out_dim', type=int, default=128,
                    help='Feature dimensionality')
parser.add_argument('--base_model', type=str, default='resnet34')
args = parser.parse_args()

use_cuda = True
num_classes = 6

class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super().__init__()
        print(base_model)
        if base_model == 'resnet18':
            self.backbone = models.resnet18(pretrained=False, num_classes=out_dim)
        elif base_model == 'resnet34':
            self.backbone = models.resnet34(pretrained=False, num_classes=out_dim)
        elif base_model == 'resnet50':
            self.backbone = models.resnet50(pretrained=False, num_classes=out_dim)
        else:
            raise ValueError('Invalid base_model!')
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)


def chunk_it(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def equal_chunks(l, n):
    """ Yield n successive chunks from l."""
    
    newn = int(1.0 * len(l) / n + 0.5)
    for i in range(0, n-1):
        yield l[i*newn:i*newn+newn]
    yield l[n*newn-newn:]


def prep_data(files, queue, finish_queue, test_transforms):
    """Loads tissue pickle and applies transforms

    Args:
        files (list): [description]
        queue (Queue): data queue to submit preprocessed samples to
        finish_queue (Queue): queue to indicate completion
        test_transforms (Transforms): list of transforms to apply.
    """
    for file in files:
        if os.path.exists(output_path + "/" + file):
            continue
        try:
            with open(base_path + file, "rb") as f:
                tiles = pickle.load(f)
        except:
            print("failed: ", file)
            continue

        processed_imgs = []
        for img in tiles:
            processed_imgs += [test_transforms(img)]
        tiles = torch.stack(processed_imgs).float()
        queue.put((tiles, file))
    
    queue.put((None,None))

    finish_queue.get()
    return



def process_data(queue, num_preprocessing_threads):
    """Loads data from a queue and processing it with a model and saves the feature vectors.

    Args:
        queue (Queue): Queue that is being filled with data.
        num_preprocessing_threads (Int): Number of threads preprocessing.
    """
    pbar = tqdm.tqdm(total=len(all_files))

    model = ResNetSimCLR(args.base_model, args.out_dim)
    # if use_cuda:
    #     model = torch.nn.DataParallel(model).cuda()
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(args.checkpoint_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    processed = []
    num_nones = 0
    while True:
        tiles, filename = queue.get()
        if tiles == None:
            num_nones += 1
            if num_nones == num_preprocessing_threads:
                break
            continue
        
        batches = list(chunk_it(tiles, 250))
        features = []
        softmax = []
        for batch in batches:
            with torch.no_grad():
                feats = model(batch.float().cuda())

                features += [feats.cpu().detach()]
        features = np.array(torch.cat(features))
        with open(output_path + "/" + filename, "wb") as f:
            pickle.dump(features, f)
            pbar.update(1)
    pbar.close()

test_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
test_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224), Image.NEAREST), transforms.ToTensor()])
print("will resize to 224 by 224")

base_path = args.input_dir
output_path = args.output_dir

all_files = os.listdir(base_path)
worker_files = equal_chunks(all_files, args.num_loaders)
worker_files = list(worker_files)
data_queue = Queue(50)
finish_queue = Queue()

# start preprocessing threads
loader_threads = []
for batch_files in list(worker_files):
    loader_thread = Process(target=prep_data, args=[batch_files, data_queue, finish_queue, test_transforms], daemon=True)
    loader_thread.start()
    loader_threads += [loader_thread]

process_data(data_queue, len(worker_files))

# let preprocessing threads know they can stop. Otherwise the last few objects they create
# are deleted when they exit.
for i in range(len(worker_files)):
    finish_queue.put(None)

data_queue.close()
