import torch

from densenet import DenseNet
use_cuda = False
num_classes = 6
model = DenseNet(num_classes)
if use_cuda:
    model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load('checkpoint-epoch100.pt')
state_dict = checkpoint.get('state_dict', checkpoint)
model.load_state_dict(state_dict)
