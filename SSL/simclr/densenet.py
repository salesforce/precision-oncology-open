
import torch
import torch.nn.functional as F
import torchvision.models as models

class DenseNet(torch.nn.Module):
    
    def __init__(self, num_classes, pretrained=True):
        super(DenseNet, self).__init__()
        self.feature_extractor = models.densenet121(pretrained=pretrained)
        self.final = torch.nn.Linear(1000, num_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        return self.final(x)
