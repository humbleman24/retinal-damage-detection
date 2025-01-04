import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm import create_model


class oct_vit(nn.Module):
    def __init__(self, num_classes=4):
        super(oct_vit, self).__init__()
        self.vit = create_model('vit_base_patch16_224', pretrained=True)  
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)  

    def forward(self, x):
        return self.vit(x)