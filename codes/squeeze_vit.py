import torch
import torch.nn as nn
from torchvision.models import squeezenet1_1
from timm import create_model

class SqueezeViT(nn.Module):
    def __init__(self, num_classes=4):
        super(SqueezeViT, self).__init__()
        # Load pre-trained SqueezeNet
        self.squeezenet = squeezenet1_1(pretrained=True)
        self.squeezenet.classifier = nn.Identity()        # Remove the final classifier layer
        
        # Load pre-trained ViT
        self.vit = create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        # Pass through SqueezeNet
        x = self.squeezenet(x)
        # Pass through ViT
        x = self.vit(x)
        return x
