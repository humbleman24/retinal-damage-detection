import torch
import torch.nn as nn
from torchvision.models import squeezenet1_1
from timm import create_model

class SqueezeViT(nn.Module):
    def __init__(self, num_classes=4):
        super(SqueezeViT, self).__init__()
        # Load pre-trained SqueezeNet
        squeezenet = squeezenet1_1(pretrained=True)
        
        # Get all layers except the classifier
        self.features = squeezenet.features
        
        # Load pre-trained ViT
        self.vit = create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)
        
        # Add adaptation layer to match ViT input dimensions (3, 224, 224)
        self.adapt = nn.Conv2d(512, 3, kernel_size=1)
        self.interpolate = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Pass through SqueezeNet features (output: B, 512, H', W')
        x = self.features(x)
        
        # Adapt number of channels and resize to ViT input size
        x = self.adapt(x)  # Change channels from 512 to 3
        x = self.interpolate(x)  # Resize to 224x224
        
        # Pass through ViT
        x = self.vit(x)
        return x