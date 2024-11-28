import torch
import torch.nn.init as init
import torch.nn as nn
from torchvision import models

class oct_resnet(nn.Module):
    def __init__(self):
        super(oct_resnet, self).__init__()
        self.model = models.resnet50()

        self.model.fc = nn.Linear(self.model.fc.in_features, 4)

        init.xavier_uniform_(self.model.fc.weight)
        init.zeros_(self.model.fc.bias)
    
    def forward(self, x):
        return self.model(x)


