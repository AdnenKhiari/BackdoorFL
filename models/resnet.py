import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from models.modelbase import ModelBase

class ResNet(ModelBase):
    def __init__(self, num_classes: int) -> None:
        super(ResNet, self).__init__(num_classes)
        self.model = resnet50()
        self.model.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, num_classes),
            nn.Softmax(dim=1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)