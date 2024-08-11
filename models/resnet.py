import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
class ResNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(ResNet, self).__init__()
        self.model = resnet50()
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512,num_classes),
            nn.Softmax(dim=1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)