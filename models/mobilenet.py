from collections import OrderedDict
from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from models.modelbase import ModelBase
class MobileNet(ModelBase):
    def __init__(self, num_classes: int) -> None:
        super(MobileNet,self).__init__(num_classes)
        self.model = mobilenet_v3_small()
        self.model.classifier = nn.Sequential(
            nn.Linear(576, num_classes),
            nn.Softmax(dim=1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
