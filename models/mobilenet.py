from collections import OrderedDict
from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import mobilenet_v3_small
class MobileNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(MobileNet,self).__init__()
        self.model = mobilenet_v3_small()
        self.model.classifier = nn.Sequential(
            nn.Linear(576, num_classes),
            nn.Softmax(dim=1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
a=MobileNet(10)

dc = [(_,val.numpy()) for _, val in a.state_dict().items()]
state_dict = dict({k: torch.from_numpy(v) for k, v in dict(dc).items()})
a.load_state_dict(state_dict, strict=True)
