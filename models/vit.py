import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16
class Vit(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Vit, self).__init__()
        self.model = vit_b_16()
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)