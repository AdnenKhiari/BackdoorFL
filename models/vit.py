import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16
from models.modelbase import ModelBase

class Vit(ModelBase):
    def __init__(self, num_classes: int) -> None:
        super(Vit, self).__init__(num_classes)
        self.model = vit_b_16()
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        self.pytorch_transforms = torch.nn.UpsamplingBilinear2d(size=(224, 224))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    def transform_input(self, batch: torch.Dict[str, torch.Any]) -> torch.Dict[str, torch.Any]:
        batch["image"] = self.pytorch_transforms(batch["image"])
        return batch