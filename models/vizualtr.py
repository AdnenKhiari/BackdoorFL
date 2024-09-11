import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16
from models.modelbase import ModelBase
from vit_pytorch.vit_for_small_dataset import ViT

class VisualTR(ModelBase):
    def __init__(self, num_classes: int) -> None:
        super(VisualTR, self).__init__(num_classes)
        self.model = ViT(
            image_size = 128,
            patch_size = 16,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        self.pytorch_transforms = torch.nn.UpsamplingBilinear2d(size=(128, 128))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    def transform_input(self, batch: torch.Dict[str, torch.Any]) -> torch.Dict[str, torch.Any]:
        batch["image"] = self.pytorch_transforms(batch["image"])
        return batch