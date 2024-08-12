from typing import Any, Dict, Mapping
import torch


class ModelBase(torch.nn.Module):
    def __init__(self,num_classes) -> None:
        super(ModelBase, self).__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def transform_input(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch
