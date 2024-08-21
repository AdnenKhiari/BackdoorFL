import torch
from typing import List

from poisoning_transforms.model.model_poisoner import ModelPoisoner

class ModelReplacement(ModelPoisoner):
    def __init__(self, global_weights: List[torch.Tensor], gamma: float = 1):
        """
        Initializes the ModelReplacement poisoner.

        Args:
            global_weights (List[torch.Tensor]): List of global model's weight tensors (G^t).
            gamma (float): The scaling factor (Γ).
        """
        super().__init__()
        self.global_weights = global_weights
        self.gamma = gamma

    def transform(self, weights: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Applies weight replacement poisoning to the list of weights.

        Args:
            weights (List[torch.Tensor]): List of weight tensors to be poisoned.

        Returns:
            List[torch.Tensor]: List of poisoned weight tensors.
        """
        
        poisoned_weights = self.gamma * (weights - self.global_weights) + self.global_weights
        return poisoned_weights