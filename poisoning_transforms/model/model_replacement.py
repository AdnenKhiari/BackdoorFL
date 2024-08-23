import numpy as np
from typing import List

from torch._tensor import Tensor

from poisoning_transforms.model.model_poisoner import ModelPoisoner

class ModelReplacement(ModelPoisoner):
    def __init__(self, global_weights: np.ndarray, gamma: float = 1):
        """
        Initializes the ModelReplacement poisoner.

        Args:
            global_weights (np.ndarray): global model's weight arrays (G^t).
            gamma (float): The scaling factor (Γ).
        """
        super().__init__()
        self.global_weights = global_weights
        self.gamma = gamma

    def fit(self, weights: List[Tensor]) -> None:
        pass
    def transform(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies weight replacement poisoning to the list of weights.

        Args:
            weights (List[np.ndarray]): List of weight arrays to be poisoned.

        Returns:
            List[np.ndarray]: List of poisoned weight arrays.
        """
        poisoned_weights = self.gamma * (weights - self.global_weights) + self.global_weights
        return poisoned_weights