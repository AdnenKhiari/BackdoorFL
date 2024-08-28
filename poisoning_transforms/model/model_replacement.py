import numpy as np
from typing import List, Union

from poisoning_transforms.model.model_poisoner import ModelPoisoner

class ModelReplacement(ModelPoisoner):
    def __init__(self, global_weights: List[np.ndarray], gamma: float = 1):
        """
        Initializes the ModelReplacement poisoner.

        Args:
            global_weights (List[np.ndarray]): List of global model's weight arrays (G^t).
            gamma (float): The scaling factor (Γ).
        """
        super().__init__()
        self.global_weights = global_weights
        self.gamma = gamma

    def fit(self, weights: List[np.ndarray]) -> None:
        """
        This method is not used in this poisoner but is required by the abstract base class.

        Args:
            weights (List[np.ndarray]): List of weight arrays to fit the poisoner.
        """
        # No fitting logic required for this poisoner
        pass

    def transform(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies weight replacement poisoning to each weight array in the list.

        Args:
            weights (List[np.ndarray]): List of weight arrays to be poisoned.

        Returns:
            List[np.ndarray]: List of poisoned weight arrays.
        """
        poisoned_weights = []
        for w, gw in zip(weights, self.global_weights):
            poisoned_w = self.gamma * (w - gw) + gw
            poisoned_weights.append(poisoned_w)
        return poisoned_weights