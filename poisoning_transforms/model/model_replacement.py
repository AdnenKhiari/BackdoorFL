import numpy as np
from typing import List

from poisoning_transforms.model.model_poisoner import ModelPoisoner

class   ModelReplacement(ModelPoisoner):
    def __init__(self, global_weights: np.ndarray, gamma: float = 1):
        """
        Initializes the ModelReplacement poisoner.

        Args:
            global_weights (np.ndarray): Global model's weight array (G^t).
            gamma (float): The scaling factor (Γ).
        """
        super().__init__()
        self.global_weights = global_weights
        self.gamma = gamma

    def fit(self, weights: np.ndarray) -> None:
        """
        This method is not used in this poisoner but is required by the abstract base class.

        Args:
            weights (np.ndarray): weight to fit the poisoner.
        """
        # No fitting logic required for this poisoner
        pass

    def transform(self, weights: np.ndarray) -> np.ndarray:
        """
        Applies weight replacement poisoning to the weights.

        Args:
            weights (np.ndarray): weight array to be poisoned.

        Returns:
            np.ndarray: weights arrays.
        """
        print("Weights Type",type(weights))
        poisoned_weights =self.gamma * (weights - self.global_weights) + self.global_weights
        return poisoned_weights
