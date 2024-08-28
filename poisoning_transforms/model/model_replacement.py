import numpy as np
from typing import List, Union

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

    def fit(self, weights:Union[np.ndarray,list]) -> None:
        """
        This method is not used in this poisoner but is required by the abstract base class.

        Args:
            weights (np.ndarray): weight to fit the poisoner.
        """
        # No fitting logic required for this poisoner
        pass

    def transform(self, weights: Union[np.ndarray,list]) -> np.ndarray:
        """
        Applies weight replacement poisoning to the weights.

        Args:
            weights (np.ndarray): weight array to be poisoned.

        Returns:
            np.ndarray: weights arrays.
        """
        poisoned_weights =self.gamma * (np.array(weights) - self.global_weights) + self.global_weights
        return poisoned_weights
