import numpy as np
from typing import List

from poisoning_transforms.model.model_poisoner import ModelPoisoner

class NormScaling(ModelPoisoner):
    def __init__(self, global_weights: List[np.ndarray], gamma: float = 1):
        """
        Initializes the NormScaling object.

        Args:
            global_weights (List[np.ndarray]): List of global model's weight arrays (G).
            gamma (float): The scaling factor (Γ).
        """
        super(NormScaling, self).__init__()
        self.global_weights = global_weights
        self.gamma = gamma

    def fit(self, weights: List[np.ndarray]) -> None:
        """
        This method is not used in this scaler but is required by the API.

        Args:
            weights (List[np.ndarray]): List of weight arrays to fit the scaler.
        """
        # No fitting logic required for this scaler
        pass

    def transform(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies norm scaling to each weight array in the list.

        Args:
            weights (List[np.ndarray]): List of weight arrays to be scaled.

        Returns:
            List[np.ndarray]: List of scaled weight arrays.
        """
        scaled_weights = []
        for w, gw in zip(weights, self.global_weights):
            diff = w - gw  # Compute the difference X - G
            norm_diff = diff / np.linalg.norm(diff)  # Normalize the difference
            scaled_w = self.gamma * norm_diff + gw  # Scale by gamma and add G back
            scaled_weights.append(scaled_w)
        
        return scaled_weights
