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
        # Step 1: Compute the total difference vector
        diff_vectors = [w - gw for w, gw in zip(weights, self.global_weights)]
        
        # Step 2: Flatten all differences into a single vector
        concatenated_diff = np.concatenate([diff.flatten() for diff in diff_vectors])
        
        # Step 3: Compute the norm of the concatenated difference vector
        total_norm = np.linalg.norm(concatenated_diff)
        
        print("Old Norm",total_norm)
        
        # Step 4: Normalize and scale each weight array using the total norm
        scaled_weights = []
        for diff, gw in zip(diff_vectors, self.global_weights):
            norm_diff = diff / total_norm  # Normalize using the total norm
            scaled_w = self.gamma * norm_diff + gw  # Scale by gamma and add G back
            scaled_weights.append(scaled_w)
        
        return scaled_weights