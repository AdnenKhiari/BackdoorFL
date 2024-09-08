from typing import List, Tuple, Optional
import numpy as np
from flwr.common import NDArrays, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy, StrategyWrapper

class KrumFilter(StrategyWrapper):
    def __init__(self, strategy: Strategy, num_malicious: int, to_keep: int):
        """
        Initialize the KrumFilter.

        Args:
            strategy (Strategy): The strategy to wrap.
            num_malicious (int): The number of malicious clients.
            to_keep (int): The number of models to keep after applying Krum.
        """
        super().__init__(strategy, [])
        self.num_malicious = num_malicious
        self.to_keep = to_keep

    def process_weights(self, weights: List[Tuple[NDArrays, int,int]]) -> List[Tuple[NDArrays, int,int]]:
        """Process the weights received from the clients."""
        
        # Apply Krum filtering to select reliable weights
        good_weights = self.aggregate_krum(
            results=weights,
            num_malicious=self.num_malicious,
            to_keep=self.to_keep
        )
        return good_weights

    def aggregate_krum(
        self,
        results: List[Tuple[NDArrays, int,int]],
        num_malicious: int,
        to_keep: int
    ) -> NDArrays:
        """Select weights using the Krum algorithm."""
        # Create a list of weights and ignore the number of examples
        weights = [weights for weights, _,_ in results]

        # Compute distances between vectors
        distance_matrix = self._compute_distances(weights)

        # For each client, take the n-f-2 closest parameters vectors
        num_closest = max(1, len(weights) - num_malicious - 2)
        closest_indices = []
        for distance in distance_matrix:
            closest_indices.append(
                np.argsort(distance)[1:num_closest + 1] 
            )

        # Compute the score for each client, which is the sum of the distances
        # of the n-f-2 closest parameters vectors
        scores = [
            np.sum(distance_matrix[i, closest_indices[i]])
            for i in range(len(distance_matrix))
        ]

        if to_keep > 0:
            # Choose to_keep clients and return their weights
            best_indices = np.argsort(scores)[::-1][len(scores) - to_keep:] 
            best_results = [results[i] for i in best_indices]
            return best_results

        # Return the weights that minimize the score (Krum)
        return results[np.argmin(scores)]

    def _compute_distances(self, weights: List[NDArrays]) -> np.ndarray:
        """Compute distances between vectors.

        Input: weights - list of weights vectors
        Output: distances - matrix distance_matrix of squared distances between the vectors
        """
        flat_w = np.array([np.concatenate(p, axis=None).ravel() for p in weights])
        distance_matrix = np.zeros((len(weights), len(weights)))
        for i, flat_w_i in enumerate(flat_w):
            for j, flat_w_j in enumerate(flat_w):
                delta = flat_w_i - flat_w_j
                norm = np.linalg.norm(delta)
                distance_matrix[i, j] = norm ** 2
        return distance_matrix
