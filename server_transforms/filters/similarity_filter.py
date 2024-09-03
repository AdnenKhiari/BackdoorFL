from typing import List, Tuple, Optional
import numpy as np
from flwr.common import NDArrays, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy, StrategyWrapper

class SimilarityFilter(StrategyWrapper):
    def __init__(self, strategy: Strategy, similarity_metric: str, threshold: float, p: Optional[int] = None):
        """
        Initialize the SimilarityFilter.

        Args:
            strategy (Strategy): The strategy to wrap.
            similarity_metric (str): The similarity metric to use ("cosine" or "lp").
            threshold (float): The similarity threshold to use for filtering.
            p (Optional[int]): The p value for Lp distance (e.g., p=2 for Euclidean distance). Ignored if similarity_metric is "cosine".
        """
        super().__init__(strategy, [])
        self.similarity_metric = similarity_metric
        self.threshold = threshold
        self.p = p

    def process_weights(self, weights: List[Tuple[NDArrays, int]]) -> List[Tuple[NDArrays, int]]:
        """
        Filter weights based on the similarity metric.

        Args:
            weights (List[Tuple[NDArrays, int]]): The list of tuples where each tuple contains numpy arrays (model weights) and the number of examples.

        Returns:
            List[Tuple[NDArrays, int]]: The list of tuples with the filtered weights and the number of examples.
        """
        # Calculate the filtered weights based on similarity
        filtered_weights = self._filter_weights(weights)
        
        # Return the filtered weights
        return filtered_weights

    def _filter_weights(self, weights: List[Tuple[NDArrays, int]]) -> List[Tuple[NDArrays, int]]:
        """
        Apply the similarity metric to filter weights.

        Args:
            weights (List[Tuple[NDArrays, int]]): The list of tuples where each tuple contains numpy arrays (model weights) and the number of examples.

        Returns:
            List[Tuple[NDArrays, int]]: The list of tuples with the filtered weights and the number of examples.
        """
        # Convert the weights to a list of NDArrays
        weight_arrays = [weights for weights, _ in weights]
        
        # Select the reference weights (first one in the list)
        reference_weights = weight_arrays[0]
        
        # Compute similarities or distances
        if self.similarity_metric == "cosine":
            similarities = [self._cosine_similarity(reference_weights, w) for w in weight_arrays]
        elif self.similarity_metric == "lp":
            if self.p is None:
                raise ValueError("Parameter 'p' must be specified for Lp distance.")
            similarities = [self._lp_distance(reference_weights, w, self.p) for w in weight_arrays]
        else:
            raise ValueError("Unsupported similarity metric. Choose 'cosine' or 'lp'.")

        # Apply the threshold to filter weights
        filtered_indices = [i for i, sim in enumerate(similarities) if sim >= self.threshold]
        
        # Return the filtered weights
        return [(weight_arrays[i], sum(num_examples for _, num_examples in weights)) for i in filtered_indices]

    def _cosine_similarity(self, weights1: NDArrays, weights2: NDArrays) -> float:
        """Compute cosine similarity between two sets of weights."""
        dot_product = sum(
            np.sum(layer1 * layer2) for layer1, layer2 in zip(weights1, weights2)
        )
        norm1 = np.sqrt(sum(np.sum(layer ** 2) for layer in weights1))
        norm2 = np.sqrt(sum(np.sum(layer ** 2) for layer in weights2))
        return dot_product / (norm1 * norm2)

    def _lp_distance(self, weights1: NDArrays, weights2: NDArrays, p: int) -> float:
        """Compute Lp distance between two sets of weights."""
        distance = 0
        for layer1, layer2 in zip(weights1, weights2):
            layer_diff = layer1 - layer2
            distance += np.sum(np.abs(layer_diff) ** p)
        return np.power(distance, 1 / p)
