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

    def process_weights(self, weights: List[Tuple[NDArrays, int, int]]) -> List[Tuple[NDArrays, int, int]]:
        filtered_weights = []
        global_weights = self._global_model

        for client_weights, num_examples, client_id in weights:
            if self.similarity_metric == "cosine":
                similarity = self._cosine_similarity(global_weights, client_weights)
                if similarity >= self.threshold:
                    filtered_weights.append((client_weights, num_examples, client_id))
            elif self.similarity_metric == "lp":
                distance = self._lp_distance(global_weights, client_weights, self.p)
                if distance <= self.threshold:
                    filtered_weights.append((client_weights, num_examples, client_id))
            else:
                raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        return filtered_weights


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
