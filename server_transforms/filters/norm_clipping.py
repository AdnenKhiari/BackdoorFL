from typing import List, Tuple, Optional
import numpy as np
from flwr.common import NDArrays, Parameters
from flwr.server.strategy import Strategy, StrategyWrapper

class NormClipping(StrategyWrapper):
    def __init__(self, strategy: Strategy, p: Optional[int] = None, norm_type: str = "lp", threshold: float = 1.0):
        """
        Initialize the LpFilter.

        Args:
            strategy (Strategy): The strategy to wrap.
            p (Optional[int]): The p value for Lp distance (e.g., p=2 for Euclidean distance). 
                               Use "inf" for infinity norm. 
                               This is ignored if norm_type is "inf".
            norm_type (str): The type of norm to use ("lp" or "inf").
            threshold (float): The threshold value for projection.
        """
        super().__init__(strategy, [])
        self.p = p
        self.norm_type = norm_type
        self.threshold = threshold

    def process_weights(self, weights: List[Tuple[NDArrays, int, int]]) -> List[Tuple[NDArrays, int, int]]:
        global_weights = self._global_model
        updated_weights = []

        for client_weights, num_examples, client_id in weights:
            delta = self._compute_delta(global_weights, client_weights)
            projected_delta = self._project_into_lp_ball(delta)
            new_weights = self._apply_delta(projected_delta)
            updated_weights.append((new_weights, num_examples, client_id))

        return updated_weights

    def _compute_delta(self, global_weights: NDArrays, client_weights: NDArrays) -> NDArrays:
        """Compute the delta between the global model weights and client weights."""
        delta = []
        for g_layer, c_layer in zip(global_weights, client_weights):
            delta.append(c_layer - g_layer)
        return delta

    def _project_into_lp_ball(self, delta: NDArrays) -> NDArrays:
        """Project the delta into the Lp ball."""
        # Flatten delta for norm computation
        delta_flat = np.concatenate([layer.flatten() for layer in delta])
        
        if self.norm_type == "lp":
            norm = self._compute_norm(delta_flat)
            if norm > self.threshold:
                scaling_factor = self.threshold / norm
                delta = [layer * scaling_factor for layer in delta]
        elif self.norm_type == "inf":
            max_abs_value = np.max(np.abs(delta_flat))
            if max_abs_value > self.threshold:
                scaling_factor = self.threshold / max_abs_value
                delta = [layer * scaling_factor for layer in delta]
        else:
            raise ValueError(f"Unknown norm type: {self.norm_type}")
        return delta

    def _apply_delta(self,delta: NDArrays) -> NDArrays:
        """Add the projected delta back to the global model weights."""
        new_weights = []
        for g_layer, delta_layer in zip(self._global_model, delta):
            new_weights.append(g_layer + delta_layer)
        return new_weights

    def _compute_norm(self, delta_flat: np.ndarray) -> float:
        """Compute the Lp norm (or infinity norm) of the delta."""
        if self.norm_type == "lp":
            if self.p is None:
                raise ValueError("p value must be provided for Lp norm")
            return np.linalg.norm(delta_flat, ord=self.p)
        elif self.norm_type == "inf":
            return np.linalg.norm(delta_flat, ord=np.inf)
        else:
            raise ValueError(f"Unknown norm type: {self.norm_type}")
