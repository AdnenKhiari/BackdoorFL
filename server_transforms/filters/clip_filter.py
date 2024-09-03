from typing import List
from click import Tuple
import numpy as np
from flwr.server.strategy import Strategy
from server_transforms.wrapper import StrategyWrapper
from flwr.common import NDArrays
class ClipFilter(StrategyWrapper):
    def __init__(self, strategy: Strategy, min_value: float, max_value: float):
        super().__init__(strategy)
        self.min_value = min_value
        self.max_value = max_value

    def process_weights(self, weights: List[Tuple[NDArrays, int]]) -> List[Tuple[NDArrays, int]]:
        """
        Clip the values in each numpy array of each model's weights to be within the specified range.

        Args:
            weights (List[Tuple[NDArrays, int]]): The list of tuples where each tuple contains numpy arrays and the number of examples.

        Returns:
            List[Tuple[NDArrays, int]]: The list of tuples with numpy arrays clipped within the specified range and the number of examples.
        """
        return [
            (
                [np.clip(array, self.min_value, self.max_value) for array in model_weights],
                num_examples
            )
            for model_weights, num_examples in weights
        ]
