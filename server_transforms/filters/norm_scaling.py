import numpy as np
from typing import List, Tuple
from flwr.common import NDArrays
from flwr.server.strategy import Strategy
from server_transforms.wrapper import StrategyWrapper
from flwr.common import NDArrays
from server_transforms.wrapper import StrategyWrapper
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays
)
class NormScalingFilter(StrategyWrapper):
    def __init__(self, strategy: Strategy, clamp_value: float):
        """
        Initialize the NormScalingFilter.

        Args:
            strategy (Strategy): The strategy to wrap.
            clamp_value (float): The maximum allowable difference between global model and client's weights.
        """
        super().__init__(strategy)
        self.clamp_value = clamp_value

    def process_weights(self, weights: List[Tuple[NDArrays, int]]) -> List[Tuple[NDArrays, int]]:
        """
        Clamp the weight updates to prevent norm boosting attacks.

        Args:
            weights (List[Tuple[NDArrays, int]]): The list of tuples where each tuple contains numpy arrays (model weights) and the number of examples.

        Returns:
            List[Tuple[NDArrays, int]]: The list of tuples with clamped weights and the number of examples.
        """

        clamped_weights = []
        
        for model_weights, num_examples in weights:
            # Compute the clamped weights
            clamped_model_weights = []
            
            for global_array, client_array in zip(self._global_model, model_weights):
                # Compute the difference
                difference = client_array - global_array
                # Clamp the difference
                clamped_difference = np.clip(difference, -self.clamp_value, self.clamp_value)
                # Apply the clamped difference to the global weights
                clamped_model_weights.append(global_array + clamped_difference)
            
            clamped_weights.append((clamped_model_weights, num_examples))
        
        return clamped_weights

    def aggregate_fit(
        self, server_round: int, results, 
        failures
    ):
        # Convert FitRes parameters to list of tuples (weights, num_examples)
        weights_list = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
        
        # Process the list of tuples using the `process_weights` method
        processed_weights = self.process_weights(weights_list)
        
        # Convert processed weights back to Parameters
        processed_results = [
            (client_proxy, FitRes(ndarrays_to_parameters(weights), num_examples))
            for (client_proxy, fit_res), (weights, num_examples) in zip(results, processed_weights)
        ]
        
        # Aggregate the processed results using the wrapped strategy's aggregate_fit
        new_global_parameters, metrics = self._strategy.aggregate_fit(server_round, processed_results, failures)
        
        return new_global_parameters, metrics
