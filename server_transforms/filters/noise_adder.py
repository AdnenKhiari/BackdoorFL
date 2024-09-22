from typing import List, Tuple, Dict, Union
from abc import ABC, abstractmethod
from flwr.server.strategy import Strategy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays
)
from flwr.server.client_proxy import ClientProxy
import numpy as np
import scipy.spatial.distance as ssd
from server_transforms.wrapper import StrategyWrapper

#NOT USED
class NoiseAdder(StrategyWrapper):
    def __init__(self, strategy: Strategy, noise_scale: float):
        """
        Initialize the NoiseAdder.

        Args:
            strategy (Strategy): The strategy to wrap.
            noise_scale (float): The scale of the noise to be added for differential privacy.
        """
        super().__init__(strategy)
        self.noise_scale = noise_scale

    def post_process_weights(self, weights: NDArrays) -> NDArrays:
        """
        Post-process the weights by adding noise for differential privacy.

        Args:
            weights (NDArrays): The weights to post-process.

        Returns:
            NDArrays: The post-processed weights with added noise.
        """
        noisy_weights = []
        for layer_weights in weights:
            noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=layer_weights.shape)
            noisy_layer_weights = layer_weights + noise
            noisy_weights.append(noisy_layer_weights)
        
        return noisy_weights
