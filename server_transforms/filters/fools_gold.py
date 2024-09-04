from typing import List, Tuple, Optional
import numpy as np
from flwr.common import NDArrays, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy, StrategyWrapper
from typing import List
from click import Tuple
import numpy as np
from flwr.server.strategy import Strategy
from server_transforms.wrapper import StrategyWrapper
from flwr.common import NDArrays
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays
)
import sklearn.metrics.pairwise as smp
class FoolsGoldStrategy(StrategyWrapper):
    def __init__(self, strategy: Strategy, total_clients: int):
        super().__init__(strategy)
        self.total_clients = total_clients
        self.history = np.zeros((total_clients, 0))  # Initialize history matrix
    
    def process_weights(self, weights: List[Tuple[NDArrays, int]]) -> List[Tuple[NDArrays, int]]:
        # Flatten and collect the weight updates for a specific layer or all layers
        layer_weights = [np.concatenate([w.flatten() for w in weight_tuple[0]]) for weight_tuple in weights]
        self.history = np.column_stack((self.history, np.array(layer_weights)))

        # Compute cosine similarities
        cosine_similarities = smp.cosine_similarity(self.history) - np.eye(self.total_clients)
        max_similarities = np.max(cosine_similarities, axis=1) + 1e-5

        # Adjust cosine similarities
        for i in range(self.total_clients):
            for j in range(self.total_clients):
                if i != j and max_similarities[i] < max_similarities[j]:
                    cosine_similarities[i][j] *= max_similarities[i] / max_similarities[j]

        # Calculate weights vector
        wv = 1 - np.max(cosine_similarities, axis=1)
        wv = np.clip(wv / np.max(wv), 0, 1)
        wv[wv == 1] = 0.99  # Prevent division by zero in logit
        wv = np.log(wv / (1 - wv) + 1e-5) + 0.5
        wv = np.clip(wv, 0, 1)

        # Reweight the client updates
        processed_weights = [
            ([w * wv[i] for w in weight_tuple[0]], weight_tuple[1]) 
            for i, weight_tuple in enumerate(weights)
        ]

        return processed_weights