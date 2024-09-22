from typing import Dict, List, Tuple
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from server_transforms.wrapper import StrategyWrapper
    
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays
)
from flwr.server.strategy import Strategy

class ReptileWrapper(StrategyWrapper):
    
    def __init__(self, strategy: Strategy,poisoned_clients,all_client_ids,wandb_active = False,reptile_lr: float = 0.1):
        """
        Initialize the ReptileWrapper.
        
        Args:
        - strategy (Strategy): The strategy to wrap.
        - reptile_lr (float): The learning rate for the Reptile algorithm.
        """
        super().__init__(strategy,poisoned_clients,all_client_ids,wandb_active)
        self.reptile_lr = reptile_lr
    
    def process_weights(self, weights):
        return self._strategy.process_weights(weights)
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[ClientProxy, FitRes] | BaseException]) -> Tuple[Parameters | None, Dict[str, bool | bytes | float | int | str]]:
        
        self.server_round = server_round
        
        # Convert FitRes parameters to list of tuples (weights, num_examples)
        weights_list = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples,cp.node_id) for cp, fit_res in results]
        
        node_id_to_cp = dict([(cp.node_id, (cp,fitres)) for cp, fitres in results])
        
        # Process the list of tuples using the `process_weights` method
        processed_weights = self.process_weights(weights_list)
        
        # Convert processed weights back to Parameters
        processed_results = [
            (node_id_to_cp[node_id][0], FitRes(node_id_to_cp[node_id][1].status,ndarrays_to_parameters(weights), num_examples,node_id_to_cp[node_id][1].metrics))
            for weights, num_examples,node_id in  processed_weights
        ]
        
        # Aggregate the processed results using the wrapped strategy's aggregate_fit
        params,metrics = self._strategy.aggregate_fit(server_round, processed_results, failures)
        if params is not None:
            params = self.post_process_weights(parameters_to_ndarrays(params))
            self._global_model = self.reptile(self._global_model,params)
        if metrics is not None:
            self._metrics = metrics
        return ndarrays_to_parameters(self._global_model),self._metrics    
    

    def reptile(self,global_weights: NDArrays, updated_weights: NDArrays) -> NDArrays:
        """
        Implements the Reptile algorithm for meta-learning by updating the global weights 
        using the difference between the current global weights and updated weights from clients.
        
        Args:
        - global_weights: The current global model weights (list of NumPy arrays).
        - updated_weights: The new weights to incorporate (list of NumPy arrays).
        - lr: Learning rate or step size for Reptile update (default: 0.1).
        
        Returns:
        - Updated global weights as a list of NumPy arrays.
        """
        # Perform the Reptile update
        new_weights = [
            global_w + self.reptile_lr * (updated_w - global_w)
            for global_w, updated_w in zip(global_weights, updated_weights)
        ]
        
        return new_weights