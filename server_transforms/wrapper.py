from typing import List, Tuple, Optional, Dict, Union
from abc import ABC, abstractmethod
from flwr.server.strategy import Strategy
from flwr.common.typing import GetParametersIns
from flwr.common import Code
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

class StrategyWrapper(Strategy, ABC):
    def __init__(self, strategy: Strategy,poisoned_clients,all_client_ids,wandb_active = False):
        """
        Initialize the StrategyWrapper.

        Args:
            strategy (Strategy): The strategy to wrap.
        """
        self._strategy = strategy
        self._global_model = None
        self.wandb_active = wandb_active
        self.server_round = 0
        self._metrics = None
        self._poisoned_clients_dict =poisoned_clients
        self._all_client_ids = all_client_ids
        
    @property
    def _poisoned_clients(self):
        return [k for k,v in self._poisoned_clients_dict.items() if v.get_properties({})["can_poison"]]
        
    def get_random_params_from_one_client(self,client_manager):
                # Get initial parameters from one of the clients
        print("Requesting initial parameters from a client")
        all_clients = client_manager.all()
        cid = list(all_clients.keys())[0]
        random_client = all_clients[cid]

        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(
            ins=ins, timeout=5000000, group_id=0
        )
        if get_parameters_res.status.code == Code.OK:
            print("Received initial parameters from one random client")
        else:
            raise Exception("Could not get init weights")
        return parameters_to_ndarrays(get_parameters_res.parameters)
        
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        # params=self._strategy.initialize_parameters(client_manager)
        # if params is not None:
        #     self._global_model = parameters_to_ndarrays(params)
        # else:
        self._global_model = self.get_random_params_from_one_client(client_manager)
        return ndarrays_to_parameters(self._global_model)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        return self._strategy.configure_fit(server_round, parameters, client_manager)

    @abstractmethod
    def process_weights(self, weights: List[Tuple[NDArrays, int,int]]) -> List[Tuple[NDArrays, int,int]]:
        """
        Abstract method to process weights (apply filters or other transformations).

        Args:
            weights (List[Tuple[NDArrays, int]]): The list of tuples where each tuple contains numpy arrays and the number of examples and node_id.

        Returns:
            List[Tuple[NDArrays, int]]: The processed list of tuples with numpy arrays and the number of examples and node_id.
        """
        pass
    
    def post_process_weights(self, weights: NDArrays) -> NDArrays:
        """
        Post-process the weights after aggregation.

        Args:
            weights (NDArrays): The weights to post-process.

        Returns:
            NDArrays: The post-processed weights.
        """
        return weights

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], 
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        self.server_round = server_round
        
        # Convert FitRes parameters to list of tuples (weights, num_examples)
        weights_list = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples,cp.node_id) for cp, fit_res in results]
        
        node_id_to_cp = dict([(cp.node_id, (cp,fitres)) for cp, fitres in results])
        
        # Process the list of tuples using the `process_weights` method
        processed_weights = self.process_weights(weights_list)
        
        filtered_ids = [node_id for (weights, num_examples,node_id) in processed_weights]
        
        filtered_results = [(cp,fit_res) for cp, fit_res in results if cp.node_id in filtered_ids ]
        
        # Convert processed weights back to Parameters
        processed_results = [
            (node_id_to_cp[node_id][0], FitRes(node_id_to_cp[node_id][1].status,ndarrays_to_parameters(weights), num_examples,node_id_to_cp[node_id][1].metrics))
            for weights, num_examples,node_id in  processed_weights
        ]
        
        # Aggregate the processed results using the wrapped strategy's aggregate_fit
        params,metrics = self._strategy.aggregate_fit(server_round, processed_results, failures)
        if params is not None:
            self._global_model = self.post_process_weights(parameters_to_ndarrays(params))
        if metrics is not None:
            self._metrics = metrics
        return ndarrays_to_parameters(self._global_model),self._metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        return self._strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self, server_round: int, results, 
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self._strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round: int,parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self._strategy.evaluate(server_round,parameters)
