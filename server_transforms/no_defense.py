from typing import Dict, List, Tuple
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from server_transforms.wrapper import StrategyWrapper
from flwr.server.strategy.fedavg import FedAvg

class NoDefense(StrategyWrapper):
    def __init__(self,strategy):
        super().__init__(strategy)
    
    def process_weights(self, weights):
        return weights
    
    def post_process_weights(self, weights):
        return weights
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[ClientProxy, FitRes] | BaseException]) -> Tuple[Parameters | None, Dict[str, bool | bytes | float | int | str]]:
        return self._strategy.aggregate_fit(server_round, results, failures)