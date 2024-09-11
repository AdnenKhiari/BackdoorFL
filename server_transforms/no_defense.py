from server_transforms.wrapper import StrategyWrapper
from flwr.server.strategy.fedavg import FedAvg

class NoDefense(StrategyWrapper):
    def __init__(self,strategy):
        super().__init__(strategy)
    
    def process_weights(self, weights):
        return weights
    
    def post_process_weights(self, weights):
        return weights