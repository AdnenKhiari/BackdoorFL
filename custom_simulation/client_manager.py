from typing import List
from flwr.server.client_manager import SimpleClientManager
from flwr.common.logger import log
from logging import INFO
import random

from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

class ClientM(SimpleClientManager):
    
    def __init__(self,seed):
        # random.seed(seed)
        super().__init__()
        
    def sample(self, num_clients: int, min_num_clients: int | None = None, criterion: Criterion | None = None) -> List[ClientProxy]:
        print("State",random.getstate())
        result = super().sample(num_clients, min_num_clients, criterion)
        print("Sampled",list(map(lambda d: d.node_id,result)))
        return result
        