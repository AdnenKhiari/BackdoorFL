from typing import List
from flwr.server.client_manager import SimpleClientManager
from flwr.common.logger import log
from logging import INFO
import random

class ClientM(SimpleClientManager):
    
    def __init__(self,seed):
        # random.seed(seed)
        super().__init__()
        
        