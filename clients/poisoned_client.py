from typing import Dict
from clients.client import FlowerClient
from poisoning_transforms.weights.model_replacement import weight_replacement_poisoning


class PoisonedFlowerClient(FlowerClient):
    def __init__(self, trainloader, vallodaer, model_cfg,optimizer) -> None:
        super(PoisonedFlowerClient).__init__(trainloader,vallodaer,model_cfg,optimizer)
        
    def set_parameters(self, parameters):
        self.global_params = parameters
        return super().set_parameters(parameters)
    
    def get_parameters(self, config: Dict[str, bool | bytes | float | int | str]):
        params = super().get_parameters(config)
        return weight_replacement_poisoning(params,self.global_params,1)
    
    
    