from typing import Dict
from clients.client import FlowerClient
from models.model import train, test
from flwr.common import NDArrays, Scalar,Context

from poisoning_transforms.data.datapoisoner import DataPoisoner
from poisoning_transforms.model.model_poisoner import ModelPoisoner

class PoisonedFlowerClient(FlowerClient):
    def __init__(self,trainloader, vallodaer,model_cfg,optimizer,data_poisoner : DataPoisoner,model_poisoner: ModelPoisoner) -> None:
        super(PoisonedFlowerClient).__init__(trainloader,vallodaer,model_cfg,optimizer)
        self.data_poisoner = data_poisoner
        self.model_poisoner = model_poisoner
    def set_parameters(self, parameters):
        self.global_params = parameters
        return super().set_parameters(parameters)
    
    def get_parameters(self, config: Dict[str, bool | bytes | float | int | str]):
        params = super().get_parameters(config)
        return params
    
    def fit(self, parameters, config):
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)
        
        # Inject Backdoor
        self.data_poisoner.fit(self.trainloader) 
        backdoored_train = self.data_poisoner.transform(self.trainloader) 
        
        # Get Config
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]
        optim = self.optimizer(self.model.parameters(), lr=lr, momentum=momentum)
        
        train(self.model,backdoored_train, optim, epochs, self.device)
        
        # Poison Weights
        params = self.get_parameters({})
        self.model_poisoner.fit(params)
        backdoored_params = self.model_poisoner.transform(params)
        
        return backdoored_params, len(backdoored_train), {"Poisoned": True}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        mt_loss, mta_metrics = test(self.model, self.valloader, self.device)
        backdoored_valid = self.data_poisoner.transform(self.valloader) 
        attack_loss, attack_metrics = test(self.model, backdoored_valid, self.device)

        return float(mt_loss), len(self.valloader), {"AttackLoss": attack_loss,"MTA": mta_metrics["accuracy"],"ASR": attack_metrics["accuracy"]}