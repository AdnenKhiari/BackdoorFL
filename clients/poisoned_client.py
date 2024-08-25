from typing import Dict
from clients.client import FlowerClient
from models.model import train, test
from flwr.common import NDArrays, Scalar,Context

from poisoning_transforms.data.datapoisoner import DataPoisoner, DataPoisoningPipeline
from poisoning_transforms.data.patch.SimplePatch import SimplePatchPoisoner
from poisoning_transforms.model import model_poisoner
from poisoning_transforms.model.model_poisoner import ModelPoisoner, ModelPoisoningPipeline
from poisoning_transforms.model.model_replacement import ModelReplacement

class PoisonedFlowerClient(FlowerClient):
    def __init__(self,trainloader, vallodaer,model_cfg,optimizer) -> None:
        super(PoisonedFlowerClient,self).__init__(trainloader,vallodaer,model_cfg,optimizer)
        self.data_poisoner : DataPoisoner = None
        self.model_poisoner : ModelPoisoner = None
    def fit(self, parameters, config):
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)
        
        # Inject Backdoor
        if self.data_poisoner is None:
            raise Exception("Implement a data poisoner")
        [_ for _ in self.data_poisoner.wrap_fit_iterator(self.trainloader)]
        backdoored_train = lambda d : self.data_poisoner.wrap_transform_iterator(self.trainloader)
        
        # Get Config
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]
        optim = self.optimizer(self.model.parameters(), lr=lr, momentum=momentum)
        
        train(self.model,backdoored_train, optim, epochs, self.device)
        
        # Poison Weights
        params = self.get_parameters({})
        if self.model_poisoner is None:
            raise Exception("Implement a model poisoner")
        self.model_poisoner.fit(params)
        backdoored_params = self.model_poisoner.transform(params)
        
        return backdoored_params, len(self.trainloader), {"Poisoned": True}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        mt_loss, mta_metrics = test(self.model, self.valloader, self.device)
        backdoored_valid = lambda d :self.data_poisoner.wrap_transform_iterator(self.valloader)
        attack_loss, attack_metrics = test(self.model, backdoored_valid, self.device)

        return float(mt_loss), len(self.valloader), {"AttackLoss": attack_loss,"MTA": mta_metrics["accuracy"],"ASR": attack_metrics["accuracy"]}