from typing import Dict

import wandb
from clients.clean_client import FlowerClient
from models.model import train, test
from flwr.common import NDArrays, Scalar,Context
from wandb.sdk.wandb_run import Run

from poisoning_transforms.data.datapoisoner import BatchPoisoner, DataPoisoner, DataPoisoningPipeline, IgnoreLabel
from poisoning_transforms.data.patch.SimplePatch import SimplePatchPoisoner
from poisoning_transforms.model import model_poisoner
from poisoning_transforms.model.model_poisoner import ModelPoisoner, ModelPoisoningPipeline
from poisoning_transforms.model.model_replacement import ModelReplacement

class PoisonedFlowerClient(FlowerClient):
    def __init__(self,node_id,model_cfg,optimizer,data_poisoner,batch_poison_num,target_poisoned,batch_size) -> None:
        super(PoisonedFlowerClient,self).__init__(node_id,model_cfg,optimizer)
        self.train_data_poisoner : DataPoisoner = BatchPoisoner(data_poisoner,batch_poison_num,target_poisoned)
        self.test_data_poisoner : DataPoisoner = IgnoreLabel(BatchPoisoner(data_poisoner,-1,target_poisoned),target_poisoned,batch_size)
        self.model_poisoner : ModelPoisoner = None
        self.target_poisoned = target_poisoned
    def report_data(self,global_run: Run):
        super().report_data(global_run)
        self.client_run._set_config_wandb("Poisoned",True)
        self.client_run._set_config_wandb("target_poisoned",self.target_poisoned)
        
    def fit(self, parameters, config):
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)
        
        # Inject Backdoor
        if self.train_data_poisoner is None:
            raise Exception("Implement a data poisoner")
        [_ for _ in self.train_data_poisoner.wrap_fit_iterator(self.trainloader)]
        backdoored_train = lambda : self.train_data_poisoner.wrap_transform_iterator(self.trainloader)
        
        # Get Config
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]
        current_round = config["current_round"]
        optim = self.optimizer(self.model.parameters(), lr=lr, momentum=momentum)
        
        train(self.model,backdoored_train, optim, epochs, self.device)
        
        # Poison Weights
        params = self.get_parameters({})
        if self.model_poisoner is None:
            raise Exception("Implement a model poisoner")
        self.model_poisoner.fit(params)
        backdoored_params = self.model_poisoner.transform(params)
        
        return backdoored_params, len(self.trainloader), {"Poisoned": 1,"current_round":current_round}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        current_round = config["current_round"]

        mt_loss, mta_metrics = test(self.model, lambda : self.valloader, self.device)
        backdoored_valid = lambda :self.test_data_poisoner.wrap_transform_iterator(self.valloader)
        attack_loss, attack_metrics = test(self.model, backdoored_valid, self.device)
        self.client_run.log(
            {"current_round":current_round ,"AttackLoss": attack_loss,"MTA": mta_metrics["accuracy"],"ASR": attack_metrics["accuracy"]}
        )
        return float(mt_loss), len(self.valloader), {"current_round":current_round ,"Poisoned": 1,"AttackLoss": attack_loss,"MTA": mta_metrics["accuracy"],"ASR": attack_metrics["accuracy"]}
    
    
    
def get_global_data_poisoner(clients : dict[str,dict[str,PoisonedFlowerClient]]) -> DataPoisoner:
    def get_poisoner():
        data_poisoner = []
        for client in clients["malicious"].values():
            if len(data_poisoner) == 0:
                data_poisoner = [client.test_data_poisoner]
            else:
                pass
                # data_poisoner = DataPoisoningPipeline(data_poisoner, client.train_data_poisoner)
        if len(data_poisoner) == 0:
            return None
        return DataPoisoningPipeline(data_poisoner) 
    return get_poisoner