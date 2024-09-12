from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar,Context
from hydra.utils import instantiate
import torch
import flwr as fl
import wandb
from wandb.sdk.wandb_run import Run
from models.model import train, test
import gc

class FlowerClient(fl.client.NumPyClient):
    """A standard FlowerClient."""

    def __init__(self,node_id,model_cfg,optimizer,pgd_conf,grad_filter) -> None:
        super(FlowerClient,self).__init__()
        self.model_cfg = model_cfg
        self.optimizer = instantiate(optimizer)
        self.node_id = node_id
        self.model = None
        self.global_run = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pgd_conf = pgd_conf
        self.grad_filter = instantiate(grad_filter.filter) if grad_filter.active else None
        
    def with_loaders(self,trainloader, vallodaer):
        self.trainloader = trainloader
        self.valloader = vallodaer
        return self
    
    def report_data(self):
        if self.global_run:
            print("Found Group",self.global_run.group)
            wandb.init(name=str(self.node_id),project=self.global_run.project,group=self.global_run.group,notes=self.global_run.notes, tags=["client"],config={
                "node_id": self.node_id,
                "poisoned": False
            })
    def register_report_data(self,global_run: Run):
        self.global_run = global_run
        
    def set_parameters(self, parameters):
        gc.collect()

        self.model = instantiate(self.model_cfg)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        if self.model is None:
            self.model = instantiate(self.model_cfg)
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        # Removed From Memory
        self.model = None
        return params

    def fit(self, parameters, config):
        # copy parameters sent by the server into client's local model
        self.report_data()
        self.set_parameters(parameters)
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]
        current_round = config["current_round"]

        optim = self.optimizer(self.model.parameters(), lr=lr, momentum=momentum)
        # if self.grad_filter != None:
        #     self.grad_filter.fit(self.model,self.trainloader)
        train(self.model, lambda : self.trainloader, optim, epochs, self.device,self.pgd_conf,None)
        

        gc.collect()

        return self.get_parameters({}), len(self.trainloader), {"current_round": current_round,"Poisoned": 0}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.report_data()

        
        self.set_parameters(parameters)
        current_round = config["current_round"]

        loss, metrics = test(self.model, lambda : self.valloader, self.device)
        if self.global_run:
            wandb.run.log({
                "current_round": current_round,
                "MTA": metrics["accuracy"]
            })
        gc.collect()
        return float(loss), len(self.valloader), {"current_round": current_round,"MTA": metrics["accuracy"],"Poisoned": 0}
