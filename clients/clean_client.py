from collections import Counter, OrderedDict
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
        self.weights = None
        self.grad_filter = grad_filter
        self.grad_filter_conf = grad_filter
        self.can_poison_now = False

    def get_properties(self, config: Dict[str, bool | bytes | float | int | str]) -> Dict[str, bool | bytes | float | int | str]:
        res= super().get_properties(config)
        res["can_poison"] = 0
        return res
        
    def get_weights(self,trainloader):
        
        self.model = instantiate(self.model_cfg)

        
        # Step 1: Calculate class distribution from the training data
        class_counts = Counter()
        
        # Assuming trainloader returns batches with (inputs, labels)
        for batch in trainloader:
            class_counts.update(batch["label"].tolist())
        
        # Step 2: Compute class weights as the inverse of the class frequencies
        total_samples = sum(class_counts.values())
        num_classes = self.model.num_classes

        # Inverse frequency for each class
        class_weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}
        
        
        new_weights = [class_weights.get(i) if i in class_weights else 1.0/num_classes for i in range(num_classes)]
        print("New Weight Arriving Not Normalized",new_weights)

        # Step 3: Convert weights to tensor
        mixed_weights = torch.tensor(new_weights, dtype=torch.float) 
        
        normalized_weights = mixed_weights / mixed_weights.sum()    
        self.model = None

        return normalized_weights
        
    def with_loaders(self,trainloader, vallodaer):
        self.trainloader = trainloader
        self.valloader = vallodaer
        self.weights = self.get_weights(trainloader)
        print("Weights Used For Balancing: ",self.weights)
        self.grad_filter = instantiate(self.grad_filter.filter,weights=self.weights) if self.grad_filter.active else None

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
        train(self.model, lambda : self.trainloader, optim, epochs, self.device,self.pgd_conf,None,self.weights)

        gc.collect()

        return self.get_parameters({}), len(self.trainloader), {"current_round": current_round,"Poisoned": 0}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.report_data()

        
        self.set_parameters(parameters)
        current_round = config["current_round"]

        loss, metrics = test(self.model, lambda : self.valloader, self.device,self.weights)
        if self.global_run:
            wandb.run.log({
                "current_round": current_round,
                "MTA": metrics["accuracy"],
                "MTR": metrics["recall"],
                "MTF1": metrics["f1"],
                "MTP": metrics["precision"]
            })
        gc.collect()
        return float(loss), len(self.valloader), {"current_round": current_round,"MTA": metrics["accuracy"],"MTR": metrics["recall"],"MTF1": metrics["f1"],"MTP": metrics["precision"],"Poisoned": 0}
