from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar,Context
from hydra.utils import instantiate
import torch
import flwr as fl
from dataset.dataset import Dataset
from models.model import train, test
from flwr_datasets.partitioner import Partitioner

class FlowerClient(fl.client.NumPyClient):
    """A standard FlowerClient."""

    def __init__(self, trainloader, vallodaer, model_cfg,optimizer) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = vallodaer

        # For further flexibility, we don't hardcode the type of model we use in
        # federation. Here we are instantiating the object defined in `conf/model/net.yaml`
        # (unless you changed the default) and by then `num_classes` would already be auto-resolved
        # to `num_classes=10` (since this was known right from the moment you launched the experiment)
        self.model = instantiate(model_cfg)
        self.optimizer = instantiate(optimizer)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device Used", self.device)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)
        print("CONF:", config)
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        optim = self.optimizer(self.model.parameters(), lr=lr, momentum=momentum)
        train(self.model, self.trainloader, optim, epochs, self.device)

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {"accuracy": accuracy}

def get_partitioner(dataset_cfg,partitioner_cfg,seed_cfg,num_partitions):
    params = {}
    # if partitioner_cfg.get("seed",None) is not None:
    #    params.update({"seed":seed_cfg})
    # if partitioner_cfg.get("partition_by",None) is not None:
    #    params.update({"partition_by":dataset_cfg.label})
    return instantiate(partitioner_cfg,num_partitions=num_partitions,**params)

def generate_client_fn(optimizer_cfg,model_cfg,dataset_cfg,partitioner_cfg,bachsize_cfg,valratio_cfg,seed_cfg):
    """Return a function to construct a FlowerClient."""

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        partitioner : Partitioner = get_partitioner(dataset_cfg,partitioner_cfg,seed_cfg,num_partitions)
        dataset : Dataset= instantiate(dataset_cfg["class"],partitioner=partitioner)
        trainloader, valloader, _ = dataset.load_datasets(partition_id,bachsize_cfg,valratio_cfg,seed_cfg)
        return FlowerClient(
            trainloader=trainloader,
            vallodaer=valloader,
            model_cfg=model_cfg,
            optimizer=optimizer_cfg
        ).to_client()

    return client_fn