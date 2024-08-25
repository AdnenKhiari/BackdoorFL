from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar,Context
from hydra.utils import instantiate
import torch
import flwr as fl
from models.model import train, test

class FlowerClient(fl.client.NumPyClient):
    """A standard FlowerClient."""

    def __init__(self, trainloader, vallodaer, model_cfg,optimizer) -> None:
        super(FlowerClient,self).__init__()

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
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        optim = self.optimizer(self.model.parameters(), lr=lr, momentum=momentum)
        train(self.model, lambda : self.trainloader, optim, epochs, self.device)

        return self.get_parameters({}), len(self.trainloader), {"Poisoned": 0}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, metrics = test(self.model, lambda : self.valloader, self.device)

        return float(loss), len(self.valloader), {"MTA": metrics["accuracy"],"Poisoned": 0}
