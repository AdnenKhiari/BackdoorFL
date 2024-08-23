from clients.clean_client import FlowerClient
from clients.poisoned_client import PoisonedFlowerClient
from clients.simple_poisoned_client import SimplePoisonedClient
from dataset.dataset import Dataset
from flwr_datasets.partitioner import Partitioner
from flwr.common import Context
from hydra.utils import instantiate

def get_partitioner(dataset_cfg,partitioner_cfg,seed_cfg,num_partitions):
    params = {}
    # if partitioner_cfg.get("seed",None) is not None:
    #    params.update({"seed":seed_cfg})
    # if partitioner_cfg.get("partition_by",None) is not None:
    #    params.update({"partition_by":dataset_cfg.label})
    return instantiate(partitioner_cfg,num_partitions=num_partitions,**params)

def generate_client_fn(honest_clients,optimizer_cfg,model_cfg,dataset_cfg,partitioner_cfg,bachsize_cfg,valratio_cfg,seed_cfg):
    """Return a function to construct a FlowerClient."""

    def good_client_fn(context: Context):
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
        
    def poisoned_client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        partitioner : Partitioner = get_partitioner(dataset_cfg,partitioner_cfg,seed_cfg,num_partitions)
        dataset : Dataset= instantiate(dataset_cfg["class"],partitioner=partitioner)
        trainloader, valloader, _ = dataset.load_datasets(partition_id,bachsize_cfg,valratio_cfg,seed_cfg)
        return SimplePoisonedClient(
            trainloader=trainloader,
            vallodaer=valloader,
            model_cfg=model_cfg,
            optimizer=optimizer_cfg
        ).to_client()
        
    def client_fn(context: Context):
        node_id = context.node_id
        if node_id in honest_clients:
            return good_client_fn(context)
        return poisoned_client_fn(context)
    return client_fn
