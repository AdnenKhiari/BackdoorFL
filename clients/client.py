import numpy as np
from clients.clean_client import FlowerClient
from clients.poisoned_client import PoisonedFlowerClient
from clients.simple_poisoned_client import SimplePoisonedClient
from custom_simulation.simulation import get_client_ids
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

def get_clients(cfg):
    client_ids = get_client_ids(cfg.num_clients)
    honest_clients_ids = np.random.choice(client_ids, int(cfg.num_clients * (1-cfg.poisoned_clients_ratio)), replace=False)
    poisoned_clients_ids = list(set(client_ids) - set(honest_clients_ids))
    clients_dict = {"malicious": {},"honest": {}}
    for node_id in honest_clients_ids:
        clients_dict["honest"][node_id] = FlowerClient(
            node_id;
            model_cfg=cfg.model,
            optimizer=cfg.optimizers
        )
    for node_id in poisoned_clients_ids:
        clients_dict["malicious"][node_id] =  SimplePoisonedClient(
            node_id,
            model_cfg=cfg.model,
            optimizer=cfg.optimizers,
            batch_poison_num=cfg.poisoned_batch_size,
            target_poisoned=cfg.poisoned_target,
            batch_size=cfg.batch_size
        )
    return client_ids,clients_dict

def generate_client_fn(dataset_cfg,partitioner_cfg,bachsize_cfg,valratio_cfg,seed_cfg,clients_dict):
    """Return a function to construct a FlowerClient."""
    def good_client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        node_id = context.node_id
        partitioner : Partitioner = get_partitioner(dataset_cfg,partitioner_cfg,seed_cfg,num_partitions)
        dataset : Dataset= instantiate(dataset_cfg["class"],partitioner=partitioner)
        trainloader, valloader, _ = dataset.load_datasets(partition_id,bachsize_cfg,valratio_cfg,seed_cfg)

        return clients_dict["honest"][node_id].with_loaders(trainloader,valloader).to_client()
    
    def poisoned_client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        node_id = context.node_id
        partitioner : Partitioner = get_partitioner(dataset_cfg,partitioner_cfg,seed_cfg,num_partitions)
        dataset : Dataset= instantiate(dataset_cfg["class"],partitioner=partitioner)
        trainloader, valloader, _ = dataset.load_datasets(partition_id,bachsize_cfg,valratio_cfg,seed_cfg)
        return clients_dict["malicious"][node_id].with_loaders(trainloader,valloader).to_client()
    
    def client_fn(context: Context):
        node_id = context.node_id
        if node_id in clients_dict["honest"]:
            return good_client_fn(context)
        return poisoned_client_fn(context)
    return client_fn
