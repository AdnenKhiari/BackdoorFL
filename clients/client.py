import numpy as np
from omegaconf import OmegaConf
from clients.attackers.frequency_poisoned_client import FrequencyPoisonedClient
from clients.attackers.randomized_simple_poisoned_client import RandomizedSimplePoisonedClient
from clients.clean_client import FlowerClient
from clients.attackers.iba_client import IbaClient
from clients.attackers.simple_poisoned_client import SimplePoisonedClient
from custom_simulation.simulation import get_client_ids
from dataset.dataset import Dataset
from flwr_datasets.partitioner import Partitioner
from flwr.common import Context
from hydra.utils import instantiate

from poisoning_transforms.data.patch.frequency import FrequencyDomainPoisoner

def get_partitioner(dataset_cfg,partitioner_cfg,seed_cfg,num_partitions):
    params = {}
    # if partitioner_cfg.get("seed",None) is not None:
    #    params.update({"seed":seed_cfg})
    # if partitioner_cfg.get("partition_by",None) is not None:
    #    params.update({"partition_by":dataset_cfg.label})
    res = instantiate(partitioner_cfg,num_partitions=num_partitions,**params)
    return res


def simple_poisoned_client_fn(node_id,cfg,poison_between):
    return  SimplePoisonedClient(
                node_id,
                model_cfg=cfg.model,
                optimizer=cfg.optimizers,
                batch_poison_num=cfg.poisoned_batch_size,
                target_poisoned=cfg.poisoned_target,
                batch_size=cfg.batch_size,
                patch_size=cfg.patch_size,
                patch_loc=cfg.patch_loc,
                patch_val=cfg.patch_val,
                norm_scaling_factor=cfg.norm_scaling_factor,
                grad_filter=cfg.grad_filter,
                pgd_conf=cfg.pgd_conf,
                poison_between=poison_between
        )
    
def randomized_simple_poisoned_client_fn(node_id,cfg,poison_between):
    return  RandomizedSimplePoisonedClient(
                node_id,
                model_cfg=cfg.model,
                optimizer=cfg.optimizers,
                batch_poison_num=cfg.poisoned_batch_size,
                target_poisoned=cfg.poisoned_target,
                batch_size=cfg.batch_size,
                norm_scaling_factor=cfg.norm_scaling_factor,
                grad_filter=cfg.grad_filter,
                pgd_conf=cfg.pgd_conf,
                patch_location_range=cfg.patch_location_range,
                patch_size_range=cfg.patch_size_range,
                patch_val=cfg.patch_val,
                seed=cfg.global_seed,
                poison_between=poison_between

        )
    
def frequency_client_fn(node_id,cfg,poison_between):
    return  FrequencyPoisonedClient(
                node_id,
                model_cfg=cfg.model,
                optimizer=cfg.optimizers,
                batch_poison_num=cfg.poisoned_batch_size,
                target_poisoned=cfg.poisoned_target,
                batch_size=cfg.batch_size,
                window_size=cfg.window_size,
                channel_list=cfg.channel_list,
                pos_list=cfg.pos_list,
                magnitude=cfg.magnitude,
                norm_scaling_factor=cfg.norm_scaling_factor,
                grad_filter=cfg.grad_filter,
                pgd_conf=cfg.pgd_conf,
                poison_between=poison_between
        )
    
    
def get_clients(cfg,global_run):
    client_ids = get_client_ids(cfg.num_clients)
    honest_clients_ids = np.random.choice(client_ids, int(cfg.num_clients * (1-cfg.poisoned_clients_ratio)), replace=False)
    poisoned_clients_ids = list(set(client_ids) - set(honest_clients_ids))
    clients_dict = {"malicious": {},"honest": {}}
    for node_id in honest_clients_ids:
        clients_dict["honest"][node_id] = FlowerClient(
            node_id,
            model_cfg=cfg.model,
            optimizer=cfg.optimizers,
            pgd_conf=cfg.pgd_conf,
            grad_filter=cfg.grad_filter
        )
        if cfg.wandb.active and cfg.wandb.report_clients:
            print("Using Wandb for honest client",node_id)
            clients_dict["honest"][node_id].register_report_data(global_run)
    for node_id in poisoned_clients_ids:
        clients_dict["malicious"][node_id] = instantiate(cfg.poisoned_client,node_id=node_id,poison_between=cfg.poison_between)
        if cfg.wandb.active and cfg.wandb.report_clients:
            print("Using Wandb for maliciours client",node_id)
            clients_dict["malicious"][node_id].register_report_data(global_run)
    return client_ids,clients_dict

def generate_client_fn(dataset_cfg,partitioner_cfg,bachsize_cfg,valratio_cfg,seed_cfg,clients_dict):
    """Return a function to construct a FlowerClient."""
    def good_client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        node_id = context.node_id
        partitioner : Partitioner = get_partitioner(dataset_cfg,partitioner_cfg,seed_cfg,num_partitions)
        dataset : Dataset= instantiate(dataset_cfg["class"],partitioner=partitioner,seed=seed_cfg)
        trainloader, valloader, _ = dataset.load_datasets(partition_id,bachsize_cfg,valratio_cfg,seed_cfg)

        return clients_dict["honest"][node_id].with_loaders(trainloader,valloader).to_client()
    
    def poisoned_client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        node_id = context.node_id
        partitioner : Partitioner = get_partitioner(dataset_cfg,partitioner_cfg,seed_cfg,num_partitions)
        dataset : Dataset= instantiate(dataset_cfg["class"],partitioner=partitioner,seed=seed_cfg)
        trainloader, valloader, _ = dataset.load_datasets(partition_id,bachsize_cfg,valratio_cfg,seed_cfg)
        return clients_dict["malicious"][node_id].with_loaders(trainloader,valloader).to_client()
    
    def client_fn(context: Context):
        node_id = context.node_id
        if node_id in clients_dict["honest"]:
            return good_client_fn(context)
        return poisoned_client_fn(context)
    return client_fn
