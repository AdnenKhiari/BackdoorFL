from collections import OrderedDict
from typing import Dict, List, Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from models.model import test
from poisoning_transforms.data.datapoisoner import DataPoisoner


def get_on_fit_config(config: DictConfig):
    """Return a function to configure the client's fit."""

    def fit_config_fn(server_round: int):
        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
            "current_round": server_round
        }

    return fit_config_fn


def fit_stats(client_metrics: List[Tuple[int, Dict[str, bool]]]) -> float:
    """
    Calculate the proportion of poisoned items in the round.

    Args:
        client_metrics (List[Tuple[int, Dict[str, bool]]]): A list of tuples where each tuple contains a weight (ignored) and a dictionary with a 'Poisoned' key.

    Returns:
        float: The proportion of poisoned items in the round.
    """
    
    # Initialize counters for poisoned and total items
    total_items = 0
    poisoned_items = 0
    
    # Count poisoned items
    for _, metrics in client_metrics:
        for key, value in metrics.items():
            if key == 'Poisoned':
                total_items += 1
                if value:
                    poisoned_items += 1
    
    # Calculate the proportion of poisoned items
    if total_items == 0:
        return 0.0  # Avoid division by zero if no items are present
    
    return {
        "PoisonedClients": poisoned_items,
        "RoundPoisoningPercentage": poisoned_items / total_items
    }

def aggregation_metrics(client_metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Aggregate metrics from all clients with individual weights for each metric.

    Args:
        client_metrics (List[Tuple[int, Dict[str, float]]]): A list of tuples where each tuple contains a weight and a dictionary of metrics.

    Returns:
        Dict[str, float]: A dictionary with aggregated metrics.
    """
    
    poisoning_stats = fit_stats(client_metrics)
    
    # Initialize dictionaries for weighted sums and total weights per metric
    weighted_sums = {}
    total_weights = {}
    
    # Aggregate metrics
    for weight, metrics in client_metrics:
        for key, value in metrics.items():
            if key == 'Poisoned':
                continue
            if key not in weighted_sums:
                weighted_sums[key] = 0
                total_weights[key] = 0
            weighted_sums[key] += value * weight
            total_weights[key] += weight

    # Calculate the weighted average for each metric
    aggregated_metrics = {}
    for key in weighted_sums:
        if total_weights[key] > 0:
            aggregated_metrics[key] = weighted_sums[key] / total_weights[key]
    
    return {
        **aggregated_metrics,
        metrics: aggregated_metrics
    }


def get_evalulate_fn(model_cfg: int, testloader,data_poisoner: DataPoisoner):
    """Return a function to evaluate the global model."""

    def evaluate_fn(server_round: int, parameters, config):
        model : torch.nn.Module = instantiate(model_cfg)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        backdoored_set = data_poisoner.transform(testloader)
        
        mt_loss, mt_metrics = test(model, testloader, device)
        attack_loss, attack_metrics = test(model, backdoored_set, device)

        return mt_loss, {
            "MTA": mt_metrics["accuracy"],
            "ASR": attack_metrics["accuracy"],
            "AttackLoss": attack_loss
        }

    return evaluate_fn
