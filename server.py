from collections import OrderedDict
from typing import Dict, List, Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import wandb
from models.model import test
from poisoning_transforms.data.datapoisoner import DataPoisoner
import numpy as np
from torchvision.utils import make_grid

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

def get_on_eval_config(config: DictConfig):
    """Return a function to configure the client's evaluate."""

    def eval_config_fn(server_round: int):
        return {
            "current_round": server_round
        }

    return eval_config_fn

def fit_stats(client_metrics: List[Tuple[int, Dict[str, bool]]]) -> dict:
    """
    Calculate the proportion of poisoned items in the round.

    Args:
        client_metrics (List[Tuple[int, Dict[str, bool]]]): A list of tuples where each tuple contains a weight (ignored) and a dictionary with a 'Poisoned' key.

    Returns:
        float: The proportion of poisoned items in the round.
    """
    
    # Initialize counters for poisoned and total items
    total_items = 0.0
    poisoned_items = 0.0
    
    print("METRICSSS",client_metrics)
    
    # Count poisoned items
    for _, metrics in client_metrics:
        for key, value in metrics.items():
            if key == 'Poisoned':
                total_items += 1.0
                if value:
                    poisoned_items += 1.0
    
    # Calculate the proportion of poisoned items
    if total_items == 0.0:
        return {
            "PoisonedClients": 0.0,
            "RoundPoisoningPercentage": 0.0
        }  # Avoid division by zero if no items are present
    
    return {
        "PoisonedClients": poisoned_items,
        "RoundPoisoningPercentage": poisoned_items / total_items
    }
def get_aggregation_metrics(global_run):
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
        
        result = {
            "poisoning_stats": poisoning_stats,
            "metrics": aggregated_metrics
        }
        if global_run is not None:
            wandb.run.log(result)

        print("Evaluation Result :",result)
        return result
    return aggregation_metrics
def get_evalulate_fn(model_cfg: int, testloader,data_poisoner: DataPoisoner,global_run):
    """Return a function to evaluate the global model."""

    def evaluate_fn(server_round: int, parameters, config):
        model : torch.nn.Module = instantiate(model_cfg)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        mt_loss, mt_metrics = test(model, lambda : testloader, device)
        
        
        global_asr = 0
        global_attack_loss = 0
        
        if data_poisoner is not None:
            backdoored_set = lambda : data_poisoner.wrap_transform_iterator(testloader)
            attack_loss, attack_metrics = test(model, backdoored_set, device)
            global_asr = attack_metrics["accuracy"]
            global_attack_loss= attack_loss
    
        result = {
            "metrics":{
                "global_loss": mt_loss,
                "global_MTA": mt_metrics["accuracy"],
                "global_ASR": global_asr ,
                "global_AttackLoss": global_attack_loss,
                "current_round": server_round
            }
        }
        if global_run is not None:
            # Randomly sample 4 clean images
            clean_images = []
            poisoned_images = []
            for i, batch in enumerate(testloader):
                if len(clean_images) >= 4:
                    break
                images = batch["image"]
                clean_images.extend([images[j].to(device) for j in range(min(len(images), 4))])

            # Create poisoned versions
            poisoned_images = [data_poisoner.poison(img.unsqueeze(0)).squeeze(0).to(device) for img in clean_images]

            # Compute the differences and amplify
            diffs = [torch.clamp((poisoned - clean) * 10, 0, 1) for clean, poisoned in zip(clean_images, poisoned_images)]

            # Stack them into a single grid
            clean_grid = make_grid(clean_images, nrow=1, normalize=True)
            poisoned_grid = make_grid(poisoned_images, nrow=1, normalize=True)
            diff_grid = make_grid(diffs, nrow=1, normalize=True)

            # Concatenate grids vertically
            grid = torch.cat((clean_grid, poisoned_grid, diff_grid), dim=1)

            # Log to wandb
            global_run.log({"evaluation_images": wandb.Image(grid)})

            # Log other metrics
            global_run.log(result)

        return mt_loss, result

    return evaluate_fn
