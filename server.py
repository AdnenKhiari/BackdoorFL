from collections import OrderedDict
import gc
from typing import Dict, List, Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import wandb
from models.model import test
from poisoning_transforms.data.datapoisoner import DataPoisoner
import numpy as np
from torchvision.utils import make_grid
import uuid  # Import the uuid module to generate random file names
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
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

def get_fit_stats_fn(global_run):
    def fit_stats_wrapper(client_metrics: List[Tuple[int, Dict[str, bool]]]) -> dict:
        res =  fit_stats(client_metrics)
        if global_run is not None:
            wandb.run.log({
                "poisoning_stats": res,
                "metrics":{
                    "current_round": client_metrics[0][1]["current_round"],
                }
            })
        return res
    

    def log_heatmap_wandb(history,clients):
        """Log the history of selected clients (poisoned/non-poisoned) as a heatmap to wandb."""
        if not history:
            return  # No history to log

        # Convert history to a NumPy array for visualization
        all_clients = list(map(lambda d: d.node_id,clients.values()))  # Retrieve all client IDs
        num_rounds = len(history)
        num_clients = len(all_clients)

        # Create a matrix (rounds x clients) to hold the heatmap data
        heatmap_data = np.zeros((num_rounds, num_clients))

        for round_num, client_data in history.items():
            for node_id, state in client_data.items():
                client_idx = all_clients.index(node_id)
                heatmap_data[round_num, client_idx] = state

        # Create a custom colormap: dark white for not selected, soft blue for selected, red for poisoned
        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap",
            [(0, "#f0f0f0"),   # 0 - dark white (not selected)
            (0.5, "#add8a7"), # 0.5 - soft blue (selected)
            (1, "#ff0000")],  # 1 - red (poisoned)
            N=256
        )

        # Create a heatmap plot
        plt.figure(figsize=(25, 15))
        sns.heatmap(heatmap_data, cmap=cmap, cbar=False, 
                    xticklabels=all_clients,  # Use client IDs as x-axis labels
                    yticklabels=[f"{i+1}" for i in range(num_rounds)])

        # Set titles and labels
        plt.title("Client Selection and Poisoning Status Heatmap")
        plt.xlabel("Client IDs")
        plt.ylabel("Rounds")

        # Rotate the x-tick labels for better visibility
        plt.xticks(rotation=90)

        # Save the plot as an image and log to wandb
        plt.savefig("client_poisoning_heatmap.png")
        wandb.log({"Poisoning Heatmap": wandb.Image("client_poisoning_heatmap.png")})
        plt.close()


    return fit_stats_wrapper

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
    
    # print("METRICSSS",client_metrics)
    
    # Count poisoned items
    for _, metrics in client_metrics:
        for key, value in metrics.items():
            if key == 'Poisoned':
                total_items += 1.0
                if value == 1:
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
        
        worst_asr = 0
        bast_asr = 0
        
        # Aggregate metrics
        for weight, metrics in client_metrics:
            for key, value in metrics.items():
                if key == 'Poisoned':
                    continue
                if key not in weighted_sums:
                    weighted_sums[key] = 0
                    total_weights[key] = 0
                if key == "ASR":
                    if worst_asr == 0 or value < worst_asr:
                        worst_asr = value
                    if bast_asr == 0 or value > bast_asr:
                        bast_asr = value
                    
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
            wandb.run.log({
                "poisoning_stats": poisoning_stats,
                "metrics":{
                    "current_round": client_metrics[0][1]["current_round"],
                }
            })

        print("Evaluation Result :",result)
        return result
    return aggregation_metrics

def get_evalulate_fn(model_cfg, testloader, data_poisoner: DataPoisoner, global_run):
    """Return a function to evaluate the global model."""
    # Generate a random file name for the model
    random_filename = f"model_{uuid.uuid4().hex}.pth"
    def evaluate_fn(server_round: int, parameters, config):
        model: torch.nn.Module = instantiate(model_cfg)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        mt_loss, mt_metrics = test(model, lambda: testloader, device)

        global_asr = 0
        global_attack_loss = 0

        if data_poisoner is not None:
            backdoored_set = lambda: data_poisoner.wrap_transform_iterator(testloader)
            attack_loss, attack_metrics = test(model, backdoored_set, device,weights=None,mode="micro")
            global_asr = attack_metrics["accuracy"]
            global_attack_loss = attack_loss

        result = {
            "metrics": {
                "global_loss": mt_loss,
                "global_MTA": mt_metrics["accuracy"],
                "global_MTP": mt_metrics["precision"],
                "global_MTR": mt_metrics["recall"],
                "global_MTF1": mt_metrics["f1"],
                "global_ASR": global_asr,
                "global_AttackLoss": global_attack_loss,
                "current_round": server_round,
            }
        }

        if global_run is not None:
            # Randomly sample 4 clean images
            clean_images = []
            poisoned_images = []
            for i, batch in enumerate(testloader):
                images = batch["image"]
                for j in range(4):
                    clean_images.append(images[j].to(device))
                break

            # Create poisoned versions
            poisoned_images = data_poisoner.transform({
                "image": torch.stack(clean_images).to(device),
                "label": torch.tensor([159] * len(clean_images)).to(device)
            })["image"]

            # Compute the differences and amplify
            diffs = [torch.clamp((poisoned - clean) * 10, 0, 1) for clean, poisoned in zip(clean_images, poisoned_images)]

            # Stack them into a single grid
            grid = make_grid(torch.cat([torch.stack(clean_images), poisoned_images, torch.stack(diffs)]), nrow=4, normalize=True)

            # Log to wandb
            global_run.log({"evaluation_images": wandb.Image(grid)})

            # Log the model as an artifact if the server round is even
            if server_round % 50 == 0:
                # Save model weights with the random file name
                model_artifact = wandb.Artifact(random_filename, type="model")
                torch.save(model.state_dict(), random_filename)
                model_artifact.add_file(random_filename)
                
                # Log the artifact to W&B
                global_run.log_artifact(model_artifact)

            # Log other metrics
            global_run.log(result)

        gc.collect()
        return mt_loss, result

    return evaluate_fn