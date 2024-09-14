from typing import List
from flwr.server.client_manager import SimpleClientManager
from flwr.common.logger import log
from logging import INFO
import random

from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
import numpy as np
import wandb
        # Custom colormap for the heatmap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns

class ClientM(SimpleClientManager):
    
    def __init__(self,poisoned_client_ids: List[int],wandb_active: bool) -> None:
        super().__init__()
        self._poisoned_client_ids = poisoned_client_ids
        self._wandb_active = wandb_active
        self.history = {}

    def sample(
        self,
        num_clients: int,
        min_num_clients=None,
        criterion=None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = np.random.choice(available_cids, num_clients, replace=False)
        result = [self.clients[cid] for cid in sampled_cids]
        print("Sampled", list(map(lambda d: d.node_id, result)))

        poisoned_count = 0
        round_data = {}  # To store the poisoning status and selection of each client for this round

        if self._wandb_active:
            for node in result:
                # Check if the client is poisoned
                is_poisoned = (node.node_id in self._poisoned_client_ids) and node.get_properties()["can_poison"]
                if is_poisoned:
                    poisoned_count += 1
                    round_data[node.node_id] = 1  # Poisoned
                else:
                    round_data[node.node_id] = 0.5  # Selected but not poisoned

            # Add non-selected clients with a value of 0 (not selected)
            for cid in self.clients:
                client = self.clients[cid]
                if client.node_id not in round_data:
                    round_data[client.node_id] = 0  # Not selected

            # Store the round data in the history dictionary
            self.history[len(self.history)] = round_data

            # Log poisoning stats to wandb
            wandb.log({
                "poisoning_stats": {
                    "PoisonedClients": poisoned_count,
                    "RoundPoisoningPercentage": poisoned_count / num_clients
                }
            })

            # Generate and log heatmap to wandb
            self.log_heatmap_wandb()

        return result

    def log_heatmap_wandb(self):
        """Log the history of selected clients (poisoned/non-poisoned) as a heatmap to wandb."""
        if not self.history:
            return  # No history to log

        # Convert history to a NumPy array for visualization
        all_clients = list(map(lambda d: d.node_id,self.clients.values()))  # Retrieve all client IDs
        num_rounds = len(self.history)
        num_clients = len(all_clients)

        # Create a matrix (rounds x clients) to hold the heatmap data
        heatmap_data = np.zeros((num_rounds, num_clients))

        for round_num, client_data in self.history.items():
            for node_id, state in client_data.items():
                client_idx = all_clients.index(node_id)
                heatmap_data[round_num, client_idx] = state

        # Create a custom colormap: dark white for not selected, soft blue for selected, red for poisoned
        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap",
            ["#f0f0f0", "#add8e6", "#ff0000"],  # dark white, soft blue, red
            N=256
        )

        # Create a heatmap plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, cmap=cmap, cbar=False, 
                    xticklabels=all_clients,  # Use client IDs as x-axis labels
                    yticklabels=[f"Round {i+1}" for i in range(num_rounds)])

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
