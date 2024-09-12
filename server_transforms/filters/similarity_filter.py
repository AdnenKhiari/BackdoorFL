import logging
from typing import List, Tuple, Optional
from matplotlib import pyplot as plt
import numpy as np
from flwr.common import NDArrays, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from server_transforms.wrapper import StrategyWrapper
import wandb
from PIL import Image
import seaborn as sns
class SimilarityFilter(StrategyWrapper):
    def __init__(self, strategy: Strategy,poisoned_clients, similarity_metric: str, threshold: float, p: Optional[int] = None, wandb_active: bool = False):
        """
        Initialize the SimilarityFilter.

        Args:
            strategy (Strategy): The strategy to wrap.
            similarity_metric (str): The similarity metric to use ("cosine" or "lp").
            threshold (float): The similarity threshold to use for filtering.
            p (Optional[int]): The p value for Lp distance (e.g., p=2 for Euclidean distance). Ignored if similarity_metric is "cosine".
            wandb_active (bool): Whether to log metrics and plots to Weights and Biases.
        """
        super(SimilarityFilter, self).__init__(strategy, poisoned_clients,wandb_active)
        self.similarity_metric = similarity_metric
        self.threshold = threshold
        self.p = p
        self.wandb_active = wandb_active
        self.similarity_values = [] 
        self.accepted_clients = []
        self.rejected_clients = []



    def process_weights(self, weights: List[Tuple[NDArrays, int, int]]) -> List[Tuple[NDArrays, int, int]]:
        filtered_weights = []
        global_weights = self._global_model
        for client_weights, num_examples, client_id in weights:
            if self.similarity_metric == "cosine":
                similarity = self._cosine_similarity(global_weights, client_weights)
                self.similarity_values.append(similarity)

                if similarity >= self.threshold:
                    filtered_weights.append((client_weights, num_examples, client_id))
                    self.accepted_clients.append(client_id)
                else:
                    self.rejected_clients.append(client_id)
                    
            elif self.similarity_metric == "lp":
                distance = self._lp_distance(global_weights, client_weights, self.p)
                self.similarity_values.append(distance)

                if distance <= self.threshold:
                    filtered_weights.append((client_weights, num_examples, client_id))
                    self.accepted_clients.append(client_id)
                else:
                    self.rejected_clients.append(client_id)
            else:
                raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        # Log information about accepted and rejected clients
        print(f"Accepted clients: {self.accepted_clients}")
        print(f"Rejected clients: {self.rejected_clients}")
        
        # Log to wandb if active
        if self.wandb_active:
            self._log_wandb_metrics(weights)

        return filtered_weights

    def _cosine_similarity(self, weights1: NDArrays, weights2: NDArrays) -> float:
        """Compute cosine similarity between two sets of weights."""
        dot_product = sum(
            np.sum(layer1 * layer2) for layer1, layer2 in zip(weights1, weights2)
        )
        norm1 = np.sqrt(sum(np.sum(layer ** 2) for layer in weights1))
        norm2 = np.sqrt(sum(np.sum(layer ** 2) for layer in weights2))
        return dot_product / (norm1 * norm2)

    def _lp_distance(self, weights1: NDArrays, weights2: NDArrays, p: int) -> float:
        """Compute Lp distance between two sets of weights."""
        distance = 0
        for layer1, layer2 in zip(weights1, weights2):
            layer_diff = layer1 - layer2
            distance += np.sum(np.abs(layer_diff) ** p)
        return np.power(distance, 1 / p)
    
    def _log_wandb_metrics(self,weights):
        """Log similarity/distribution information to wandb."""
        if self.similarity_metric == "cosine":
            metric_name = "Cosine Similarity"
        elif self.similarity_metric == "lp":
            metric_name = f"L{self.p} Distance"
        else:
            metric_name = "Unknown Metric"
            
        client_ids = [client_id for _, _, client_id in weights]
        plt.subplots(figsize=(10, 5))   
        # Log histogram of similarity values
        wandb.log({
            "Metric Distribution": wandb.Image(sns.barplot(x=client_ids,y=self.similarity_values),caption=f'{metric_name} Distribution'),
            "metrics.current_round": self.server_round
        })

        # Log the number of accepted/rejected clients
        wandb.log({
            "Accepted Clients": len(self.accepted_clients),
            "Rejected Clients": len(self.rejected_clients),
            "metrics.current_round": self.server_round
        })

        # Reset for next round of filtering
        self.similarity_values = []
        self.accepted_clients = []
        self.rejected_clients = []
