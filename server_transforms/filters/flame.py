import numpy as np
import sklearn.metrics.pairwise as smp
from typing import List, Tuple
from copy import deepcopy
from typing import List, Tuple, Dict, Union
from abc import ABC, abstractmethod
from flwr.server.strategy import Strategy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays
)
import wandb

from server_transforms.wrapper import StrategyWrapper
import numpy as np
import hdbscan
import sklearn.metrics.pairwise as smp
from copy import deepcopy
from typing import List, Tuple
from flwr.common import NDArrays
from scipy.spatial.distance import cosine

class FLAMEStrategyWrapper(StrategyWrapper):
    def __init__(self, strategy: Strategy,poisoned_clients,client_ids, lamda: float = 0.001, min_cluster_size: int = None,wandb_active=False):
        """
        Initialize the FLAME StrategyWrapper.

        Args:
            strategy (Strategy): The strategy to wrap.
            lamda (float): The noise addition parameter for FLAME.
            min_cluster_size (int): Minimum cluster size for HDBSCAN. If not specified, defaults to half the number of clients + 1.
        """
        super().__init__(strategy,poisoned_clients,client_ids,wandb_active)
        self.lamda = lamda
        self.min_cluster_size = min_cluster_size
    
    def process_weights(self, weights: List[Tuple[NDArrays, int, int]]) -> List[Tuple[NDArrays, int, int]]:
        # Step 1: Collect local parameters
        local_params = []
        norm_distances = []
        
        client_ids = []
        poisoned_clients = []
        
        for params, _, client_id in weights:
            # Collect parameters into a single flat array
            flat_params = np.concatenate([layer.flatten() for layer in params])
            local_params.append(flat_params)
            
            #W&B
            client_ids.append(client_id)
            if client_id in self._poisoned_clients:
                poisoned_clients.append(client_id)
            
            # Calculate the norm distance for norm clipping later
            norm_distances.append(np.linalg.norm(flat_params - np.concatenate([layer.flatten() for layer in self._global_model])))
        
        local_params = np.array(local_params)
        norm_distances = np.array(norm_distances)
        
        # Step 2: Compute cosine distances and perform HDBSCAN clustering
        cosine_distances = smp.cosine_distances(local_params)
        hdbscan_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size or (len(weights) // 2 + 1), 
            min_samples=1, 
            metric='precomputed',
            allow_single_cluster=True
        )
        cluster_labels = hdbscan_clusterer.fit_predict(cosine_distances)
        
        # Step 3: Norm-clipping
        norm_threshold = np.median(norm_distances)
        reweighted_weights = []
        
        rejected_list = []
        accepted_list = []
        
        for i, (params, num_examples, client_id) in enumerate(weights):
            if cluster_labels[i] == -1:  # Ignore outliers
                rejected_list.append(client_id)
                continue
            
            if norm_threshold / norm_distances[i] < 1:
                scale_factor = norm_threshold / norm_distances[i]
                scaled_params = [
                    layer * scale_factor for layer in params
                ]
            else:
                scaled_params = params
            
            reweighted_weights.append((scaled_params, num_examples, client_id))
        
        # Step 4: Add noise
        for i, (params, num_examples, client_id) in enumerate(reweighted_weights):
            noisy_params = [
                layer + np.random.normal(0, self.lamda * norm_threshold, layer.shape) for layer in params
            ]
            reweighted_weights[i] = (noisy_params, num_examples, client_id)
            
        
        # Benign recall
        benign_clients = [cid for cid in client_ids if cid not in poisoned_clients]
        benign_recall = len([cid for cid in accepted_list if cid in benign_clients]) / len(benign_clients) if benign_clients else 0
        print(f"Benign Recall (accuracy in accepting benign clients): {benign_recall:.2f}")

        # Weakness percentage
        poisoned_accepted = len([cid for cid in accepted_list if cid in poisoned_clients])
        weakness_percentage = poisoned_accepted / len(poisoned_clients) if len(poisoned_clients) > 0 else 0
        print(f"Weakness Percentage (poisoned clients mistakenly accepted): {weakness_percentage:.2f}")
    
        
        if self.wandb_active:
            wandb.log({
                "Accepted Clients": len(accepted_list),
                "Rejected Clients": len(rejected_list),
                "metrics.current_round": self.server_round,
                "benign_recall": benign_recall,
                "weakness_percentage": weakness_percentage
            })

        
        return reweighted_weights
