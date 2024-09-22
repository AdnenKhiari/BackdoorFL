import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple, Union
from flwr.common import NDArrays
from flwr.server.strategy import Strategy

from server_transforms.wrapper import StrategyWrapper

#NOT USED
class HDBSCANFilter(StrategyWrapper):
    def __init__(self, strategy: Strategy, distance_metric: str = 'cosine', min_cluster_size: int = 2, min_samples: int = 1):
        """
        Initialize the HDBSCANFilter.

        Args:
            strategy (Strategy): The base strategy to wrap.
            distance_metric (str): The distance metric to use ('cosine' or 'lp').
            min_cluster_size (int): The minimum size of clusters.
            min_samples (int): The minimum number of samples in a neighborhood for a point to be considered a core point.
        """
        super().__init__(strategy)
        self.distance_metric = distance_metric
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def compute_distance_matrix(self, local_params: np.ndarray) -> np.ndarray:
        """
        Compute the distance matrix based on the chosen distance metric.

        Args:
            local_params (np.ndarray): Array of local model parameters.

        Returns:
            np.ndarray: Distance matrix.
        """
        if self.distance_metric == 'cosine':
            return cosine_distances(local_params)
        else:  # Lp norm distance
            p = int(self.distance_metric[1:]) if self.distance_metric != 'linf' else np.inf
            return squareform(pdist(local_params, metric='minkowski', p=p))

    def process_weights(self, weights: List[Tuple[NDArrays, int, int]]) -> List[Tuple[NDArrays, int, int]]:
        """
        Apply HDBSCAN clustering to the weights.

        Args:
            weights (List[Tuple[NDArrays, int, int]]): The list of tuples where each tuple contains numpy arrays, the number of examples, and node_id.

        Returns:
            List[Tuple[NDArrays, int, int]]: The filtered list of tuples.
        """
        # Flatten and concatenate weights from all clients
        local_params = np.array([np.concatenate([w.flatten() for w in client_weights]) for client_weights, _, _ in weights])

        # Compute the distance matrix
        distance_matrix = self.compute_distance_matrix(local_params)

        # Apply HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size  or (len(weights) // 2 + 1),
            min_samples=self.min_samples,
            metric='precomputed',
            allow_single_cluster=True
        )
        labels = clusterer.fit_predict(distance_matrix)

        # Filter out the weights that are considered outliers (label -1)
        filtered_weights = [
            (weights[idx][0], weights[idx][1], weights[idx][2]) for idx, label in enumerate(labels) if label != -1
        ]

        return filtered_weights
