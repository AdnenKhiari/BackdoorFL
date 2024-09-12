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
from flwr.server.client_proxy import ClientProxy
import numpy as np
import scipy.spatial.distance as ssd

from server_transforms.wrapper import StrategyWrapper

# TODO : Make it a matrix from the begin , use Module for removing and adding according to memory budget
class FoolsGoldWrapper(StrategyWrapper):
    def __init__(self, strategy: Strategy, num_client_round: int, client_ids: List[int], num_features: int, num_classes: int, memory_budget: int, clip: int = 0, importance: bool = True, importance_hard: bool = False, topk_prop: float = 0.5):
        """
        Initializes the FoolsGoldWrapper class with the given parameters.

        Args:
            strategy (Strategy): The strategy object that this wrapper will use.
            num_client_round (int): The number of clients participating in each round.
            client_ids (List[int]): List of client IDs to initialize the history and client ID to index mapping.
            num_features (int): The number of features in the model.
            num_classes (int): The number of classes in the classification task.
            memory_budget (int): The number of deltas to keep in the history for each client.
            clip (int, optional): Number of clients to exclude based on Krum scores. Defaults to 0 (no clipping).
            importance (bool, optional): Flag to enable or disable importance weighting. Defaults to True.
            importance_hard (bool, optional): Flag to enable or disable hard importance weighting. Defaults to False.
            topk_prop (float, optional): Proportion of features to consider significant for importance weighting. Defaults to 0.5.

        Attributes:
            num_clients (int): The number of clients participating in each round.
            client_ids (List[int]): List of client IDs.
            num_features (int): The number of features in the model.
            num_classes (int): The number of classes in the classification task.
            memory_budget (int): The number of deltas to keep in the history for each client.
            clip (int): Number of clients to exclude based on Krum scores.
            importance (bool): Flag to enable or disable importance weighting.
            importance_hard (bool): Flag to enable or disable hard importance weighting.
            topk_prop (float): Proportion of features to consider significant for importance weighting.
            history (Dict[int, np.ndarray]): Dictionary to keep track of the latest deltas for each client, with client IDs as keys.
            client_id_to_idx (Dict[int, int]): Dictionary mapping client IDs to their indices in the `client_ids` list.
        """
        super().__init__(strategy)
        self.num_clients = num_client_round
        self.client_ids = client_ids
        self.num_features = num_features
        self.num_classes = num_classes
        self.memory_budget = memory_budget
        self.clip = clip
        self.importance = importance
        self.importance_hard = importance_hard
        self.topk_prop = topk_prop
        
        # Initialize history with empty matrices
        self.history = {client_id: np.zeros((memory_budget, num_features)) for client_id in client_ids}
        self.client_id_to_idx = {client_id: self.client_ids.index(client_id) for client_id in self.client_ids}

   
    def update_history(self, client_id: int, flattened_delta: NDArrays):
        # Flatten delta
        history_matrix = self.history[client_id]
        # Add new delta to the history, shift the old ones
        history_matrix = np.roll(history_matrix, shift=1, axis=0)
        history_matrix[0] = flattened_delta
        self.history[client_id] = history_matrix

    def get_cos_similarity(self, full_deltas):
        '''
        Returns the pairwise cosine similarity of client gradients
        '''
        if np.isnan(full_deltas).any():
            raise ValueError("NaN values detected in deltas")
        return 1 - ssd.pdist(full_deltas, 'cosine')

    def importanceFeatureMapGlobal(self, model):
        return np.abs(model) / np.sum(np.abs(model))

    def importanceFeatureMapLocal(self, model, topk_prop=0.5):
        d = self.num_features
        class_d = int(d / self.num_classes)
        M = np.reshape(model, (self.num_classes, class_d))
        
        for i in range(self.num_classes):
            if (M[i].sum() == 0):
                raise ValueError("Zero sum detected in model features")
            M[i] = np.abs(M[i] - M[i].mean())
            M[i] = M[i] / M[i].sum()
            topk = int(class_d * topk_prop)
            sig_features_idx = np.argpartition(M[i], -topk)[0:-topk]
            M[i][sig_features_idx] = 0
        
        return M.flatten()

    def importanceFeatureHard(self, model, topk_prop=0.5):
        class_d = int(self.num_features / self.num_classes)
        M = np.reshape(model, (self.num_classes, class_d))
        importantFeatures = np.ones((self.num_classes, class_d))
        topk = int(class_d * topk_prop)
        for i in range(self.num_classes):
            sig_features_idx = np.argpartition(M[i], -topk)[0:-topk]
            importantFeatures[i][sig_features_idx] = 0
        return importantFeatures.flatten()

    def get_krum_scores(self, X, groupsize):
        krum_scores = np.zeros(len(X))
        distances = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None] - 2 * np.dot(X, X.T)
        for i in range(len(X)):
            krum_scores[i] = np.sum(np.sort(distances[i])[1:(groupsize - 1)])
        return krum_scores
    
    def get_summed_deltas(self, history: Dict[int, np.ndarray]) -> np.ndarray:
        # Initialize an array to hold summed deltas for each client
        summed_deltas = np.zeros((self.num_clients, self.num_features))
        
        # Iterate over each client and sum the latest deltas from the history
        for i, client_id in enumerate(self.client_ids):
            if history[client_id].size > 0:
                summed_deltas[i] = np.sum(history[client_id], axis=0)
        
        return summed_deltas

    def foolsgold(self, deltas, summed_deltas, sig_features_idx, topk_prop=0, importance=False, importanceHard=False, clip=0):
        epsilon = 1e-5
        sd = summed_deltas.copy()
        sig_filtered_deltas = np.take(sd, sig_features_idx, axis=1)

        if importance or importanceHard:
            if importance:
                importantFeatures = self.importanceFeatureMapLocal(np.mean(deltas, axis=0), topk_prop)
            if importanceHard:
                importantFeatures = self.importanceFeatureHard(np.mean(deltas, axis=0), topk_prop)
            for i in range(self.num_clients):
                sig_filtered_deltas[i] = np.multiply(sig_filtered_deltas[i], importantFeatures)
                
        N, _ = sig_filtered_deltas.shape
        cs = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    cs[i, i] = 1
                    continue
                if cs[i, j] != 0 and cs[j, i] != 0:
                    continue
                dot_i = sig_filtered_deltas[i][np.newaxis, :] @ sig_filtered_deltas[j][:, np.newaxis]
                norm_mul = np.linalg.norm(sig_filtered_deltas[i]) * np.linalg.norm(sig_filtered_deltas[j])
                cs[i, j] = cs[j, i] = dot_i / norm_mul

        cs = cs - np.eye(N)
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(self.num_clients):
            for j in range(self.num_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99
        wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        if clip != 0:
            scores = self.get_krum_scores(deltas, self.num_clients - clip)
            bad_idx = np.argpartition(scores, self.num_clients - clip)[(self.num_clients - clip):self.num_clients]
            wv[bad_idx] = 0

        wv = wv / np.sum(wv)
        return wv


    def process_weights(self, weights: List[Tuple[NDArrays, int, int]]) -> List[Tuple[NDArrays, int, int]]:
        deltas = [
            np.concatenate([layer.flatten() for layer in params]) - np.concatenate([layer.flatten() for layer in self._global_model])
            for params, _, client_id in weights
        ]
        
        # Normalize deltas if their norm is higher than 1 ( Norm Clipping )
        for i in range(len(deltas)):
            norm = np.linalg.norm(deltas[i])
            if norm > 1:
                deltas[i] /= norm
            
        # Update history
        for i, (_, _, client_id) in enumerate(weights):
            self.update_history(client_id, deltas[i])
        
        # Sum of deltas
        summed_deltas = self.get_summed_deltas(self.history)
        sig_features_idx = np.arange(self.num_features)
        
        wv = self.foolsgold(
            deltas=np.array(deltas),
            summed_deltas=summed_deltas,
            sig_features_idx=sig_features_idx,
            topk_prop=self.topk_prop,
            importance=self.importance,
            importanceHard=self.importance_hard,
            clip=self.clip
        )
        
        # Map client_id to their respective weight vector
        client_id_to_wv = {client_id: wv[self.client_id_to_idx[client_id]] for _, _, client_id in weights}

        # Reweight the params based on the computed weights
        reweighted_weights = []
        for params, num_examples, client_id in weights:
            # Get the corresponding weight vector for the client
            weight_value = client_id_to_wv[client_id]
            
            # Apply the weight vector to each layer's weights
            reweighted_params = [
                weight_value * layer
                for layer in params
            ]
            
            reweighted_weights.append((reweighted_params, num_examples, client_id))
        
        return reweighted_weights
