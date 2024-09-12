from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, NDArrays
from server_transforms.wrapper import StrategyWrapper
from flwr.server.server import Strategy
import sklearn.metrics.pairwise as smp

class RFLBATWrapper(StrategyWrapper):
    def __init__(self, strategy: Strategy,poisoned_clients, epsilon1: float = 10, epsilon2: float = 4, num_sampling: int = 5, K_max: int = 10,wandb_active = False):
        super().__init__(strategy,poisoned_clients,wandb_active)
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.num_sampling = num_sampling
        self.K_max = K_max

    def gap_statistics(self, data, num_sampling, K_max, n):
        # Implementation of gap statistics function as provided
        data = np.reshape(data, (data.shape[0], -1))
        data_c = np.zeros(shape=data.shape)
        for i in range(data.shape[1]):
            data_c[:, i] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]))

        gap = []
        s = []
        for k in range(1, K_max + 1):
            k_means = KMeans(n_clusters=k, init='k-means++').fit(data_c)
            predicts = k_means.labels_
            centers = k_means.cluster_centers_
            v_k = sum(np.linalg.norm(centers[i] - data_c[predicts == i], axis=1).sum() for i in range(k))

            v_kb = []
            for _ in range(num_sampling):
                data_fake = np.random.uniform(0, 1, (n, data.shape[1]))
                k_means_b = KMeans(n_clusters=k, init='k-means++').fit(data_fake)
                predicts_b = k_means_b.labels_
                centers_b = k_means_b.cluster_centers_
                v_kb_i = sum(np.linalg.norm(centers_b[i] - data_fake[predicts_b == i], axis=1).sum() for i in range(k))
                v_kb.append(v_kb_i)

            gap.append(np.mean(np.log(v_kb)) - np.log(v_k))
            sd = np.sqrt(np.var(np.log(v_kb)) / num_sampling)
            s.append(sd * np.sqrt(1 + 1 / num_sampling))

        for k in range(1, K_max + 1):
            if k == K_max or gap[k - 1] - gap[k] + s[k - 1] > 0:
                return k

    def viz_pca_with_colors(self,dataAll,poisoned_clients_indicies,X_dr):
        # Create a mask for benign clients
        benign_clients_indicies = np.setdiff1d(np.arange(dataAll.shape[0]), poisoned_clients_indicies)

        # Visualize the data
        fig = plt.figure(figsize=(10, 6))

        # Plot poisoned clients in red
        plt.scatter(X_dr[poisoned_clients_indicies, 0], X_dr[poisoned_clients_indicies, 1],
                    color='red', label='Poisoned Clients', alpha=0.7)

        # Plot benign clients in blue
        plt.scatter(X_dr[benign_clients_indicies, 0], X_dr[benign_clients_indicies, 1],
                    color='blue', label='Benign Clients', alpha=0.7)

        # Add labels, title, and legend
        plt.title('PCA of Clients (Poisoned vs Benign)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
       
        return fig

    def process_weights(self, weights: List[Tuple[NDArrays, int, int]]) -> List[Tuple[NDArrays, int, int]]:
        dataAll = []

        # poisoned_clients_indicies : For Viz
        poisoned_clients_indicies = []
        
        for index,(weight_set, _, node_id) in enumerate(weights):
            flat_weights = np.concatenate([w.flatten() for w in weight_set])
            dataAll.append(flat_weights)
            
            #For Viz 
            if node_id in self._poisoned_clients:
                poisoned_clients_indicies.append(index)
                
        dataAll = np.array(dataAll)

        # PCA reduction to 2 components
        pca = PCA(n_components=2)
        X_dr = pca.fit_transform(dataAll)
        
        # For Viz
        self.viz_pca_with_colors(dataAll,poisoned_clients_indicies,X_dr)

        # Compute sum of Euclidean distances and initial filtering
        eu_list = [np.sum([np.linalg.norm(X_dr[i] - X_dr[j]) for j in range(len(X_dr)) if i != j]) for i in range(len(X_dr))]
        accept = [i for i in range(len(eu_list)) if eu_list[i] < self.epsilon1 * np.median(eu_list)]

        X_filtered = X_dr[accept]

        # Clustering using gap statistics to determine the optimal number of clusters
        num_clusters = self.gap_statistics(X_filtered, self.num_sampling, self.K_max, len(X_filtered))
        k_means = KMeans(n_clusters=num_clusters, init='k-means++').fit(X_filtered)
        predicts = k_means.labels_

        # Select the most suitable cluster based on cosine similarity
        v_med = []
        for i in range(num_clusters):
            cluster_indices = [idx for idx, pred in enumerate(predicts) if pred == i]
            cluster_data = [dataAll[accept[j]] for j in cluster_indices]
            if len(cluster_data) <= 1:
                v_med.append(1)
            else:
                v_med.append(np.median(np.mean(smp.cosine_similarity(cluster_data), axis=1)))

        best_cluster = v_med.index(min(v_med))
        accept = [accept[i] for i in range(len(predicts)) if predicts[i] == best_cluster]

        # Recalculate Euclidean distances and further filtering
        X_final = X_filtered[[i in accept for i in range(len(X_filtered))]]
        eu_list_final = [np.sum([np.linalg.norm(X_final[i] - X_final[j]) for j in range(len(X_final)) if i != j]) for i in range(len(X_final))]
        final_accept = [accept[i] for i in range(len(eu_list_final)) if eu_list_final[i] < self.epsilon2 * np.median(eu_list_final)]

        # Return the filtered weights
        filtered_weights = [weights[i] for i in final_accept]
        return filtered_weights 