from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, NDArrays
import wandb
from server_transforms.wrapper import StrategyWrapper
from flwr.server.server import Strategy
import sklearn.metrics.pairwise as smp

class RFLBATWrapper(StrategyWrapper):
    def __init__(self, strategy: Strategy, poisoned_clients, epsilon1: float = 10, epsilon2: float = 4, num_sampling: int = 5, K_max: int = 10, wandb_active=False):
        super().__init__(strategy, poisoned_clients, wandb_active)
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.num_sampling = num_sampling
        self.K_max = K_max

    def gap_statistics(self, data, num_sampling, K_max, n):
        print("Calculating gap statistics...")
        data = np.reshape(data, (data.shape[0], -1))
        data_c = np.zeros(shape=data.shape)
        for i in range(data.shape[1]):
            data_c[:, i] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]))

        gap = []
        s = []
        for k in range(1, K_max + 1):
            print(f"Evaluating k={k} for gap statistics...")
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
                print(f"Optimal number of clusters determined: {k}")
                return k

    def viz_pca_with_colors(self, dataAll, poisoned_clients_indicies, X_dr):
        benign_clients_indicies = np.setdiff1d(np.arange(dataAll.shape[0]), poisoned_clients_indicies)
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(X_dr[poisoned_clients_indicies, 0], X_dr[poisoned_clients_indicies, 1], color='red', label='Poisoned Clients', alpha=0.7)
        plt.scatter(X_dr[benign_clients_indicies, 0], X_dr[benign_clients_indicies, 1], color='blue', label='Benign Clients', alpha=0.7)
        plt.title('PCA of Clients (Poisoned vs Benign)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        
        if(self.wandb_active):
            wandb.log({
                "Initial PCA of Clients (Poisoned vs Benign)": wandb.Image(plt),
                "metrics.current_round": self.server_round
            })
        
        return fig

    def process_weights(self, weights: List[Tuple[NDArrays, int, int]]) -> List[Tuple[NDArrays, int, int]]:
        dataAll = []
        poisoned_clients_indicies = []
        client_ids = []

        print("Starting to process weights...")

        for index, (weight_set, _, node_id) in enumerate(weights):
            flat_weights = np.concatenate([w.flatten() for w in weight_set])
            dataAll.append(flat_weights)
            client_ids.append(node_id)
            if node_id in self._poisoned_clients:
                poisoned_clients_indicies.append(index)
        
        dataAll = np.array(dataAll)

        print(f"Total number of clients: {len(weights)}")
        print(f"Poisoned clients (IDs): {[weights[i][2] for i in poisoned_clients_indicies]}")

        # PCA reduction to 2 components
        pca = PCA(n_components=2)
        X_dr = pca.fit_transform(dataAll)
        
        # Visualize PCA (optional)
        self.viz_pca_with_colors(dataAll, poisoned_clients_indicies, X_dr)

        # Step 1: Initial filtering based on Euclidean distances
        print("Filtering based on Euclidean distances...")
        eu_list = [np.sum([np.linalg.norm(X_dr[i] - X_dr[j]) for j in range(len(X_dr)) if i != j]) for i in range(len(X_dr))]
        accept = [i for i in range(len(eu_list)) if eu_list[i] < self.epsilon1 * np.median(eu_list)]
        accepted_client_ids_1 = [client_ids[i] for i in accept]
        print(f"Clients accepted after first filtering (IDs): {accepted_client_ids_1}")
        rejected_clients_1 = [client_ids[i] for i in range(len(X_dr)) if i not in accept]
        print(f"Clients rejected after first filtering (IDs): {rejected_clients_1}")

        X_filtered = X_dr[accept]

        # Step 2: Clustering with gap statistics
        print("Performing clustering using gap statistics...")
        num_clusters = self.gap_statistics(X_filtered, self.num_sampling, self.K_max, len(X_filtered))
        k_means = KMeans(n_clusters=num_clusters, init='k-means++').fit(X_filtered)
        predicts = k_means.labels_

        # Step 3: Select the best cluster based on cosine similarity
        print("Selecting the best cluster based on cosine similarity...")
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
        accepted_client_ids_2 = [client_ids[i] for i in accept]
        print(f"Clients accepted after clustering (IDs): {accepted_client_ids_2}")

        # Step 4: Final filtering based on Euclidean distances
        print("Final filtering based on Euclidean distances...")
        X_final = X_filtered[[i in accept for i in range(len(X_filtered))]]
        eu_list_final = [np.sum([np.linalg.norm(X_final[i] - X_final[j]) for j in range(len(X_final)) if i != j]) for i in range(len(X_final))]
        final_accept = [accept[i] for i in range(len(eu_list_final)) if eu_list_final[i] < self.epsilon2 * np.median(eu_list_final)]
        final_accepted_client_ids = [client_ids[i] for i in final_accept]
        print(f"Final accepted clients (IDs): {final_accepted_client_ids}")
        rejected_clients_2 = [client_ids[i] for i in accept if i not in final_accept]
        print(f"Clients rejected after final filtering (IDs): {rejected_clients_2}")
        
        if self.wandb_active:
            wandb.log({
                "Accepted Clients": len(final_accepted_client_ids),
                "Rejected Clients": len(rejected_clients_2),
                "metrics.current_round": self.server_round
            })

        # Return the filtered weights
        filtered_weights = [weights[i] for i in final_accept]
        return filtered_weights
