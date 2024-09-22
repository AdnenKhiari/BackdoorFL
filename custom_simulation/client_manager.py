from typing import List
from flwr.server.client_manager import SimpleClientManager
from flwr.common.logger import log
from logging import INFO
import random
from flwr.common.typing import GetPropertiesIns,GetPropertiesRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
import numpy as np
import wandb
        # Custom colormap for the heatmap


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

        # poisoned_count = 0
        # round_data = {}  # To store the poisoning status and selection of each client for this round
        # ins = GetPropertiesIns({})
        # if self._wandb_active and num_clients > 0:
        #     for node in result:
        #         # Check if the client is poisoned
        #         round_data[node.node_id] = 0.5  # Selected but not poisoned
        #         is_poisoned = (node.node_id in self._poisoned_client_ids) and node.get_properties(ins=ins,timeout=1000,group_id=0).properties["can_poison"]
        #         if is_poisoned:
        #             poisoned_count += 1
        #             round_data[node.node_id] = 1  # Poisoned

        #     # Add non-selected clients with a value of 0 (not selected)
        #     for cid in self.clients:
        #         client = self.clients[cid]
        #         if client.node_id not in round_data:
        #             round_data[client.node_id] = 0  # Not selected

        #     # Store the round data in the history dictionary
        #     self.history[len(self.history)] = round_data

        return result

