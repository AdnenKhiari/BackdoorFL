import os
import pickle
from pathlib import Path
import ray
from flwr.simulation.app import _create_node_id_to_partition_mapping
import hydra
from hydra.utils import instantiate, call
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl

from clients.client import generate_client_fn, get_partitioner
from custom_simulation.simulation import get_client_ids, start_simulation
from server import aggregation_metrics, fit_stats, get_evalulate_fn
import numpy as np

@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir
    client_ids = get_client_ids(cfg.num_clients)
    honest_clients = np.random.choice(client_ids, int(cfg.num_clients * (1-cfg.poisoned_clients_ratio)), replace=False)
    poisoned_clients = list(set(client_ids) - set(honest_clients))
    
    # Generate Classes for poisoned clients with partitions and poisons ready
    
    
    # dataset = instantiate(cfg.dataset["class"],test_partitioner)
    # evaluate_fn = get_evalulate_fn(cfg.model, dataset.get_test_set(cfg.batch_size),data_poisoner)
    # test_partitioner = get_partitioner(cfg.dataset,cfg.partitioners,cfg.global_seed,num_partitions=1)
    
    client_fn = generate_client_fn(honest_clients,cfg.optimizers,cfg.model,cfg.dataset,cfg.partitioners,cfg.batch_size,cfg.valratio,cfg.global_seed,cfg.poisoned_batch_size,cfg.poisoned_target)
    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=None,
        fit_metrics_aggregation_fn=fit_stats,
        evaluate_metrics_aggregation_fn=aggregation_metrics,
    )
   
    ## 5. Start Simulation
    # As you'll notice, we can start the simulation in exactly the same way as we did in the previous project.
    history = start_simulation(
        client_fn=client_fn,
        clients_ids=client_ids,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": cfg.num_cpus_per_client, "num_gpus": cfg.num_gpus_per_client},
    )

    ## 6. Save your results
    # now we save the results of the simulation.
    results_path = Path(save_path) / "results.pkl"

    results = {"history": history, "anythingelse": "here"}

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(str(save_path),"report.txt"), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
