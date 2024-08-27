import os
import pickle
from pathlib import Path
import hydra
from hydra.utils import instantiate, call
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl

from clients.client import generate_client_fn, get_clients, get_partitioner
from clients.poisoned_client import get_global_data_poisoner
from custom_simulation.simulation import start_simulation
from server import aggregation_metrics, fit_stats, get_evalulate_fn
import numpy as np
import wandb
@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    
    wandb_config = {
        "num_clients": cfg.num_clients,
        "num_rounds": cfg.num_rounds,
        "num_cpus_per_client": cfg.num_cpus_per_client,
        "num_gpus_per_client": cfg.num_gpus_per_client,
        "poisoned_clients_ratio": cfg.poisoned_clients_ratio,
        "global_seed": cfg.global_seed,
        "batch_size": cfg.batch_size,
        "valratio": cfg.valratio,
        "poisoned_batch_size": cfg.poisoned_batch_size,
        "poisoned_target": cfg.poisoned_target,
        "dataset": cfg.dataset,
        "model": cfg.model,
        "optimizers": cfg.optimizers,
        "partitioners": cfg.partitioners,
        "strategy": cfg.strategy,
        "num_classes": cfg.num_classes,
    }
    global_run = wandb.init(project="federated-1", notes="test", tags=[],config=wandb_config)

    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir
    
    client_ids,clients_dict = get_clients(cfg)
    
    global_data_poisoner = get_global_data_poisoner(clients_dict)
    client_fn = generate_client_fn(cfg.dataset,cfg.partitioners,cfg.batch_size,cfg.valratio,cfg.global_seed,clients_dict)

    test_partitioner = get_partitioner(cfg.dataset,cfg.partitioners,cfg.global_seed,num_partitions=1)
    dataset = instantiate(cfg.dataset["class"],partitioner=test_partitioner)
    evaluate_fn = get_evalulate_fn(cfg.model, dataset.get_test_set(cfg.batch_size),global_data_poisoner)

    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=evaluate_fn,
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
        
    global_run.finish()
if __name__ == "__main__":
    main()
