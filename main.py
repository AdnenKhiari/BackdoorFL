import pickle
from pathlib import Path

import hydra
from hydra.utils import instantiate, call
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from clients.client import generate_client_fn, get_partitioner
from server import get_evalulate_fn


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir
    client_fn = generate_client_fn(cfg.optimizers,cfg.model,cfg.dataset,cfg.partitioners,cfg.batch_size,cfg.valratio,cfg.global_seed)
    
    test_partitioner = get_partitioner(cfg.dataset,cfg.partitioners,cfg.global_seed,num_partitions=1)
    dataset = instantiate(cfg.dataset["class"],test_partitioner)
    strategy = instantiate(
        cfg.strategy, evaluate_fn=get_evalulate_fn(cfg.model, dataset.get_test_set(cfg.batch_size))
    )

    ## 5. Start Simulation
    # As you'll notice, we can start the simulation in exactly the same way as we did in the previous project.
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
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


if __name__ == "__main__":
    main()
