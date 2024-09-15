#!/bin/bash
script_path=$1

# Run the Python command with the script path
HYDRA_FULL_ERROR=1 && python "$script_path" \
  num_gpus_per_client=0.25 \
  num_cpus_per_client=1 \
  num_rounds=100 \
  num_clients=20 \
  num_clients_per_round_fit=10 \
  num_clients_per_round_eval=0 \
  config_fit.local_epochs=3 \
  config_fit.lr=0.008 \
  optimizers=sgd \
  dataset=breast \
  model=mobilenet \
  batch_size=512 \
  poisoned_clients_ratio=0.15 \
  poisoned_batch_size=64 \
  global_merger=single \
  poisoned_client=frequency \
  poisoned_client.cfg.norm_scaling_factor=1.2 \
  grad_filter.active=true \
  grad_filter.ratio=0.97 \
  pgd_conf.active=true \
  global_seed=42 \
  wandb.active=true \
  poison_between=[[20,50]] \
  wandb.main_run_name="mobilenet_breast_iid_full_attack" \
  wandb.group_name="benign" \
  wandb.project_name="official"
