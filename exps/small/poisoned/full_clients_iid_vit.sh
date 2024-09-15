#!/bin/bash
script_path=$1

# Run the Python command with the script path
HYDRA_FULL_ERROR=1 && python "$script_path" \
  num_gpus_per_client=0.5 \
  num_cpus_per_client=2 \
  num_rounds=40 \
  num_clients=20 \
  num_clients_per_round_fit=10 \
  num_clients_per_round_eval=0 \
  config_fit.local_epochs=2 \
  config_fit.lr=0.007 \
  optimizers=sgd \
  dataset=breast \
  model=vit \
  batch_size=256 \
  poisoned_clients_ratio=0.15 \
  poisoned_batch_size=32 \
  global_merger=single \
  poisoned_client=frequency \
  poisoned_client.cfg.norm_scaling_factor=1.2 \
  grad_filter.active=true \
  grad_filter.filter.ratio=0.95 \
  pgd_conf.active=true \
  global_seed=42 \
  wandb.active=true \
  poison_between=[[5,20]] \
  wandb.main_run_name="vit_breast_iid_full_attack" \
  wandb.group_name="benign" \
  wandb.project_name="official"