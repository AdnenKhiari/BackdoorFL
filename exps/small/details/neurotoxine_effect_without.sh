#!/bin/bash
script_path=$1

# Run the Python command with the script path
HYDRA_FULL_ERROR=1 && python "$script_path" \
  num_gpus_per_client=0.5 \
  num_cpus_per_client=2 \
  num_rounds=100 \
  num_clients=10 \
  num_clients_per_round_fit=2 \
  num_clients_per_round_eval=0 \
  config_fit.local_epochs=2 \
  config_fit.lr=0.09 \
  optimizers=sgd \
  dataset=breast \
  model=vit \
  batch_size=256 \
  poisoned_clients_ratio=0.2 \
  poisoned_batch_size=32 \
  global_merger=single \
  poisoned_client=frequency \
  poisoned_client.cfg.norm_scaling_factor=2 \
  poisoned_client.cfg.magnitude=60 \
  grad_filter.active=false \
  grad_filter.filter.ratio=1 \
  pgd_conf.active=true \
  global_seed=42 \
  wandb.active=true \
  poison_between=[[8,25]] \
  wandb.main_run_name="vit_breast_iid_neurotoxine_test_without" \
  wandb.group_name="neurotoxine" \
  wandb.project_name="official"


