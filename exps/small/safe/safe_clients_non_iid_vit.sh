#!/bin/bash
script_path=$1


  # Run the Python command with the script path
HYDRA_FULL_ERROR=1 && python "$script_path" \
  num_gpus_per_client=0.5 \
  num_cpus_per_client=2 \
  num_rounds=50 \
  num_clients=20 \
  num_clients_per_round_fit=10 \
  num_clients_per_round_eval=0 \
  config_fit.local_epochs=2 \
  config_fit.lr=0.006 \
  optimizers=sgd \
  dataset=breast \
  model=vit \
  batch_size=64 \
  poisoned_clients_ratio=0 \
  poisoned_batch_size=8 \
  global_merger=single \
  poisoned_client=simple_patch \
  partitioners=noniid \
  partitioners.alpha=2 \
  grad_filter.active=false \
  pgd_conf.active=false \
  global_seed=42 \
  wandb.active=true \
  wandb.main_run_name="vit_breast_non_iid" \
  wandb.group_name="benign" \
  wandb.project_name="official"