#!/bin/bash
script_path=$1

# Run the Python command with the script path
HYDRA_FULL_ERROR=1 && python "$script_path" \
  num_gpus_per_client=0.25 \
  num_cpus_per_client=1 \
  num_rounds=60 \
  num_clients=50 \
  num_clients_per_round_fit=10 \
  num_clients_per_round_eval=0 \
  config_fit.local_epochs=1 \
  config_fit.lr=0.008 \
  optimizers=sgd \
  dataset=breast \
  model=mobilenet \
  batch_size=512 \
  poisoned_clients_ratio=0 \
  poisoned_batch_size=64 \
  global_merger=single \
  poisoned_client=simple_patch \
  grad_filter.active=false \
  pgd_conf.active=false \
  global_seed=42 \
  wandb.active=true \
  wandb.main_run_name="mobilenet_breast_iid" \
  wandb.group_name="benign" \
  wandb.project_name="federated_official_report"


HYDRA_FULL_ERROR=1 && python "$script_path" \
  num_gpus_per_client=0.25 \
  num_cpus_per_client=1 \
  num_rounds=80 \
  num_clients=50 \
  num_clients_per_round_fit=10 \
  num_clients_per_round_eval=0 \
  config_fit.local_epochs=1 \
  config_fit.lr=0.007 \
  optimizers=sgd \
  dataset=breast \
  model=mobilenet \
  batch_size=256 \
  poisoned_clients_ratio=0 \
  poisoned_batch_size=64 \
  global_merger=single \
  poisoned_client=simple_patch \
  grad_filter.active=false \
  pgd_conf.active=false \
  partitioners=noniid \
  partitioners.alpha=0.9 \
  global_seed=42 \
  wandb.active=true \
  wandb.main_run_name="mobilenet_breast_non_iid" \
  wandb.group_name="benign" \
  wandb.project_name="federated_official_report"