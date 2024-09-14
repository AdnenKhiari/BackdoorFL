#!/bin/bash
script_path=$1
HYDRA_FULL_ERROR=1 && python "$script_path" \
  num_gpus_per_client=0.5 \
  num_cpus_per_client=2 \
  num_rounds=200 \
  num_clients=50 \
  num_clients_per_round_fit=10 \
  num_clients_per_round_eval=0 \
  config_fit.local_epochs=3 \
  config_fit.lr=0.008 \
  optimizers=sgd \
  dataset=breast \
  model=resnet \
  batch_size=512 \
  poisoned_clients_ratio=0.1 \
  poisoned_batch_size=64 \
  poison_between=[[35,85]] \
  global_merger=distributed \
  poisoned_client=randomized_simple_patch \
  poisoned_client.cfg.norm_scaling_factor=2 \
  grad_filter.active=false \
  pgd_conf.active=false \
  global_seed=42 \
  partitioners=noniid \
  wandb.active=true \
  wandb.main_run_name="resnet_breast_randomized_boosted_distributed_non_iid" \
  wandb.group_name="attacker_only" \
  wandb.project_name="federated_official"

