cd ./..


python3 experiment_dt.py --env halfcheetah --dataset medium-replay --warmup_steps 10000 --embed_dim 213 --n_head 3 --log_to_wandb True --device cuda:0

python3 experiment_dt.py --env walker2d --dataset medium --warmup_steps 10000 --embed_dim 213 --n_head 3 --log_to_wandb True --device cuda:0
python3 experiment_dt.py --env walker2d --dataset medium-replay --warmup_steps 10000 --embed_dim 213 --n_head 3 --log_to_wandb True --device cuda:0

python3 experiment_dt.py --env hopper --dataset medium-replay --warmup_steps 10000 --embed_dim 213 --n_head 3 --log_to_wandb True --device cuda:0
  