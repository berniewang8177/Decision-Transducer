cd ./..


# python3 experiment_dt.py --env hopper --dataset medium-replay --warmup_steps 10000 --embed_dim 213 --n_head 3 --seed 1 --log_to_wandb True --device cuda:1
# python3 experiment_dt.py --env walker2d --dataset medium-replay --warmup_steps 10000 --embed_dim 213 --n_head 3 --seed 1 --log_to_wandb True --device cuda:1

# python3 experiment_dt_small.py --env hopper --dataset medium-replay --warmup_steps 10000 --seed 0 --log_to_wandb True --device cuda:1
# python3 experiment_dt_small.py --env hopper --dataset medium-replay --warmup_steps 10000 --seed 1 --log_to_wandb True --device cuda:1

# python3 experiment_dt_small.py --env hopper --dataset medium --warmup_steps 10000 --seed 0 --log_to_wandb True --device cuda:1
# python3 experiment_dt_small.py --env hopper --dataset medium --warmup_steps 10000 --seed 1 --log_to_wandb True --device cuda:1

# python3 experiment_dt_small.py --env hopper --dataset medium-expert --warmup_steps 10000 --seed 0 --log_to_wandb True --device cuda:1
# python3 experiment_dt_small.py --env hopper --dataset medium-expert --warmup_steps 10000 --seed 1 --log_to_wandb True --device cuda:1

# python3 experiment_dt_small.py --env walker2d --dataset medium --warmup_steps 10000 --seed 0 --log_to_wandb True --device cuda:1
# python3 experiment_dt_small.py --env walker2d --dataset medium --warmup_steps 10000 --seed 1 --log_to_wandb True --device cuda:1

# python3 experiment_dt_small.py --env walker2d --dataset medium-replay --warmup_steps 10000 --seed 0 --log_to_wandb True --device cuda:1
# python3 experiment_dt_small.py --env walker2d --dataset medium-replay --warmup_steps 10000 --seed 1 --log_to_wandb True --device cuda:1

# python3 experiment_dt_small.py --env walker2d --dataset medium-expert --warmup_steps 10000 --seed 0 --log_to_wandb True --device cuda:1
# python3 experiment_dt_small.py --env walker2d --dataset medium-expert --warmup_steps 10000 --seed 1 --log_to_wandb True --device cuda:1

python3 experiment_dt_small.py --env halfcheetah --dataset medium-expert --warmup_steps 10000 --seed 0 --log_to_wandb True --device cuda:1
python3 experiment_dt_small.py --env halfcheetah --dataset medium-expert --warmup_steps 10000 --seed 1 --log_to_wandb True --device cuda:1
  