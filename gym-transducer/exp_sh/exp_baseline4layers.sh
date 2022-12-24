python3 experiment_dt.py --env hopper --dataset medium --warmup_steps 10000 --log_to_wandb True --device cuda:0
python3 experiment_dt.py --env hopper --dataset medium-expert --warmup_steps 10000 --log_to_wandb True --device cuda
python3 experiment_dt.py --env hopper --dataset medium-replay --warmup_steps 10000 --log_to_wandb True --device cuda

