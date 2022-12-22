# python3 experiment_transducer.py --env hopper --dataset medium --warmup_steps 8000 --log_to_wandb True
# python3 experiment_transducer.py --env hopper --dataset medium-expert --warmup_steps 8000 --log_to_wandb True
# python3 experiment_transducer.py --env hopper --dataset medium-replay --warmup_steps 8000 --log_to_wandb True

# norm experiment. b0
python3 experiment_transducer.py --env hopper --dataset medium --warmup_steps 8000 --device cuda --bias b0 --norm_mode n1 --log_to_wandb True
python3 experiment_transducer.py --env hopper --dataset medium --warmup_steps 8000 --device cuda --bias b0 --norm_mode n2 --log_to_wandb True
python3 experiment_transducer.py --env hopper --dataset medium --warmup_steps 8000 --device cuda --bias b0 --norm_mode n3 --log_to_wandb True