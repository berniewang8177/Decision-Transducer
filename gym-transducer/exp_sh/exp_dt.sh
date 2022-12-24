cd ./..

# norm experiment. b0
# python3 experiment_transducer.py --env hopper --dataset medium --warmup_steps 8000 --device cuda:0 --bias b0 --norm_mode n1 --log_to_wandb True
# python3 experiment_transducer.py --env hopper --dataset medium --warmup_steps 8000 --device cuda:1 --bias b0 --norm_mode n2 --log_to_wandb True
# python3 experiment_transducer.py --env hopper --dataset medium --warmup_steps 8000 --device cuda:2 --bias b0 --norm_mode n3 --log_to_wandb True