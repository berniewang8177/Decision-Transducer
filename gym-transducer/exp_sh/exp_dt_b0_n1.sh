cd ./..

# norm experiment. b0
python3 experiment_transducer.py --env hopper --dataset medium  --device cuda:1 --bias b0 --norm_mode n1 --log_to_wandb True
python3 experiment_transducer.py --env hopper --dataset medium  --device cuda:1 --bias b0 --norm_mode n1 --log_to_wandb True
python3 experiment_transducer.py --env hopper --dataset medium  --device cuda:1 --bias b0 --norm_mode n1 --log_to_wandb True