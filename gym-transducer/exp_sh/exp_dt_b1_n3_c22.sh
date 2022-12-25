cd ./..

# norm experiment. b1
python3 experiment_transducer.py --env hopper --dataset medium  --device cuda:2 --bias b1 --norm_mode n3 --comb c22 --log_to_wandb True
python3 experiment_transducer.py --env hopper --dataset medium  --device cuda:2 --bias b1 --norm_mode n3 --comb c22 --log_to_wandb True
python3 experiment_transducer.py --env hopper --dataset medium  --device cuda:2 --bias b1 --norm_mode n3 --comb c22 --log_to_wandb True