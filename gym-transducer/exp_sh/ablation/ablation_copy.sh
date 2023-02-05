cd ../..

# hopper
# b0
# python3 experiment_transducer.py --env halfcheetah --dataset medium-expert --bias b0 --log_to_wandb True --seed 0 --device cuda:0
# python3 experiment_transducer.py --env halfcheetah --dataset medium-expert --bias b0 --log_to_wandb True --seed 1 --device cuda:1
# python3 experiment_transducer.py --env halfcheetah --dataset medium-expert --bias b0 --log_to_wandb True --seed 2 --device cuda:2
# python3 experiment_transducer.py --env halfcheetah --dataset medium-expert --bias b0 --log_to_wandb True --seed 3 --device cuda:3

# python3 experiment_transducer.py --env hopper --dataset medium-replay --bias left --learning_rate 1.5e-4 --log_to_wandb True --seed 0 --device cuda:2
# python3 experiment_transducer.py --env hopper --dataset medium-replay --bias left --learning_rate 1.5e-4 --log_to_wandb True --seed 1 --device cuda:2
# bias left


python3 experiment_transducer.py --env hopper --dataset medium-replay --bias b0 --learning_rate 1.5e-4 --log_to_wandb True --seed 0 --device cuda:1
python3 experiment_transducer.py --env hopper --dataset medium-replay --bias b0 --learning_rate 1.5e-4 --log_to_wandb True --seed 1 --device cuda:1

python3 experiment_transducer.py --env halfcheetah --dataset medium-replay --bias b0 --log_to_wandb True --seed 0 --device cuda:1
python3 experiment_transducer.py --env halfcheetah --dataset medium-replay --bias b0 --log_to_wandb True --seed 1 --device cuda:1

python3 experiment_transducer.py --env walker2d --dataset medium-replay --bias b0 --log_to_wandb True --seed 0 --device cuda:1 --batch_size 128
python3 experiment_transducer.py --env walker2d --dataset medium-replay --bias b0 --log_to_wandb True --seed 1 --device cuda:1 --batch_size 128