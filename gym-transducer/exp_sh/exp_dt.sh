cd ./..

#hopper
# python3 experiment_dt.py --env hopper --dataset medium --warmup_steps 10000 --embed_dim 213 --n_head 3 --log_to_wandb True --device cuda:0
# halfcheetah
# python3 experiment_transducer.py --env halfcheetah --dataset medium --modality_emb 3 --log_to_wandb True --device cuda:0

# # python3 experiment_transducer.py --env walker2d --dataset medium-replay --modality_emb 3 --log_to_wandb True --device cuda:0
# python3 experiment_transducer.py --env halfcheetah --dataset medium-replay --modality_emb 3 --log_to_wandb True --device cuda:0

# # python3 experiment_transducer.py --env hopper --dataset medium-replay --modality_emb 3 --log_to_wandb True --device cuda:0

# medium-replay
# python3 experiment_transducer.py --env hopper --dataset medium-replay --device cuda:0 --seed 0 --learning_rate 1.5e-4 --log_to_wandb True
# python3 experiment_transducer.py --env walker2d --dataset medium-replay --device cuda:0 --seed 0 --learning_rate 1.5e-4 --log_to_wandb True
# python3 experiment_transducer.py --env walker2d --dataset medium-replay --device cuda:0 --seed 0 --learning_rate 1.0e-4 --log_to_wandb True


# python3 experiment_transducer.py --env walker2d --dataset medium-replay --device cuda:0 --seed 0 --learning_rate 1.0e-4 --batch_size 128 --log_to_wandb True

# python3 experiment_transducer.py --env halfcheetah --dataset medium-replay --device cuda:0 --seed 0 --learning_rate 1.0e-4 --batch_size 256 --log_to_wandb True
# python3 experiment_transducer.py --env halfcheetah --dataset medium-replay --device cuda:0 --seed 0 --learning_rate 1.0e-4 --batch_size 128 --log_to_wandb True

python3 experiment_transducer.py --env hopper --dataset medium-replay --device cuda:0 --seed 0 --learning_rate 1.5e-4 --batch_size 256 --comb c23 --log_to_wandb True





