cd ./..

#hopper
# python3 experiment_dt.py --env hopper --dataset medium --warmup_steps 10000 --embed_dim 213 --n_head 3 --log_to_wandb True --device cuda:1
# python3 experiment_transducer.py --env halfcheetah --dataset medium --modality_emb 3 --log_to_wandb True --device cuda:1

# python3 experiment_transducer.py --env walker2d --dataset medium-replay --modality_emb 3 --log_to_wandb True --device cuda:1
# python3 experiment_transducer.py --env halfcheetah --dataset medium-replay --modality_emb 3 --log_to_wandb True --device cuda:1
python3 experiment_transducer.py --env hopper --dataset medium-replay --modality_emb 3 --log_to_wandb True --device cuda:1