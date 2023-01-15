cd ../..


# python3 experiment_transducer.py --dataset medium --bias b2 --log_to_wandb True --dropout 0.2 --device cuda:2
# python3 experiment_transducer.py --dataset medium --bias b2 --log_to_wandb True --dropout 0.1 --device cuda:2
# python3 experiment_transducer.py --dataset medium --bias b2 --log_to_wandb True --dropout 0.2 --modality_emb 3 --device cuda:2
python3 experiment_transducer.py --dataset medium --bias b2 --log_to_wandb True --dropout 0.1 --modality_emb 0 --device cuda:2