import numpy as np
import torch
from datetime import datetime
import os
import time

class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.loss_fn1 = None
        self.loss_fn2 = None
        self.critics = None
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, plot_dict=dict(), print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        if num_steps > 0:
            self.model.train()
            times = 5
            for _ in range(num_steps):
                if _ % int(num_steps / times) == 0:
                    print("step:", _)
                train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()


            logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        # get the mean return of this Iteration
        plot_dict['mean_return'] = []
        for eval_fn in self.eval_fns: # self.eval_fns gives results for each target Rtg
            outputs = eval_fn(self.model)
            plot_dict['mean_return'].append(outputs[ list(outputs.keys())[0]])
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start

        plot_dict['mean_loss'] = np.mean(train_losses)
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/learning_rate'] = self.scheduler.get_last_lr()[0]

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs, plot_dict

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

 