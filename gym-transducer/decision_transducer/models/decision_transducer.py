import numpy as np
import torch
import torch.nn as nn

from decision_transducer.models.model import TrajectoryModel
from decision_transducer.models.encoders import Encoder
from decision_transducer.models.join_net import JoinNet
from decision_transducer.models.biasing_combine import BiasCombineNet


class DecisionTransducer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            bias_mode = 'b1',
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)

        # encoders for 3 modalities
        self.state_encoder = Encoder(hidden_size)
        self.action_encoder = Encoder(hidden_size)
        self.bias_mode = bias_mode

        if self.bias_mode != "b0":
            self.rtg_encoder = Encoder(hidden_size)

        # join network with tanh out
        self.join = JoinNet(hidden_size)

        # biasing amd combine
        self.bias1 = BiasCombineNet(hidden_size)
        self.bias2 = BiasCombineNet(hidden_size)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)



    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]
        # 1 means the position want to attend. Reverse it
        
        attention_mask = ( 1.0 - attention_mask)
        
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.zeros((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_ln(state_embeddings + time_embeddings )
        action_embeddings = self.embed_ln( action_embeddings + time_embeddings )
        returns_embeddings = self.embed_ln( returns_embeddings + time_embeddings )
        

        # encoding with causal mask
        encoded_state = self.state_encoder(state_embeddings, attention_mask)
        encoded_action = self.action_encoder(action_embeddings, attention_mask)
        if self.bias_mode != "b0":
            encoded_rtg = self.rtg_encoder(returns_embeddings, attention_mask)
        
        # combiner & biasing
        if self.bias_mode == "b0":
            pass
        else:
            encoded_state = self.bias1(encoded_state, encoded_rtg, attention_mask)
            if self.bias_mode == "b2":
                encoded_action = self.bias2(encoded_action, encoded_rtg, attention_mask)
        
        # join network
        join_encoded = self.join(encoded_state, encoded_action, attention_mask)

        # # get predictions
        action_preds = self.predict_action(join_encoded)  # predict next action given state

        if True in torch.isnan( action_preds[0,-1]).detach().cpu().numpy():
            print(attention_mask[0])
            print(action_preds[0])
            assert False
        #     print(join_encoded[0,-1])
        #     print(encoded_state)
        #     print()
        #     print(encoded_action)
        #     print()
        #     print(state_embeddings)
        #     print(action_embeddings)

        return None, action_preds, None

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            # 1 means the position we want to attend. Reverse it for padding inside forward.
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None
        # assert False,f"{actions[0]}\n{attention_mask[0]}"
        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)
        # print( f"action_preds: {action_preds[0]}" )
        return action_preds[0,-1]
