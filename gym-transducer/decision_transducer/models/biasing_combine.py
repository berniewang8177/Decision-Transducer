import torch
import torch.nn as nn
from decision_transducer.models.encoders import Encoder, get_lookahead_mask

class BiasCombineNet(torch.nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(hidden_size, hidden_size*2)
        self.w2 = nn.Linear(hidden_size, hidden_size*2)
        self.w3 = nn.Linear(2*hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.norm2 = torch.nn.LayerNorm(hidden_size)
        self.norm3 = torch.nn.LayerNorm(hidden_size)

        self.attn = nn.MultiheadAttention(hidden_size, 1, batch_first=True)
    
    def forward(self, data, rtgs, pad_mask):

        attn_mask = get_lookahead_mask(rtgs)
        data_0 = data
        data_1, _ = self.attn(
            query = data,
            key = rtgs,
            value = rtgs,
            key_padding_mask = pad_mask,
            attn_mask = attn_mask
            )
        fuse = torch.cat([data_0,data_1], dim = -1)
        final = self.w3( fuse )
        return final




    # concatenation does not work
    # def forward(self, states, actions, pad_mask):
    #     states = self.norm1(states)
    #     actions = self.norm2(actions)
    #     fuse = torch.cat([states,actions], dim = -1)
    #     states =  states + self.dropout1( self.w2( self.gelu( self.w3(fuse) ) ) )
    #     return self.norm3(states)

    # def forward(self, states, actions, pad_mask):
    #     attn_mask = get_lookahead_mask(actions)
    #     states_0 = states
    #     states_1, _ = self.attn(
    #         query = actions,
    #         key = states,
    #         value = states,
    #         key_padding_mask = pad_mask,
    #         attn_mask = attn_mask
    #         )
    #     final = self.w3( self.tanh( self.w1(states_0) + self.w2(states_1)) )
    #     return self.norm1(final)