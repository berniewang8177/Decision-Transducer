import torch
import torch.nn as nn
from decision_transducer.models.encoders import Encoder

def get_lookahead_mask(padded_input):
    """Creates a binary mask for each sequence which maskes future frames.
    Arguments
    ---------
    padded_input: torch.Tensor
        Padded input tensor.
    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_lookahead_mask(a)
    tensor([[0., -inf, -inf],
            [0., 0., -inf],
            [0., 0., 0.]])
    """
    seq_len = padded_input.shape[1]
    mask = (
        torch.triu(torch.ones((seq_len, seq_len), device=padded_input.device))
        == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.detach().to(padded_input.device)

class JoinNet(torch.nn.Module):

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

        self.join_enc = Encoder(hidden_size, n_layers = 1)
    
    def forward(self, states, actions, pad_mask):
        batch_size, seq_length = states.shape[0], states.shape[1]

        stacked_inputs = torch.stack(
                (states, actions), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)

    
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_pad_mask = torch.stack(
            (pad_mask, pad_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)
        
        x = self.join_enc(stacked_inputs, stacked_pad_mask)
        # the 0 dim is batch, 1 dim is state,action
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)
        # retrieve state
        return x[:,0]

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

    # def forward(self, states, actions, pad_mask):
    #     attn_mask = get_lookahead_mask(actions)
    #     states_0 = states
    #     states_1, _ = self.attn(
    #         query = states_0,
    #         key = actions,
    #         value = actions,
    #         key_padding_mask = pad_mask,
    #         attn_mask = attn_mask
    #         )
    #     # add & norm
    #     states = states + self.dropout1(states_1)
    #     states = self.norm1(states)

    #     # FFN
    #     states_0 = states
    #     states_1 = self.w2( self.gelu( self.w1(states_0)))

    #     # add & norm
    #     states = states + self.dropout2( states_1 )
    #     return self.norm2(states)



    # def forward(self, a,b, pad_mask):
    #     return a
        # attn_mask = get_lookahead_mask(b)
        # return self.gelu( self.w1(a) + self.w2(b) )
        # return self.tanh( self.w1(a) + self.w2(b) )
        # c = torch.cat([a,b], dim = -1)
        # return self.tanh( self.w3(c) )
        # c,_ = self.attn(
        #     query = a,
        #     key = b,
        #     value = b,
        #     key_padding_mask = pad_mask,
        #     attn_mask = attn_mask
        #     )
        # 
        # return self.tanh( self.w2(c) )