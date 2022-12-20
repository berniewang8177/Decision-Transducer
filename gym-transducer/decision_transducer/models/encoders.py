import torch
import torch.nn as nn

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

class Encoder(torch.nn.Module):

    def __init__(self, hidden_size, n_layers = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=1, batch_first=True, dim_feedforward = 256)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers )
    
    def forward(self, x, padding_mask):
        causal_mask = get_lookahead_mask( x )
        # print( f"padding_mask: {padding_mask[0]}" )
        out = self.transformer_encoder(x, mask = causal_mask, src_key_padding_mask = padding_mask)
        # assert False, f"out: {out[0]}"
        return out