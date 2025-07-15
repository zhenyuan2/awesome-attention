import math
import torch
from torch import nn, Tensor


class Attention(nn.Module):
    def __init__(self,
        d_model,
        n_heads,
        max_seq_len,
        bias=True,
        device=None,
        dtype=None
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.Wk = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.Wv = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

        self.Wo = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

    def forward(self, x: Tensor, mask=None):
        batch_size, seq_len, dim = x.shape
        q = self.Wq(x).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

        score = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            score = score + mask
        attns = torch.softmax(score, dim=-1)

        attn_output = attns @ v

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)

        return self.Wo(attn_output)