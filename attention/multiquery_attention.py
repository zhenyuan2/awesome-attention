import math
import torch
from torch import nn, Tensor
from einops import rearrange


class MultiQueryAttention(nn.Module):
    def __init__(self,
        d_model, 
        n_heads,
        attention_bais=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device":device, "dtype": dtype}
        super().__init__()
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        
        self.Wq = nn.Linear(d_model, d_model, bias=attention_bais, **factory_kwargs)
        self.Wk = nn.Linear(d_model, self.head_dim, bias=attention_bais, **factory_kwargs)
        self.Wv = nn.Linear(d_model, self.head_dim, bias=attention_bais, **factory_kwargs)

        self.Wo = nn.Linear(d_model, d_model, bias=attention_bais, **factory_kwargs)
        
    def forward(self, x:Tensor, mask=None):
        batch_size, seq_len, d_model = x.shape
        q = self.Wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).unsqueeze(1)
        v = self.Wv(x).unsqueeze(1)

        scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, None, :]
            scores = scores + mask
        attns = torch.softmax(scores, dim=-1)
        output = attns @ v 
        output = output.contiguous().view(batch_size, seq_len, d_model)

        return self.Wo(output)
