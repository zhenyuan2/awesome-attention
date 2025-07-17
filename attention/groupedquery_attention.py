from typing import Optional

import math
import torch
from torch import nn, Tensor
from einops import rearrange, einsum



class GroupedQueryAttention(nn.Module):
    def __init__(self, 
        d_model, 
        n_heads,
        n_groups,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device":device, "dtype": dtype}
        super().__init__()
        self.n_heads = n_heads
        self.n_groups = n_groups
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        
        self.Wq = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.Wk = nn.Linear(d_model, self.head_dim * n_groups, bias=bias, **factory_kwargs)
        self.Wv = nn.Linear(d_model, self.head_dim * n_groups, bias=bias, **factory_kwargs)

        self.Wo = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        
    def forward(self, x:Tensor, mask: Optional[Tensor] = None):
        """
            Einstein notation:
            - b: batch size
            - n / s: sequence length
            - h: number of heads
            - g: number of groups
            - d: dimension of query/key/value
        """
        batch_size, seq_len, _ = x.shape
        q = self.Wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(batch_size, seq_len, self.n_groups, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(batch_size, seq_len, self.n_groups, self.head_dim).transpose(1, 2)

        q = rearrange(q, "b (g h) n d -> b g h n d", g=self.n_groups)

        attn_score = torch.einsum("b g h n d, b g s d -> b g h n s", q, k) / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, None, :]
            attn_score = attn_score + mask

        attn_prob = torch.softmax(attn_score, dim=-1).to(q)
        out = torch.einsum("b g h n s, b g s d -> b g h n d", attn_prob, v)
        out = rearrange(out, "b g h n d -> b n (h g d)")

        return self.Wo(out)