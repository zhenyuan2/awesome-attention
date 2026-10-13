import math

import torch
from torch import Tensor, nn


class GroupedQueryAttention(nn.Module):
    """Each contiguous group of query heads shares a key/value head."""

    def __init__(self, d_model, n_heads, n_groups, bias=True, device=None, dtype=None):
        super().__init__()
        assert n_heads > 0 and d_model > 0 and d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert n_groups > 0 and n_heads % n_groups == 0
        self.n_groups = n_groups

        factory_kwargs = {"device": device, "dtype": dtype}
        self.Wq = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.Wk = nn.Linear(d_model, n_groups * self.head_dim, bias=bias, **factory_kwargs)
        self.Wv = nn.Linear(d_model, n_groups * self.head_dim, bias=bias, **factory_kwargs)
        self.Wo = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

    def forward(self, x: Tensor, mask=None, *, return_attention=False):
        batch_size, seq_len, d_model = x.shape
        # Split the projected features into heads: [B,N,H,D] -> [B,H,N,D].
        q = self.Wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(batch_size, seq_len, self.n_groups, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(batch_size, seq_len, self.n_groups, self.head_dim).transpose(1, 2)
        # H = G * heads_per_group; keep the original head order.
        q = q.reshape(batch_size, self.n_groups, self.n_heads // self.n_groups, seq_len, self.head_dim)

        # Each query group attends only to its corresponding K/V head.
        scores = torch.einsum("bghnd,bgsd->bghns", q, k)
        scores = scores.reshape(batch_size, self.n_heads, seq_len, seq_len)
        scores = scores / math.sqrt(self.head_dim)
        if mask is not None:
            # [N,N] -> [1,1,N,N]; [B,N] -> [B,1,1,N].
            if mask.ndim == 2:
                if mask.shape == (seq_len, seq_len):
                    mask = mask[None, None, :, :]
                else:
                    mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask

        attns = torch.softmax(scores, dim=-1)

        grouped_attns = attns.reshape(batch_size, self.n_groups, self.n_heads // self.n_groups, seq_len, seq_len)
        output = torch.einsum("bghns,bgsd->bghnd", grouped_attns, v)
        # [B,G,h,N,D] -> [B,N,G,h,D] -> [B,N,d_model].
        output = output.permute(0, 3, 1, 2, 4).reshape(batch_size, seq_len, d_model)
        output = self.Wo(output)
        return (output, attns) if return_attention else output
