import math

import torch
from torch import Tensor, nn


class MultiQueryAttention(nn.Module):
    """All query heads share one key/value head."""

    def __init__(self, d_model, n_heads, bias=True, device=None, dtype=None):
        super().__init__()
        assert n_heads > 0 and d_model > 0 and d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        factory_kwargs = {"device": device, "dtype": dtype}
        self.Wq = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.Wk = nn.Linear(d_model, self.head_dim, bias=bias, **factory_kwargs)
        self.Wv = nn.Linear(d_model, self.head_dim, bias=bias, **factory_kwargs)
        self.Wo = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

    def forward(self, x: Tensor, mask=None, *, return_attention=False):
        batch_size, seq_len, d_model = x.shape
        # Split the projected features into heads: [B,N,H,D] -> [B,H,N,D].
        q = self.Wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # A single K/V head broadcasts across all query heads.
        k = self.Wk(x).unsqueeze(1)  # [B,1,N,D]
        v = self.Wv(x).unsqueeze(1)

        scores = q @ k.transpose(-2, -1)
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

        output = attns @ v  # [B,H,N,D]
        # Move tokens before heads, then concatenate the heads.
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.Wo(output)
        return (output, attns) if return_attention else output
