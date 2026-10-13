import math

import torch
from torch import Tensor, nn


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor):
    """Rotate adjacent feature pairs; x is [B,H,N,rope_dim]."""
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-V2 MLA core: joint KV compression and decoupled RoPE."""

    def __init__(
        self, d_model, n_heads, q_rank, kv_rank, head_dim, rope_dim,
        bias=False, device=None, dtype=None, rope_base=10000.0,
    ):
        super().__init__()
        assert n_heads > 0 and q_rank > 0 and kv_rank > 0 and head_dim > 0
        assert rope_dim > 0 and rope_dim % 2 == 0
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        factory_kwargs = {"device": device, "dtype": dtype}

        # Compress queries, then produce per-head content and RoPE queries.
        self.Wdq = nn.Linear(d_model, q_rank, bias=bias, **factory_kwargs)
        self.Wuq = nn.Linear(q_rank, n_heads * head_dim, bias=bias, **factory_kwargs)
        self.Wqr = nn.Linear(q_rank, n_heads * rope_dim, bias=bias, **factory_kwargs)

        # One latent vector jointly represents all content keys and values.
        self.Wdkv = nn.Linear(d_model, kv_rank, bias=bias, **factory_kwargs)
        self.Wuk = nn.Linear(kv_rank, n_heads * head_dim, bias=bias, **factory_kwargs)
        self.Wuv = nn.Linear(kv_rank, n_heads * head_dim, bias=bias, **factory_kwargs)

        # The positional key comes from x and is shared by every head.
        self.Wkr = nn.Linear(d_model, rope_dim, bias=bias, **factory_kwargs)
        self.Wo = nn.Linear(n_heads * head_dim, d_model, bias=bias, **factory_kwargs)

        inv_freq = 1.0 / (rope_base ** (torch.arange(0, rope_dim, 2, **factory_kwargs) / rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, mask=None, *, return_attention=False):
        batch_size, seq_len, _ = x.shape

        c_q = self.Wdq(x)    # [B,N,q_rank]
        c_kv = self.Wdkv(x)  # [B,N,kv_rank]
        q_content = self.Wuq(c_q).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k_content = self.Wuk(c_kv).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.Wuv(c_kv).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q_rope = self.Wqr(c_q).view(batch_size, seq_len, self.n_heads, self.rope_dim).transpose(1, 2)
        k_rope = self.Wkr(x).unsqueeze(1)  # [B,1,N,rope_dim], shared across heads
        positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        angles = positions[:, None] * self.inv_freq[None, :]
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)

        # Equivalent to dotting concatenated [q_content; q_rope] and
        # [k_content; k_rope], without explicitly concatenating them.
        scores = q_content @ k_content.transpose(-2, -1)
        scores = scores + q_rope @ k_rope.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim + self.rope_dim)

        if mask is not None:
            # Same mask convention as the other attention implementations.
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
        output = attns @ v  # [B,H,N,head_dim]; V has no RoPE component
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.head_dim)
        output = self.Wo(output)
        return (output, attns) if return_attention else output
