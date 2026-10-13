<h1 align="center">Awesome Attention 🧠🔥</h1>

Four independent PyTorch self-attention implementations for learning.
Each file contains the complete flow: projections, head splitting, attention scores,
masking, softmax, and output projection. Some code is intentionally repeated so you
can read and experiment with each mechanism on its own.

| Module | Query heads | Key/value heads |
|---|---|---|
| `MultiHeadAttention` | `n_heads` | `n_heads` |
| `MultiQueryAttention` | `n_heads` | One shared head |
| `GroupedQueryAttention` | `n_heads` | `n_groups`, shared by contiguous groups of query heads |
| `MultiHeadLatentAttention` | `n_heads` | Per-head content K/V restored from a shared latent vector, plus one shared RoPE key |

## Setup

Python 3.9+ and PyTorch 2.0+ are required.

```bash
python -m pip install -e .
# Optional: install and run the correctness tests.
python -m pip install -e '.[test]'
python -m pytest -q
```

## Example

```python
import torch
from attention import GroupedQueryAttention

attn = GroupedQueryAttention(d_model=512, n_heads=8, n_groups=4)
x = torch.randn(2, 128, 512)

# Without a mask, each token can attend to all tokens.
causal_mask = torch.ones(128, 128, dtype=torch.bool).tril()
out = attn(x, causal_mask)  # [B,N,d_model]
out, weights = attn(x, causal_mask, return_attention=True)
# weights: [B,n_heads,N,N]
```

All modules accept `bias`, `device`, and `dtype`. For MHA/MQA/GQA, `d_model` must be divisible by
`n_heads`; GQA also requires `n_heads` to be divisible by `n_groups`.

## Multi-Head Latent Attention (MLA)

The implementation follows the core formulas in
[DeepSeek-V2, Sections 2.1.2–2.1.3](https://arxiv.org/html/2405.04434v5#S2.SS1.SSS2).
Read [multihead_latent_attention.py](attention/multihead_latent_attention.py) on its own.

```python
from attention import MultiHeadLatentAttention

mla = MultiHeadLatentAttention(
    d_model=512,
    n_heads=8,
    q_rank=128,    # compressed query dimension
    kv_rank=64,    # joint compressed K/V dimension
    head_dim=64,   # content Q/K and V dimension per head
    rope_dim=32,   # separate positional Q/K dimension; must be even
)
out = mla(x, causal_mask)  # x and causal_mask from the example above
```

`Wdq` compresses Q; `Wuq` and `Wqr` produce its content and positional branches.
`Wdkv` jointly compresses K/V; `Wuk` and `Wuv` restore per-head content K/V.
`Wkr` produces one positional key directly from the input, shared across heads.
RoPE acts only on the positional branches. Attention scores sum the content and
positional dot products, scaled by `sqrt(head_dim + rope_dim)`.

Unlike the other modules, MLA specifies `head_dim` explicitly; `n_heads * head_dim`
does not need to equal `d_model`. This teaching version expands K/V for full-sequence
attention and omits latent RMSNorm, YaRN, KV caching, and inference matrix absorption.
It implements the core equations rather than an exact DeepSeek checkpoint format.
With a compressed cache implementation, each token would store only `kv_rank + rope_dim`
elements per layer: the shared KV latent and the rotated positional key.

## Masks

Boolean masks use **True = allow, False = block**, matching PyTorch SDPA.
Floating masks are added to the scores: use `0` for no change and `-inf` to block.
Finite negative values reduce attention rather than fully blocking it.

Supported shapes:

- `[N,N]`: shared attention mask, such as the causal mask above.
- `[B,N]`: key padding mask.
- `[B,N,N]`: attention mask for each batch item.
- Four-dimensional masks broadcast to `[B,H,N,N]`, e.g. `[B,1,1,N]` for padding.

When `B == N`, a two-dimensional mask is treated as `[N,N]`. Pass padding masks as
`[B,1,1,N]` to avoid ambiguity. For example, combine padding with causal masking:

```python
padding_mask = torch.ones(2, 128, dtype=torch.bool)
padding_mask[0, 100:] = False
mask = causal_mask[None, None] & padding_mask[:, None, None, :]
out = attn(x, mask)
```

Each query row should have at least one unmasked key. Pass masks on the same device
as the input, and use the same dtype for additive masks. Padding masks block keys;
they do not zero padded query outputs.

## Notes

- Tests compare outputs and gradients with PyTorch SDPA and check causal masking.
- These implementations explicitly materialize attention scores and use quadratic
  memory in sequence length. They are intended for understanding the algorithms.
- MHA now returns only the output by default; use `return_attention=True` to get
  the weights too. MQA's old `attention_bais` parameter is now named `bias`.
- MQA's token/head layout and GQA's head ordering have been corrected, so old
  checkpoints may produce different outputs.

MIT license; see [LICENSE](LICENSE).
