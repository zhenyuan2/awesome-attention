# Awesome Attention 🧠🔥

A curated and well-documented PyTorch implementation of modern **attention mechanisms**, designed for learning, research, and real-world applications.

> ⭐ Focused, Minimal, and Extensible — perfect for understanding the core logic behind various attention variants.

## 🚀 Overview

This repository provides clean and efficient implementations of popular **attention mechanisms** in PyTorch. It aims to serve as both a learning resource and a foundation for research and production.

### ✅ Implemented Modules

| Module Name                | Description                                                      |
|---------------------------|------------------------------------------------------------------|
| `MultiHeadAttention`      | Standard scaled dot-product multi-head attention (Vaswani et al.) |
| `MultiQueryAttention`     | Uses one query head and multiple key-value heads (Shazeer et al.) |
| `GroupedQueryAttention`   | A hybrid form where groups of query heads share key-value heads   |
| *(More coming soon!)*     | Long-range attention, sparse attention, flash attention, etc.     |

---

## 📦 Features

- ✅ **Readable and modular** PyTorch code  
- ✅ Works with both **float32 / bfloat16 / float16**  
- ✅ Simple interface, **plug-and-play** with your models  
- ✅ Designed for **easy extension** and customization  
- ✅ Clean gradient flow and proper masking support  
- ✅ Ideal for both **educational** and **experimental** use

---

## 🧩 Example Usage

```python
from attention import GroupedQueryAttention

attn = GroupedQueryAttention(
    d_model=512,
    n_heads=8,
    n_groups=4,
)

x = torch.randn(2, 128, 512)  # [batch, seq_len, d_model]
out = attn(x)                 # [batch, seq_len, d_model]
