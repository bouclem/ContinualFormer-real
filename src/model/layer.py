import torch.nn as nn

from .attention import LinearAttention
from .ffn import FreezableFFN


class TransformerLayer(nn.Module):
    def __init__(self, dim, ffn_hidden, heads=2, dropout=0.1):
        super().__init__()
        self.attn = LinearAttention(dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = FreezableFFN(dim, ffn_hidden, dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None, causal=False):
        x = x + self.attn(self.norm1(x), mask, causal=causal)
        x = x + self.ffn(self.norm2(x))
        return x
