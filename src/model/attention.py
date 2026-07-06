import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=2, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, causal=False):
        B, L, D = x.shape
        H, Hd = self.heads, self.head_dim
        q = F.elu(self.to_q(x).view(B, L, H, Hd).transpose(1, 2)) + 1
        k = F.elu(self.to_k(x).view(B, L, H, Hd).transpose(1, 2)) + 1
        v = self.to_v(x).view(B, L, H, Hd).transpose(1, 2)
        if mask is not None:
            k = k * mask[:, None, :, None].float()

        if causal:
            # Cumulative KV sum — O(n) vectorized, each position only sees past
            kv_t = torch.einsum('bhld,bhlv->bhldv', k, v)
            kv_cum = torch.cumsum(kv_t, dim=2)
            qkv = torch.einsum('bhld,bhldv->bhlv', q, kv_cum)
            k_cum = torch.cumsum(k, dim=2)
            denom = torch.einsum('bhld,bhld->bhl', q, k_cum).clamp(min=1e-6)
            out = qkv / denom.unsqueeze(-1)
        else:
            kv = torch.einsum('bhld,bhlv->bhdv', k, v)
            qkv = torch.einsum('bhld,bhdv->bhlv', q, kv)
            k_sum = k.sum(dim=2)
            qks = torch.einsum('bhld,bhd->bhl', q, k_sum).clamp(min=1e-6).unsqueeze(-1)
            out = qkv / qks

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.dropout(self.to_out(out))
