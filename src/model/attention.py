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
            # Cumulative KV sum — O(1) per token, each position only sees past
            kv = torch.zeros(B, H, Hd, Hd, device=x.device, dtype=x.dtype)
            outputs = []
            for t in range(L):
                kv = kv + torch.einsum('bhd,bhv->bhdv', k[:, :, t], v[:, :, t])
                q_t = q[:, :, t]
                out_t = torch.einsum('bhd,bhdv->bhv', q_t, kv)
                denom = torch.einsum('bhd,bhd->bh', q_t, k[:, :, :t+1].sum(dim=2)).clamp(min=1e-6)
                out_t = out_t / denom.unsqueeze(-1)
                outputs.append(out_t)
            out = torch.stack(outputs, dim=2)
        else:
            kv = torch.einsum('bhld,bhlv->bhdv', k, v)
            qkv = torch.einsum('bhld,bhdv->bhlv', q, kv)
            k_sum = k.sum(dim=2)
            qks = torch.einsum('bhld,bhd->bhl', q, k_sum).clamp(min=1e-6).unsqueeze(-1)
            out = qkv / qks

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.dropout(self.to_out(out))
