import torch
import torch.nn as nn


class FreezableFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.frozen_mask = torch.zeros(hidden_dim, dtype=torch.bool)

    def freeze_neurons(self, indices):
        for idx in indices:
            i = idx.item() if torch.is_tensor(idx) else int(idx)
            self.frozen_mask[i] = True
            self.fc1.weight.data[i].requires_grad = False
            self.fc1.bias.data[i].requires_grad = False
            self.fc2.weight.data[:, i].requires_grad = False

    def active_count(self):
        return int((~self.frozen_mask).sum().item())

    def get_importance(self, x, mask=None):
        with torch.no_grad():
            h = self.act(self.fc1(x))
            if mask is not None:
                h = h * mask[:, :, None].float()
            return h.abs().mean(dim=(0, 1))

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))
