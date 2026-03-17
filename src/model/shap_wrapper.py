import torch.nn as nn
from model.attn_model import AttentionModel


class ClsWrapper(nn.Module):
    def __init__(self, model: AttentionModel):
        super().__init__()
        self.model = model

    def forward(self, x):
        is_nonzero, _, _ = self.model(x)
        return is_nonzero.unsqueeze(-1)


class CountWrapper(nn.Module):
    def __init__(self, model: AttentionModel):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, count, _ = self.model(x)
        return count.unsqueeze(-1)
