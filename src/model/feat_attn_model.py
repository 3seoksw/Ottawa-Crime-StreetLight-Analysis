import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
    ):
        super().__init__()

        self.linear = nn.Linear(1, d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1)  # B, 14, 128
        h = self.linear(x)  # B, 14, 128
        out, weights = self.mha(
            query=h,
            key=h,
            value=h,
            need_weights=True,
            average_attn_weights=False,
        )
        self.attn_weights = weights
        return self.norm(out), weights


if __name__ == "__main__":
    model = FeatureAttention()
    x = torch.randn((64, 14), dtype=torch.float32)
    out = model(x)
    print(out.shape)
