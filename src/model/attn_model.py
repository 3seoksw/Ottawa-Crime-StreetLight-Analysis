import torch
import torch.nn as nn
from model.feat_attn_model import FeatureAttention


class AttentionModel(nn.Module):
    def __init__(
        self,
        n_features: int = 14,
        d_model: int = 128,
        n_heads: int = 4,
    ):
        super().__init__()
        self.feat_attn = FeatureAttention(d_model, n_heads)
        self.input = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.embedding = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)

        # Shared output head
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(d_model * n_features),
            nn.Linear(d_model * n_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
        )
        # Outputs classification result; nonzero or zero
        self.output_cls = nn.Linear(d_model // 2, 1)
        # If the classification results in nonzero, outputs crime count
        self.output_count = nn.Linear(d_model // 2, 1)

    def forward(self, x: torch.Tensor):
        feat_attn, attn_weights = self.feat_attn(x)  # (B, 14, 128) & (B, 4, 14, 14)

        h = self.input(feat_attn)
        h_emb = h + self.embedding

        output = self.output(h_emb)
        is_nonzero = self.output_cls(output).squeeze(1)
        count = self.output_count(output).squeeze(1)
        return is_nonzero, count, attn_weights

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            is_nonzero, count, attn_weights = self.forward(x)
        return is_nonzero, count, attn_weights


if __name__ == "__main__":
    model = AttentionModel()
    x = torch.randn((64, 14), dtype=torch.float32)
    nonzero, count, attn_weights = model(x)
    print(nonzero.shape, count.shape, attn_weights.shape)
