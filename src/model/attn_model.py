import torch
import torch.nn as nn
from model.feat_attn_model import FeatureAttention


class AttentionModel(nn.Module):
    def __init__(
        self,
        n_features: int = 14,
        k: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        # Equal to the window size + 2 (history + current + feature)
        self.seq_len = k + 2

        self.feat_attn = FeatureAttention(d_model, n_heads)
        self.input = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Shared output head
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(d_model * self.seq_len),
            nn.Linear(d_model * self.seq_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
        )
        # Outputs classification result; nonzero or zero
        self.output_cls = nn.Linear(d_model // 2, 1)
        # If the classification results in nonzero, outputs crime count
        self.output_count = nn.Linear(d_model // 2, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        current = x[:, -1, :]
        feat_attn = self.feat_attn(current)
        # NOTE: Might be changed to GlobalPooling
        feat_attn = feat_attn.mean(dim=1, keepdim=True)

        h = self.input(x)
        h_cat = torch.concat([h, feat_attn], dim=1)
        h_pos = h_cat + self.pos_emb

        feat_mask = torch.ones(mask.shape[0], 1, dtype=torch.bool).to(mask.device)
        mask = torch.concat([mask, feat_mask], dim=1)
        encoder = self.encoder(h_pos, src_key_padding_mask=~mask)

        output = self.output(encoder)
        is_nonzero = self.output_cls(output).squeeze(1)
        count = self.output_count(output).squeeze(1)
        return is_nonzero, count

    def predict(self, x, mask):
        self.eval()
        with torch.no_grad():
            is_nonzero, count = self.forward(x, mask)
        return is_nonzero, count


if __name__ == "__main__":
    model = AttentionModel()
    x = torch.randn((64, 4, 14), dtype=torch.float32)
    mask = torch.ones((64, 4), dtype=torch.bool)
    model(x, mask)
