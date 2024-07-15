import math
import torch
from torch import nn
from models.pe import PositionalEncoding, RotaryPositionalEmbedding

# Baseline model
class TE(nn.Module):
    def __init__(self, input_dim=9, scalar_dim=16, output_dim=14, dim_model=256, num_heads=8, num_encoder_layers=8, use_pe=2):
        super().__init__()
        self.dim_model = dim_model
        self.embed = nn.Linear(input_dim, dim_model)
        self.use_pe = use_pe
        if use_pe == 1:
            self.pe = PositionalEncoding(dim_model + scalar_dim)
        elif use_pe == 2:
            self.pe = RotaryPositionalEmbedding(dim_model + scalar_dim)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model + scalar_dim,
                nhead=num_heads,
                dim_feedforward=dim_model * 4,
                batch_first=True,
                dropout=.1
            ),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(dim_model + scalar_dim, output_dim)

    def forward(self, src, scalars):
        x = self.embed(src) * math.sqrt(self.dim_model)
        x = torch.cat([x, scalars], dim=-1)
        if self.use_pe > 0:
            x = self.pe(x)
        x = self.te(x)
        x = self.fc(x)
        return x
