import torch
from torch import nn
from models.pe import *
from models.kan import *

class KF(nn.Module):
    def __init__(self, input_dim=9, scalar_dim=16, output_dim=14, dim_model=512, num_heads=4, num_encoder_layers=6,  use_pe=0):
        super().__init__()
        self.use_pe = use_pe
        self.embed = nn.ModuleList([KANLinear(input_dim + scalar_dim, dim_model) for _ in range(60)])
        if use_pe == 1:
            self.pe = PositionalEncoding(dim_model)
        elif use_pe == 2:
            self.pe = RotaryPositionalEmbedding(dim_model)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dim_feedforward=dim_model*4,
                batch_first=True,
            )
            , num_layers=num_encoder_layers
        )
        self.fc = nn.ModuleList([KANLinear(dim_model, output_dim) for _ in range(60)])
        self.loss = nn.MSELoss()

    def forward(self, src, scalars):
        x = torch.cat([src, scalars], dim=-1)
        x = torch.stack([self.embed[i](x[:,i,:]) for i in range(x.size(1))], dim=1)
        if self.use_pe > 0:
            x = self.pe(x)
        x = self.te(x)
        x = torch.stack([self.fc[i](x[:,i,:]) for i in range(x.size(1))], dim=1)
        return x

class KFC(nn.Module):
    def __init__(
            self, 
            input_dim=9, 
            scalar_dim=16, 
            output_dim=14, 
            dim_model=512, 
            num_heads=4, 
            num_encoder_layers=6, 
            conv_kernals=[9,7,5,3], 
            use_pe=2):
        super().__init__()
        self.use_pe = use_pe
        self.embed = nn.ModuleList([KANLinear(input_dim + scalar_dim, dim_model) for _ in range(60)])
        if use_pe == 1:
            self.pe = PositionalEncoding(dim_model)
        elif use_pe == 2:
            self.pe = RotaryPositionalEmbedding(dim_model)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dim_feedforward=dim_model*4,
                batch_first=True,
            )
            , num_layers=num_encoder_layers
        )
        self.decoder = nn.ModuleList()
        dim_in = dim_model
        dim_out = dim_model // 2
        for ks in conv_kernals[:-1]:
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose1d(dim_in, dim_out, kernel_size=ks, padding=int(ks//2)), 
                    nn.GELU()
                )
            )
            dim_in = dim_out
            dim_out = dim_out // 2

        self.decoder.append(
            nn.ConvTranspose1d(
                in_channels=dim_in, 
                out_channels=output_dim, 
                kernel_size=conv_kernals[0], 
                padding=int(conv_kernals[0]//2)
            )
        )
        self.loss = nn.MSELoss()

    def forward(self, src, scalars):
        x = torch.cat([src, scalars], dim=-1)
        x = torch.stack([self.embed[i](x[:,i,:]) for i in range(x.size(1))], dim=1)
        if self.use_pe > 0:
            x = self.pe(x)
        x += self.te(x)
        x = x.permute(0,2,1)
        for layer in self.decoder:
            x = layer(x)
        return x.permute(0,2,1)

class KFL(nn.Module):
    def __init__(
            self, 
            input_dim=9, 
            scalar_dim=16, 
            output_dim=14, 
            dim_model=256, 
            num_heads=8, 
            num_encoder_layers=8,
            use_pe=2):
        super().__init__()
        self.use_pe = use_pe
        self.embed = nn.ModuleList([KANLinear(input_dim + scalar_dim, dim_model) for _ in range(60)])
        if use_pe == 1:
            self.pe = PositionalEncoding(dim_model)
        elif use_pe == 2:
            self.pe = RotaryPositionalEmbedding(dim_model)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dim_feedforward=dim_model*4,
                batch_first=True,
                dropout=.01
            )
            , num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(dim_model, output_dim)
        self.loss = nn.MSELoss()

    def forward(self, src, scalars):
        x = torch.cat([src, scalars], dim=-1)
        x = torch.stack([self.embed[i](x[:,i,:]) for i in range(x.size(1))], dim=1)
        if self.use_pe > 0:
            x_pe = self.pe(x)
            x += self.te(x_pe)
        else:
            x += self.te(x)
        return self.fc(x)
