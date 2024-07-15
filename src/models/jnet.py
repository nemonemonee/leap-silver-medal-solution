import torch
from torch import nn
from models.pe import *
from models.kan import *


class JNet(nn.Module):
    def __init__(self,
                 input_dim=9, 
                 scalar_dim=16, 
                 output_dim=14,
                 dim_model=16, 
                 conv_kernals=[3,5,9,7,5,3],
                 num_heads=4, 
                 num_encoder_layers=6,
                 use_pe=0):
        super().__init__()
        
        self.use_pe = use_pe
        self.conv_layers = nn.ModuleList()
        dim_in = input_dim
        dim_sum = 0
        for i, ks in enumerate(conv_kernals):
            dim_out = dim_model * (2 ** i)
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(dim_in, dim_out, kernel_size=ks, padding=int(ks//2)), 
                    nn.GELU()
                )
            )
            dim_in = dim_out
            dim_sum += dim_out

        if use_pe == 1:
            self.pe = PositionalEncoding(dim_sum + scalar_dim)
        elif use_pe == 2:
            self.pe = RotaryPositionalEmbedding(dim_sum + scalar_dim)

        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_sum + scalar_dim,
                nhead=num_heads,
                dim_feedforward=dim_out * 8,
                batch_first=True,
            )
            , num_layers=num_encoder_layers
        )
        self.fc = nn.ModuleList([KANLinear(dim_sum + scalar_dim, output_dim) for _ in range(60)])

    def forward(self, src, scalars): 
        x = src.permute(0, 2, 1)
        embed = []
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            embed.append(x)
        x = torch.cat(embed, dim=1).permute(0, 2, 1)
        x = torch.cat([x, scalars], dim=-1)
        if self.use_pe > 0:
            x = self.pe(x)
        x = self.te(x)
        return torch.stack([self.fc[i](x[:,i,:]) for i in range(x.size(1))], dim=1)

class JCNet(nn.Module):
    def __init__(self,
                 input_dim=9, 
                 scalar_dim=16, 
                 output_dim=14,
                 dim_model=16, 
                 conv_kernals=[3,5,9,7],
                 num_heads=8, 
                 num_encoder_layers=8,
                 use_pe=2):
        super().__init__()
        
        self.use_pe = use_pe
        self.conv_layers = nn.ModuleList()
        dim_in = input_dim
        dim_sum = 0
        for i, ks in enumerate(conv_kernals):
            dim_out = dim_model * (2 ** i)
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(dim_in, dim_out, kernel_size=ks, padding=int(ks//2)), 
                    nn.GELU()
                )
            )
            dim_in = dim_out
            dim_sum += dim_out

        if use_pe == 1:
            self.pe = PositionalEncoding(dim_sum + scalar_dim)
        elif use_pe == 2:
            self.pe = RotaryPositionalEmbedding(dim_sum + scalar_dim)

        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_sum + scalar_dim,
                nhead=num_heads,
                dim_feedforward=dim_out * 8,
                batch_first=True,
                dropout=.01
            )
            , num_layers=num_encoder_layers
        )
        self.decoder = nn.ModuleList()
        dim_in = dim_sum + scalar_dim
        for ks in reversed(conv_kernals[1:]):
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

    def forward(self, src, scalars): 
        x = src.permute(0, 2, 1)
        embed = []
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            embed.append(x)
        x = torch.cat(embed, dim=1).permute(0, 2, 1)
        x = torch.cat([x, scalars], dim=-1)
        if self.use_pe > 0:
            x = self.pe(x)
        x = self.te(x)
        x = x.permute(0,2,1)
        for layer in self.decoder:
            x = layer(x)
        return x.permute(0,2,1)

class JLNet(nn.Module):
    def __init__(self,
                 input_dim=9, 
                 scalar_dim=16, 
                 output_dim=14,
                 dim_model=16, 
                 conv_kernals=[3,5,9,7],
                 num_heads=8, 
                 num_encoder_layers=8,
                 use_pe=2):
        super().__init__()
        
        self.use_pe = use_pe
        self.conv_layers = nn.ModuleList()
        dim_in = input_dim
        dim_sum = 0
        for i, ks in enumerate(conv_kernals):
            dim_out = dim_model * (2 ** i)
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(dim_in, dim_out, kernel_size=ks, padding=int(ks//2)), 
                    nn.GELU()
                )
            )
            dim_in = dim_out
            dim_sum += dim_out

        if use_pe == 1:
            self.pe = PositionalEncoding(dim_sum + scalar_dim)
        elif use_pe == 2:
            self.pe = RotaryPositionalEmbedding(dim_sum + scalar_dim)

        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_sum + scalar_dim,
                nhead=num_heads,
                dim_feedforward=dim_out * 8,
                batch_first=True,
                dropout=.1
            )
            , num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(dim_sum + scalar_dim, output_dim)

    def forward(self, src, scalars): 
        x = src.permute(0, 2, 1)
        embed = []
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            embed.append(x)
        x = torch.cat(embed, dim=1).permute(0, 2, 1)
        x = torch.cat([x, scalars], dim=-1)
        if self.use_pe > 0:
            x = self.pe(x)
        x = self.te(x)
        return self.fc(x)
