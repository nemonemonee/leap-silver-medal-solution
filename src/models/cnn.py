import torch
from torch import nn
from models.pe import *

class CNN(nn.Module):
    def __init__(self, input_dim=9, scalar_dim=16, output_dim=14, dim_model=16, conv_kernals=[3,5,9,7,3]):
        super().__init__()
        self.encoder = nn.ModuleList()
        dim_in = input_dim
        dim_out = dim_model
        for i, ks in enumerate(conv_kernals):
            dim_out = dim_model * (2 ** i)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(dim_in, dim_out, kernel_size=ks, padding=int(ks//2)), 
                    nn.GELU()
                )
            )
            dim_in = dim_out
        self.decoder = nn.ModuleList()
        dim_in = dim_out + scalar_dim
        for ks in reversed(conv_kernals[1:]):
            dim_out = dim_out // 2
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose1d(dim_in, dim_out, kernel_size=ks, padding=int(ks//2)), 
                    nn.GELU()
                )
            )
            dim_in = dim_out
        self.decoder.append(
            nn.ConvTranspose1d(
                in_channels=dim_model, 
                out_channels=output_dim, 
                kernel_size=conv_kernals[0], 
                padding=int(conv_kernals[0]//2)
            )
        )

    def forward(self, src, scl):
        encoded = src.permute(0,2,1)
        for layer in self.encoder:
            encoded = layer(encoded)
        scl = scl.permute(0,2,1)
        decoded = torch.cat([encoded, scl], dim=1)
        for layer in self.decoder:
            decoded = layer(decoded)
        output = decoded.permute(0,2,1)
        return output
    
class CNNTE(nn.Module):
    def __init__(self, input_dim=9, scalar_dim=16, output_dim=14, dim_model=16, conv_kernals=[3,5,9,7], use_pe=2):
        super().__init__()
        self.use_pe = use_pe
        self.encoder = nn.ModuleList()
        dim_in = input_dim
        dim_out = dim_model
        for i, ks in enumerate(conv_kernals):
            dim_out = dim_model * (2 ** i)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(dim_in, dim_out, kernel_size=ks, padding=int(ks//2)), 
                    nn.GELU()
                )
            )
            dim_in = dim_out
        if use_pe == 1:
            self.pe = PositionalEncoding(dim_out+scalar_dim)
        elif use_pe == 2:
            self.pe = RotaryPositionalEmbedding(dim_out+scalar_dim)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_out+scalar_dim,
                nhead=8,
                dim_feedforward=512,
                batch_first=True,
            )
            , num_layers=8
        )
        self.decoder = nn.ModuleList()
        dim_in = dim_out + scalar_dim
        for ks in reversed(conv_kernals[1:]):
            dim_out = dim_out // 2
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose1d(dim_in, dim_out, kernel_size=ks, padding=int(ks//2)), 
                    nn.GELU()
                )
            )
            dim_in = dim_out
        self.decoder.append(
            nn.ConvTranspose1d(
                in_channels=dim_model, 
                out_channels=output_dim, 
                kernel_size=conv_kernals[0], 
                padding=int(conv_kernals[0]//2)
            )
        )

    def forward(self, src, scl):
        encoded = src.permute(0,2,1)
        for layer in self.encoder:
            encoded = layer(encoded)
        scl = scl.permute(0,2,1)
        encoded = torch.cat([encoded, scl], dim=1)
        if self.use_pe > 0:
            encoded = self.pe(encoded.permute(0,2,1))
        encoded += self.te(encoded)
        decoded = encoded.permute(0,2,1)
        for layer in self.decoder:
            decoded = layer(decoded)
        output = decoded.permute(0,2,1)
        return output
    