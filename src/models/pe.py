import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len=60):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=60):
        super(RotaryPositionalEmbedding, self).__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        sinusoid_inp = torch.einsum('ij->ij', pos @ inv_freq.unsqueeze(0))
        self.register_buffer('sin', sinusoid_inp.sin())
        self.register_buffer('cos', sinusoid_inp.cos())

    def forward(self, x):
        # x: (batch_size, seq_len, dim)
        batch_size, seq_len, dim = x.shape
        sin = self.sin[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, seq_len, dim)
        cos = self.cos[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, seq_len, dim)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
        return x