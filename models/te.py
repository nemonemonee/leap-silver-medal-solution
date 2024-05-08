import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        self.register_buffer('pe', self.encoding)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DoubleTransformer(pl.LightningModule):
    def __init__(self, mask, input_dim, scalar_dim, output_dim, dim_model=64, num_heads=4, num_encoder_layers=4, p=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.embed = nn.Linear(input_dim, dim_model)
        self.embed_ = nn.Linear(input_dim, dim_model)
        self.pe = PositionalEncoding(dim_model)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dim_feedforward=dim_model * 4,
                batch_first=True,
                dropout=p
            ),
            num_layers=num_encoder_layers
        )
        self.te_ = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dim_feedforward=dim_model * 4,
                batch_first=True,
                dropout=p
            ),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(dim_model + scalar_dim, output_dim)
        self.fc_ = nn.Linear(dim_model + scalar_dim, output_dim)
        self.mask = mask
        self.loss = nn.MSELoss()

    def forward(self, src, scalars):
        x = self.embed(src) * math.sqrt(self.embed.out_features)
        x = self.pe(x)
        x = self.te(x)
        x = self.fc(torch.cat([x, scalars], dim=-1))
        x_ = self.embed_(src) * math.sqrt(self.embed_.out_features)
        x_ = self.te_(x_)
        x_ = self.fc_(torch.cat([x_, scalars], dim=-1))
        return x + x_

    def training_step(self, batch, batch_idx):
        src, slr, tgt = batch
        batch_size = src.size(0)
        output = self(src, slr)
        part1 = output[:,:,:6].permute(0, 2, 1).reshape(batch_size, -1)
        part2 = torch.mean(output[:,:,6:], axis=1)
        preds = torch.cat((part1, part2), dim=1)
        loss = self.loss(preds[:, self.mask], tgt[:, self.mask])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, slr, tgt = batch
        batch_size = src.size(0)
        output = self(src, slr)
        part1 = output[:,:,:6].permute(0, 2, 1).reshape(batch_size, -1)
        part2 = torch.mean(output[:,:,6:], axis=1)
        preds = torch.cat((part1, part2), dim=1)
        loss = self.loss(preds[:, self.mask], tgt[:, self.mask])
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        src, slr= batch
        batch_size = src.size(0)
        output = self(src, slr)
        part1 = output[:,:,:6].permute(0, 2, 1).reshape(batch_size, -1)
        part2 = torch.mean(output[:,:,6:], axis=1)
        preds = torch.cat((part1, part2), dim=1)
        return preds
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.85)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}
    
