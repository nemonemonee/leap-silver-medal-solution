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
    

class AttentionFCN(pl.LightningModule):
    def __init__(self, mask, input_dim, scalar_dim, output_dim, dim_model=16, num_heads=4, num_encoder_layers=4, p=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.mask = mask
        self.enc1 = nn.Sequential(nn.Conv1d(input_dim, dim_model, kernel_size=5, padding=2), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv1d(dim_model, dim_model * 2, kernel_size=5, padding=2), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv1d(dim_model * 2, dim_model * 4, kernel_size=5, padding=2), nn.GELU())
        self.enc4 = nn.Sequential(nn.Conv1d(dim_model * 4, dim_model * 8, kernel_size=5, padding=2), nn.GELU())
        self.enc5 = nn.Sequential(nn.Conv1d(dim_model * 8, dim_model * 16, kernel_size=5, padding=2), nn.GELU())
        self.pe = PositionalEncoding(dim_model * 31 + scalar_dim)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model * 31 + scalar_dim,
                nhead=num_heads,
                dim_feedforward=dim_model * 128,
                batch_first=True,
                dropout=p
            )
            , num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(dim_model * 31 + scalar_dim, output_dim)
        self.loss = nn.MSELoss()

    def forward(self, src, scalars):
        src = src.permute(0, 2, 1) 
        enc1 = self.enc1(src)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        x = torch.cat([enc1, enc2, enc3, enc4, enc5], dim=1).permute(0, 2, 1)
        x = self.te(self.pe(torch.cat([x, scalars], dim=-1)))
        x = self.fc(x)
        return x

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
    


class FCN(pl.LightningModule):
    def __init__(self, mask, input_dim, scalar_dim, output_dim=368, dim_model=16, num_heads=8, num_encoder_layers=4, p=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.mask = mask
        self.enc1 = nn.Sequential(nn.Conv1d(input_dim, dim_model, kernel_size=5, padding=2), nn.LeakyReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(dim_model, dim_model * 2, kernel_size=5, padding=2), nn.LeakyReLU())
        self.enc3 = nn.Sequential(nn.Conv1d(dim_model * 2, dim_model * 4, kernel_size=5, padding=2), nn.LeakyReLU())
        self.enc4 = nn.Sequential(nn.Conv1d(dim_model * 4, dim_model * 8, kernel_size=5, padding=2), nn.LeakyReLU())
        self.enc5 = nn.Sequential(nn.Conv1d(dim_model * 8, dim_model * 16, kernel_size=5, padding=2), nn.LeakyReLU())
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model * 16 + scalar_dim,
                nhead=num_heads,
                dim_feedforward=dim_model * 64,
                batch_first=True,
                dropout=p
            )
            , num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(dim_model * 16 + scalar_dim, output_dim)
        self.loss = nn.MSELoss()

    def forward(self, src, scalars):
        src = src.permute(0, 2, 1) 
        enc1 = F.max_pool1d(self.enc1(src), 2)  # 30
        enc2 = F.max_pool1d(self.enc2(enc1), 2) # 15
        enc3 = F.max_pool1d(self.enc3(enc2), 2) # 7
        enc4 = F.max_pool1d(self.enc4(enc3), 2) # 3
        enc5 = F.max_pool1d(self.enc5(enc4), 2) # 1
        x = enc5.permute(0, 2, 1)
        x = torch.cat([x, scalars], dim=-1)
        x = self.te(x)
        x = self.fc(x[:,0,:])
        return x

    def training_step(self, batch, batch_idx):
        src, slr, tgt = batch
        preds = self(src, slr)
        loss = self.loss(preds[:, self.mask], tgt[:, self.mask])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, slr, tgt = batch
        preds = self(src, slr)
        loss = self.loss(preds[:, self.mask], tgt[:, self.mask])
        loss = self.loss(preds[:, self.mask], tgt[:, self.mask])
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        batch_size = batch.size(0)
        src, slr = batch
        return self(src, slr)  

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.85)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}
    


class DoubleAttentionFCN(pl.LightningModule):
    def __init__(self, mask, input_dim, scalar_dim, output_dim, dim_model=16, num_heads=4, num_encoder_layers=4, p=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.mask = mask
        self.enc1 = nn.Sequential(nn.Conv1d(input_dim, dim_model, kernel_size=5, padding=2), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv1d(dim_model, dim_model * 2, kernel_size=5, padding=2), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv1d(dim_model * 2, dim_model * 4, kernel_size=5, padding=2), nn.GELU())
        self.enc4 = nn.Sequential(nn.Conv1d(dim_model * 4, dim_model * 8, kernel_size=5, padding=2), nn.GELU())
        self.enc5 = nn.Sequential(nn.Conv1d(dim_model * 8, dim_model * 16, kernel_size=5, padding=2), nn.GELU())
        self.pe = PositionalEncoding(dim_model * 31 + scalar_dim)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model * 31 + scalar_dim,
                nhead=num_heads,
                dim_feedforward=dim_model * 128,
                batch_first=True,
                # dropout=p
            )
            , num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(dim_model * 31 + scalar_dim, output_dim)

        self.enc1_ = nn.Sequential(nn.Conv1d(input_dim, dim_model, kernel_size=5, padding=2), nn.GELU())
        self.enc2_ = nn.Sequential(nn.Conv1d(dim_model, dim_model * 2, kernel_size=5, padding=2), nn.GELU())
        self.enc3_ = nn.Sequential(nn.Conv1d(dim_model * 2, dim_model * 4, kernel_size=5, padding=2), nn.GELU())
        self.enc4_ = nn.Sequential(nn.Conv1d(dim_model * 4, dim_model * 8, kernel_size=5, padding=2), nn.GELU())
        self.enc5_ = nn.Sequential(nn.Conv1d(dim_model * 8, dim_model * 16, kernel_size=5, padding=2), nn.GELU())
        self.te_ = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model * 31 + scalar_dim,
                nhead=4,
                dim_feedforward=dim_model * 128,
                batch_first=True,
            )
            , num_layers=num_encoder_layers
        )
        self.fc_ = nn.Linear(dim_model * 31 + scalar_dim, output_dim)
        self.loss = nn.MSELoss()

    def forward(self, src, scalars): 
        src = src.permute(0, 2, 1)
        enc1 = self.enc1(src)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        x = torch.cat([enc1, enc2, enc3, enc4, enc5], dim=1).permute(0, 2, 1)
        x = self.te(self.pe(torch.cat([x, scalars], dim=-1)))
        x = self.fc(x)

        enc1_ = self.enc1_(src)
        enc2_ = self.enc2_(enc1_)
        enc3_ = self.enc3_(enc2_)
        enc4_ = self.enc4_(enc3_)
        enc5_ = self.enc5_(enc4_)
        x_ = torch.cat([enc1_, enc2_, enc3_, enc4_, enc5_], dim=1).permute(0, 2, 1)
        x_ = self.te_(torch.cat([x_, scalars], dim=-1))
        x_ = self.fc_(x_)
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
        src, slr = batch
        batch_size = src.size(0)
        output = self(src, slr)
        part1 = output[:,:,:6].permute(0, 2, 1).reshape(batch_size, -1)
        part2 = torch.mean(output[:,:,6:], axis=1)
        preds = torch.cat((part1, part2), dim=1)
        return preds
    

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.75)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}
    