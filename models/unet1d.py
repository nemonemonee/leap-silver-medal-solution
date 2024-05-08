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
    
class UNet1D(pl.LightningModule):
    def __init__(self, mask, input_dim, scalar_dim, output_dim, dim_model=16, num_heads=8, num_encoder_layers=4, p=0):
        super().__init__()
        self.save_hyperparameters()
        self.mask = mask
        self.enc1 = nn.Sequential(nn.Conv1d(input_dim, dim_model, kernel_size=5, padding=2), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv1d(dim_model, dim_model * 2, kernel_size=5, padding=2), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv1d(dim_model * 2, dim_model * 4, kernel_size=3, padding=1), nn.GELU())
        self.enc4 = nn.Sequential(nn.Conv1d(dim_model * 4, dim_model * 8, kernel_size=3, padding=1), nn.GELU())
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model * 8 + scalar_dim,
                nhead=num_heads,
                dim_feedforward=dim_model * 32,
                batch_first=True,
                dropout=p
            )
            , num_layers=num_encoder_layers
        )
        self.dec1 = nn.Sequential(nn.ConvTranspose1d(dim_model * 8 + scalar_dim, dim_model * 4, kernel_size=2, stride=2), nn.GELU()) 
        self.dec2 = nn.Sequential(nn.ConvTranspose1d(dim_model * 8, dim_model * 2, kernel_size=2, stride=2), nn.GELU()) 
        self.dec3 = nn.Sequential(nn.ConvTranspose1d(dim_model * 4, dim_model, kernel_size=2, stride=2), nn.GELU()) 
        self.dec4 = nn.Sequential(nn.ConvTranspose1d(dim_model * 2, dim_model, kernel_size=2, stride=2), nn.GELU())
        self.final_conv = nn.Conv1d(dim_model, output_dim, kernel_size=1)
        self.loss = nn.MSELoss()

    def forward(self, src, scalars):
        src = src.permute(0, 2, 1) 
        enc1 = F.avg_pool1d(self.enc1(src), 2)  # 30
        enc2 = F.avg_pool1d(self.enc2(enc1), 2) # 15
        enc3 = F.avg_pool1d(self.enc3(enc2), 2) # 7
        enc4 = F.avg_pool1d(self.enc4(enc3), 2) # 3
        x = enc4.permute(0, 2, 1)
        x = torch.cat([x, scalars], dim=-1)
        x = self.te(x)
        x = x.permute(0, 2, 1) # 3
        dec1 = self.dec1(x)  # 6
        dec1 = F.interpolate(dec1, 7)
        dec2 = self.dec2(torch.cat([dec1, enc3], dim=1))  # 14
        dec2 = F.interpolate(dec2, 15)
        dec3 = self.dec3(torch.cat([dec2, enc2], dim=1))  # 30
        dec4 = self.dec4(torch.cat([dec3, enc1], dim=1))  # 60
        x = self.final_conv(dec4)
        x = x.permute(0, 2, 1)
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
        batch_size = batch.size(0)
        src, slr = batch
        output = self(src, slr)
        part1 = output[:,:,:6].permute(0, 2, 1).reshape(batch_size, -1)
        part2 = torch.mean(output[:,:,6:], axis=1)
        preds = torch.cat((part1, part2), dim=1)
        return preds
    

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.85)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}
    


class AttentionUNet1D(pl.LightningModule):
    def __init__(self, mask, input_dim, scalar_dim, output_dim, dim_model=16, num_heads=4, num_encoder_layers=4, p=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.mask = mask
        self.enc1 = nn.Sequential(nn.Conv1d(input_dim, dim_model, kernel_size=5, padding=2), nn.LeakyReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(dim_model, dim_model * 2, kernel_size=5, padding=2), nn.LeakyReLU())
        self.enc3 = nn.Sequential(nn.Conv1d(dim_model * 2, dim_model * 4, kernel_size=3, padding=1), nn.LeakyReLU())
        self.enc4 = nn.Sequential(nn.Conv1d(dim_model * 4, dim_model * 8, kernel_size=3, padding=1), nn.LeakyReLU())

        self.pe3 = PositionalEncoding(dim_model * 4)
        self.pe4 = PositionalEncoding(dim_model * 8 + scalar_dim)
       
        self.te3 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim_model * 4, nhead=num_heads, dim_feedforward=dim_model * 16, batch_first=True), num_layers=num_encoder_layers)
        self.te4 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim_model * 8 + scalar_dim, nhead=num_heads, dim_feedforward=dim_model * 32, batch_first=True), num_layers=num_encoder_layers)
        
        self.dec1 = nn.Sequential(nn.ConvTranspose1d(dim_model * 8 + scalar_dim, dim_model * 4, kernel_size=2, stride=2), nn.LeakyReLU()) 
        self.dec2 = nn.Sequential(nn.ConvTranspose1d(dim_model * 8, dim_model * 2, kernel_size=2, stride=2), nn.LeakyReLU()) 
        self.dec3 = nn.Sequential(nn.ConvTranspose1d(dim_model * 4, dim_model, kernel_size=2, stride=2), nn.LeakyReLU()) 
        self.dec4 = nn.Sequential(nn.ConvTranspose1d(dim_model * 2, dim_model, kernel_size=2, stride=2), nn.LeakyReLU())
        self.final_conv = nn.Conv1d(dim_model, output_dim, kernel_size=1)
        self.loss = nn.MSELoss()

    def forward(self, src, scalars):
        src = src.permute(0, 2, 1) 
        enc1 = F.avg_pool1d(self.enc1(src), 2)  # 30
        enc2 = F.avg_pool1d(self.enc2(enc1), 2) # 15
        enc3 = F.avg_pool1d(self.enc3(enc2), 2) # 7
        enc4 = F.avg_pool1d(self.enc4(enc3), 2) # 3
        x = self.pe4(torch.cat([enc4.permute(0, 2, 1), scalars], dim=-1))
        x = self.te4(x).permute(0, 2, 1)
        dec1 = self.dec1(x)  # 6
        dec1 = F.interpolate(dec1, 7)
        enc3 = self.te3(self.pe3(enc3.permute(0,2,1))).permute(0,2,1)
        dec2 = self.dec2(torch.cat([dec1, enc3], dim=1))  # 14
        dec2 = F.interpolate(dec2, 15)
        dec3 = self.dec3(torch.cat([dec2, enc2], dim=1))  # 30
        dec4 = self.dec4(torch.cat([dec3, enc1], dim=1))  # 60
        x = self.final_conv(dec4).permute(0, 2, 1)
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
        batch_size = batch.size(0)
        src, slr = batch
        output = self(src, slr)
        part1 = output[:,:,:6].permute(0, 2, 1).reshape(batch_size, -1)
        part2 = torch.mean(output[:,:,6:], axis=1)
        preds = torch.cat((part1, part2), dim=1)
        return preds
    

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.85)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

