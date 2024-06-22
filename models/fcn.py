import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from torch.autograd import Variable
from torchmetrics.regression import R2Score
import sys
sys.path.append('.')
sys.path.append('..')
from models.pe import *

def r2_score(pred:torch.Tensor, tgt:torch.Tensor) -> float:
    ss_res = torch.sum((tgt - pred) ** 2)
    ss_tot = torch.sum((tgt - torch.mean(tgt)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


class RoFCN(pl.LightningModule):
    def __init__(self, mask, input_dim, scalar_dim, output_dim, dim_model=16, num_heads=4, num_encoder_layers=6):
        super().__init__()
        self.save_hyperparameters()
        self.mask = mask
        self.inverse_mask = ~self.mask.bool()
        
        self.enc1 = nn.Sequential(nn.Conv1d(input_dim, dim_model, kernel_size=3, padding=1), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv1d(dim_model, dim_model * 2, kernel_size=5, padding=2), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv1d(dim_model * 2, dim_model * 4, kernel_size=7, padding=3), nn.GELU())
        self.enc4 = nn.Sequential(nn.Conv1d(dim_model * 4, dim_model * 8, kernel_size=7, padding=3), nn.GELU())
        self.enc5 = nn.Sequential(nn.Conv1d(dim_model * 8, dim_model * 16, kernel_size=5, padding=2), nn.GELU())
        self.enc6 = nn.Sequential(nn.Conv1d(dim_model * 16, dim_model * 32, kernel_size=3, padding=1), nn.GELU())
        self.pe = RotaryPositionalEmbedding(dim_model * 63 + scalar_dim)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model * 63 + scalar_dim,
                nhead=num_heads,
                dim_feedforward=dim_model * 256,
                batch_first=True,
            )
            , num_layers=num_encoder_layers
        )
        self.fc = nn.ModuleList([KANLinear(dim_model * 63 + scalar_dim, output_dim) for _ in range(60)])
        self.loss = nn.MSELoss()

    def forward(self, src, scalars): 
        src = src.permute(0, 2, 1)
        enc1 = self.enc1(src)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        x = torch.cat([enc1, enc2, enc3, enc4, enc5, enc6], dim=1).permute(0, 2, 1)
        x = self.te(self.pe(torch.cat([x, scalars], dim=-1)))
        return torch.stack([self.fc[i](x[:,i,:]) for i in range(x.size(1))], dim=1)

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
        preds[:, self.inverse_mask] = 0.
        r2 = r2_score(preds, tgt)
        self.log("val_score", r2, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        src, slr = batch
        batch_size = src.size(0)
        output = self(src, slr)
        part1 = output[:,:,:6].permute(0, 2, 1).reshape(batch_size, -1)
        part2 = torch.mean(output[:,:,6:], axis=1)
        preds = torch.cat((part1, part2), dim=1)
        return preds

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=4e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.85)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}


class Ro2(pl.LightningModule):
    def __init__(self, mask, input_dim, scalar_dim, output_dim, dim_model=16, num_heads=8, num_encoder_layers=8):
        super().__init__()
        self.save_hyperparameters()
        self.mask = mask
        self.inverse_mask = ~self.mask.bool()
        self.enc1 = nn.Sequential(nn.Conv1d(input_dim, dim_model, kernel_size=3, padding=1), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv1d(dim_model, dim_model * 2, kernel_size=5, padding=2), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv1d(dim_model * 2, dim_model * 4, kernel_size=7, padding=3), nn.GELU())
        self.enc4 = nn.Sequential(nn.Conv1d(dim_model * 4, dim_model * 8, kernel_size=5, padding=2), nn.GELU())
        self.enc5 = nn.Sequential(nn.Conv1d(dim_model * 8, dim_model * 16, kernel_size=3, padding=1), nn.GELU())
        self.pe = RotaryPositionalEmbedding(dim_model * 31 + scalar_dim)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model * 31 + scalar_dim,
                nhead=num_heads,
                dim_feedforward=dim_model * 128,
                batch_first=True,
            )
            , num_layers=num_encoder_layers
        )
        self.fc = nn.ModuleList([KANLinear(dim_model * 31 + scalar_dim, output_dim) for _ in range(60)])
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
        self.fc_ = nn.ModuleList([KANLinear(dim_model * 31 + scalar_dim, output_dim) for _ in range(60)])
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
        x = torch.stack([self.fc[i](x[:,i,:]) for i in range(x.size(1))], dim=1)

        enc1_ = self.enc1_(src)
        enc2_ = self.enc2_(enc1_)
        enc3_ = self.enc3_(enc2_)
        enc4_ = self.enc4_(enc3_)
        enc5_ = self.enc5_(enc4_)
        x_ = torch.cat([enc1_, enc2_, enc3_, enc4_, enc5_], dim=1).permute(0, 2, 1)
        x_ = self.te_(torch.cat([x_, scalars], dim=-1))
        x_ = torch.stack([self.fc_[i](x_[:,i,:]) for i in range(x_.size(1))], dim=1)
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
        preds[:, self.inverse_mask] = 0.
        r2 = r2_score(preds, tgt)
        self.log("val_score", r2, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
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
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.75)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}
  

class RoKAN(nn.Module):
    def __init__(self, input_dim, scalar_dim, output_dim, dim_model=16, num_heads=4, num_encoder_layers=4,  use_pe=True):
        super().__init__()
        self.use_pe = use_pe
        self.enc1 = nn.Sequential(nn.Conv1d(input_dim, dim_model, kernel_size=3, padding=1), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv1d(dim_model, dim_model * 2, kernel_size=5, padding=2), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv1d(dim_model * 2, dim_model * 4, kernel_size=7, padding=3), nn.GELU())
        self.enc4 = nn.Sequential(nn.Conv1d(dim_model * 4, dim_model * 8, kernel_size=5, padding=2), nn.GELU())
        self.enc5 = nn.Sequential(nn.Conv1d(dim_model * 8, dim_model * 16, kernel_size=3, padding=1), nn.GELU())
        self.pe = RotaryPositionalEmbedding(dim_model * 31 + scalar_dim)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model * 31 + scalar_dim,
                nhead=num_heads,
                dim_feedforward=dim_model * 128,
                batch_first=True,
            )
            , num_layers=num_encoder_layers
        )
        self.fc = nn.ModuleList([KANLinear(dim_model * 31 + scalar_dim, output_dim) for _ in range(60)])
        self.loss = nn.MSELoss()

    def forward(self, src, scalars): 
        src = src.permute(0, 2, 1)
        enc1 = self.enc1(src)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        x = torch.cat([enc1, enc2, enc3, enc4, enc5], dim=1).permute(0, 2, 1)
        x = torch.cat([x, scalars], dim=-1)
        if self.use_pe:
            x = self.pe(x)
        x = self.te(x)
        x = torch.stack([self.fc[i](x[:,i,:]) for i in range(x.size(1))], dim=1)
        return x 


class RoKANLit():
    pass

class KANFormer(nn.Module):
    def __init__(self, input_dim, scalar_dim, output_dim, dim_model=16, num_heads=4, num_encoder_layers=4,  use_pe=True):
        super().__init__()
        self.use_pe = use_pe
        self.embed = nn.ModuleList([KANLinear(input_dim, dim_model * 32) for _ in range(60)])
        self.pe = RotaryPositionalEmbedding(dim_model * 32 + scalar_dim)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model * 32 + scalar_dim,
                nhead=num_heads,
                dim_feedforward=dim_model * 128,
                batch_first=True,
            )
            , num_layers=num_encoder_layers
        )
        self.fc = nn.ModuleList([KANLinear(dim_model * 32 + scalar_dim, output_dim) for _ in range(60)])
        self.loss = nn.MSELoss()

    def forward(self, src, scalars):
        x = torch.stack([self.embed[i](src[:,i,:]) for i in range(src.size(1))], dim=1)
        x = torch.cat([x, scalars], dim=-1)
        if self.use_pe:
            x = self.pe(x)
        x = self.te(x)
        x = torch.stack([self.fc[i](x[:,i,:]) for i in range(x.size(1))], dim=1)
        return x

class RoKAN2(pl.LightningModule):
    def __init__(self, mask, input_dim, scalar_dim, output_dim, dim_model=16, num_heads=4, num_encoder_layers=4):
        super().__init__()
        self.save_hyperparameters()
        self.mask = mask
        self.output_dim = output_dim
        self.inverse_mask = ~self.mask.bool()
        self.enc1 = nn.Sequential(nn.Conv1d(input_dim, dim_model, kernel_size=3, padding=1), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv1d(dim_model, dim_model * 2, kernel_size=5, padding=2), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv1d(dim_model * 2, dim_model * 4, kernel_size=7, padding=3), nn.GELU())
        self.enc4 = nn.Sequential(nn.Conv1d(dim_model * 4, dim_model * 8, kernel_size=5, padding=2), nn.GELU())
        self.enc5 = nn.Sequential(nn.Conv1d(dim_model * 8, dim_model * 16, kernel_size=3, padding=1), nn.GELU())
        self.pe = RotaryPositionalEmbedding(dim_model * 31 + scalar_dim)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model * 31 + scalar_dim,
                nhead=num_heads,
                dim_feedforward=dim_model * 128,
                batch_first=True,
            )
            , num_layers=num_encoder_layers
        )
        self.fc = nn.ModuleList([KANLinear(dim_model * 31 + scalar_dim, output_dim) for _ in range(60)])

        self.embed = nn.ModuleList([KANLinear(input_dim + scalar_dim, dim_model * 32) for _ in range(60)])
        self.te_ = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model * 32,
                nhead=4,
                dim_feedforward=dim_model * 128,
                batch_first=True,
            )
            , num_layers=num_encoder_layers
        )
        self.pe_ = RotaryPositionalEmbedding(dim_model * 32)
        self.fc_ = nn.Linear(dim_model * 32, output_dim)
        self.loss = nn.MSELoss()

    def forward(self, src, scalars): 
        x_ = torch.cat([src, scalars], dim=-1)
        x_ = torch.stack([self.embed[i](x_[:,i,:]) for i in range(src.size(1))], dim=1)
        x_ = self.te_(self.pe_(x_))
        x_ = self.fc_(x_)
        src = src.permute(0, 2, 1)
        enc1 = self.enc1(src)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        x = torch.cat([enc1, enc2, enc3, enc4, enc5], dim=1).permute(0, 2, 1)
        x = self.te(self.pe(torch.cat([x, scalars], dim=-1)))
        x = torch.stack([self.fc[i](x[:,i,:]) for i in range(x.size(1))], dim=1)
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
        preds[:, self.inverse_mask] = 0.
        r2 = r2_score(preds, tgt)
        self.log("val_score", r2, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
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
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.65)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

class KanFormer2(pl.LightningModule):
    def __init__(self, mask, input_dim, scalar_dim, output_dim, dim_model=16, num_heads=8, num_encoder_layers=8):
        super().__init__()
        self.save_hyperparameters()
        self.mask = mask
        self.output_dim = output_dim
        self.inverse_mask = ~self.mask.bool()

        self.model_1 = KANFormer(input_dim, scalar_dim, output_dim, dim_model, num_heads, num_encoder_layers, True)
        self.model_2 = KANFormer(input_dim, scalar_dim, output_dim, dim_model, num_heads, num_encoder_layers, False)
        
        self.loss = nn.MSELoss()

    def forward(self, src, scalars):
        x = self.model_1(src, scalars)
        x_ = self.model_2(src, scalars)
        return (x + x_) / 2.

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
        preds[:, self.inverse_mask] = 0.
        r2 = r2_score(preds, tgt)
        self.log("val_score", r2, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
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
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.75)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

class Ensemble(pl.LightningModule):
    def __init__(self, mask, input_dim, scalar_dim, output_dim, dim_model=16, num_heads=4, num_encoder_layers=4):
        super().__init__()
        self.save_hyperparameters()
        self.mask = mask
        self.output_dim = output_dim
        self.inverse_mask = ~self.mask.bool()

        self.model_1 = KANFormer(input_dim, scalar_dim, output_dim, dim_model, num_heads, num_encoder_layers, True)
        self.model_2 = KANFormer(input_dim, scalar_dim, output_dim, dim_model, num_heads, num_encoder_layers, False)
        self.model_3 = RoKAN(input_dim, scalar_dim, output_dim, dim_model, num_heads, num_encoder_layers, True)
        self.model_4 = RoKAN(input_dim, scalar_dim, output_dim, dim_model, num_heads, num_encoder_layers, False)
        self.weights = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32), requires_grad=True)
        self.loss = nn.MSELoss()

    def forward(self, src, scalars):
        x1 =  self.model_1(src, scalars)
        x2 =  self.model_2(src, scalars)
        x3 =  self.model_3(src, scalars)
        x4 =  self.model_4(src, scalars)
        weights = torch.softmax(self.weights, dim=0)
        output = weights[0] * x1 + weights[1] * x2 + weights[2] * x3 + weights[3] * x4
        return output

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
        preds[:, self.inverse_mask] = 0.
        r2 = r2_score(preds, tgt)
        self.log("val_score", r2, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        src, slr = batch
        batch_size = src.size(0)
        output = self(src, slr)
        part1 = output[:,:,:6].permute(0, 2, 1).reshape(batch_size, -1)
        part2 = torch.mean(output[:,:,6:], axis=1)
        preds = torch.cat((part1, part2), dim=1)
        return preds

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=2e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}
