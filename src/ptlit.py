import sys
import torch
import lightning.pytorch as pl
import torch.nn as nn
sys.path.append('..')
from utils import r2_score


class PTLit(pl.LightningModule):
    def __init__(self, mask, lr, step_size, gamma, models):
        super().__init__()
        self.save_hyperparameters()
        self.mask = mask
        self.inverse_mask = ~self.mask.bool()

        self.models = nn.ModuleList()
        for model in models:
            self.models.append(model)
        self.loss = nn.MSELoss()

    def forward(self, src, scalars):
        xs =  torch.stack([self.models[i](src, scalars) for i in range(len(self.models))])
        return torch.mean(xs, dim=0)

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
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}
    