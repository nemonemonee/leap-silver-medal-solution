import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from ptlit import PTLit
from models.te import TE
from datasets import LEAPDataset

label = torch.load("../data/labeln.pt").to(torch.float32)
src_seq = torch.load("../data/src_seq_p.pt").to(torch.float32)
src_scl = torch.load("../data/src_scl_p.pt").to(torch.float32)
mask = torch.load("../data/weight.pt").to(torch.float16).bool()

# base = PTLit.load_from_checkpoint("../ckpt/jnet/xlarge-n-2-epoch=44-val_score=0.717.ckpt").models
mdlit = PTLit(mask, 1e-5, 1, 0.81, [TE(num_encoder_layers=10)])

torch.set_float32_matmul_precision('high')
checkpoint_callback = ModelCheckpoint(
    dirpath='../ckpt/jnet/',
    filename='xlarge-n-2-{epoch:02d}-{val_score:.3f}',
    save_top_k=10,
    monitor='val_score',
    mode='max'
)
logger = TensorBoardLogger(save_dir="logger")
trainer = pl.Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    max_epochs=30,
    accelerator="gpu",
    devices=[2,3],
    check_val_every_n_epoch=1,
)
ds = LEAPDataset(src_seq, src_scl, label)
train_size = int(0.9 * len(ds))
val_size = len(ds) - train_size
train_dataset, val_dataset = random_split(ds, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=256, num_workers=33, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1024, num_workers=8, shuffle=False, pin_memory=True)
trainer.fit(mdlit, train_loader, val_loader)