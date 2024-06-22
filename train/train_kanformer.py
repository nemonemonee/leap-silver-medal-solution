import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import sys
sys.path.append('..')
from models.datasets import *
from models.fcn import *

label = torch.load("../data/labeln.pt").to(torch.float32)
label_std = torch.load("../data/labelstd.pt").to(torch.float32)
src_seq = torch.load("../data/seq.pt").to(torch.float32)
src_scl = torch.load("../data/scl.pt").to(torch.float32)
mask = torch.load("../data/weight.pt").to(torch.float32).bool()
input_dim = src_seq.size(-1)
scalar_dim = src_scl.size(-1)


model = KanFormer2(mask=mask, input_dim=input_dim, scalar_dim=scalar_dim, output_dim=14)
checkpoint_callback = ModelCheckpoint(
    dirpath='../ckpt/',
    filename='kf2-{epoch:02d}-{val_score:.3f}',
    save_top_k=5,
    monitor='val_score',
    mode='max'
)
logger = TensorBoardLogger(save_dir="logger")
trainer = Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    max_epochs=500,
    accelerator="gpu",
    devices=[2,3,4,5,6,7],
    strategy='ddp'
)
torch.set_float32_matmul_precision('medium')

ds = MixDataset(src_seq, src_scl, label)
train_loader = DataLoader(ds, batch_size=1024, num_workers=5, shuffle=True, pin_memory=True)
val_loader = DataLoader(ds, batch_size=1024, shuffle=False, pin_memory=True)

trainer.fit(model, train_loader, val_loader)
