import polars
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

df = polars.read_csv('../data/train.csv')

SRC_COLS = df.columns[1:557]
TGT_COLS = df.columns[557:]

for col in SRC_COLS:
    df = df.with_columns(polars.col(col).cast(polars.Float32))

for col in TGT_COLS:
    df = df.with_columns(polars.col(col).cast(polars.Float32))

w = torch.load('../data/weight.pt')
mask = w != 0

src = torch.load("../data/src.pt")
label = torch.load("../data/label.pt")

src_mu = src.mean(axis=0)
src_std = torch.maximum(src.std(axis=0), torch.tensor(1e-8))
src = (src - src_mu) / src_std
label_mu = label.mean(axis=0)
label_std = torch.maximum(label.std(axis=0), torch.tensor(1e-8))
label = (label - label_mu) / label_std

input_seq_name = [
    "state_t", "state_q0001", "state_q0002", "state_q0003", "state_u", "state_v", "pbuf_ozone", "pbuf_CH4", "pbuf_N2O"
]
input_scl_name = [
    "state_ps", "pbuf_SOLIN", "pbuf_LHFLX", "pbuf_SHFLX", "pbuf_TAUX", "pbuf_TAUY", "pbuf_COSZRS", "cam_in_ALDIF", "cam_in_ALDIR", "cam_in_ASDIF", "cam_in_ASDIR", "cam_in_LWUP", "cam_in_ICEFRAC", "cam_in_LANDFRAC", "cam_in_OCNFRAC", "cam_in_SNOWHLAND"
]
input_seq_idx = [[idx - 1 for idx, column in enumerate(df.columns) if 
                    column.startswith(var)] for var in input_seq_name]
input_scl_idx = [[idx - 1 for idx, column in enumerate(df.columns) if 
                    column.startswith(var)] for var in input_scl_name]
src_seq = torch.stack([src[:, i] for i in input_seq_idx], dim=-1)
src_scl = torch.stack([src[:, i].repeat(1, 60) for i in input_scl_idx], dim=-1)
input_dim = src_seq.size(-1)
scalar_dim = src_scl.size(-1)

ds = MixDataset(src_seq, src_scl, label)
train_size = int(0.9 * len(ds))
val_size = len(ds) - train_size
train_dataset, val_dataset = random_split(ds, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2048, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048, num_workers=4, shuffle=False)

torch.set_float32_matmul_precision('high')
new_mask = torch.logical_and(mask, label_std > 1.1e-8)
# model = AttentionFCN(mask=new_mask, input_dim=input_dim, scalar_dim=scalar_dim, output_dim=14)
model = DoubleAttentionFCN.load_from_checkpoint("../ckpt/db-fnc-epoch=194-val_loss=0.206.ckpt")
# model = FCN(mask=new_mask, input_dim=input_dim, scalar_dim=scalar_dim)

checkpoint_callback = ModelCheckpoint(
    dirpath='../ckpt/',
    filename='fnc-{epoch:02d}-{val_loss:.3f}',
    save_top_k=3,
    monitor='val_loss',
    mode='min'
)
logger = TensorBoardLogger(save_dir="logger")
trainer = Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    max_epochs=500,
    accelerator="gpu",
    devices=4,
    strategy='ddp'
)
trainer.fit(model, train_loader, val_loader)
