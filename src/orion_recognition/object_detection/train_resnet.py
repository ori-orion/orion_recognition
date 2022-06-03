import argparse
import datetime
import os

from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import DataLoader
import pytorch_lightning as pl

import sys

sys.path.append(".")
sys.path.append("./src")

from orion_recognition.datasets.athome_dataset import AtHomeImageDataset
from orion_recognition.object_detection.utils import ResNetModule, get_transforms, collate_fn

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--save_path", default="./data/model")
parser.add_argument("--checkpoint", default=None)

args = parser.parse_args()
train_tf, val_tf = get_transforms()

train_dataset = AtHomeImageDataset(is_train=True, transforms=train_tf)
val_dataset = AtHomeImageDataset(is_train=False, transforms=val_tf)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True, num_workers=args.n_workers, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, drop_last=False, num_workers=args.n_workers, collate_fn=collate_fn)

version_name = f"{datetime.datetime.now().strftime('%m%d_%H%M%S')}{'_' + args.name if args.name else ''}"

module = ResNetModule(train_dataset.n_classes, save_path=os.path.join(args.save_path, version_name))

tb_logger = TensorBoardLogger(save_dir=".", version=version_name)
trainer = pl.Trainer(accelerator="gpu", logger=tb_logger, gpus=1)
trainer.fit(module, train_loader, val_loader, ckpt_path=args.checkpoint)
