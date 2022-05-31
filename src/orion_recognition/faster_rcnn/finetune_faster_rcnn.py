import torch
import argparse
import sys
# to train on gpu if selected.
from torch.utils.data import DataLoader
import datetime
import os

sys.path.insert(0, ".")

from orion_recognition.faster_rcnn.utils import collate_fn
from orion_recognition.datasets.athome_dataset import LABELS_SUBSET_PATH, AtHomeImageDataset
from orion_recognition.faster_rcnn.engine import train_one_epoch, evaluate
from orion_recognition.object_detector import ObjectDetector

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "model")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", "-bs", type=int, default=8)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--model_path", type=str, default=MODEL_PATH)

args = parser.parse_args()

model_path = args.model_path

if not os.path.exists(model_path):
    os.makedirs(model_path)

# use our dataset and defined transformations
train_dataset = AtHomeImageDataset(is_train=True)
val_dataset = AtHomeImageDataset(is_train=False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                          collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                        collate_fn=collate_fn)

device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

with open(LABELS_SUBSET_PATH, "r") as f:
    labels = f.read().split("\n")

# get the model using our helper function
model = ObjectDetector(labels)

# move model to the right device
model.to(device)

# construct an optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(args.epochs):
    # training for one epoch
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100, model_path=args.model_path, save_freq=10000)

    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, val_loader, device=device)
