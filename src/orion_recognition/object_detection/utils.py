from datetime import datetime

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Identity
from torch.optim import Adam
from torchvision import datasets, models, transforms
import pytorch_lightning as pl
from torchvision.transforms import RandomChoice, CenterCrop, RandomCrop, Resize, ToTensor, Compose, RandomRotation, \
    RandomAffine, ColorJitter


def load_resnet(num_classes):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


def get_transforms(input_size=224, rotation=15):
    train_tf_list = [
        Resize(int(input_size * 1.2)),
        RandomChoice([
            RandomRotation(rotation),
            RandomAffine(rotation,
                         scale=(0.9, 1.1),
                         translate=(0.1, 0.1),
                         shear=10),
            Identity()
        ]),
        RandomChoice([
            RandomCrop(input_size),
            CenterCrop(input_size)
        ]),
        ColorJitter(brightness=0.4, contrast=0.4,
                    saturation=0.4, hue=0.125),
        ToTensor()
    ]

    val_tf_list = [
        Resize(input_size),
        ToTensor()
    ]

    return Compose(train_tf_list), Compose(val_tf_list)


class ResNetModule(pl.LightningModule):
    def __init__(self, num_classes, save_path=None):
        super().__init__()
        self.model = load_resnet(num_classes)
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_hyperparameters()

    def _step_common(self, batch):
        images, labels = batch
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step_common(batch)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step_common(batch)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)
        return {"loss": loss, "acc": acc}

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self.infer_step(batch, batch_idx, "pred")

    def validation_epoch_end(self, outputs):
        acc = torch.stack([o['acc'] for o in outputs]).mean().item()
        print(acc)

    def on_train_epoch_end(self):
        torch.save(self.model, os.path.join(self.save_path, f"resnet_epoch_{self.current_epoch}.pth"))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)


def collate_fn(batch):
    return tuple(zip(*batch))
