import torch
import os

import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from einops import rearrange
from torch.utils.data import Dataset
from torchvision.transforms import RandomHorizontalFlip, Compose, ToTensor
import argparse


import sys
sys.path.append(".")

from orion_recognition.datasets.utils import get_bbox

ATHOME_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "athome")
LABELS_SUBSET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "athome_labels_subset.txt")


class AtHomeImageDataset(Dataset):

    def __init__(self, dataset_dir = ATHOME_DATASET_DIR, is_train=True, transforms=None, label_path=LABELS_SUBSET_PATH, bbox_brightness_thresh=0.95):
        self.dataset_dir = dataset_dir
        self.transforms = transforms or (Compose([RandomHorizontalFlip(0.5), ToTensor()]) if is_train else ToTensor())
        self.bbox_brightness_thresh = bbox_brightness_thresh

        with open(os.path.join(dataset_dir, "labels.txt"), "r") as f:
            base_labels = f.read().split("\n")

        with open(label_path, "r") as f:
            self.labels = f.read().split("\n")

        self.image_label_pairs = []

        with open(os.path.join(dataset_dir, "training_split.txt" if is_train else "validation_split.txt"), "r") as f:
            for line in f.read().strip().split("\n"):
                path, base_label_id = line.split("\t")
                image_path = path.replace("/PATH/TO/DATABASE/DIRECTORY/athome", ATHOME_DATASET_DIR)
                label_index = self.labels.index(base_labels[int(base_label_id)])
                self.image_label_pairs.append((image_path, label_index))

    def __getitem__(self, idx):
        image_path, label = self.image_label_pairs[idx]
        image = Image.open(image_path)

        image = self.transforms(image)
        bbox = get_bbox(image, self.bbox_brightness_thresh)

        # convert boxes into a torch.Tensor
        boxes = torch.tensor([bbox], dtype=torch.float)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            "boxes": boxes,
            "labels": torch.tensor([label], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.tensor([0], dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        return image, target

    def __len__(self):
        return len(self.image_label_pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", "-i", type=int, default=100)

    args = parser.parse_args()

    dataset = AtHomeImageDataset()
    image, target = dataset[args.index]
    print(target)
    plt.imshow(rearrange(image, "c h w -> h w c"))
    plt.show()
