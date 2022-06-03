import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

import torch

from orion_recognition.object_detection.utils import get_transforms

sys.path.append(".")

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="path to checkpoint")

args = parser.parse_args()

module = PLCluster.load_from_checkpoint(args.checkpoint)

with torch.no_grad():
    pred_full = module(pov_full)

pred_full = pred_full.argmax(dim=1)
n_clusters = pred_full.max() + 1

image = vis_clusters(pov_full, pred_full, n_clusters, interval=1)

plt.imshow(image)
plt.show()

im = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
im.save("your_file.png")
