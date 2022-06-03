import argparse

import sys
import os

sys.path.append(".")
sys.path.append("./src")

from orion_recognition.object_detection.utils import load_finetuned_resnet

dir_path = os.path.join(os.path.abspath(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="path to checkpoint", default=os.path.join(dir_path, "finetuned", "resnet_epoch_3.pth"))
parser.add_argument("--n_classes", help="number of classes", type=int, default=180)

args = parser.parse_args()

model = load_finetuned_resnet(args.checkpoint, args.n_classes)