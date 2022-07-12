#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os, psutil, sys, time
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2
import gc
from PIL import Image
from orion_recognition.utils import load_finetuned_resnet
import rospkg


class ObjectClassifer(torch.nn.Module):
    def __init__(self):
        super(ObjectClassifer, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('orion_recognition')
        model_path = pkg_path + "/src/orion_recognition/resnet_epoch_999.pth"
        self.model = load_finetuned_resnet(model_path, 100, eval=True)
        self.model.to(self.device).float()

    def forward(self, img):
        assert len(img.size()) == 3, "Assumes a single image"
        # img = transforms.functional.resize(img, size=[224])
        # print("Image shape: {}".format(img.shape))
        output = self.model(img.unsqueeze(0))
        probabilities = torch.softmax(output[0], dim=0)
        return torch.argmax(probabilities).item(), torch.max(probabilities).item()
