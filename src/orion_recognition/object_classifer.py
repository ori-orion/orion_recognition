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

class ObjectClassifer(torch.nn.Module):
    def __init__(self):
        super(ObjectClassifer, self).__init__()
        self.device= torch.device( "cuda:0" if torch.cuda.is_available() else  "cpu")
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model.to(self.device).float()
        
    def forward(self, img):
        output = self.model(img)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return torch.argmax(probabilities), torch.max(probabilities)
