#!/usr/bin/env python
# coding: utf-8

import os
import torch
from orion_recognition.object_detection.utils import load_finetuned_resnet


class ObjectClassifer(torch.nn.Module):
    def __init__(self, device: torch.device):
        super(ObjectClassifer, self).__init__()
        self.device = device
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resnet_epoch_999.pth")
        self.model = load_finetuned_resnet(model_path, 100, eval=True)
        self.model.to(self.device).float()

    def forward(self, img_tensor):
        assert len(img_tensor.size()) == 3, "Assumes a single image"
        # img = transforms.functional.resize(img, size=[224])
        # print("Image shape: {}".format(img.shape))
        output = self.model(img_tensor.unsqueeze(0))
        probabilities = torch.softmax(output[0], dim=0)
        return torch.argmax(probabilities).item(), torch.max(probabilities).item()
