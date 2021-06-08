#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image



class ObjectDetector(torch.nn.Module):
    def __init__(self):
        super(ObjectDetector, self).__init__()

        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.device = torch.device( "cuda:0" if torch.cuda.is_available() else  "cpu")
        self.model.to(self.device).float()

    def forward(self, img):
        img = np.concatenate(img)
        x = torch.as_tensor(img).to(self.device).float().unsqueeze(0)
        y = self.model(x)
        
        y = [{k: v.cpu().detach().numpy() for k,v in y[i].items()} for i in range(len(y))]
        return y 



    def detect_random(self):
        self.model.eval()
        x = [torch.rand(3, 300, 400).to(self.device).float(), torch.rand( 3, 500, 400).to(self.device).float()]
        predictions = self.model(x) 
        return predictions


    def detect_test(self):
        self.model.eval()
        path2data = "./val2017"
        path2json = "./annotations/instances_val2017.json"
        coco_dataset = datasets.CocoDetection(root=path2data, annFile=path2json, transform=transforms.ToTensor())
        test = coco_dataset[3][0].to(self.device).float()
        output = self.model([test])
        return output


if __name__ == '__main__':
    detector = ObjectDetector()
    print(detector.device)
    #result = detector.detect_test()
    result = detector.detect_random()
    #print(result[0]['boxes'])
    #print(result[0]['labels'])
    #print(result[0]['scores'])
