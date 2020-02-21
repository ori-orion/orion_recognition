#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image

#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util


class ObjectDetector(torch.nn.Module):
    def __init__(self):
        super(ObjectDetector, self).__init__()

        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.device = torch.device( "cuda:0" if torch.cuda.is_available() else  "cpu")
        self.model.to(self.device).float()

    def forward(self, img):
        x.to(self.device).float()
        x = torch.as_tensor(img, trainable=False)
        y = self.model(x)
        
        return y.cpu().numpy()



    def detect_random(self):
        """
        Doesn't work due to dependencies - hope it works on a ubuntu machine
        """
        self.model.eval()
        x = [torch.rand( 3, 300, 400).to(self.device).float(), torch.rand( 3, 500, 400).to(self.device).float()]
        predictions = self.model(x) 
        print(predicitons)


#    def detect_test(self):
#        self.model.eval()
#        path2data = "./val2017"
#        path2json = "./annotations/instances_val2017.json"
#        coco_dataset = datasets.CocoDetection(root=path2data, annFile=path2json, transform=transforms.ToTensor())
#        print(coco_dataset[3][0].size())
#        print(torch.version.cuda)
        #print(coco_dataset[3][1])
#        test = coco_dataset[3][0].reshape(1, 3, 500, 375).to(self.device).float(#)
#        print(test.size())
        #output = self.model(test)
        #print(output)


if __name__ == '__main__':
    detector = ObjectDetector()
    print(detector.device)
    detector.detect_random()
