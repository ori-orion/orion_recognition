#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2

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

    def train_on_data(self):
        images = []
        targets = []
        labels_file = open('labels.txt', 'r')
        labels = labels_file.read().split("\n")
        for i in range(len(labels)):
            image_dir = "./"+labels[i]+"/"
            for image_location in os.listdir(image_dir):
                image = Image.open(image_dir+image_location)
                image_tensor = transforms.ToTensor()(image)
                images.append(image_tensor)
                size = image_tensor.size()
                d  = {}
                d['boxes'] = [[0, 0, size[1], size[2]]]
                d['labels'] = [i+91]
                targets.append(d)
        self.model(images, targets)
        torch.save(self.model.state_dict(), "model_weights.pth")
        # model.load_state_dict(torch.load("model_weights.pth"))



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

    def detect_video(self):
        self.model.eval()
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)

        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
            image_tensor = transforms.ToTensor()(frame)
        else:
            rval = False

        while rval:
            detections = self.model([image_tensor])[0]
            for detection, label in zip(detections['boxes'], detections['labels']):
                cv2.rectangle(frame, (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3])), (255, 0, 0), 3)
                cv2.putText(frame, str(label), (int(detection[0]), int(detection[1])), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            image_tensor = transforms.ToTensor()(frame)
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break

        vc.release()
        cv2.destroyWindow("preview")


if __name__ == '__main__':
    detector = ObjectDetector()
    print(detector.device)
    #result = detector.detect_video()
    detector.train_on_data()
    #result = detector.detect_random()
    #print(result[0]['boxes'])
    #print(result[0]['labels'])
    #print(result[0]['scores'])
