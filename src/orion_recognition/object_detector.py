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
from object_classifer import ObjectClassifer
from PIL import Image


classifer=True

class ObjectDetector(torch.nn.Module):
    def __init__(self):
        super(ObjectDetector, self).__init__()
        print(torch.cuda.memory_allocated(0))
        self.device= torch.device( "cuda:0" if torch.cuda.is_available() else  "cpu")
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device).float()
        if classifer:
            self.classfier = ObjectClassifer()
            self.classfier.eval()
        self.coco_labels = []
        with open('coco_labels2017.txt', 'r') as in_file:
            self.coco_labels = in_file.readlines()
        with open('labels.txt', 'r') as in_file:
            self.imagenet_labels = in_file.readlines()
        
    def forward(self, img):
        img = np.concatenate(img)
        x = torch.as_tensor(img).to(self.device).float().unsqueeze(0)
        y = self.model(x)
        
        y = [{k: v.cpu().detach().numpy() for k,v in y[i].items()} for i in range(len(y))]
        if classifer:
            y_new = [{}]
            new_labels = []
            new_scores = []
            for box, label, score in zip(y[0]['boxes'], y[0]['labels'], y[0]['scores']):
                if label == 1:
                    new_labels.append(self.convert_label_index_to_string(label))
                    new_scores.append(score)
                else:
                    label, score = self.classfier(x[:, :, int(box[0]):int(box[2]), int(box[1]):int(box[3])])
                    new_labels.append(self.convert_label_index_to_string(label, coco=False))
                    new_scores.append(score)
            y_new[0]['boxes']=y[0]['boxes']
            y_new[0]['labels']=new_labels
            y_new[0]['scores']=new_scores
            return y_new
        else:
            return y 
        
    def convert_label_index_to_string(self, index, coco=True):
        if coco:
            return self.coco_labels[index-1]
        else:
            return self.imagenet_labels[index-1]
        

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
            image_tensor = transforms.ToTensor()(frame).to(self.device).float()
        else:
            rval = False

        while rval:
            detections = self.forward([image_tensor])[0]
            for detection, label in zip(detections['boxes'], detections['labels']):
                cv2.rectangle(frame, (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3])), (255, 0, 0), 3)
                cv2.putText(frame, str(label), (int(detection[0]), int(detection[1])), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            image_tensor = transforms.ToTensor()(frame).to(self.device).float()
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break

        vc.release()
        cv2.destroyWindow("preview")


if __name__ == '__main__':
    detector = ObjectDetector()
    print(detector.device)
    result = detector.detect_video()
    #detector.train_on_data()
    #result = detector.detect_random()
    #print(type(result[0]['boxes']))
    #print(result[0]['labels'])
    #print(result[0]['scores'])
