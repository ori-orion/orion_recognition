#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import os, psutil, sys, time
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2
import gc

from orion_recognition.bbox_utils import non_max_supp
from orion_recognition.object_classifer import ObjectClassifer
from PIL import Image
import rospkg

classifer = True
buffer = 20

PERSON_LABEL = 1

min_acceptable_score = 0.6


class ObjectDetector(torch.nn.Module):
    def __init__(self):
        super(ObjectDetector, self).__init__()
        print("Is cuda available? {}".format(torch.cuda.is_available()))
        print("cuda memory allocated: {}".format(torch.cuda.memory_allocated()))  # does not cause seg fault
        # print(torch.cuda.memory_allocated(0)) # causes seg fault
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        if classifer:
            self.classfier = ObjectClassifer()
            self.classfier.eval()

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('orion_recognition')
        labels_path = pkg_path + "/src/orion_recognition"

        self.coco_labels = []
        with open(labels_path + '/coco_labels2017.txt', 'r') as in_file:
            self.coco_labels = in_file.read().strip().split("\n")
        with open(labels_path + '/labels_short.txt', 'r') as in_file:
            self.imagenet_labels = in_file.read().strip().split("\n")

        print(self.coco_labels)

        self.all_labels = self.coco_labels + self.imagenet_labels
        self.label_map = {}
        for i in range(len(self.all_labels)):
            self.label_map[self.all_labels[i]] = i + 1

    def forward(self, img):
        assert len(img.size()) == 3, "Assumes a single image input"
        C, H, W = img.size()

        x = torch.as_tensor(img, device=self.device, dtype=torch.float)
        # print("forward started with img size: {}".format(x.shape))
        y = self.model(x.unsqueeze(0))[0]

        y = {k: v.cpu().detach().numpy() for k, v in y.items()}
        # print("Detected COCO objects: {}".format([self.convert_label_index_to_string(l,coco=True) for l in y[0]['labels']]))

        if classifer:
            y_new = {}
            new_labels = []
            new_scores = []
            new_boxes = []
            for box, label, score in zip(y['boxes'], y['labels'], y['scores']):
                if score < min_acceptable_score:
                    continue
                w_min, h_min, w_max, h_max = box
                # print("box corners: {}, x-size: {}, y-size: {}, label: {}".format(box, box[2]-box[0], box[3]-box[1], label))
                min_dim_size = 25
                if (h_max - h_min < min_dim_size) or (w_max - w_min < min_dim_size):
                    # dont take box that is too small
                    continue
                new_labels.append(self.convert_label_index_to_string(label))
                new_scores.append(score)
                new_boxes.append(box)
                if label != PERSON_LABEL:
                    new_label, new_score = self.classfier(
                        x[:, max(0, int(h_min) - buffer):min(int(h_max) + buffer, H),
                        max(0, int(w_min - buffer)):min(int(w_max) + buffer, W)])
                    if new_score < min_acceptable_score:
                        continue
                    new_labels.append(self.convert_label_index_to_string(new_label, coco=False))
                    new_scores.append(new_score)
                    new_boxes.append(box)
            y_new['boxes'] = new_boxes
            y_new['labels'] = new_labels
            y_new['scores'] = new_scores
            print("Detected objects (COCO + RoboCup): {}\n".format(y_new['labels']))
            return y_new
        else:
            print("Detected objects (COCO only): {}\n".format(y['labels']))
            return y

    def convert_label_index_to_string(self, index, coco=True):
        if coco:
            return self.coco_labels[index - 1]
        else:
            return self.imagenet_labels[index - 1]

    def detect_random(self):
        self.model.eval()
        x = torch.rand(3, 300, 400).to(self.device).float()
        predictions = self.model(x)
        return predictions

    def detect_test(self):
        self.model.eval()
        path2data = "./val2017"
        path2json = "./annotations/instances_val2017.json"
        coco_dataset = datasets.CocoDetection(root=path2data, annFile=path2json, transform=transforms.ToTensor())
        test = coco_dataset[3][0].to(self.device).float()
        output = self.model(test)
        return output

    def detect_video(self):
        self.model.eval()
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)

        while vc.isOpened():  # try to get the first frame
            rval, frame = vc.read()
            if not rval:
                break
            image_tensor = transforms.ToTensor()(frame)
            detections = self(image_tensor)
            for detection, label in zip(detections['boxes'], detections['labels']):
                cv2.rectangle(frame, (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3])),
                              (255, 0, 0), 3)
                cv2.putText(frame, str(label), (int(detection[0]), int(detection[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (0, 255, 0), 1)
            cv2.imshow("preview", frame)
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break

        vc.release()
        cv2.destroyWindow("preview")


if __name__ == '__main__':
    detector = ObjectDetector()
    # print(detector.device)
    result = detector.detect_video()
    # detector.train_on_data()
    # result = detector.detect_random()
    # print(type(result[0]['boxes']))
    # print(result[0]['labels'])
    # print(result[0]['scores'])
