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

from einops import rearrange

from orion_recognition.bbox_utils import non_max_supp
from orion_recognition.object_classifer import ObjectClassifer
from PIL import Image

classifer = True
buffer = 20

PERSON_LABEL = 1

min_acceptable_score = 0.0


class ObjectDetector(torch.nn.Module):
    def __init__(self):
        super(ObjectDetector, self).__init__()
        print("Is cuda available? {}".format(torch.cuda.is_available()))
        print("cuda memory allocated: {}".format(torch.cuda.memory_allocated()))  # does not cause seg fault
        # print(torch.cuda.memory_allocated(0)) # causes seg fault
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

        self.model.to(self.device)
        if classifer:
            self.classfier = ObjectClassifer(self.device)
            self.classfier.eval()

        labels_path = os.path.dirname(os.path.abspath(__file__))

        self.coco_labels = []
        with open(os.path.join(labels_path, 'coco_labels2017.txt'), 'r') as in_file:
            self.coco_labels = in_file.read().strip().split("\n")
        with open(os.path.join(labels_path, 'labels_short.txt'), 'r') as in_file:
            self.imagenet_labels = in_file.read().strip().split("\n")

        print(self.coco_labels)

        self.all_labels = self.coco_labels + self.imagenet_labels
        self.label_map = {}
        for i in range(len(self.all_labels)):
            self.label_map[self.all_labels[i]] = i + 1

    def forward(self, img):
        assert len(img.size()) == 3, "Assumes a single image input"
        C, H, W = img.size()

        img = img.cpu().numpy()
        x = torch.as_tensor(img, device=self.device, dtype=torch.float)

        results = self.model("tmp.jpg")
        img = rearrange(img, "c h w -> h w c")
        Image.fromarray(np.uint8(img*255)).save("tmp.jpg")

        bbox_results = {}
        labels = []
        scores = []
        bboxes = []
        # y = {k: v.cpu().detach().numpy() for k, v in results.items()}
        # for box, label, score in zip(y['boxes'], y['labels'], y['scores']):
        #     w_min, h_min, w_max, h_max = box
        for i, result in results.pandas().xyxy[0].iterrows():
            w_min, h_min, w_max, h_max, score, class_index, label = result
            box = w_min, h_min, w_max, h_max
            if score < min_acceptable_score:
                continue

            min_dim_size = 25
            if (h_max - h_min < min_dim_size) or (w_max - w_min < min_dim_size):
                # dont take box that is too small
                continue
            labels.append(label)
            scores.append(score)
            bboxes.append(box)
            if label != "person" and self.classfier:
                new_label, new_score = self.classfier(
                    x[:, max(0, int(h_min) - buffer):min(int(h_max) + buffer, H),
                    max(0, int(w_min - buffer)):min(int(w_max) + buffer, W)])
                if new_score < min_acceptable_score:
                    continue
                labels.append(label)
                scores.append(new_score)
                bboxes.append(box)
        bbox_results['boxes'] = bboxes
        bbox_results['labels'] = labels
        bbox_results['scores'] = scores

        print(f"Detected objects (COCO{' + RoboCup' if self.classfier else ''}): {bbox_results['labels']}")
        return bbox_results

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

            bbox_tuples = []

            detections = self(image_tensor)
            for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
                bbox_tuples.append((box, label, score, None))

            clean_bbox_tuples = non_max_supp(bbox_tuples)
            for ((x_min, y_min, x_max, y_max), label, score, detection) in clean_bbox_tuples:

                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                              (255, 0, 0), 3)
                cv2.putText(frame, str(label), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
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
