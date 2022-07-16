#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import os
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2

from einops import rearrange

from orion_recognition.bbox_utils import non_max_supp
from orion_recognition.object_classifer import ObjectClassifer
from PIL import Image

from orion_recognition.utils import data_path

use_classifier = True
buffer = 20

PERSON_LABEL = 1

min_acceptable_score = 0.0

tmp_image_dir = "tmp.jpg"

torch.hub.set_dir(data_path)


class ObjectDetector(torch.nn.Module):
    def __init__(self, algorithm="yolo"):
        """
        :param algorithm: 'yolo' or 'rcnn' ('yolo' recommended)
        """
        super(ObjectDetector, self).__init__()
        print("Is cuda available? {}".format(torch.cuda.is_available()))
        print("cuda memory allocated: {}".format(torch.cuda.memory_allocated()))  # does not cause seg fault
        # print(torch.cuda.memory_allocated(0)) # causes seg fault
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if algorithm == "yolo":
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l', force_reload=True,
                                            pretrained=True)  # needs internet
            except:
                self.model = torch.hub.load(os.path.join(data_path, "ultralytics_yolov5_master"),
                                            'custom', source="local",
                                            path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov5l.pt"))
        elif algorithm == "rcnn":
            self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            raise NotImplementedError

        self.model.to(self.device)
        if use_classifier:
            self.classfier = ObjectClassifer(self.device)
            self.classfier.eval()

        labels_path = os.path.dirname(os.path.abspath(__file__))

        self.coco_labels = []
        with open(os.path.join(labels_path, 'coco_labels2017.txt'), 'r') as in_file:
            self.coco_labels = in_file.read().strip().split("\n")
        with open(os.path.join(labels_path, 'labels_short.txt'), 'r') as in_file:
            self.robocup_labels = in_file.read().strip().split("\n")

        print(self.coco_labels)

        self.all_labels = self.coco_labels + self.robocup_labels
        self.label_map = {}
        for i in range(len(self.all_labels)):
            self.label_map[self.all_labels[i]] = i + 1

        self.algorithm = algorithm

    def forward(self, img):
        """
        :param img: Requires RGB image (Open-CV and ROS images are BGR by default so be careful!)
        :return: bbox results (dict)
        """
        assert len(img.size()) == 3, "Assumes a single image input"

        C, H, W = img.size()
        img = torch.as_tensor(img, device=self.device, dtype=torch.float)

        if self.algorithm == "yolo":
            Image.fromarray(np.uint8(rearrange(img.cpu().numpy(), "c h w -> h w c") * 255)).save(tmp_image_dir)
            results = self.model(tmp_image_dir)
            bbox_iterator = results.pandas().xyxy[0].iterrows()
        elif self.algorithm == "rcnn":
            results = self.model(img.unsqueeze(0))[0]
            results = {k: v.cpu().detach().numpy() for k, v in results.items()}
            bbox_iterator = enumerate(
                (*box, score, label, self.convert_label_index_to_string(label - 1, dataset="coco"))
                for box, label, score in zip(results['boxes'], results['labels'], results['scores']))
        else:
            raise NotImplementedError

        bbox_results = {
            'boxes': [],
            'scores': [],
            'labels': []
        }
        # y = {k: v.cpu().detach().numpy() for k, v in results.items()}
        # for box, label, score in zip(y['boxes'], y['labels'], y['scores']):
        #     w_min, h_min, w_max, h_max = box
        for i, result in bbox_iterator:
            w_min, h_min, w_max, h_max, score, class_index, label = result
            box = w_min, h_min, w_max, h_max
            if score < min_acceptable_score:
                continue

            min_dim_size = 25
            if (h_max - h_min < min_dim_size) or (w_max - w_min < min_dim_size):
                # dont take box that is too small
                continue
            bbox_results['labels'].append(label)
            bbox_results['scores'].append(score)
            bbox_results['boxes'].append(box)

            if label != "person" and self.classfier:
                new_class_index, new_score = self.classfier(
                    img[:, max(0, int(h_min) - buffer):min(int(h_max) + buffer, H),
                    max(0, int(w_min - buffer)):min(int(w_max) + buffer, W)])
                if new_score < min_acceptable_score:
                    continue
                bbox_results['labels'].append(self.convert_label_index_to_string(new_class_index, dataset="robocup"))
                bbox_results['scores'].append(new_score)
                bbox_results['boxes'].append(box)

        print(f"Detected objects (COCO{' + RoboCup' if self.classfier else ''}): {bbox_results['labels']}")
        return bbox_results

    def convert_label_index_to_string(self, index, dataset="coco"):
        if dataset == "coco":
            return self.coco_labels[index]
        elif dataset == "robocup":
            return self.robocup_labels[index].split("/")[-1]
        else:
            raise NotImplementedError

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
