#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import os
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2
import rospkg
from ultralytics import YOLO
from einops import rearrange

from orion_recognition.bbox_utils import non_max_supp
from orion_recognition.object_classifer import ObjectClassifer
from PIL import Image

from orion_recognition.utils import data_path

# try tensorrt
from orion_recognition.models import TRTModule
from orion_recognition.models.torch_utils import det_postprocess
from orion_recognition.models.utils import blob, letterbox, path_to_list

use_classifier = False
buffer = 20

PERSON_LABEL = 1

min_acceptable_score = 0.0

tmp_image_dir = "tmp.jpg"

torch.hub.set_dir(data_path)


class ObjectDetector(torch.nn.Module):
    def __init__(self, algorithm="yolov5"):
        """
        :param algorithm: 'yolo' or 'yolotrt'
        """
        if algorithm == 'yolo':
            suffix = '.pt'
        elif algorithm == 'yolotrt':
            suffix = '.engine'
        elif algorithm == 'yolov5':
            suffix = '.pt'
        else:
            NotImplementedError
        # get yolo weights path
        rospack = rospkg.RosPack()
        self.yolo_weights_path = rospack.get_path('orion_recognition') + '/src/orion_recognition/weights/yolov5ycb' + suffix
        print(self.yolo_weights_path)
        super(ObjectDetector, self).__init__()
        print("Is cuda available? {}".format(torch.cuda.is_available()))
        print("cuda memory allocated: {}".format(torch.cuda.memory_allocated()))  # does not cause seg fault
        # print(torch.cuda.memory_allocated(0)) # causes seg fault
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if algorithm == "yolo":
            try:
                self.model = YOLO(self.yolo_weights_path) # load saved copy of pretrained weights
                self.model.to(self.device)
            except Exception as e:
                print(e)
                print("Cannot find yolo weights in weights folder, Downloading YOLO weights...")
                self.model = YOLO('yolov8l.pt') # needs internet
                self.model.to(self.device)
        elif algorithm == "yolotrt":
            
            Engine = TRTModule(self.yolo_weights_path, self.device)
            self.E_H, self.E_W = Engine.inp_info[0].shape[-2:]
            Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
            self.model = Engine
        elif algorithm == "yolov5":
            try: 
                self.model = torch.hub.load(os.path.join(data_path, "ultralytics_yolov5_master"),
                    'custom', source="local",
                    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov5ycb.pt"))
                # self.model = torch.hub.load('~/yolov5', 'custom', source = "local", path = self.yolo_weights_path)
                self.model.to(self.device)
            except Exception as e:
                print(e)
                print("Cannot find yolov5 weights in weights folder, default to yolov8")
                self.model = YOLO('yolov8l.pt') # needs internet
                self.model.to(self.device)
        else:
            raise NotImplementedError

        if use_classifier:
            self.classfier = ObjectClassifer(self.device)
            self.classfier.eval()
        else:
            self.classfier = None

        labels_path = os.path.dirname(os.path.abspath(__file__))

        self.coco_labels = []
        self.ycb_labels = []
        with open(os.path.join(labels_path, 'coco_labels2017.txt'), 'r') as in_file:
            self.coco_labels = in_file.read().strip().split("\n")
        with open(os.path.join(labels_path, 'labels_short.txt'), 'r') as in_file:
            self.robocup_labels = in_file.read().strip().split("\n")
        with open(os.path.join(labels_path, 'label_ycb.txt'), 'r') as in_file:
            self.ycb_labels = in_file.read().strip().split("\n")

        print(self.coco_labels)
        print(self.ycb_labels)

        self.all_labels = self.coco_labels + self.robocup_labels
        self.label_map = {}
        for i in range(len(self.all_labels)):
            self.label_map[self.all_labels[i]] = i + 1

        self.algorithm = algorithm

    def forward(self, img):
        """
        :param img: Requires RGB image (Open-CV and ROS images are BGR by default so be careful!)
        :input is an img of shape (C, H, W)
        :return: bbox results (dict)
        """
        bbox_results = {
            'boxes': [], # in xyxy format
            'scores': [], # confidence score
            'labels': [] # string class
        }
        
        # assert len(img.size()) == 3, "Assumes a single image input"
        assert len(img.shape) == 3, "Assumes a single image input"
        H, W, C = img.shape
        # img = torch.as_tensor(img, device=self.device, dtype=torch.float)

        if self.algorithm == "yolotrt":
            bgr, ratio, dwdh = letterbox(img, (self.E_W, self.E_H))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = blob(rgb, return_seg=False)
            dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=self.device)
            tensor = torch.asarray(tensor, device=self.device)
            # inference
            data = self.model(tensor)
            bboxes, scores, labels = det_postprocess(data)
            bboxes -= dwdh
            bboxes /= ratio
            for (bbox, score, label) in zip(bboxes, scores, labels):
                bbox = bbox.tolist()
                w_min, h_min, w_max, h_max = bbox
                label = self.convert_label_index_to_string(int(label), dataset="coco")
                score = np.float32(score.item())
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
                    # convert image to tensor to be used for classifier
                    img_tensor = transforms.ToTensor()(img)
                    img_tensor = torch.as_tensor(img_tensor, device=self.device, dtype=torch.float)
                    new_class_index, new_score = self.classfier(
                        img_tensor[:, max(0, int(h_min) - buffer):min(int(h_max) + buffer, H),
                        max(0, int(w_min - buffer)):min(int(w_max) + buffer, W)])
                    if new_score < min_acceptable_score:
                        continue
                    bbox_results['labels'].append(self.convert_label_index_to_string(new_class_index, dataset="robocup"))
                    bbox_results['scores'].append(new_score)
                    bbox_results['boxes'].append(box)
        elif self.algorithm == "yolo":
            results = self.model(img) # add in batch dimension
            bbox_iterator = results[0].cpu().boxes
            for result in bbox_iterator:
                w_min, h_min, w_max, h_max = result.xyxy.numpy()[0]
                score = result.conf.numpy()[0]
                label = result.cls.numpy()[0]
                label = self.convert_label_index_to_string(int(label), dataset="coco")
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
                    # convert image to tensor to be used for classifier
                    img_tensor = transforms.ToTensor()(img)
                    img_tensor = torch.as_tensor(img_tensor, device=self.device, dtype=torch.float)
                    new_class_index, new_score = self.classfier(
                        img_tensor[:, max(0, int(h_min) - buffer):min(int(h_max) + buffer, H),
                        max(0, int(w_min - buffer)):min(int(w_max) + buffer, W)])
                    if new_score < min_acceptable_score:
                        continue
                    bbox_results['labels'].append(self.convert_label_index_to_string(new_class_index, dataset="robocup"))
                    bbox_results['scores'].append(new_score)
                    bbox_results['boxes'].append(box)    
                    
            # Image.fromarray(np.uint8(rearrange(img.cpu().numpy(), "c h w -> h w c") * 255)).save(tmp_image_dir)
            # results = self.model(tmp_image_dir) # add in batch dimension
            # bbox_iterator = results[0].cpu().boxes
        elif self.algorithm == "yolov5":
            image_tensor = transforms.ToTensor()(img)
            image_tensor = torch.as_tensor(image_tensor, device=self.device, dtype=torch.float)
            Image.fromarray(np.uint8(rearrange(image_tensor.cpu().numpy(), "c h w -> h w c") * 255)).save(tmp_image_dir)
            results = self.model(tmp_image_dir)
            bbox_iterator = results.pandas().xyxy[0].iterrows()
            # print(results.pandas().xyxy[0])
            for i, result in bbox_iterator:
                w_min, h_min, w_max, h_max, score, class_index, label = result
                label = self.convert_label_index_to_string(int(class_index), dataset="ycb")
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
            
        else:
            raise NotImplementedError
        print(f"Detected objects (YCB{' + RoboCup' if self.classfier else ''}): {bbox_results['labels']}")
        return bbox_results

    def convert_label_index_to_string(self, index, dataset="coco"):
        if dataset == "coco":
            return self.coco_labels[index]
        elif dataset == "robocup":
            return self.robocup_labels[index].split("/")[-1]
        elif dataset == "ycb":
            return self.ycb_labels[index]
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
