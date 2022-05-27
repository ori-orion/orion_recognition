#!/usr/bin/env python3
# coding: utf-8
from typing import List

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn


class ObjectDetector(torch.nn.Module):
    def __init__(self, labels: List[str]):
        super(ObjectDetector, self).__init__()
        # load a model pre-trained pre-trained on COCO
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(labels))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, images):
        outputs = self.model(images)
        outputs = [{k: v.cpu().detach().numpy() for k, v in output.items()} for output in outputs]
        return outputs

    def forward(self, images, targets):
        return self.model(images, targets)

    def detect_random(self):
        self.model.eval()
        x = [torch.rand(3, 300, 400).to(self.device).float(), torch.rand(3, 500, 400).to(self.device).float()]
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

        if vc.isOpened():  # try to get the first frame
            while True:
                rval, frame = vc.read()
                if not rval:
                    break
                image_tensor = transforms.ToTensor()(frame).to(self.device).float()
                detections = self.model([image_tensor])[0]
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
    print(detector.device)
    result = detector.detect_video()
    # detector.train_on_data()
    # result = detector.detect_random()
    # print(type(result[0]['boxes']))
    # print(result[0]['labels'])
    # print(result[0]['scores'])
