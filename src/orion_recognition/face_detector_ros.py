#!/usr/bin/env python
# coding: utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from face_detector_new import FaceDetector
from skimage.transform import resize
import argparse
import tensorflow as tf
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import pdb
from keras import backend as K

class FaceDetector_ros(object):
    def __init__(self):
        MODEL_PATH = '/home/ori/code/recognition/face_age_gender/mix/model.pb'
        self.face_detector = FaceDetector(MODEL_PATH, gpu_memory_fraction=0.25, visible_device_list='0')
        #self.depth = 16
        #self.k = 8
        #self.weight_file = '/home/ori/code/recognition/face_age_gender/mix/pretrained_models/weights.28-3.73.hdf5'
        #self.margin = 0.4
        #self.img_size = 64
        #magic_face=np.zeros([64,64,3])
        #dd=self.face_detector.analyse(magic_face)
        #pdb.set_trace()
        #self.graph = tf.Graph()
        #self.config = tf.ConfigProto()
        #self.config.gpu_options.allow_growth = True
        #self.model = WideResNet(self.img_size, depth=self.depth, k=self.k)()
        #self.model.load_weights(self.weight_file)
        #self.sess = tf.Session(graph=self.graph, config=self.config)
       
    def detect(self, image_np):
        output_dict={}
        boxes, scores = self.face_detector(image_np, score_threshold=0.3)
        #faces = np.empty((len(boxes), self.img_size, self.img_size, 3))
        if len(boxes) > 0:
            for i, b in enumerate(boxes):
                ymin, xmin, ymax, xmax = b
                #faces[i, :, :, :] = resize(image_np[int(ymin):int(ymax) + 1, int(xmin):int(xmax) + 1, :], (self.img_size, self.img_size), anti_aliasing=True)
                #face = resize(image_np[int(ymin):int(ymax) + 1, int(xmin):int(xmax) + 1, :], (self.img_size, self.img_size), anti_aliasing=True)
                face = image_np[int(ymin):int(ymax) + 1, int(xmin):int(xmax) + 1, :]
                results = self.face_detector.analyse(face*255.0)
                predicted_genders = results[0]
                predicted_age_group = results[1]
                predicted_age_indx = results[2]
                people_id = i
                face_bbox = b
                score = scores[i]
                output_dict[str(people_id)] = {'face_bbox':face_bbox, 'score':score, 'age_group':predicted_age_group, 'age_indx':int(predicted_age_indx), 'gender':predicted_genders}
            #pdb.set_trace()
            #results = self.model.predict(faces)
            #results = self.face_detector.analyse(faces*255.0)
            #predicted_genders = results[0][0]
            #predicted_age_group = results[0][1]
            #predicted_age_indx = results[0][2]     
            #ages = np.arange(0, 101).reshape(101, 1)
            #predicted_ages = results[1].dot(ages).flatten()
        #for j in range(len(boxes)):
            #people_id = j
            #face_bbox = boxes[j]
            #score = scores[j]
            #age = int(predicted_ages[j])
            #gender = "M" if predicted_genders[j][0] < 0.5 else "F"
            #output_dict[str(people_id)] = {'face_bbox':face_bbox, 'score':score, 'age_group':age, 'gender':gender}
        return output_dict
     

