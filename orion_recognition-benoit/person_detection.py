#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from os.path import expanduser

path_to_tf_model = os.path.join('./object_detection')
sys.path.insert(0, path_to_tf_model)

from utils import label_map_util
from utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


class PersonDetection():
    def __init__(self, model_name, downloaded_model_path):
        self.threshold_detection = 0.4
        self.only_detect_person = True

        # # Model preparation 
        model_file = model_name + '.tar.gz'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        path_to_frozen_graph = model_name + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        path_to_labels = os.path.join(path_to_tf_model, 'data', 'mscoco_label_map.pbtxt')

        #extract downloaded model
        tar_file = tarfile.open(downloaded_model_path)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())

        #Load a (frozen) Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        self.sess = tf.Session(graph=self.detection_graph)
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        #Loading label map
        self.category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

        # Get handles to input and output tensors
        with self.detection_graph.as_default():
            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)


    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)


    def run_inference_for_single_image(self, image, graph, tensor_dict, image_tensor):

        # Run inference
        start = time.time()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        output_dict = self.sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})
        end = time.time()
        print (end-start)
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict


    def filter_predictions(self, input_dict):
        output_dict = {'detection_scores':[], 'detection_classes':[], 'detection_boxes':np.empty((0,4), int), 'num_detections':0}

        for i, score in enumerate(input_dict['detection_scores']):
            if score > self.threshold_detection:
                if self.only_detect_person and input_dict['detection_classes'][i] != 1:
                    continue
                output_dict['detection_scores'].append(score)
                output_dict['num_detections'] += 1
                output_dict['detection_boxes'] = np.vstack((output_dict['detection_boxes'], 
                                                        input_dict['detection_boxes'][i]))
                output_dict['detection_classes'].append(input_dict['detection_classes'][i])
        return output_dict


    def detect(self, np_image):
        # detect 
        with self.detection_graph.as_default():            
            # The following processing is only for single image
            detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])

            # Actual detection.
            inference_dict = self.run_inference_for_single_image(np_image, self.detection_graph, self.tensor_dict, self.image_tensor)
            output_dict = self.filter_predictions(inference_dict)
            return output_dict['detection_boxes']

     
    def detect_test(self):
        # # Detection
        PATH_TO_TEST_IMAGES_DIR = './object_detection/test_images'
        TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

        # Size, in inches, of the output images.
        IMAGE_SIZE = (24, 16)
        with self.detection_graph.as_default():
            #initialization
            # Get handles to input and output tensors
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                  tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                  detection_masks_reframed, 0)

            for image_path in TEST_IMAGE_PATHS:
                print(image_path)
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = self.load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                #image_np_expanded = np.expand_dims(image_np, axis=0)

                # Actual detection.
                inference_dict = self.run_inference_for_single_image(image_np, self.detection_graph, tensor_dict, image_tensor)
                output_dict = self.filter_predictions(inference_dict)
                print(output_dict['detection_boxes'])

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  output_dict['detection_boxes'],
                  output_dict['detection_classes'],
                  output_dict['detection_scores'],
                  self.category_index,
                  instance_masks=output_dict.get('detection_masks'),
                  use_normalized_coordinates=True,
                  line_thickness=8)
                plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(image_np)



#
#model_name = 'ssd_mobilenet_v1_coco_2018_01_28'
#downloaded_model_path = os.path.join('./', model_name + '.tar.gz')
#detector = PersonDetection(model_name, downloaded_model_path)
#detector.detect_test()


