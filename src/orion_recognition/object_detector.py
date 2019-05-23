#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import tensorflow as tf
import time

from matplotlib import pyplot as plt

from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class ObjectDetector(object):
    def __init__(self, model_filename, path_to_model, detect_object=True, detect_person=True, threshold_detection=0.4):
        self.detect_person = detect_person
        self.detect_object = detect_object
        self.threshold_detection = threshold_detection

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        path_to_frozen_graph = os.path.join(path_to_model, model_filename, 'frozen_inference_graph.pb')

        # List of the strings that is used to add correct label for each box.
        path_to_labels = os.path.join(path_to_model, 'data', 'mscoco_label_map.pbtxt')

        # Load a (frozen) Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        self.sess = tf.Session(graph=self.detection_graph)
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Load label map
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
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    def run_inference_for_single_image(self, image_np):
        # Run inference
        start = time.time()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: np.expand_dims(image_np, 0)})
        end = time.time()
        print (end - start)
        # All outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def filter_predictions(self, input_dict):
        output_dict = {'detection_scores': [], 'detection_classes': [], 'detection_boxes': np.empty((0, 4), int),
                       'num_detections': 0}

        for i, score in enumerate(input_dict['detection_scores']):
            if score > self.threshold_detection:
                if not self.detect_person and input_dict['detection_classes'][i] == 1:
                    continue
                if not self.detect_object and input_dict['detection_classes'][i] != 1:
                    continue
                output_dict['detection_scores'].append(score)
                output_dict['num_detections'] += 1
                output_dict['detection_boxes'] = np.vstack(
                    (output_dict['detection_boxes'], input_dict['detection_boxes'][i]))
                output_dict['detection_classes'].append(input_dict['detection_classes'][i])
        return output_dict

    def add_detection_classes_string(self, output_dict):
        size = len(output_dict['detection_classes'])
        output_dict['detection_classes_string'] = ["" for x in range(size)]
        for i in range(size):
            output_dict['detection_classes_string'][i] = self.category_index[output_dict['detection_classes'][i]][
                'name']
        return output_dict

    def detect(self, image_np):
        '''
        Detect bounding boxes in an image.
        :param image_np: image as numpy array.
        :return: bounding_boxes['detection_boxes']: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
                 The coordinates are in normalized format between [0, 1].
                 category index example: {1: {'id': 1, 'name': u'person'}, 2: {'id': 2, 'name': u'bicycle'}}
        '''
        inference_dict = self.run_inference_for_single_image(image_np)
        output_dict = self.filter_predictions(inference_dict)
        output_dict = self.add_detection_classes_string(output_dict)
        return output_dict
     
    def detect_test(self):
        # # Detection
        # PATH_TO_TEST_IMAGES_DIR = './object_detection/test_images'
        PATH_TO_TEST_IMAGES_DIR = '/home/chiaman/git/models/research/object_detection/test_images'
        TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

        # Size, in inches, of the output images.
        IMAGE_SIZE = (24, 16)
        with self.detection_graph.as_default():
            for image_path in TEST_IMAGE_PATHS:
                print(image_path)
                image = Image.open(image_path)
                # The array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)

                # Actual detection
                output_dict = self.detect(image_np)
                print(output_dict['detection_boxes'])
                print(output_dict['detection_classes'])
                print(output_dict['detection_scores'])

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
                plt.show(block=True)  # for pycharm


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


if __name__ == '__main__':
    model_name = 'ssd_mobilenet_v1_coco_2018_01_28'
    path_to_tf_model = os.path.join('/home/chiaman/git/models/research/object_detection')
    detect_obj = True
    detect_pers = True
    detector = ObjectDetector(model_name, path_to_tf_model, detect_obj, detect_pers)
    detector.detect_test()
