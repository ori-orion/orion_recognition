#!/usr/bin/env python

import object_detector
from orion_actions.msg import Detection, DetectionArray, Label

import cv2
import numpy as np
import os
import rospy
import std_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

    
class BboxPublisher(object):
    def __init__(self, image_topic, model_filename, path_to_tf_model, detect_object, detect_person,
                 threshold_detection):
        self.detector = object_detector.ObjectDetector(model_filename, path_to_tf_model, detect_object, detect_person,
                                                       threshold_detection)
        self.subscriber = rospy.Subscriber(image_topic, Image, self.callback, queue_size=100)
        self.bbox_pub = rospy.Publisher('/vision/bbox_image', Image, queue_size=100)
        self.detections_pub = rospy.Publisher('/vision/bbox_detections', DetectionArray, queue_size=100)
        self.bridge = CvBridge()

    def callback(self, ros_image):
        image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        np_image = np.asarray(image)
        height = ros_image.height
        width = ros_image.width
        output_dict = self.detector.detect(np_image)
        detection_boxes = output_dict['detection_boxes']
        detection_classes_string = output_dict['detection_classes_string']
        detection_scores = output_dict['detection_scores']

        detections = []
        for i, row in enumerate(detection_boxes):
            print 'Detection'
            top_left = (int(row[1] * width), int(row[0] * height))
            bottom_right = (int(row[3] * width), int(row[2] * height))
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 3)

            center_x = (top_left[0] + bottom_right[0]) / 2
            center_y = (top_left[1] + bottom_right[1]) / 2
            label = Label(detection_classes_string[i], detection_scores[i])
            detection = Detection(label, center_x, center_y, width, height)
            detections.append(detection)
        try:
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            self.detections_pub.publish(DetectionArray(header, detections))
            self.bbox_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
        except CvBridgeError as e:
            print(e)

 
if __name__ == '__main__':
    rospy.init_node('bbox_publisher')
    img_topic = "/hsrb/head_rgbd_sensor/rgb/image_rect_color"
    model_name = 'ssd_mobilenet_v1_coco_2018_01_28'
    path_to_model = os.path.join('/home/chiaman/git/models/research/object_detection')
    detect_pers = True
    detect_obj = True
    thresh = 0.4
    sub = BboxPublisher(img_topic, model_name, path_to_model, detect_obj, detect_pers, thresh)
    rospy.spin() 
