#!/usr/bin/env python

import object_detector

import cv2
import numpy as np
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

    
class BboxPublisher(object):
    def __init__(self, image_topic, model_filename, path_to_tf_model, detect_object, detect_person,
                 threshold_detection):
        self.detector = object_detector.ObjectDetector(model_filename, path_to_tf_model, detect_object, detect_person,
                                                       threshold_detection)
        rospy.init_node('listener', anonymous=True)
        self.subscriber = rospy.Subscriber(image_topic, Image, self.callback, queue_size=100)
        self.pub = rospy.Publisher('/image_listener/image', Image, queue_size=100)
        self.bridge = CvBridge()

    def callback(self, ros_image):
        image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        np_image = np.asarray(image)
        height = ros_image.height
        width = ros_image.width
        detection_boxes = self.detector.detect(np_image)
        for row in detection_boxes:
            print 'Detection'
            top_left = (int(row[1] * width), int(row[0] * height))
            bottom_right = (int(row[3] * width), int(row[2] * height))
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 3)

        try:
            self.pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
        except CvBridgeError as e:
            print(e)

 
if __name__ == '__main__':
    img_topic = "/hsrb/head_rgbd_sensor/rgb/image_rect_color"
    model_name = 'ssd_mobilenet_v1_coco_2018_01_28'
    path_to_model = os.path.join('/home/chiaman/git/models/research/object_detection')
    detect_pers = True
    detect_obj = True
    thresh = 0.4
    sub = BboxPublisher(img_topic, model_name, path_to_model, detect_obj, detect_pers, thresh)
    rospy.spin() 
