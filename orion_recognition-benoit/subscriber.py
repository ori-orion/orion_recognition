#!/usr/bin/env python

import person_detection

import cv2
import numpy as np

import os

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


 
    
class imageListener():
    def __init__(self, model_name, downloaded_model_path, image_topic):
        self.detector = person_detection.PersonDetection(model_name, downloaded_model_path)

        rospy.init_node('listener', anonymous=True)
        self.subscriber = rospy.Subscriber(image_topic, Image, self.callback, queue_size = 100)
        self.pub = rospy.Publisher('/image_listener/image', Image, queue_size = 100)
        self.bridge = CvBridge()


    def callback(self, ros_image):
        image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        np_image = np.asarray(image)
        height = ros_image.height
        width = ros_image.width
        detection_boxes = self.detector.detect(np_image)
        for row in detection_boxes:
            print 'Detection'
            top_left = (int(row[1]*width), int(row[0]*height))
            bottom_right = (int(row[3]*width), int(row[2]*height))
            cv2.rectangle(image, top_left, bottom_right, (255,0,0), 3)

        try:
            self.pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
        except CvBridgeError as e:
            print(e)
 
        

 
if __name__ == '__main__':
    #rosrun image_transport republish compressed in:=/realsense_d435_forward/color/image_raw/ raw out:=/realsense_d435_forward/color/image_raw/decompressed/
    model_name = 'ssd_mobilenet_v1_coco_2018_01_28'
    downloaded_model_path = os.path.join('./', model_name + '.tar.gz')
    sub = imageListener(model_name, downloaded_model_path, "/hsrb/head_rgbd_sensor/rgb/image_rect_color")
    rospy.spin() 
