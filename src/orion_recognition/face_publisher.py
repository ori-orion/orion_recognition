#!/usr/bin/env python

from face_detector_ros import FaceDetector_ros
from orion_actions.msg import FaceDetection, FaceDetectionArray
import pdb
import cv2
import numpy as np
import os
import rospy
import std_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from PIL import ImageDraw, ImageFont
from PIL import Image as Im_PIL
    
class FacePublisher(object):
    def __init__(self, image_topic):
        self.detector = FaceDetector_ros()
        #rospy.init_node('bbox', anonymous=True)
        self.subscriber = rospy.Subscriber(image_topic, Image, self.callback, queue_size=100)
        self.bbox_pub = rospy.Publisher('/vision/face_bbox_image', Image, queue_size=100)
        self.detections_pub = rospy.Publisher('/vision/face_bbox_detections', FaceDetectionArray, queue_size=100)
        self.bridge = CvBridge()
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 15, encoding="unic")


    def callback(self, ros_image):
        image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        np_image = np.asarray(image)
        height = ros_image.height
        width = ros_image.width
        true_image =Im_PIL.fromarray(image)
        try:
            output_dict = self.detector.detect(np_image)
            detections = []
            all_ids = output_dict.keys()
            for indx in all_ids:
                print 'Detection'
                people_id = int(indx)
                face_bbox = output_dict[indx]['face_bbox']
                score = output_dict[indx]['score']
                age_group = output_dict[indx]['age_group']
                age_indx  = output_dict[indx]['age_indx']
                gender = output_dict[indx]['gender']
                ymin, xmin, ymax, xmax = face_bbox
                top_left = (int(xmin), int(ymax))
                bottom_right = (int(xmax), int(ymin))
                cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 5)
                center_x = (top_left[0] + bottom_right[0])/2
                center_y = (top_left[1] + bottom_right[1])/2
                #center_x, center_y = (top_left + bottom_right) / 2
                detection = FaceDetection(people_id, score, center_x, center_y, width, height, gender, age_group,age_indx)
                detections.append(detection)
                true_image =Im_PIL.fromarray(image)
                draw = ImageDraw.Draw(true_image, 'RGBA')
                label = "{}, {}".format(age_group, gender)
                draw.text((xmin, ymax), text=label, font = self.font, fill=(0,255,0))
            h = std_msgs.msg.Header()
            h.stamp = rospy.Time.now()
            self.detections_pub.publish(FaceDetectionArray(h, detections))
            self.bbox_pub.publish(self.bridge.cv2_to_imgmsg(np.asarray(true_image), "bgr8"))
        except CvBridgeError as e:
                print(e)

 
if __name__ == '__main__':
    rospy.init_node('face_publisher')
    img_topic = "/hsrb/head_rgbd_sensor/rgb/image_rect_color"
    pub = BboxPublisher(img_topic)
    rospy.spin() 
