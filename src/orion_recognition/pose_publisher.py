#!/usr/bin/env python

from pose_detector_ros import PoseDetector
from orion_actions.msg import PoseDetection, PoseDetectionArray
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
    
class PosePublisher(object):
    def __init__(self, image_topic):
        self.detector = PoseDetector()
        #rospy.init_node('bbox', anonymous=True)
        self.subscriber = rospy.Subscriber(image_topic, Image, self.callback, queue_size=100)
        self.bbox_pub = rospy.Publisher('/vision/pose_image', Image, queue_size=100)
        self.detections_pub = rospy.Publisher('/vision/pose_detections', PoseDetectionArray, queue_size=100)
        self.bridge = CvBridge()
        #self.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 50, encoding="unic")


    def callback(self, ros_image):
        image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        np_image = np.asarray(image)
        try:
            img,outputs = self.detector.detect(np_image)
            #print(outputs)
            if isinstance(outputs, int):
                print 'No Detection'
                img=image
            else:
                detections = []
                for det in outputs.keys():
                    people_id = int(det)
                    Nose = outputs[det]['Nose']
                    LEye = outputs[det]['LEye']
                    REye = outputs[det]['REye']
                    LEar = outputs[det]['LEar']
                    REar = outputs[det]['REar']
                    LShoulder = outputs[det]['LShoulder']
                    RShoulder = outputs[det]['RShoulder']
                    LElbow = outputs[det]['LElbow']
                    RElbow = outputs[det]['RElbow']
                    LWrist = outputs[det]['LWrist']
                    RWrist = outputs[det]['RWrist']
                    LHip = outputs[det]['LHip']
                    RHip = outputs[det]['RHip']
                    LKnee = outputs[det]['LKnee']
                    Rknee = outputs[det]['Rknee']
                    LAnkle = outputs[det]['LAnkle']
                    RAnkle = outputs[det]['RAnkle']
                    Neck = outputs[det]['Neck']
                    detection = PoseDetection(people_id, Nose[0], Nose[1], LEye[0], LEye[1], REye[0], REye[1], LEar[0], LEar[1], REar[0], REar[1], LShoulder[0], LShoulder[1],RShoulder[0],RShoulder[1],LElbow[0],LElbow[1],RElbow[0],RElbow[1],LWrist[0],LWrist[1],RWrist[0], RWrist[1],LHip[0],LHip[1],RHip[0],RHip[1], LKnee[0], LKnee[1],Rknee[0],Rknee[1],LAnkle[0],LAnkle[1],RAnkle[0],RAnkle[1],Neck[0],Neck[1])
                    detections.append(detection)
    
                h = std_msgs.msg.Header()
                h.stamp = rospy.Time.now()
                self.detections_pub.publish(PoseDetectionArray(h, detections))
            self.bbox_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except CvBridgeError as e:
                print(e)

 
if __name__ == '__main__':
    rospy.init_node('pose_publisher')
    img_topic = "/hsrb/head_rgbd_sensor/rgb/image_rect_color"
    pub = BboxPublisher(img_topic)
    rospy.spin() 
