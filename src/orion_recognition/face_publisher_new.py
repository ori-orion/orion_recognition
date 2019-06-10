#!/usr/bin/env python

from new_face.src.test.face_detector import FaceDetector
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
        self.detector = FaceDetector()
        #rospy.init_node('bbox', anonymous=True)
        self.subscriber = rospy.Subscriber(image_topic, Image, self.callback, queue_size=100)
        self.bbox_pub = rospy.Publisher('/vision/face_bbox_image', Image, queue_size=100)
        self.detections_pub = rospy.Publisher('/vision/face_bbox_detections', FaceDetectionArray, queue_size=100)
        self.bridge = CvBridge()
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 35, encoding="unic")
        self.emotion_label = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
        self.gender_label = {0:'woman', 1:'man'}


    def callback(self, ros_image):
        image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        
        np_image = np.asarray(image)
        height = ros_image.height
        width = ros_image.width
        true_image =Im_PIL.fromarray(image)
        try:
            bboxes, predicted_ages, predicted_genders, predicted_emotions = self.detector.detect(np_image)
            
            if bboxes is None:
                print 'No face detected'
                return 0
            detections = []
            all_ids = range(len(bboxes))
            for indx in all_ids:
                print 'Detection'
                people_id = int(indx)
                face_bbox = bboxes[indx]
                age = predicted_ages[indx]
                gender = predicted_genders[indx]
                emotion = predicted_emotions[indx]
                gender = self.gender_label[gender]
                emotion = self.emotion_label[emotion]
                ymin, xmin, ymax, xmax = face_bbox
                top_left = (int(xmin), int(ymax))
                bottom_right = (int(xmax), int(ymin))
                cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 5)
                center_x = (top_left[0] + bottom_right[0])/2
                center_y = (top_left[1] + bottom_right[1])/2
                #center_x, center_y = (top_left + bottom_right) / 2
                detection = FaceDetection(people_id, center_x, center_y, width, height, gender, age, emotion)
                detections.append(detection)
                true_image =Im_PIL.fromarray(image)
                draw = ImageDraw.Draw(true_image, 'RGBA')
                label = "{}, {}, {}".format(age, gender, emotion)
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
