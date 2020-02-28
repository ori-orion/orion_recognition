#!/usr/bin/env python

import orion_recognition.object_detector
import message_filters
from orion_actions.msg import Detection, DetectionArray, Label
import pdb
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import os
import rospy
import std_msgs.msg
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
from orion_recognition.colornames import ColorNames
import torchvision.transforms as transforms

class BboxPublisher(object):
    def __init__(self, image_topic, depth_topic):
        self.detector = orion_recognition.object_detector.ObjectDetector()
        self.detector.eval()
        # Subscribers        
        self.image_sub = message_filters.Subscriber(image_topic, Image, queue_size=100)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image)

        #synchronise subscribers
        self.subscribers = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 30, 0.5)

        #Publishers
        self.image_pub = rospy.Publisher('/vision/bbox_image', Image, queue_size=100)
        self.detections_pub = rospy.Publisher('/vision/bbox_detections', DetectionArray, queue_size=100)
        
        #Image calibrator
        camera_info = rospy.wait_for_message(
            "/hsrb/head_rgbd_sensor/depth_registered/camera_info", CameraInfo)
        self._invK = np.linalg.inv(np.array(camera_info.K).reshape(3, 3))

        # Define bridge open cv -> RosImage
        self.bridge = CvBridge()

        #Register a subscriber
        self.subscribers.registerCallback(self.callback)



    def callback(self, ros_image, depth_data):
        # get images from cv bridge
        image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        depth = np.array(self.bridge.imgmsg_to_cv2(depth_data, 'passthrough'),
                         dtype=np.float32)

        image_np = np.asarray(image)
        image_tensor = transforms.ToTensor()(image)
        img_x = image_tensor.size()[2]
        img_y = image_tensor.size()[1]
        
        #apply model to image
        detections = self.detector([image_tensor])[0]
        
        boxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']
        
        detections = []
        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]
            score = scores[i]
            
            #draw rectangle onto the image
            top_left = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 3)
            
            # Dimensions of bounding box
            center_x = (box[0]+box[2])/2
            width = box[2]-box[0]
            center_y = (box[1]+box[3])/2
            height = box[3]-box[1]
            
            # Get depth
            trim_depth = depth[int(box[0]):int(box[2]), int(box[1]):int(box[2])]
            valid = trim_depth[np.nonzero(trim_depth)]
            
            if valid.size != 0:
                z = np.min(valid) * 1e-3
                top_left_3d = np.array([top_left[0], top_left[1], 0])
                top_left_camera = np.dot(self._invK, top_left_3d)*z
                bottom_right_3d = np.array([bottom_right[0], bottom_right[1], 0])
                bottom_right_camera = np.dot(self._invK, bottom_right_3d)*z
                corner_to_corner = top_left_camera - bottom_right_camera
                x_size = abs(corner_to_corner[0])
                y_size = abs(corner_to_corner[1])
                z_size = (x_size + y_size)/2.0
                size = Point(x_size, y_size, z_size)
            else:
                size = Point(0.0, 0.0, 0.0)
                print('no valid depth for object size')
                continue

            # Find object position
            image_point = np.array([int(center_x), int(center_y), 1])
            obj = np.dot(self._invK, image_point) * z

            # Get Colour
            crop = image_np[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
            RGB = np.mean(crop, axis=(0,1))
            RGB = (RGB[2], RGB[1], RGB[0])
            colour = ColorNames.findNearestOrionColorName(RGB)
            
            # create label
            score_lbl = Label(str(label).encode('ascii','ignore'), np.float64(score))
            
            # create detection instance
            detection = Detection(score_lbl, center_x, center_y, width, height, size, colour, obj[0], obj[1], obj[2])
            
            detections.append(detection)

        
        # Publish nodes
        try:
            h = std_msgs.msg.Header()
            h.stamp = rospy.Time.now()
            self.detections_pub.publish(DetectionArray(h, detections))
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    rospy.init_node('bbox_publisher')
    img_topic = "/hsrb/head_rgbd_sensor/rgb/image_rect_color"
    depth_topic = "/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw"
    sub = BboxPublisher(img_topic, depth_topic)
    rospy.spin()
