#!/usr/bin/env python

import object_detector
import message_filters
from orion_actions.msg import Detection, DetectionArray, Label
import pdb
import cv2
import numpy as np
import os
import rospy
import std_msgs.msg
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
from colornames import ColorNames
import torchvision.transforms as transforms

class BboxPublisher(object):
    def __init__(self, image_topic, detect_object, detect_person, threshold_detection):
        self.detector = object_detector.ObjectDetector()
        # Subscribers        
        self.image_sub = message_filters.Subscriber(image_topic, Image, queue_size=100)
        self.depth_sub = message_filters.Subscriber("/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw", Image)

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
        depth = self.bridge.imgmsg_to_cv2(depth_data, "bgr8")
        
        image_tensor = transforms.ToTensor(image)
        
        #apply model to image
        detections = self.detector([image_tensor])[0]
        
        boxes = detections['boxes']
        labels = detections['labels']
        scores = detectons['scores']

            label = Label(class_string.encode('ascii','ignore'), np.float64(detection_scores[i]))
            crop = np_image[int(row[0] * height):int(row[2] * height), int(row[1] * width):int(row[3] * width)]
            RGB = np.mean(crop, axis=(0,1))
            RGB = (RGB[2], RGB[1], RGB[0])
            color = ColorNames.findNearestOrionColorName(RGB)

            # Get depth and size
            min_v = int(center_y - (height_box / 2.0))
            max_v = int(center_y + (height_box / 2.0))
            min_u = int(center_x - (width_box / 2.0))
            max_u = int(center_x + (width_box / 2.0))
            trim_depth = depth_array[min_v:max_v, min_u:max_u]
            valid = trim_depth[np.nonzero(trim_depth)]

            if valid.size != 0:
                z = np.min(valid) * 1e-3
                top_left_image = np.array([top_left[0], top_left[1], 0])
                bottom_right_image = np.array([bottom_right[0], bottom_right[1], 0])
                top_left_camera = np.dot(self._invK, top_left_image) * z
                bottom_right_camera = np.dot(self._invK, bottom_right_image) * z
                corner_to_corner = top_left_camera - bottom_right_camera
                x_size = abs(corner_to_corner[0])
                y_size = abs(corner_to_corner[1])
                z_size = (x_size + y_size)/2.0
                size = Point(x_size, y_size, z_size)
            else:
                size = Point(0.0, 0.0, 0.0)
                print('no valid depth for object size')
                continue

            image_point = np.array([int(center_x), int(center_y), 1])
            object_point = np.dot(self._invK, image_point) * z
    
            detection = Detection(label, center_x, center_y, width_box, height_box, size, color, object_point[0], object_point[1], object_point[2])
            detections.append(detection)
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
    model_name = 'faster_rcnn_inception_v2_coco_2018_01_28'
    path_to_model = os.path.join('/home/ori/code/recognition/models-master/research/object_detection')
    detect_pers = True
    detect_obj = True
    thresh = 0.4
    sub = BboxPublisher(img_topic, model_name, path_to_model, detect_obj, detect_pers, thresh)
    rospy.spin()
