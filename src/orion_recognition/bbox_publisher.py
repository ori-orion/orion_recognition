#!/usr/bin/env python3
from collections import defaultdict

import orion_recognition.object_detector as object_detector
import message_filters
from orion_actions.msg import Detection, DetectionArray, Label
import sys
import cv2
import numpy as np
import math
import json
import rospy
import std_msgs.msg
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError

from orion_recognition.util.bbox_utils import non_max_supp
from orion_recognition.colornames import ColorNames
import torchvision.transforms as transforms
import rospkg
import torch
import os
import MemoryManager;

# Approximate maximum dimension size limits - Up to this length
# In meters
max_size_limits = {
    "small": 0.5,
    "medium": 1,
    "large": 10000
}

min_size_limits = {
    "small": 0.04,
    "medium": 0.2,
    "large": 0.4
}


class BboxPublisher(object):
    def __init__(self, image_topic, depth_topic):
        self.detector = object_detector.ObjectDetector()
        #self.detector.eval()

        rospack = rospkg.RosPack()

        # Subscribers
        self.image_sub = message_filters.Subscriber(image_topic, Image, queue_size=1)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image, queue_size=1)

        # synchronise subscribers
        self.subscribers = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 1, 0.5)

        # Publishers
        self.image_pub = rospy.Publisher('/vision/bbox_image', Image, queue_size=10)
        # Do we actually need to publish the detections to a publisher?
        self.detections_pub = rospy.Publisher('/vision/bbox_detections', DetectionArray, queue_size=10)
        self.pymongo_interface = MemoryManager.PerceptionInterface(MemoryManager.MemoryManager(connect_to_current_latest=True));

        # Image calibrator, Default: #/hsrb/head_rgbd_sensor/depth_registered/camera_info
        camera_info = rospy.wait_for_message(
            "/camera/aligned_depth_to_color/camera_info", CameraInfo)
        self._invK = np.linalg.inv(np.array(camera_info.K).reshape(3, 3))

        # Define bridge open cv -> RosImage
        self.bridge = CvBridge()

        # Register a subscriber
        # In this callback function, subscribed messages are synchronized and processed and then published
        self.subscribers.registerCallback(self.callback)

        # Read the data on how large the objects should be
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "object_sizes.json"), "r") as json_file:
            self.size_dict = json.load(json_file)

    def callback(self, ros_image: Image, depth_data: Image):
        stamp = ros_image.header.stamp

        # get images from cv bridge
        image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        depth = np.array(self.bridge.imgmsg_to_cv2(depth_data, 'passthrough'),
                         dtype=np.float32)

        image_np = np.asarray(image)
        #image_tensor = transforms.ToTensor()(image)

        # apply model to image
        # !!! This part is changed to adapt to YOLOv8 !!!
        with torch.no_grad():
            detections, nouse = self.detector.decode_result_Boxes(self.detector.detect_img_single(image_np))

        boxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']

        bbox_tuples = []

        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x_min, y_min, x_max, y_max = box[0]

            # Dimensions of bounding box
            center_x = (x_min + x_max) / 2
            width = x_max - x_min
            center_y = (y_min + y_max) / 2
            height = y_max - y_min

            # Get depth
            trim_depth = depth[int(y_min):int(y_max), int(x_min):int(x_max)]
            valid = trim_depth[np.nonzero(trim_depth)]

            # If depth is not valid, discard bounding box
            if valid.size == 0:
                # print('no valid depth for object size')
                continue

            # Use depth to get position
            z = np.min(valid) * 1e-3
            top_left_3d = np.array([int(x_min), int(y_min), 1])         # Homogenous coordinates
            top_left_camera = np.dot(self._invK, top_left_3d) * z       # 3D Point (top left of bbox)
            bottom_right_3d = np.array([int(x_max), int(y_max), 1])     # Homogenous coordinates
            bottom_right_camera = np.dot(self._invK, bottom_right_3d) * z  # 3D Point (bottom right of bbox)
            corner_to_corner = top_left_camera - bottom_right_camera
            x_size = abs(corner_to_corner[0])
            y_size = abs(corner_to_corner[1])
            z_size = (x_size + y_size) / 2.0                            # Assumes the 3D BoundBox depth (z) is the mean of x3D and y3D values
            size = Point(x_size, y_size, z_size)

            # Check if the size of the 3D bounding box makes sense
            if max(x_size, y_size, z_size) > max_size_limits[self.size_dict.get(label, "large")]:
                continue
            elif max(x_size, y_size, z_size) < min_size_limits[self.size_dict.get(label, "small")]:
                continue

            # Find object position
            image_point = np.array([int(center_x), int(center_y), 1])         # Homogenous coordinates
            obj = np.dot(self._invK, image_point) * z

            # Get Colour
            crop = image_np[int(y_min):int(y_max), int(x_min):int(x_max)]
            RGB = np.mean(crop, axis=(0, 1))
            colour = ColorNames.findNearestOrionColorName(RGB)

            # create label
            label_str = label.replace(" ", "_")
            label_str = label_str.rstrip()  # Remove any white spaces at the end of the string
            score_lbl = Label(label_str, np.float64(score))

            # create detection instance
            detection = Detection(score_lbl, center_x, center_y, width, height,
                                  size, colour, obj[0], obj[1], obj[2], stamp)
            bbox_tuples.append((box[0], label, score[0], detection))

        clean_bbox_tuples = non_max_supp(bbox_tuples)
        clean_detections = []

        image_bgr = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")

        for ((x_min, y_min, x_max, y_max), label, score, detection) in clean_bbox_tuples:
            clean_detections.append(detection)
            top_left = (int(x_min), int(y_min))
            bottom_right = (int(x_max), int(y_max))
            cv2.rectangle(image_bgr, top_left, bottom_right, (255, 0, 0), 3)
            cv2.putText(image_bgr, f"{label}: {score}", top_left, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

        if clean_detections:
            # Publish nodes
            try:
                header = std_msgs.msg.Header()
                header.stamp = rospy.Time.now()
                self.detections_pub.publish(DetectionArray(header, clean_detections))
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(image_bgr, "bgr8"))
            except CvBridgeError as e:
                print(e)


if __name__ == '__main__':
    rospy.init_node('bbox_publisher')
    img_topic = "/camera/color/image_raw" #/hsrb/head_rgbd_sensor/rgb/image_rect_color"
    depth_topic = "/camera/aligned_depth_to_color/image_raw" #"/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw"
    sub = BboxPublisher(img_topic, depth_topic)
    rospy.spin()
