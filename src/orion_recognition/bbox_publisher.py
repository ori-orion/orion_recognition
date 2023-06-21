#!/usr/bin/env python3
from collections import defaultdict

import orion_recognition.object_detector
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

from orion_recognition.bbox_utils import non_max_supp
from orion_recognition.colornames import ColorNames
import torchvision.transforms as transforms
import rospkg
import torch
import os

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
        self.detector = orion_recognition.object_detector.ObjectDetector()
        # self.detector.eval()

        # rospack = rospkg.RosPack()

        # Subscribers
        self.image_sub = message_filters.Subscriber(image_topic, Image, queue_size=1)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image, queue_size=1)

        # synchronise subscribers
        self.subscribers = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 1, 0.5)

        # Publishers
        self.image_pub = rospy.Publisher('/vision/bbox_image', Image, queue_size=10)
        self.detections_pub = rospy.Publisher('/vision/bbox_detections', DetectionArray, queue_size=10)

        # Image calibrator
        print("\tWaiting for camera info");
        camera_info = rospy.wait_for_message(
            "/hsrb/head_rgbd_sensor/depth_registered/camera_info", CameraInfo)
        self._invK = np.linalg.inv(np.array(camera_info.K).reshape(3, 3))
        print("\tCamera info recieved");

        # Define bridge open cv -> RosImage
        self.bridge = CvBridge()

        # Register a subscriber
        self.subscribers.registerCallback(self.callback)

        # Read the data on how large the objects should be
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "object_sizes.json"), "r") as json_file:
            self.size_dict = json.load(json_file)

        print("bbox_publisher.__init__()");

    # def getMeanDepth_gaussian(self, depth):
    #     """Ok so we want to mean over the depth image using a gaussian centred at the
    #     mid point
    #     Gausian is defined as e^{-(x/\sigma)^2}
    #
    #     Now, e^{-(x/sigma)^2}|_{x=1.5, sigma=1}=0.105 which is small enough. I'll therefore set the width of
    #     the depth image to be 3 standard deviations. (Remember, there're going to be two distributions
    #     multiplied together here! so that makes the corners 0.011 times as strong as the centre of the image.)
    #
    #     I'm then going to do (2D_gaussian \cdot image) / sum(2D_gaussian)
    #         (accounting for valid and invalid depth pixels on the way.)
    #
    #     This should give a fairly good approximation for the depth.
    #     """
    #
    #     def shiftedGaussian(x: float, shift: float, s_dev: float) -> float:
    #         return math.exp(-pow((x - shift) / s_dev, 2))
    #
    #     width: int = depth.shape[0]
    #     height: int = depth.shape[1]
    #     x_s_dev: float = width / 3
    #     y_s_dev: float = height / 3
    #     x_shift: float = width / 2
    #     y_shift: float = height / 2
    #
    #     # We need some record of the total amount of gaussian over the image so that we can work out
    #     # what to divide by.
    #     gaussian_sum: float = 0
    #     depth_sum: float = 0
    #
    #     for x in range(width):
    #         x_gaussian = shiftedGaussian(x, x_shift, x_s_dev)
    #         for y in range(height):
    #             if (depth[x, y] != 0):
    #                 point_multiplier: float = x_gaussian * shiftedGaussian(y, y_shift, y_s_dev)
    #                 gaussian_sum += point_multiplier
    #                 depth_sum += depth[x, y] * point_multiplier
    #             pass
    #         pass
    #
    #     return depth_sum / gaussian_sum

    def callback(self, ros_image: Image, depth_data: Image):
        stamp = ros_image.header.stamp

        # get images from cv bridge
        image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        depth = np.array(self.bridge.imgmsg_to_cv2(depth_data, 'passthrough'),
                         dtype=np.float32)

        image_np = np.asarray(image)
        # image_tensor = transforms.ToTensor()(image)
        # apply model to image
        with torch.no_grad():
            detections = self.detector(image_np)

        boxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']

        bbox_tuples = []

        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x_min, y_min, x_max, y_max = box

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
            top_left_camera = np.dot(self._invK, top_left_3d) * z
            bottom_right_3d = np.array([int(x_max), int(y_max), 1])     # Homogenous coordinates
            bottom_right_camera = np.dot(self._invK, bottom_right_3d) * z
            corner_to_corner = top_left_camera - bottom_right_camera
            x_size = abs(corner_to_corner[0])
            y_size = abs(corner_to_corner[1])
            z_size = (x_size + y_size) / 2.0
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
            bbox_tuples.append((box, label, score, detection))

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
                # header = std_msgs.msg.Header()
                # header.stamp = rospy.Time.now()
                header = depth_data.header;
                self.detections_pub.publish(DetectionArray(header, clean_detections))
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(image_bgr, "bgr8"))
            except CvBridgeError as e:
                print(e)


if __name__ == '__main__':
    rospy.init_node('bbox_publisher')
    img_topic = "/hsrb/head_rgbd_sensor/rgb/image_rect_color"
    depth_topic = "/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw"
    sub = BboxPublisher(img_topic, depth_topic)
    rospy.spin()
