#!/usr/bin/env python

import rospy
import os
from orion_recognition.bbox_publisher import BboxPublisher


def main():
    rospy.init_node('bbox_publisher_node')
    img_topic = "/hsrb/head_rgbd_sensor/rgb/image_rect_color"
    model_name = 'ssd_mobilenet_v1_coco_2018_01_28'
    path_to_model = os.path.join('/home/chiaman/git/models/research/object_detection')  # TODO: No hardcoded location
    detect_pers = True
    detect_obj = True
    thresh = 0.4
    BboxPublisher(img_topic, model_name, path_to_model, detect_obj, detect_pers, thresh)
    rospy.spin()


if __name__ == '__main__':
    main()
