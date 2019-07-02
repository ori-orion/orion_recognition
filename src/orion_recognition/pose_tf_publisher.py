#!/usr/bin/env python

from cv_bridge import CvBridge, CvBridgeError
import geometry_msgs.msg
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import tf2_ros
from orion_actions.msg import PoseDetectionArray
import pdb


class DetectionTFPublisher(object):
    def __init__(self):
        self.bridge = CvBridge()
        pose_sub = message_filters.Subscriber(
            "/vision/pose_detections", PoseDetectionArray)
        depth_sub = message_filters.Subscriber(
            "/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw", Image)
        camera_info = rospy.wait_for_message(
            "/hsrb/head_rgbd_sensor/depth_registered/camera_info", CameraInfo)
        self._objects = rospy.get_param('~objects', [])
        rospy.loginfo('objects = {0}'.format(self._objects))
        self._use_center = False
        self._invK = np.linalg.inv(np.array(camera_info.K).reshape(3, 3))
        self._ts = message_filters.ApproximateTimeSynchronizer(
            [depth_sub,pose_sub], 30, 0.5)
        self._ts.registerCallback(self.callback)
        self._br = tf2_ros.TransformBroadcaster()

    def callback(self, depth_data, pose_detection):
    	label_name = 'person_pose'
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, 'passthrough')
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        trans = []
        depth_array = np.array(depth_image, dtype=np.float32)
        human_det ={'person_pose':[]}
        for det in pose_detection.detections:
            # pdb.set_trace()
            z = 0.0
            upLeft_x = det.upLeft_x
            upLeft_y = det.upLeft_y
            bottomRight_x = det.bottomRight_x
            bottomRight_y = det.bottomRight_y
            center_x = (upLeft_x + bottomRight_x) / 2
            center_y = (upLeft_y + bottomRight_y) / 2
            width = bottomRight_x - upLeft_x
            height =  bottomRight_y - upLeft_y
            if True: # detection.label.name in self._objects:
                if self._use_center:
                    # use center depth
                    z = depth_array[int(center_y)][int(center_x)] * 1e-3
                else:
                    # use min depth in the BoundingBox
                    min_v = int(center_y - (height / 2.0))
                    max_v = int(center_y + (height / 2.0))
                    min_u = int(center_x - (width / 2.0))
                    max_u = int(center_x + (width / 2.0))
                    trim_depth = depth_array[min_v:max_v, min_u:max_u]
                    valid = trim_depth[np.nonzero(trim_depth)]
                    if valid.size != 0:
                        z = np.min(valid) * 1e-3
                    else:
                        print('skip1')
                        continue
                # no valid point
                if z == 0.0:
                    print('skip2')
                    continue
                image_point = np.array([int(center_x), int(center_y), 1])
                object_point = np.dot(self._invK, image_point) * z
                human_det[label_name].append(object_point)
        print(human_det)
        for i, pos in enumerate(human_det[label_name]):
            t = geometry_msgs.msg.TransformStamped()
            t.header = depth_data.header
            t.child_frame_id = 'person_pose' + '_' + str(i)
            t.transform.translation.x = pos[0]
            t.transform.translation.y = pos[1]
            t.transform.translation.z = pos[2]
            # compute the tf frame
            # rotate -90 degrees along z-axis
            t.transform.rotation.z = np.sin(-np.pi / 4)
            t.transform.rotation.w = np.cos(-np.pi / 4)
            trans.append(t)
        self._br.sendTransform(trans)


if __name__ == '__main__':
    rospy.init_node('detection_tf_publisher')
    DetectionTFPublisher()
    rospy.spin()
