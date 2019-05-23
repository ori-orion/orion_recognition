#!/usr/bin/env python

from cv_bridge import CvBridge, CvBridgeError
import geometry_msgs.msg
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import tf2_ros
from orion_actions.msg import DetectionArray


class DetectionTFPublisher(object):
    def __init__(self):
        self.bridge = CvBridge()
        detection_sub = message_filters.Subscriber(
            "/vision/bbox_detections", DetectionArray)
        depth_sub = message_filters.Subscriber(
            "/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw", Image)
        camera_info = rospy.wait_for_message(
            "/hsrb/head_rgbd_sensor/depth_registered/camera_info", CameraInfo)
        self._objects = rospy.get_param('~objects', [])
        rospy.loginfo('objects = {0}'.format(self._objects))
        self._use_center = False
        self._invK = np.linalg.inv(np.array(camera_info.K).reshape(3, 3))
        self._ts = message_filters.ApproximateTimeSynchronizer(
            [detection_sub, depth_sub], 30, 0.5)
        self._ts.registerCallback(self.callback)
        self._br = tf2_ros.TransformBroadcaster()

    def callback(self, detections, depth_data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, 'passthrough')
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        depth_array = np.array(depth_image, dtype=np.float32)
        objects = {key: [] for key in self._objects}
        trans = []
        for detection in detections.detections:
            if detection.label.name in self._objects:
                if self._use_center:
                    # use center depth
                    z = depth_array[int(detection.y)][int(detection.x)] * 1e-3
                else:
                    # use min depth in the BoundingBox
                    min_v = int(detection.y - (detection.height / 2.0))
                    max_v = int(detection.y + (detection.height / 2.0))
                    min_u = int(detection.x - (detection.width / 2.0))
                    max_u = int(detection.x + (detection.width / 2.0))
                    trim_depth = depth_array[min_v:max_v, min_u:max_u]
                    valid = trim_depth[np.nonzero(trim_depth)]
                    if valid.size != 0:
                        z = np.min(valid) * 1e-3
                    else:
                        continue
                # no valid point
                if z == 0.0:
                    continue
                image_point = np.array([int(detection.x), int(detection.y), 1])
                object_point = np.dot(self._invK, image_point) * z
                objects[detection.label.name].append(object_point)

        for obj in objects:
            for i, pos in enumerate(objects[obj]):
                t = geometry_msgs.msg.TransformStamped()
                t.header = depth_data.header
                t.child_frame_id = obj + '_' + str(i)
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
