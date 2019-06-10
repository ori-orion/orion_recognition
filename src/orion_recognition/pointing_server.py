#!/usr/bin/env python

import actionlib
import rospy
import orion_actions
from orion_actions.msg import Detection, DetectionArray, PoseDetection, PoseDetectionArray, Pointing, PointingArray


class PointingServer(object):
    def __init__(self):
        self.omitted_objects=['table','chair']
        rospy.loginfo("pointing -- initialising pointing server")
        self._pointing_as = actionlib.SimpleActionServer('Pointing',
                                                          orion_actions.msg.PointingAction,
                                                          execute_cb=self._pointing_cb, auto_start=False)

        rospy.loginfo("Pointing -- starting pointing server")
        self._pointing_as.start()

        rospy.loginfo("Pointing -- all initialised")

    def _pointing_cb(self):
        rospy.loginfo("pointing -- executing pointing callback function")
        if self._check_and_send_result_if_succeeded():
            rospy.loginfo("pointing -- callback done")
            return

        result = orion_actions.msg.PointingActionResult()
        result.pointing_array = None
        result.is_present = False
        self._pointing_as.set_succeeded(result)
        rospy.loginfo("Pointing -- callback done")

    def _is_object_in_detection(self, detections, humans):
        rospy.loginfo("Pointing -- check if any one pointing any objects")
        header=detections.header
        pointing_array=[]
        for hum in humans.detections:
            all_pointed_objects=[]
            LElbow_x = hum.LElbow_x
            LElbow_y = hum.LElbow_y
            RElbow_x = hum.RElbow_x
            RElbow_y = hum.RElbow_y
            LWrist_x = hum.LWrist_x
            LWrist_y = hum.LWrist_y
            RWrist_x = hum.RWrist_x
            RWrist_y = hum.RWrist_y
            color = human.color
            slope_L = (LWrist_y - LElbow_y)/ (LWrist_x - LElbow_x)
            slope_R = (LWrist_y - LElbow_y)/ (LWrist_x - LElbow_x)
            for det in detections.detections:
                if LWrist_x < LElbow_x:
                #check for left hand.
                    if det.x < LWrist_x:
                        intersect_y = LWrist_y + slope_L * (det.x - LWrist_x)
                        if intersect_y > (det.y - det.height/2) and intersect_y < (det.y + det.height/2):
                            all_pointed_objects.append(det)    
                else:
                    if det.x > LWrist_x:
                        intersect_y = LWrist_y + slope_L * (det.x - LWrist_x)
                        if intersect_y > (det.y - det.height/2) and intersect_y < (det.y + det.height/2):
                            all_pointed_objects.append(det)
                if RWrist_x < RElbow_x:
                #check for left hand.
                    if det.x < RWrist_x:
                        intersect_y = RWrist_y + slope_R * (det.x - RWrist_x)
                        if intersect_y > (det.y - det.height/2) and intersect_y < (det.y + det.height/2):
                            all_pointed_objects.append(det)    
                else:
                    if det.x > RWrist_x:
                        intersect_y = RWrist_y + slope_R * (det.x - RWrist_x)
                        if intersect_y > (det.y - det.height/2) and intersect_y < (det.y + det.height/2):
                            all_pointed_objects.append(det)
                

            if len(all_pointed_objects)!=0:
                point_msg = Pointing(all_pointed_objects, color) 
                pointing_array.append(point_msg)
        
        if len(pointing_array!=0):
            pointing_array_msg = PointingArray(header,pointing_array)
            return True, pointing_array_msg
        rospy.loginfo("People are not detected.")
        return False, None

    def _check_and_send_result_if_succeeded(self):
        rospy.loginfo("Pointing -- checking and sending result if succeeded")
        detections = rospy.wait_for_message('/vision/bbox_detections', DetectionArray, 5 * 60)
        humans = rospy.wait_for_message('/vision/pose_detections', PoseDetectionArray, 5 * 60)
        rospy.loginfo("Pointing -- detections received")

        is_present, pointing_array_msg = self._is_object_in_detections(detections, humans)
        if is_present:
            result = orion_actions.msg.PointingActionResult()
            result.is_present = True
            result.pointing_array = pointing_array_msg
            self._check_for_object_as.set_succeeded(result)
            return True
        return False
