# orion_recognition

```
$ sim_mode
<local>~$ source catkin_ws/devel/setup.bash
```
or 
```
$ hsrb_mode
<hsrb>~$ source catkin_ws/devel/setup.bash
```

## Launch
The launch file does not work.
```
<hsrb>~$ roslaunch orion_recognition recognition.launch
```
Ideally, this would launch `bbox_publisher_node.py` and `detection_tf_publisher_node.py`.

Alternatively, run them separately.
```
<hsrb>~$ rosrun orion_recognition bbox_publisher_node.py
<hsrb>~$ rosrun orion_recognition detection_tf_publisher_node.py
```

## To run check for object action server
```
<hsrb>~$ rosrun orion_recognition check_for_object_server_node.py
```

## What do the scripts do?
`bbox_publisher_node.py` subscribes to `/hsrb/head_rgbd_sensor/rgb/image_rect_color` (Image)
and publishes to
`/vision/bbox_detections` (DetectionArray, information about detected bounding boxes)
`/vision/bbox_image`, (Image, bounding boxes in an image for visualisation)

`detection_tf_publisher_node.py` subscribes to `/vision/bbox_detections` and `/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw` (Image, depth image) and broadcasts TF frames.
