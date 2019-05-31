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
## Object detection
### Installation
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

### Launch
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

### To run check for object action server
```
<hsrb>~$ rosrun orion_recognition check_for_object_server_node.py
```

### What do the scripts do?
`bbox_publisher_node.py` subscribes to `/hsrb/head_rgbd_sensor/rgb/image_rect_color` (Image)

and publishes to

`/vision/bbox_detections` (DetectionArray, information about detected bounding boxes)

`/vision/bbox_image`, (Image, bounding boxes in an image for visualisation)

`detection_tf_publisher_node.py` subscribes to `/vision/bbox_detections` and `/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw` (Image, depth image) and broadcasts TF frames.

## Human pose detection
### Launch
```
<hsrb>~$ rosrun orion_recognition pose_publisher_node.py
```

### What do the scripts do?
`pose_publisher_node.py` subscribes to `/hsrb/head_rgbd_sensor/rgb/image_rect_color` (Image)

and publishes to

`/vision/pose_detections` (DetectionArray, coordinates of each marker point, e.g. nose_x, nose_y ...)

Available marker points: Nose, LEye, REye, LEar, REar, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck

`/vision/pose_image`, (Image, skeletons in an image for visualisation)

## Face detection
### Launch
```
<hsrb>~$ rosrun orion_recognition face_publisher_node.py
```

### What do the scripts do?
`face_publisher_node.py` subscribes to `/hsrb/head_rgbd_sensor/rgb/image_rect_color` (Image)

and publishes to

`/vision/face_detections` (DetectionArray, information about bounding boxes locations, scores, age_group(string), e.g. '18-25', age_index(int), e.g. if '18-25' is the third old age group then age_index will be 3 for '18-25', gender, e.g.'M'/'F')

`/vision/face_image`, (Image, bounding boxes in an image with age and gender information for visualisation)


