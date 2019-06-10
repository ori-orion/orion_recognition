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

### To change to a new model
Download a new model from:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

Unzip it and place it under `src/orion_recognition`.

Change the `model_name` in `scripts/bbox_publisher_node.py`.

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

`/vision/bbox_detections` (DetectionArray, information about detected bounding boxes, you can also find the mean color and the size(not accurate) of the object in this messagenow.)

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
Now this message also gives you two additional bool values: waving, sitting, to predict whether this person is waving or sitting.
`/vision/pose_image`, (Image, skeletons in an image for visualisation)

## Face detection
### Launch
```
<hsrb>~$ rosrun orion_recognition face_publisher_node.py
```

### What do the scripts do?
`face_publisher_node.py` subscribes to `/hsrb/head_rgbd_sensor/rgb/image_rect_color` (Image)

and publishes to

`/vision/face_detections` (DetectionArray, information about bounding boxes locations, age(not accurate but at least is somehow reasonable now), gender, e.g.'M'/'F', emotion, e.g. 'happy')


`/vision/face_image`, (Image, bounding boxes in an image with age and gender information for visualisation)


