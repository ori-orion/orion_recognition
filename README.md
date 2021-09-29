Branch `noetic` is based off `clarissa`, but is changed over to python3 instead of python2.

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

### To train a custom model
See https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md and https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md.

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

`/vision/bbox_detections` (DetectionArray, information about detected bounding boxes, 3d location, you can also find the mean color and the size (not accurate) of the object in this message now.)

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

`pose_tf_publisher_node.py` subscribes to `/vision/pose_detections` (PoseDetectionArray) and `/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw` (Image) and broadcasts TF frames in the format of person_red_0, person_red_1, person_blue_0, etc.

## Face detection
### Launch
```
<hsrb>~$ rosrun orion_recognition face_publisher_node.py
```

### What do the scripts do?
`face_publisher_node.py` subscribes to `/hsrb/head_rgbd_sensor/rgb/image_rect_color` (Image)

and publishes to

`/vision/face_bbox_detections` (FaceDetectionArray, information about bounding boxes locations, age(not accurate but at least is somehow reasonable now), gender, e.g.'M'/'F', emotion, e.g. 'happy')


`/vision/face_bbox_image`, (Image, bounding boxes in an image with age and gender information for visualisation)


