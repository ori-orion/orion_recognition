#!/bin/bash

# Tmux session name
SESSION=vision

# Set this variable in order to source ORIon ROS workspace
ORION_WS=/home/$USER/orion_ws/devel/setup.bash

# Set this variable to run in everything in simulation, with a local roscore. Otherwise, run on HSRB
SIM_MODE=true

if [ $SIM_MODE==true ]; then
ROS_MASTER_CMD="sim_mode"
else
ROS_MASTER_CMD="hsrb_mode"
fi

_SRC_ENV="tmux send-keys source Space $ORION_WS C-m "

PREFIX="$ROS_MASTER_CMD; source $ORION_WS; source activate tf2;"

tmux -2 new-session -d -s $SESSION
tmux rename-window -t $SESSION:0 'bbox_publisher'
tmux new-window -t $SESSION:1 -n 'detection_tf_publisher'
tmux new-window -t $SESSION:2 -n 'visualise'
tmux new-window -t $SESSION:3 -n 'gpu_stats'
tmux new-window -t $SESSION:4 -n 'battery_checker'
tmux new-window -t $SESSION:5 -n 'human_pose_detector'

tmux select-window -t $SESSION:bbox_publisher
[ -f $ORION_WS ] && `$_SRC_ENV`
tmux send-keys "$PREFIX rosrun orion_recognition bbox_publisher_node.py" C-m

tmux select-window -t $SESSION:detection_tf_publisher
[ -f $ORION_WS ] && `$_SRC_ENV`
tmux send-keys "$PREFIX rosrun orion_recognition detection_tf_publisher_node.py" C-m

tmux select-window -t $SESSION:visualise
[ -f $ORION_WS ] && `$_SRC_ENV`
tmux send-keys "$PREFIX rqt_image_view" C-m

tmux select-window -t $SESSION:gpu_stats
[ -f $ORION_WS ] && `$_SRC_ENV`
tmux send-keys "nvidia-smi"

tmux select-window -t $SESSION:battery_checker
[ -f $ORION_WS ] && `$_SRC_ENV`
tmux send-keys "$PREFIX rosrun orion_battery_check laptop_battery_publisher.py" C-m

tmux select-window -t $SESSION:human_pose_detector
[ -f $ORION_WS ] && `$_SRC_ENV`
tmux send-keys "$PREFIX rosrun orion_recognition pose_tf_publisher_node.py"

