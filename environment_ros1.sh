#! /bin/bash

#################### ROS1 ####################
source /opt/ros/noetic/setup.bash
source ./ros1_ws/devel/setup.bash

if [ $# -gt 0 ]; then
	export ROS_MASTER_IP=$1
    echo "ROS_MASTER_IP set to $ROS_MASTER_IP"
    source set_ros1_master.sh $ROS_MASTER_IP
else
    source set_ros1_master.sh 127.0.0.1
fi

if [ $# -gt 0 ]; then
	export ROS_IP=$2
    echo "ROS_IP set to $ROS_IP"
    source set_ros1_ip.sh $ROS_IP
else
    source set_ros1_ip.sh 127.0.0.1
fi

