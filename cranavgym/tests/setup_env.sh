#!/bin/bash
killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient rviz python python3
# Set ROS environment variables
echo "Setting ROS environment variables..."
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_PORT_SIM=11311
export GAZEBO_RESOURCE_PATH=~/ros-rl-env/catkin_ws/src/multi_robot_scenario/launch
export ROSCONSOLE_CONFIG_FILE=~/cranfield-navigation-gym/cranavgym/ros_interface/rosconsole.config
export ROSCONSOLE_FORMAT='[${severity}] [${time}] [${logger}]: ${message}'

# Source the .bashrc file
echo "Sourcing ~/.bashrc..."
source ~/.bashrc

# Source the workspace setup.bash file
echo "Sourcing devel_isolated/setup.bash..."
source ~/ros-rl-env/catkin_ws/devel_isolated/setup.bash

# Change directory to the TD3 folder
# echo "Changing directory to the TD3 folder..."
cd ~/cranfield-navigation-gym/cranavgym/tests

# Add the root directory to the Python path
export PYTHONPATH=~/cranfield-navigation-gym:$PYTHONPATH

