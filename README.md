# Lane Follower


## Getting started

```
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## Launch Gazebo & Rviz
```
roslaunch lane_follow lane_follow.launch
```

## Run lane detector for rosbags
```
python3 src/lane_follow/src/studentVision_rosbag.py
```

## Play rosbags
Make sure to create a directory under source called "bags" and put your rosbag files in there
```
cd src/lane_follow/src/bags
rosbag play -l <bagfile>
```

