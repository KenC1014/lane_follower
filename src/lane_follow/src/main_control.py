import rospy
import numpy as np
import argparse

from gazebo_msgs.msg import  ModelState
from controller_stanley import Stanley
import time
from utils import euler_to_quaternion, quaternion_to_euler
from matplotlib import pyplot as plt

def run_model():
    rospy.init_node("model_dynamics")
    controller = Stanley()

    def shutdown():
        """Stop the car when this ROS node shuts down"""
        controller.stop()
        rospy.loginfo("Stop the car")

    rospy.on_shutdown(shutdown)

    rate = rospy.Rate(100)  # 100 Hz
    rospy.sleep(0.0)
    start_time = rospy.Time.now()
    prev_wp_time = start_time

    while not rospy.is_shutdown():
        rate.sleep()  # Wait a while before trying to get a new state

        try:
            controller.start_stanley()
        except rospy.ROSInterruptException:
            pass

if __name__ == "__main__":
    try:
        status, num_waypoints, time_taken = run_model()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutting down")