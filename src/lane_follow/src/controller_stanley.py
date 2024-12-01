import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from std_msgs.msg import Float32MultiArray, Int16MultiArray

import math
from utils import euler_to_quaternion, quaternion_to_euler
import time
#!/usr/bin/env python3

# Python Headers
import os 
import csv
from numpy import linalg as la
import scipy.signal as signal

# ROS Headers
# import alvinxy.alvinxy as axy 
from std_msgs.msg import String, Bool, Float32, Float64
# from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# GEM PACMod Headers
# from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt


#==============================================================================
# File name          : gem_gnss_tracker_stanley_rtk.py                                                                  
# Description        : gnss waypoints tracker using pid and Stanley controller                                                              
# Author             : Hang Cui (hangcui3@illinois.edu)                                       
# Date created       : 08/08/2022                                                                 
# Date last modified : 08/18/2022                                                          
# Version            : 1.0                                                                    
# Usage              : rosrun gem_gnss_control gem_gnss_tracker_stanley_rtk.py                                                                      
# Python version     : 3.8   
# Longitudinal ctrl  : Ji'an Pan (pja96@illinois.edu), Peng Hang (penghan2@illinois.edu)                                                            
#==============================================================================



class OnlineFilter(object):

    def __init__(self, cutoff, fs, order):
        
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        # Get the filter coct_errorficients 
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        # Initialize
        self.z = signal.lfilter_zi(self.b, self.a)
    
    def get_data(self, data):

        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted


class PID(object):

    def __init__(self, kp, ki, kd, wg=None):

        self.iterm  = 0
        self.last_t = None
        self.last_e = 0
        self.kp     = kp
        self.ki     = ki
        self.kd     = kd
        self.wg     = wg
        self.derror = 0

    def reset(self):
        self.iterm  = 0
        self.last_e = 0
        self.last_t = None

    def get_control(self, t, e, fwd=0):

        if self.last_t is None:
            self.last_t = t
            de = 0
        else:
            de = (e - self.last_e) / (t - self.last_t)

        if abs(e - self.last_e) > 0.5:
            de = 0

        self.iterm += e * (t - self.last_t)

        # take care of integral winding-up
        if self.wg is not None:
            if self.iterm > self.wg:
                self.iterm = self.wg
            elif self.iterm < -self.wg:
                self.iterm = -self.wg

        self.last_e = e
        self.last_t = t
        self.derror = de

        return fwd + self.kp * e + self.ki * self.iterm + self.kd * de


class Stanley(object):
    
    def __init__(self):

        self.rate   = rospy.Rate(30)

        self.olat   = 40.0928232 
        self.olon   = -88.2355788

        self.offset = 1.1 # meters

        # PID for longitudinal control
        self.desired_speed = 0.6  # m/s
        self.max_accel     = 0.48 # % of acceleration
        self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)

        # self.gnss_sub   = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)
        # self.lat        = 0.0
        # self.lon        = 0.0
        # self.heading    = 0.0


        #gem
        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        #gazebo
        #self.speed      = 0.0

        #gem
        self.stanley_pub = rospy.Publisher('/gem/stanley_gnss_cmd', AckermannDrive, queue_size=1)
        #gazebo
        #self.stanley_pub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size=1)

        self.ackermann_msg                         = AckermannDrive()
        self.ackermann_msg.steering_angle_velocity = 0.0
        self.ackermann_msg.acceleration            = 0.0
        self.ackermann_msg.jerk                    = 0.0
        self.ackermann_msg.speed                   = 0.0 
        self.ackermann_msg.steering_angle          = 0.0

        # read waypoints into the system           
        self.read_waypoints() 

        self.waypoint_x = 0
        self.waypoint_y = 0
        self.waypoint_x = 0
        self.waypoint_y = 0
        self.waypoint_heading = 0

        # Hang 
        # self.steer = 0.0 # degrees
        # self.steer_sub = rospy.Subscriber("/pacmod/parsed_tx/steer_rpt", SystemRptFloat, self.steer_callback)

    # # Get GNSS information
    # def inspva_callback(self, inspva_msg):
    #     self.lat     = inspva_msg.latitude  # latitude
    #     self.lon     = inspva_msg.longitude # longitude
    #     self.heading = inspva_msg.azimuth   # heading in degrees


    # Get vehicle speed
    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s


    # Get value of steering wheel
    def steer_callback(self, msg):
        self.steer = round(np.degrees(msg.output),1)

    def waypoint_callback(self, data):
        x_coords = np.array(data.data)[0::2]
        y_coords = np.array(data.data)[1::2]

        x2 = (x_coords[0] + x_coords[1]) / 2
        x1 = (x_coords[3] + x_coords[2]) / 2

        y2 = (y_coords[0] + y_coords[1]) / 2
        y1 = (y_coords[2] + y_coords[3]) / 2

        heading = np.arctan2(y2-y1, x2-x1)

        self.waypoint_x = x2
        self.waypoint_y = y2
        self.waypoint_heading = heading

    # Get predefined waypoints based on GNSS
    def read_waypoints(self):

        # read recorded GPS lat, lon, heading
        # dirname  = os.path.dirname(__file__)
        # filename = os.path.join(dirname, './wps.csv')

        # with open(filename) as f:
        #     path_points = [tuple(line) for line in csv.reader(f)]

        # # x towards East and y towards North
        # self.path_points_lon_x   = [float(point[0]) for point in path_points] # longitude
        # self.path_points_lat_y   = [float(point[1]) for point in path_points] # latitude
        # self.path_points_heading = [float(point[2]) for point in path_points] # heading

        # subscribe to waypoints
        self.sub_waypoints = rospy.Subscriber('lane_detection/waypoints', Int16MultiArray, self.waypoint_callback, queue_size=1)

    # Conversion of front wheel to steering wheel
    def front2steer(self, f_angle):
        if(f_angle > 35):
            f_angle = 35
        if (f_angle < -35):
            f_angle = -35
        if (f_angle > 0):
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        elif (f_angle < 0):
            f_angle = -f_angle
            steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        else:
            steer_angle = 0.0
        return steer_angle


    # # Conversion of Lon & Lat to X & Y
    # def wps_to_local_xy_stanley(self, lon_wp, lat_wp):
    #     # convert GNSS waypoints into local fixed frame reprented in x and y
    #     lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
    #     return -lon_wp_x, -lat_wp_y   


    # # Conversion of GNSS heading to vehicle heading
    # def heading_to_yaw_stanley(self, heading_curr):
    #     if (heading_curr >= 0 and heading_curr < 90):
    #         yaw_curr = np.radians(-heading_curr-90)
    #     else:
    #         yaw_curr = np.radians(-heading_curr+270)
    #     return yaw_curr


    # # Get vehicle states: x, y, yaw
    # def get_gem_state(self):

    #     # vehicle gnss heading (yaw) in degrees
    #     # vehicle x, y position in fixed local frame, in meters
    #     # rct_errorerence point is located at the center of GNSS antennas
    #     local_x_curr, local_y_curr = self.wps_to_local_xy_stanley(self.lon, self.lat)

    #     # heading to yaw (degrees to radians)
    #     # heading is calculated from two GNSS antennas
    #     curr_yaw = self.heading_to_yaw_stanley(self.heading) 

    #     # rct_errorerence point is located at the center of front axle
    #     curr_x = local_x_curr + self.offset * np.cos(curr_yaw)
    #     curr_y = local_y_curr + self.offset * np.sin(curr_yaw)

    #     return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)


    # # Find close yaw in predefined GNSS waypoint list
    # def find_close_yaw(self, arr, val):
    #     diff_arr = np.array( np.abs( np.abs(arr) - np.abs(val) ) )
    #     idx = np.where(diff_arr < 0.5)
    #     return idx


    # Conversion to -pi to pi
    def pi_2_pi(self, angle):

        if angle > np.pi:
            return angle - 2.0 * np.pi

        if angle < -np.pi:
            return angle + 2.0 * np.pi

        return angle

    # # Computes the Euclidean distance between two 2D points
    # def dist(self, p1, p2):
    #     return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

    # Start Stanley controller
    def start_stanley(self):
        
        while not rospy.is_shutdown():

            current_yaw = -np.pi/2
            curr_x = 320
            curr_y = 480
            vec_target_2_front = np.array([[curr_x-self.waypoint_x],[curr_y-self.waypoint_y]])
            front_axle_vec_rot_90 = np.array([[np.cos(current_yaw - np.pi / 2.0)], [np.sin(current_yaw - np.pi / 2.0)]])

            # crosstrack error
            ct_error = np.dot(vec_target_2_front.T, front_axle_vec_rot_90)
            ct_error = float(np.squeeze(ct_error))

            # heading error
            theta_e = self.pi_2_pi(self.waypoint_heading-current_yaw) 

            # theta_e = self.target_path_points_yaw[target_point_idx]-curr_yaw 
            theta_e_deg = round(np.degrees(theta_e), 1)
            print("Crosstrack Error: " + str(round(ct_error,3)) + ", Heading Error: " + str(theta_e_deg))

            # --------------------------- Longitudinal control using PD controller ---------------------------

            filt_vel = np.squeeze(self.speed_filter.get_data(self.speed))
            # print(filt_vel)
            a_expected = self.pid_speed.get_control(rospy.get_time(), self.desired_speed - filt_vel)
            # print(a_expected)

            if a_expected > 0.64 :
                throttle_percent = 0.5

            if a_expected < 0.0 :
                throttle_percent = 0.0

            throttle_percent = (a_expected+2.3501) / 7.3454

            if throttle_percent > self.max_accel:
                throttle_percent = self.max_accel

            if throttle_percent < 0.3:
                throttle_percent = 0.37

            # -------------------------------------- Stanley controller --------------------------------------

            f_delta        = round(theta_e + np.arctan2(ct_error*0.4, filt_vel), 3)
            f_delta        = round(np.clip(f_delta, -0.61, 0.61), 3)
            f_delta_deg    = np.degrees(f_delta)
            # steering_angle = f_delta_deg
            steering_angle = self.front2steer(f_delta_deg)

            if (filt_vel < 0.2):
                self.ackermann_msg.acceleration   = throttle_percent
                self.ackermann_msg.steering_angle = 0
                print("Steering angle: " + str(self.ackermann_msg.steering_angle))
            else:
                self.ackermann_msg.acceleration   = throttle_percent
                self.ackermann_msg.steering_angle = round(steering_angle,1)
                print("Steering angle: " + str(self.ackermann_msg.steering_angle))
            # print(self.ackermann_msg.acceleration)
            
            if (steering_angle > 0.5):
                self.ackermann_msg.speed = 2
            else:
                self.ackermann_msg.speed = 4
            # ------------------------------------------------------------------------------------------------ 

            self.stanley_pub.publish(self.ackermann_msg)

            self.rate.sleep()

    # def getModelState(self):
    #         # Get the current state of the vehicle
    #         # Input: None
    #         # Output: ModelState, the state of the vehicle, contain the
    #         #   position, orientation, linear velocity, angular velocity
    #         #   of the vehicle
    #         rospy.wait_for_service('/gazebo/get_model_state')
    #         try:
    #             serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    #             resp = serviceResponse(model_name='gem')
    #         except rospy.ServiceException as exc:
    #             rospy.loginfo("Service did not process request: "+str(exc))
    #             resp = GetModelStateResponse()
    #             resp.success = False
    #         return resp
    # def extract_vehicle_info(self, currentPose):

    #     ####################### TODO: Your TASK 1 code starts Here #######################
    #     pos_x, pos_y, vel, yaw = 0, 0, 0, 0
    #     pos_x = currentPose.pose.position.x
    #     pos_y = currentPose.pose.position.y
    #     pos_z = currentPose.pose.position.z
    #     vel_x = currentPose.twist.linear.x
    #     vel_y = currentPose.twist.linear.y
    #     vel_z = currentPose.twist.linear.z
    #     vel = np.sqrt(vel_x*vel_x+vel_y*vel_y+vel_z*vel_z)
    #     current_orientation = currentPose.pose.orientation
    #     roll, pitch, yaw = quaternion_to_euler(current_orientation.x, current_orientation.y, current_orientation.z, current_orientation.w)

    #     ####################### TODO: Your Task 1 code ends Here #######################

    #     return pos_x, pos_y, vel, yaw # note that yaw is in radian
    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = 0
        self.stanley_pub.publish(newAckermannCmd)
