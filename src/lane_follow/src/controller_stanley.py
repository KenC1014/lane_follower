import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from std_msgs.msg import Float32MultiArray, Int16MultiArray

import math
from utils import euler_to_quaternion, quaternion_to_euler
import time
from callbacks import waypoints_callback_helper
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
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt

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

        # PID for longitudinal control
        self.desired_speed = 0.6  # m/s
        self.max_accel     = 0.48 # % of acceleration
        self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)

        # GEM
        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)
        
        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.speed      = 0.0
        # self.stanley_pub = rospy.Publisher('/gem/stanley_gnss_cmd', AckermannDrive, queue_size=1)
        
        #stop sign
        self.sign_sub = rospy.Subscriber("/stop", String, self.sign_callback, queue_size=1)
        self.sign_value = False


        # SIM
        # self.speed      = 0.0
        # self.stanley_pub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size=1)

        # Control pub
        self.control_pub = rospy.Publisher("control", Int16MultiArray, queue_size=1)
        
        self.ackermann_msg                         = AckermannDrive()
        self.ackermann_msg.steering_angle_velocity = 0.0
        self.ackermann_msg.acceleration            = 0.0
        self.ackermann_msg.jerk                    = 0.0
        self.ackermann_msg.speed                   = 0.0 
        self.ackermann_msg.steering_angle          = 0.0

        # read waypoints into the system           
        self.read_waypoints() 

        self.waypoint_x1 = 0
        self.waypoint_y1 = 0
        self.waypoint_x2 = 0
        self.waypoint_y2 = 0

        self.waypoint_heading = 0

        # Hang 
        # self.steer = 0.0 # degrees
        # self.steer_sub = rospy.Subscriber("/pacmod/parsed_tx/steer_rpt", SystemRptFloat, self.steer_callback)
        
        # -------------------- PACMod setup --------------------

        self.gem_enable    = False
        self.pacmod_enable = False

        # GEM vehicle enable, publish once
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.enable_cmd = Bool()
        self.enable_cmd.data = False

        # GEM vehicle gear control, neutral, forward and reverse, publish once
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 2 # SHIFT_NEUTRAL

        # GEM vehilce brake control
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = False
        self.brake_cmd.clear  = True
        self.brake_cmd.ignore = True

        # GEM vechile forward motion control
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear  = True
        self.accel_cmd.ignore = True

        # GEM vechile turn signal control
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1 # None

        # GEM vechile steering wheel control
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
        self.steer_cmd.angular_velocity_limit = 5.0 # radians/second

    # Get vehicle speed
    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    # Get value of steering wheel
    def steer_callback(self, msg):
        self.steer = round(np.degrees(msg.output),1)

    def sign_callback(self, msg):
        self.sign = msg.data

    def waypoint_callback(self, data):
       
        new_wp = waypoints_callback_helper(data) #/ 10000
        print(new_wp)

        if len(new_wp) > 1:
            wp = new_wp
        else:
            wp = wp

        x2 = int((wp[4][0]+wp[3][0])/2)
        y2 = int((wp[4][1]+wp[3][1])/2)
        x1 = int((wp[1][0]+wp[2][0])/2)
        y1 = int((wp[1][1]+wp[2][1])/2)
        heading = np.arctan2(y2-y1, x2-x1)
        self.waypoint_x1 = x1
        self.waypoint_y1 = y1
        self.waypoint_x2 = x2
        self.waypoint_y2 = y2

        self.waypoint_heading = heading

    # Get predefined waypoints based on GNSS
    def read_waypoints(self):
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

    # Conversion to -pi to pi
    def pi_2_pi(self, angle):

        if angle > np.pi:
            return angle - 2.0 * np.pi

        if angle < -np.pi:
            return angle + 2.0 * np.pi

        return angle

    # Start Stanley controller
    def start_stanley(self):
        f_delta_prev = 0
        steering_angle = 0
        while not rospy.is_shutdown():

            # if (self.gem_enable == False):

            #     if(self.pacmod_enable == True):

    # ---------- enable PACMod ----------

            # enable forward gear
            self.gear_cmd.ui16_cmd = 3

            # enable brake
            self.brake_cmd.enable  = True
            self.brake_cmd.clear   = False
            self.brake_cmd.ignore  = False
            self.brake_cmd.f64_cmd = 0.0

            # enable gas 
            self.accel_cmd.enable  = True
            self.accel_cmd.clear   = False
            self.accel_cmd.ignore  = False
            self.accel_cmd.f64_cmd = 0.0

            self.gear_pub.publish(self.gear_cmd)
            print("Foward Engaged!")

            self.turn_pub.publish(self.turn_cmd)
            print("Turn Signal Ready!")
            
            self.brake_pub.publish(self.brake_cmd)
            print("Brake Engaged!")

            self.accel_pub.publish(self.accel_cmd)
            print("Gas Engaged!")

            self.gem_enable = True

            current_yaw = -np.pi/2

            # GEM
            curr_x = 640 #/ 10000
            curr_y = 720 #/ 10000
            
            # SIM
            # curr_x = 320 / 10000
            # curr_y = 480 / 10000

            x1 = self.waypoint_x1
            y1 = self.waypoint_y1
            x2 = self.waypoint_x2
            y2 = self.waypoint_y2


            print(x1, y1)
            print(x2, y2, self.waypoint_heading)

            e_numerator = (x2-x1) * (y1-curr_y) - (x1-curr_x) * (y2-y1)

            if e_numerator == 0:
                continue

            e_denominator = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            ct_error = e_numerator / e_denominator
            ct_error = float(ct_error)

            # heading error
            theta_e = self.pi_2_pi(self.waypoint_heading-current_yaw) 

            theta_e_deg = round(np.degrees(theta_e), 1)
            # print("Crosstrack Error: " + str(round(ct_error,3)) + ", Heading Error: " + str(theta_e_deg))

            # --------------------------- Longitudinal control using PD controller ---------------------------
            
            # SIM
            # curr_state = self.getModelState()
            # curr_x2, curr_y2, curr_vel, curr_yaw2 = self.extract_vehicle_info(curr_state)
            # self.speed = curr_vel
            # print(curr_vel)

            #rosbag speed
            # self.speed = 0.21

            filt_vel = np.squeeze(self.speed_filter.get_data(self.speed))
            # filt_vel = self.speed
            # print(filt_vel)
            
            ## object detection
            # if self.sign == "stop" & self.desired_speed > 0:
            #     self.state = False
            # elif self.sign == "stop" & filt_vel == 0:
            #     self.state = True

            self.state = True

            if self.state == True:
                if abs(steering_angle) < 3:
                    # In a straight line, accelerate smoothly
                    self.desired_speed = 0.6
                    self.brake = 0
                else:
                    # In a turn, reduce speed
                    self.desired_speed = 0.4
                    
                    # If speed is too high for the turn, apply brake
                    if filt_vel > self.desired_speed * 5:
                        self.brake = 0.5
                    else:
                        self.brake = 0
            else:
                self.desired_speed = 0
                self.brake = 0.2
            a_expected = self.pid_speed.get_control(rospy.get_time(), self.desired_speed - filt_vel)
            print(a_expected)
            

            if a_expected > 0.64 :
                throttle_percent = 0.5

            if a_expected < 0.0 :
                throttle_percent = 0.0

            throttle_percent = (a_expected+2.3501) / 7.3454
            print(throttle_percent)

            if throttle_percent > self.max_accel:
                throttle_percent = self.max_accel

            # if throttle_percent < 0.3:
            #     if self.brake == 0:
            #         throttle_percent = 0.37
            #     else:
            #         throttle_percent = (a_expected+2.3501) / 7.3454

            # -------------------------------------- Stanley controller --------------------------------------

            a_delta = np.arctan2(ct_error*0.0004, filt_vel)
            a_delta_deg = round(np.degrees(a_delta), 1)
            f_delta        = round(theta_e + a_delta, 3)
            # f_delta        = round(theta_e + np.arctan2(ct_error*0.4, filt_vel), 3)
            print("CT Error: " + str(round(ct_error,3)))
            print("CT delta Error: " + str(round(a_delta_deg,3)) + ", Heading Error: " + str(theta_e_deg))
            print("delta: " + str(round(f_delta,3)) + ", vel: " + str(filt_vel))
            print("brake: " + str(round(self.brake,3)) + ", throttle: " + str(throttle_percent))
            print("desired speed: " + str(round(self.desired_speed,3)))

            # print(f_delta_prev)
            f_delta        = round(np.clip(f_delta, -0.61, 0.61), 3)
            # print(f_delta)
            f_delta = np.clip(f_delta, f_delta_prev-0.2, f_delta_prev+0.2)
            # print(f_delta)
            if np.isnan(f_delta):
                f_delta_prev = 0
            else:
                f_delta_prev = f_delta

            f_delta_deg    = np.degrees(f_delta)

            # steering_angle = -f_delta_deg
            steering_angle = self.front2steer(-f_delta_deg)
            steering_angle = np.radians(steering_angle)

            if (filt_vel < 0.2):
                self.ackermann_msg.acceleration   = throttle_percent
                self.ackermann_msg.steering_angle = 0
                print("Steering angle: " + str(self.ackermann_msg.steering_angle))
            else:
                self.ackermann_msg.acceleration   = throttle_percent
                self.ackermann_msg.steering_angle = round(steering_angle,1)
                print("Steering angle: " + str(self.ackermann_msg.steering_angle))
            # print("YYYYYYYYY Steering angle: " + str(self.ackermann_msg.steering_angle))
            # self.ackermann_msg.steering_angle = 10
            
            # ------------------------------------------------------------------------------------------------ 

            # publish errors and commands
            control_data = Int16MultiArray()
            if not np.isnan(steering_angle):
                data = (ct_error, theta_e, steering_angle, self.brake, throttle_percent, x1, y1, x2, y2)
                # data = (ct_error*10000, theta_e, steering_angle, throttle_percent, x1*10000, y1*10000, x2*10000, y2*10000)

                control_data.data = tuple(map(int, data))
                self.control_pub.publish(control_data)
            
            # GEM
            self.accel_cmd.f64_cmd = self.ackermann_msg.acceleration
            self.steer_cmd.angular_position = self.ackermann_msg.steering_angle
            self.brake_cmd.f64_cmd = self.brake

            self.accel_pub.publish(self.accel_cmd)
            self.steer_pub.publish(self.steer_cmd)
            self.brake_pub.publish(self.brake_cmd)
            

            # SIM

            # if (abs(steering_angle) > 10):
            #     self.ackermann_mnt < 0.3:
            #     if self.brake == 0:
            #         throttle_percent = 0.37
            #     else:
            #         throttle_percent = (a_expesg.speed = 2
            # else:
            #     self.ackermann_msg.speed = 3

            # self.stanley_pub.publish(self.ackermann_msg)

            self.rate.sleep()

    def getModelState(self):
            # Get the current state of the vehicle
            # Input: None
            # Output: ModelState, the state of the vehicle, contain the
            #   position, orientation, linear velocity, angular velocity
            #   of the vehicle
            rospy.wait_for_service('/gazebo/get_model_state')
            try:
                serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
                resp = serviceResponse(model_name='gem')
            except rospy.ServiceException as exc:
                rospy.loginfo("Service did not process request: "+str(exc))
                resp = GetModelStateResponse()
                resp.success = False
            return resp
    def extract_vehicle_info(self, currentPose):

        ####################### TODO: Your TASK 1 code starts Here #######################
        pos_x, pos_y, vel, yaw = 0, 0, 0, 0
        pos_x = currentPose.pose.position.x
        pos_y = currentPose.pose.position.y
        pos_z = currentPose.pose.position.z
        vel_x = currentPose.twist.linear.x
        vel_y = currentPose.twist.linear.y
        vel_z = currentPose.twist.linear.z
        vel = np.sqrt(vel_x*vel_x+vel_y*vel_y+vel_z*vel_z)
        current_orientation = currentPose.pose.orientation
        roll, pitch, yaw = quaternion_to_euler(current_orientation.x, current_orientation.y, current_orientation.z, current_orientation.w)

        ####################### TODO: Your Task 1 code ends Here #######################

        return pos_x, pos_y, vel, yaw # note that yaw is in radian

    def stop(self):
        newAckermannCmd = AckermannDrive()
        # newAckermannCmd.speed = 0
        # self.accel_pub.publish(newAckermannCmd)
