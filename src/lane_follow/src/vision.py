from unittest.mock import right

import rospy
from callbacks import img_callback_helper, gnss_imu_callback_helper, gnss_nav_callback_helper
from sensor_msgs.msg import Image, Imu, NavSatFix
from std_msgs.msg import Int16MultiArray
from cv_bridge import CvBridge
from Line import Line
from detection_utils import combinedBinaryImage, perspective_transform
from line_fit import line_fit, tune_fit, bird_fit, final_viz, control_viz
import cv2

import time

import numpy as np

# check point: import libs
import message_filters
from detection_utils import get_three_view_birdeye, get_three_view_forward, get_three_view_birdeye_trans_first


class lanenet_detector():
    def __init__(self):
        self.bridge = CvBridge()

        ######### vehicle model ##########
        self.model = "e4"  # "e2" or "e4"

        ######### view mode ##########
        self.view_mode = "three_views"  # "front_only" or "three_views"

        self.front_view = None
        self.left_view = None
        self.right_view = None


        # control topic
        self.control = []
        self.sub_control = rospy.Subscriber('control', Int16MultiArray, self.control_callback, queue_size=1)

        # Publishers
        # front detection topics
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        # three view topics
        self.pub_three_views = rospy.Publisher("three_views/annotate", Image, queue_size=1)
        self.pub_bird_three_views = rospy.Publisher("three_views/birdseye", Image, queue_size=1)
        self.pub_three_view_forward = rospy.Publisher("three_views/forward", Image, queue_size=1)

        # waypoint topic
        self.waypoints = rospy.Publisher("lane_detection/waypoints", Int16MultiArray, queue_size=1)

        # # Left side camera topic
        # self.left_image = rospy.Subscriber('/camera_fl/arena_camera_node/image_raw', Image, self.img_left_callback, queue_size=1)
        # # Right side camera topic
        # self.right_image = rospy.Subscriber('/camera_fr/arena_camera_node/image_raw', Image, self.img_right_callback, queue_size=1)
        # # GNSS IMU topic
        # self.gnss_imu = rospy.Subscriber('/septentrio_gnss/imu', Imu, self.gnss_imu_callback, queue_size=1)
        # # GNSS Nav topic
        # self.gnss_nav = rospy.Subscriber('/septentrio_gnss/navsatfix', NavSatFix, self.gnss_nav_callback, queue_size=1)

        # Node states
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True
        self.prev_wps = []
        self.centerx_current = None
        self.turn = "front"

        # for save image identification
        self.timestamp = time.time()


        # Subscribers
        # front view callback processing
        if self.view_mode == "front_only":
            if self.model == "e2":
                self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.img_front_callback,
                                                  queue_size=1)
            elif self.model == "e4":
                self.sub_image = rospy.Subscriber('/oak/rgb/image_raw', Image, self.img_front_callback, queue_size=1)
        # three view processing
        elif self.view_mode == "three_views":
            # if self.model == "e2":
            #     self.sub_front_image = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.get_front_img_callback, ueue_size=1)
            # elif self.model == "e4":
            #     self.sub_front_image = rospy.Subscriber('/oak/rgb/image_raw', Image, self.get_front_img_callback, queue_size=1)
            # self.sub_left_image = rospy.Subscriber('/camera_fl/arena_camera_node/image_raw', Image, self.get_left_img_callback, queue_size=1)
            # self.sub_right_image = rospy.Subscriber('/camera_fr/arena_camera_node/image_raw', Image, self.get_right_img_callback, queue_size=1)
            # # self.process_three_view()
            # while not rospy.core.is_shutdown():
            #     if self.front_view is not None and self.left_view is not None and self.right_view is not None:
            #         self.threw_view_callback(self.front_view, self.left_view, self.right_view)
            #     rospy.rostime.wallsleep(0.1)

            # use message_filters lib to subscribe and synchronize messages
            front_image = None
            if self.model == "e2":
                front_image = message_filters.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image)
            elif self.model == "e4":
                front_image = message_filters.Subscriber('/oak/rgb/image_raw', Image)
            left_image = message_filters.Subscriber('/camera_fl/arena_camera_node/image_raw', Image)
            right_image = message_filters.Subscriber('/camera_fr/arena_camera_node/image_raw', Image)

            assert front_image is not None, "Front image subscriber is not initialized."
            assert left_image is not None, "Left image subscriber is not initialized."
            assert right_image is not None, "Right image subscriber is not initialized."

            ats = message_filters.ApproximateTimeSynchronizer([front_image, left_image, right_image], queue_size=1000,
                                                            slop=0.5)
            ats.registerCallback(self.threw_view_callback)

            rospy.spin()
        

    def control_callback(self, data):
        self.control = np.array(data.data)

    ##### for three view processing from here #####
    def get_front_img_callback(self, data):
        self.front_view = img_callback_helper(data)
        # cv2.imwrite('/home/gem/anr_wins_ws/lane_follower-tests/src/lane_follow/src/front_view.jpg', image)
    
    def get_left_img_callback(self, data):
        self.left_view = img_callback_helper(data)

    def get_right_img_callback(self, data):
        self.right_view = img_callback_helper(data)

    # def process_three_view(self):
    #     # use message_filters lib to subscribe and synchronize messages
    #     front_image = None
    #     if self.model == "e2":
    #         front_image = message_filters.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image)
    #     elif self.model == "e4":
    #         front_image = message_filters.Subscriber('/oak/rgb/image_raw', Image)
    #     left_image = message_filters.Subscriber('/camera_fl/arena_camera_node/image_raw', Image)
    #     right_image = message_filters.Subscriber('/camera_fr/arena_camera_node/image_raw', Image)

    #     assert front_image is not None, "Front image subscriber is not initialized."
    #     assert left_image is not None, "Left image subscriber is not initialized."
    #     assert right_image is not None, "Right image subscriber is not initialized."

    #     ats = message_filters.ApproximateTimeSynchronizer([front_image, left_image, right_image], queue_size=20,
    #                                                       slop=0.2)
    #     ats.registerCallback(self.threw_view_callback)

    #     rospy.spin()

    def threw_view_callback(self, front_image, left_image, right_image):
        # if front_image is None:
        #     print("front none")
        #     return
        # if left_image is None:
        #     print("left none")
        #     return
        # if right_image is None:
        #     print("right none")
        #     return
        # if front_image is None or left_image is None or right_image is None:
        #     return
            
        front_image = img_callback_helper(front_image)
        left_image = img_callback_helper(left_image)
        right_image = img_callback_helper(right_image)

        mask_image, bird_image, waypoints, three_view_forward_image = self.three_view_detection(front_image, left_image, right_image)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')
            
            # cv2.imwrite('src/lane_follow/src/mask_image.jpg', mask_image)

            # Publish image message in ROS
            self.pub_three_views.publish(out_img_msg)
            self.pub_bird_three_views.publish(out_bird_msg)
            waypoint_topic = Int16MultiArray()
            waypoint_topic.data = waypoints
            self.waypoints.publish(waypoint_topic)

            ## forward view three view image
            if three_view_forward_image is not None:
                three_view_forward_image = cv2.normalize(three_view_forward_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                out_forward_msg = self.bridge.cv2_to_imgmsg(three_view_forward_image.astype(np.uint8), 'bgr8')
                self.pub_three_view_forward.publish(out_forward_msg)

    def three_view_detection(self, front_image, left_image, right_image):
        mode = "front"
        output_h, output_w = front_image.shape[:2]
        if self.model == "e2":
            output_h, output_w = 2880, 1920
        elif self.model == "e4":
            output_h, output_w = 3360, 2400
        scale = 0.25
        img = front_image.astype(np.uint8)

        ##### method 1: edge detection before transformation. simple but left and right front camera lane detection peform worse.
        # binary_front = combinedBinaryImage(img)
        # binary_left = combinedBinaryImage(left_image.astype(np.uint8))
        # binary_right = combinedBinaryImage(right_image.astype(np.uint8))
        # img_birdeye, M, Minv = get_three_view_birdeye(binary_front, binary_left, binary_right, output_h=output_h, output_w=output_w, scale=scale, model=self.model)

        # cv2.imwrite("src/lane_follow/src/test_front.jpg", binary_front * 255)
        # cv2.imwrite("src/lane_follow/src/test_left.jpg", binary_left * 255)
        # cv2.imwrite("src/lane_follow/src/test_right.jpg", binary_right * 255)

        ##### method 2: transform before edge detection. better lane detection, more computational cost.
        img_birdeye, M, Minv = get_three_view_birdeye_trans_first(img, left_image, right_image,
                                                                 output_h=output_h, output_w=output_w, 
                                                                 scale=scale, model=self.model)
        print("birdeye shape", img_birdeye.shape)
        # cv2.imwrite(f'src/lane_follow/src/record/three_view_bird_binary_{self.timestamp}.jpg', img_birdeye * 255)

        three_view_forward_image = None
        ######## for visualize birdeye transformation only, do not open for high efficiency real-time demand, computational costly
        three_view_birdeye, M, Minv = get_three_view_birdeye(front_image, left_image, right_image,
                                                              output_h=output_h, output_w=output_w, scale=scale, model=self.model)
        front_h, front_w = front_image.shape[:2]
        out_size = (int(front_h * 1.5), int(front_w * 1.5))
        three_view_forward_image = get_three_view_forward(three_view_birdeye, front_h, front_w, Minv, out_size=out_size)
        print("forward shape", three_view_forward_image.shape)
        # cv2.imwrite(f'src/lane_follow/src/record/three_view_birdeye_{self.timestamp}.jpg', three_view_birdeye)
        # cv2.imwrite('src/lane_follow/src/three_view_forward.jpg', three_view_forward_image)

        combine_fit_img, bird_fit_img, waypoints = self.detection(img, img_birdeye, M, Minv, mode="front")

        return combine_fit_img, bird_fit_img, waypoints, three_view_forward_image

    ##### end three view processing #####

    ##### for front view only #####
    def img_front_callback(self, data):
        raw_img = img_callback_helper(data)

        # cv2.imwrite("src/lane_follow/src/raw_image_front.jpg", raw_img)

        mask_image, bird_image, waypoints = self.front_only_detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)
            waypoint_topic = Int16MultiArray()
            waypoint_topic.data = waypoints
            self.waypoints.publish(waypoint_topic)

    def front_only_detection(self, img):
        binary_img = combinedBinaryImage(img)
        img_birdeye, M, Minv = perspective_transform(binary_img, model=self.model)

        combine_fit_img, bird_fit_img, waypoints = self.detection(img, img_birdeye, M, Minv, mode="front")

        return combine_fit_img, bird_fit_img, waypoints

    ##### end front view only #####

    ##### work for both view modes
    def detection(self, img, img_birdeye, M, Minv, mode="front"):
        left_start = 0
        left_end = None
        right_start = None
        right_end = None
        centerx_current = self.centerx_current
        waypoints = []
        wps_left = []
        wps_right = []
        turn = self.turn

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye, left_start, left_end, right_start, right_end, self.prev_wps, centerx_current,
                           self.turn)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            centerx_current = ret['centerx_current']
            waypoints = ret['waypoints']
            wps_left = ret['wps_left']
            wps_right = ret['wps_right']
            turn = ret["turn"]

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye, left_start, left_end, right_start, right_end, self.prev_wps,
                               centerx_current, self.turn)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    centerx_current = ret['centerx_current']
                    waypoints = ret['waypoints']
                    wps_left = ret['wps_left']
                    wps_right = ret['wps_right']
                    turn = ret["turn"]

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    # Update previous waypoints
                    self.prev_wps = waypoints

                    # Update center
                    self.centerx_current = centerx_current

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    centerx_current = ret['centerx_current']
                    waypoints = ret['waypoints']
                    wps_left = ret['wps_left']
                    wps_right = ret['wps_right']
                    turn = ret["turn"]

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    # Update previous waypoints
                    self.prev_wps = waypoints

                    # Update center
                    self.centerx_current = centerx_current

                else:
                    self.detected = False

            # Update sharp turn status
            self.turn = turn

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None          

            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, mode, save_file=None)
                if len(self.control) > 0:
                    bird_fit_img = control_viz(bird_fit_img, self.control)
                combine_fit_img = final_viz(img, Minv, waypoints, wps_left, wps_right, self.turn)
            else:
                print("Unable to detect lanes")

            # publish the original image if no lane detected
            if bird_fit_img is None:
                img_birdeye = cv2.normalize(img_birdeye, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                if len(img_birdeye.shape) == 2:  # Single-channel grayscale
                    bird_fit_img = cv2.cvtColor(img_birdeye, cv2.COLOR_GRAY2BGR)
                else:  # Multi-channel, no conversion needed
                    bird_fit_img = img
            if combine_fit_img is None:
                if len(img.shape) == 2:  # Single-channel grayscale
                    combine_fit_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:  # Multi-channel, no conversion needed
                    combine_fit_img = img

            return combine_fit_img, bird_fit_img, waypoints


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
