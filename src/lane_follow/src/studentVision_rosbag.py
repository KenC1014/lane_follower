import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology


class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()

        # Front camera topic
        self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.img_front_callback, queue_size=1)
        # Left side camera topic
        self.left_image = rospy.Subscriber('/camera_fl/arena_camera_node/image_raw', Image, self.img_left_callback, queue_size=1)
        # Right side camera topic
        self.right_image = rospy.Subscriber('/camera_fr/arena_camera_node/image_raw', Image, self.img_right_callback, queue_size=1)

        # Run test
        # img = cv2.imread("test_left.jpg")
        # binary_img = self.combinedBinaryImage(img)
        # img_birdeye, M, Minv = self.perspective_transform(binary_img, 'left')
        # cv2.imwrite("test_transform.jpg", img_birdeye*255)

        # Publishers
        # front detection topic
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)

        # left detection topic
        self.pub_image_left = rospy.Publisher("left_lane_detection/annotate", Image, queue_size=1)
        self.pub_bird_left = rospy.Publisher("left_lane_detection/birdseye", Image, queue_size=1)

        # right detection topic
        self.pub_image_right = rospy.Publisher("right_lane_detection/annotate", Image, queue_size=1)
        self.pub_bird_right = rospy.Publisher("right_lane_detection/birdseye", Image, queue_size=1)

        # Others
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True


    def img_front_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        # cv2.imwrite(f"images/{time.time()}_front.jpg", raw_img)
        mask_image, bird_image = self.detection(raw_img, mode="front")

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)

    def img_left_callback(self, data):
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        # cv2.imwrite(f"images/{time.time()}_left_raw.jpg", raw_img)
        mask_image, bird_image = self.detection(raw_img, mode="left")

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image_left.publish(out_img_msg)
            self.pub_bird_left.publish(out_bird_msg)

    def img_right_callback(self, data):
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        # cv2.imwrite(f"images/{time.time()}_right_raw.jpg", raw_img)
        mask_image, bird_image = self.detection(raw_img, mode="right")

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image_right.publish(out_img_msg)
            self.pub_bird_right.publish(out_bird_msg)



    def gradient_thresh(self, img):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to find derivatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to uint8, then apply threshold to get binary image

        ## TODO
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
        sigma = 1
        ksize = 7
        img = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=sigma)
        img = cv2.medianBlur(img, ksize=3)
        # edge_x = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=7)
        # edge_y = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=7)
        # img_edges = cv2.addWeighted(edge_x, 0.5, edge_y, 0.5, 0)
        # sigma2 = 5
        # ksize2 = 31
        # img_edges = img - cv2.GaussianBlur(img, ksize=(ksize2, ksize2), sigmaX=sigma2)
        # img_edges[img_edges > 255] = 255
        # img_edges[img_edges < 0] = 0
        # cv2.imwrite("test_edges.jpg", img_edges.astype(np.uint8))
        img_edges = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=7)
        min_threshold = 128
        max_threshold = 255
        binary_output = np.zeros_like(img).astype(np.uint8)
        binary_output[(min_threshold <= img_edges) & (img_edges < max_threshold)] = 1
        ####

        return binary_output


    def color_thresh(self, img):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        #2. Apply threshold on S channel to get binary image
        #Hint: threshold on H to remove green grass
        ## TODO
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(img)
        min_s, max_s = (100, 255)
        binary_output = np.zeros_like(s).astype(np.uint8)
        binary_output[(min_s <= s) & (s <= max_s)] = 1
        min_h, max_h = (15, 85)
        mask_h = np.zeros_like(h).astype(np.uint8)
        mask_h[(min_h <= h) & (h <= max_h)] = 1
        binary_output[mask_h == 1] = 0
        ####

        return binary_output


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs
        ## Here you can use as many methods as you want.
        # cv2.imwrite("test_raw.jpg", img)
        ## TODO
        SobelOutput = self.gradient_thresh(img)
        ColorOutput = self.color_thresh(img)
        ####

        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 1
        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)

        return binaryImage


    def perspective_transform(self, img, mode="front", verbose=False):
        """
        Get bird's eye view from input image
        """
        #1. Visually determine 4 source points and 4 destination points
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        #3. Generate warped image in bird view using cv2.warpPerspective()

        ## TODO
        img = img.astype(np.uint8)
        h, w = img.shape
        # # test image
        # transform_points = [130, 180, 120, 0, 260, 160, 0, 10]
        # gazebo

        # front camera view
        transform_points = {"x_tl": 517,
                            "x_tr": 760,
                            "y_t": 427,
                            "x_bl": 302,
                            "x_br": 1050,
                            "y_b": 660,
                            "y_t_trans": 0,
                            "y_b_shift": 0
                            }
        
        # left camera view
        if mode == "left":
            transform_points = {"x_tl": 1667,
                            "x_tr": 1872,
                            "y_t": 574,
                            "x_bl": 1104,
                            "x_br": 1502,
                            "y_b": 1072,
                            "y_t_trans": 0,
                            "y_b_shift": 0
                            }
        # right camera view
        elif mode == "right":
           transform_points = {"x_tl": 28,
                            "x_tr": 246,
                            "y_t": 463,
                            "x_bl": 135,
                            "x_br": 853,
                            "y_b": 982,
                            "y_t_trans": 0,
                            "y_b_shift": 0
                            }

        
        x_tl = transform_points["x_tl"]
        x_tr = transform_points["x_tr"]
        y_t = transform_points["y_t"]
        x_bl = transform_points["x_bl"]
        x_br = transform_points["x_br"]
        y_b = transform_points["y_b"]
        x_l_trans = 100
        x_r_trans = w - 100
        y_t_trans = transform_points["y_t_trans"]
        y_b_shift = transform_points["y_b_shift"]
        if y_b + y_b_shift > h:
            y_b_shift = h - y_b
        y_b_trans = y_b + y_b_shift
        
        camera_points = np.array([[x_tl, y_t], [x_tr, y_t],[x_br, y_b], [x_bl, y_b]], dtype=np.float32)
        birdeye_points = np.array([[x_l_trans, y_t_trans], [x_r_trans, y_t_trans],
                                   [x_r_trans, y_b_trans], [x_l_trans, y_b_trans]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(camera_points, birdeye_points)
        Minv = cv2.getPerspectiveTransform(birdeye_points, camera_points)

     
        #cv2.imwrite("test_prebird.jpg", img * 255)
        warped_img = cv2.warpPerspective(img, M, dsize=(w, h))
        #cv2.imwrite("test_postbird.jpg", warped_img * 255)
        ####

        return warped_img, M, Minv


    def detection(self, img, mode="front"):
        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img, mode)

        h, w, _ = img.shape
        # img_birdeye_raw = cv2.warpPerspective(img, M, dsize=(w, h))
        # cv2.imwrite("test_bird.jpg", img_birdeye_raw)

        left_start=0
        left_end=None
        right_start=None
        right_end=None

        # if mode == "left":
        #     right_start=0
        #     right_end=1
        # elif mode == "right":
        #     left_start=0
        #     left_end=1

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye, left_start, left_end, right_start, right_end)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye, left_start, left_end, right_start, right_end)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, mode, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
