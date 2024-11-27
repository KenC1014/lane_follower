import rospy
from callbacks import img_callback_helper, gnss_imu_callback_helper
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
from Line import Line
from detection_utils import combinedBinaryImage, perspective_transform
from line_fit import line_fit, tune_fit, bird_fit, final_viz


class lanenet_detector():
    def __init__(self):
        self.bridge = CvBridge()

        # Subscribers
        # Front camera topic
        self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.img_front_callback, queue_size=1)
        # Left side camera topic
        self.left_image = rospy.Subscriber('/camera_fl/arena_camera_node/image_raw', Image, self.img_left_callback, queue_size=1)
        # Right side camera topic
        self.right_image = rospy.Subscriber('/camera_fr/arena_camera_node/image_raw', Image, self.img_right_callback, queue_size=1)
        # GNSS IMU topic
        self.gnss_imu = rospy.Subscriber('/septentrio_gnss/imu', Imu, self.gnss_imu_callback, queue_size=1)

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

        # Node states
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True

    def img_front_callback(self, data):
        raw_img = img_callback_helper(data)
        mask_image, bird_image = self.detection(raw_img, mode="front")
        
        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)

    def img_left_callback(self, data):
        raw_img = img_callback_helper(data)
        mask_image, bird_image = self.detection(raw_img, mode="left")

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image_left.publish(out_img_msg)
            self.pub_bird_left.publish(out_bird_msg)

    def img_right_callback(self, data):
        raw_img = img_callback_helper(data)
        mask_image, bird_image = self.detection(raw_img, mode="right")

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image_right.publish(out_img_msg)
            self.pub_bird_right.publish(out_bird_msg)

    def gnss_imu_callback(self, data):
        gnss_imu_callback_helper(data)


    def detection(self, img, mode="front"):
        binary_img = combinedBinaryImage(img)
        img_birdeye, M, Minv = perspective_transform(binary_img, mode)

        left_start=0
        left_end=None
        right_start=None
        right_end=None

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
