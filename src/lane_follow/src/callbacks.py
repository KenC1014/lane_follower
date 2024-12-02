import numpy as np
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

def img_callback_helper(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    raw_img = cv_image.copy()
    return raw_img

def waypoints_callback_helper(data):
    data_points = np.array(data.data)
    waypoints = data_points.reshape(data_points.shape[0]//2, 2)
    return waypoints

def gnss_imu_callback_helper(data):
    try:
        print("\n")
        print("gnss_imu: ", data)
    except CvBridgeError as e:
        print(e)

def gnss_nav_callback_helper(data):
    try:
        print("\n")
        print("gnss_nav: ", data)
    except CvBridgeError as e:
        print(e)