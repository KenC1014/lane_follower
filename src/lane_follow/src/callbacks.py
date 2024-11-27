from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

def img_callback_helper(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    raw_img = cv_image.copy()
    return raw_img


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