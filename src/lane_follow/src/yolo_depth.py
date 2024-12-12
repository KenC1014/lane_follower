import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import message_filters

# Initialize ROS node
rospy.init_node('yolo_object_detection_with_distance', anonymous=True)

# Initialize CvBridge
bridge = CvBridge()

# Load YOLO model
model = YOLO('yolo11n.pt')  # Load pretrained YOLO model

# Define target classes
TARGET_CLASSES = ["person", "stop sign"]

# Create publisher for filtered video
output_pub_filtered = rospy.Publisher('/object_detection/output_video_filtered', Image, queue_size=10)

# Create publisher for "stop" messages
stop_pub = rospy.Publisher('/stop', String, queue_size=10)

# Callback for synchronized depth and RGB frames
def callback(depth_msg, rgb_msg):
    try:
        # Convert depth image to a NumPy array
        depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # Convert RGB image to OpenCV format
        rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

        # Perform object detection on the RGB frame
        results = model(rgb_image)
        result = results[0]

        # Process detections
        detections = result.boxes
        filtered_detections = []
        stop_flag = False

        for detection in detections:
            confidence = detection.conf[0]  # Confidence score
            cls_id = int(detection.cls[0])  # Class ID
            class_name = model.names[cls_id]  # Class name

            # Strictly process only filtered objects (person and stop sign) with confidence > 0.7
            if class_name in TARGET_CLASSES and confidence > 0.7:
                # Add detection to filtered detections
                filtered_detections.append(detection)

                # Extract bounding box coordinates
                x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])  # YOLO bounding box format

                # Calculate the center pixel coordinates of the bounding box
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2

                # Clip the coordinates to ensure they are within depth image bounds
                center_x = max(0, min(center_x, depth_image.shape[1] - 1))
                center_y = max(0, min(center_y, depth_image.shape[0] - 1))

                # Get the depth value at the center pixel
                center_depth = depth_image[center_y, center_x]

                # Check for valid depth values
                if not np.isfinite(center_depth) or center_depth <= 0:
                    rospy.logwarn(f"Invalid depth for {class_name} at center ({center_x}, {center_y}): {center_depth}")
                    continue

                # Log the depth value
                rospy.loginfo(f"Depth at center of {class_name}: {center_depth} meters (center: ({center_x}, {center_y}))")

                # Check if the depth is within the stop range (5m to 10m)
                if 5.0 <= center_depth <= 10.0:
                    rospy.loginfo(f"Object {class_name} within stop range: {center_depth} meters")
                    stop_flag = True

        # Publish "stop" message if any object is within the range
        if stop_flag:
            stop_pub.publish("stop")

        # Update and publish filtered detections video
        if filtered_detections:
            result.boxes = filtered_detections
            output_image_filtered = result.plot()
        else:
            output_image_filtered = rgb_image

        # Convert processed images back to ROS messages
        output_ros_image_filtered = bridge.cv2_to_imgmsg(output_image_filtered, encoding="bgr8")

        # Publish the processed filtered image
        output_pub_filtered.publish(output_ros_image_filtered)

    except Exception as e:
        rospy.logerr(f"Error processing messages: {e}")

# Subscriptions to depth and RGB topics
depth_sub = message_filters.Subscriber('/zed2/zed_node/depth/depth_registered', Image)
rgb_sub = message_filters.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image)

# Synchronize the subscriptions
sync = message_filters.ApproximateTimeSynchronizer([depth_sub, rgb_sub], queue_size=10, slop=0.1)
sync.registerCallback(callback)

# Keep the node running
rospy.spin()
