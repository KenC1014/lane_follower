import rospy
from callbacks import waypoints_callback_helper
from std_msgs.msg import Int16MultiArray
from Line import Line


class action_unit():
    def __init__(self):
        # Subscribers
        # Waypoints topic
        self.waypoints = rospy.Subscriber('/lane_detection/waypoints', Int16MultiArray, self.waypoints_callback, queue_size=1)

        # Node states
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True
        self.turn = "front"

    def waypoints_callback(self, data):
        waypoints = waypoints_callback_helper(data)
        print("waypoints", waypoints)

    

if __name__ == '__main__':
    # init args
    rospy.init_node('action_node', anonymous=True)
    action_unit()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)