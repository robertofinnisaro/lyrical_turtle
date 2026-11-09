"""Task 2 predator: pure pursuit of the prey's live position.

Subscribes to /prey_location, publishes its own pose on /predator_location
(so the prey can run away from it) and stops when /game_status arrives.
"""

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from .common import Pose, heading_controller, latched_qos


class Predator(Node):

    def __init__(self):
        super().__init__('predator')
        self.max_linear = self.declare_parameter('max_linear', 1.5).value
        self.pose = None
        self.prey = None
        self.done = False

        self.create_subscription(Pose, '/predator/pose', self.on_pose, 10)
        self.create_subscription(Pose, '/prey_location', self.on_prey, 10)
        self.create_subscription(String, '/game_status', self.on_status,
                                 latched_qos())
        self.cmd_pub = self.create_publisher(Twist, '/predator/cmd_vel', 10)
        self.location_pub = self.create_publisher(Pose, '/predator_location',
                                                  10)
        self.create_timer(0.05, self.control)

    def on_pose(self, msg):
        self.pose = msg
        self.location_pub.publish(msg)

    def on_prey(self, msg):
        self.prey = msg

    def on_status(self, msg):
        self.get_logger().info(f'game over: {msg.data}')
        self.cmd_pub.publish(Twist())
        self.done = True

    def control(self):
        if self.done or self.pose is None or self.prey is None:
            return
        dx = self.prey.x - self.pose.x
        dy = self.prey.y - self.pose.y
        linear, angular = heading_controller(
            self.pose.theta, math.atan2(dy, dx), self.max_linear)
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = Predator()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.cmd_pub.publish(Twist())
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
