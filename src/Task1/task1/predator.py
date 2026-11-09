"""Task 1 predator: drive to the location published on /prey_location.

Each control tick computes the Euclidean distance and bearing to the prey,
turns towards it with a proportional controller, and stops (ending the
simulation) once it is within ``capture_distance``.
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
        self.max_linear = self.declare_parameter('max_linear', 2.0).value
        self.capture_distance = self.declare_parameter(
            'capture_distance', 0.5).value
        self.pose = None
        self.prey = None
        self.done = False
        self.start_time = None
        self.last_log = 0.0

        self.create_subscription(Pose, '/predator/pose', self.on_pose, 10)
        self.create_subscription(Pose, '/prey_location', self.on_prey, 10)
        self.cmd_pub = self.create_publisher(Twist, '/predator/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/game_status',
                                                latched_qos())
        self.create_timer(0.05, self.control)

    def on_pose(self, msg):
        self.pose = msg

    def on_prey(self, msg):
        self.prey = msg

    def control(self):
        if self.done or self.pose is None or self.prey is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.start_time is None:
            self.start_time = now
            self.get_logger().info('prey located - starting pursuit')

        dx = self.prey.x - self.pose.x
        dy = self.prey.y - self.pose.y
        dist = math.hypot(dx, dy)            # Euclidean distance
        angle = math.atan2(dy, dx)           # bearing to the prey

        if now - self.last_log > 1.0:
            self.last_log = now
            self.get_logger().info(
                f'distance={dist:.2f} m  bearing={math.degrees(angle):.0f} deg')

        if dist < self.capture_distance:
            self.cmd_pub.publish(Twist())
            elapsed = now - self.start_time
            self.get_logger().info(
                f'CAPTURED the prey in {elapsed:.1f} s - stopping simulation')
            self.status_pub.publish(String(data='capture'))
            self.done = True
            return

        linear, angular = heading_controller(
            self.pose.theta, angle, self.max_linear, distance_to_goal=dist)
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
        # Flush the stop command / status before exiting.
        for _ in range(5):
            node.cmd_pub.publish(Twist())
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
