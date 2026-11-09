"""Task 1 prey: stationary, publishes its world coordinates on /prey_location."""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .common import Pose, latched_qos


class StaticPrey(Node):

    def __init__(self):
        super().__init__('prey')
        self.pose = None
        self.done = False
        self.create_subscription(Pose, '/prey/pose', self.on_pose, 10)
        self.create_subscription(String, '/game_status', self.on_status,
                                 latched_qos())
        self.location_pub = self.create_publisher(Pose, '/prey_location', 10)
        self.create_timer(0.1, self.publish_location)

    def on_pose(self, msg):
        if self.pose is None:
            self.get_logger().info(
                f'prey sitting at ({msg.x:.2f}, {msg.y:.2f})')
        self.pose = msg

    def publish_location(self):
        if self.pose is not None:
            self.location_pub.publish(self.pose)

    def on_status(self, msg):
        self.get_logger().info(f'game over: {msg.data}')
        self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = StaticPrey()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
