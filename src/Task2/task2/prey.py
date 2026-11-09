"""Task 2 prey: flee from the predator while steering clear of the walls.

Every planning tick (5 Hz) the prey reads /predator_location, asks
``prey_logic.plan_escape`` for the best move (x, y, theta) and then drives
towards that point with a heading controller at 20 Hz.
"""

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from .common import Pose, cruise_controller, distance, latched_qos
from .prey_logic import plan_escape


class Prey(Node):

    def __init__(self):
        super().__init__('prey')
        self.max_linear = self.declare_parameter('max_linear', 1.45).value
        self.pose = None
        self.predator = None
        self.move = None
        self.done = False

        self.create_subscription(Pose, '/prey/pose', self.on_pose, 10)
        self.create_subscription(Pose, '/predator_location', self.on_predator,
                                 10)
        self.create_subscription(String, '/game_status', self.on_status,
                                 latched_qos())
        self.cmd_pub = self.create_publisher(Twist, '/prey/cmd_vel', 10)
        self.location_pub = self.create_publisher(Pose, '/prey_location', 10)
        self.create_timer(0.2, self.plan)
        self.create_timer(0.05, self.control)

    def on_pose(self, msg):
        self.pose = msg
        self.location_pub.publish(msg)

    def on_predator(self, msg):
        self.predator = msg

    def on_status(self, msg):
        self.get_logger().info(f'game over: {msg.data}')
        self.cmd_pub.publish(Twist())
        self.done = True

    def plan(self):
        if self.done or self.pose is None or self.predator is None:
            return
        threats = [(self.predator.x, self.predator.y)]
        previous = self.move[2] if self.move is not None else None
        # threat_step: how far the predator moves while we step 1.2 m.
        self.move, score = plan_escape(self.pose.x, self.pose.y,
                                       self.pose.theta, threats,
                                       threat_step=1.25, previous=previous)
        x, y, theta = self.move
        self.get_logger().debug(
            f'move -> ({x:.2f}, {y:.2f}, {theta:.2f}) score {score:.2f}')

    def control(self):
        if self.done or self.pose is None or self.move is None:
            return
        x, y, theta = self.move
        linear, angular = cruise_controller(self.pose.theta, theta,
                                            self.max_linear)
        # Slow down a little when the predator is far away (saves "energy").
        d = distance(self.pose.x, self.pose.y, self.predator.x,
                     self.predator.y)
        if d > 6.0:
            linear *= 0.6
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = Prey()
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
