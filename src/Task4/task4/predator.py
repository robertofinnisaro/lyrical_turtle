"""Task 4 predator (run three times as predator1/2/3).

Each predator acts independently: it only reads /prey_location and its own
pose, and aims at the prey's position extrapolated by its own ``lead``
factor (predator1 chases, predator3 intercepts).  Teammate positions are
read only to avoid colliding with them, and obstacles are steered round.
"""

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from .common import Pose, latched_qos
from .core import PREDATOR_LEAD, PREDATOR_NAMES, PREDATOR_SPEED, predator_command


class Predator(Node):

    def __init__(self):
        super().__init__('predator')
        self.name_ = self.declare_parameter('robot_name', 'predator1').value
        self.lead = self.declare_parameter(
            'lead', PREDATOR_LEAD.get(self.name_, 0.0)).value
        self.max_linear = self.declare_parameter(
            'max_linear', PREDATOR_SPEED).value
        self.pose = None
        self.prey = None
        self.mates = {}
        self.done = False

        self.create_subscription(Pose, f'/{self.name_}/pose', self.on_pose,
                                 10)
        self.create_subscription(Pose, '/prey_location', self.on_prey, 10)
        for mate in PREDATOR_NAMES:
            if mate != self.name_:
                self.create_subscription(
                    Pose, f'/{mate}_location',
                    lambda m, n=mate: self.mates.__setitem__(n, (m.x, m.y)),
                    10)
        self.create_subscription(String, '/game_status', self.on_status,
                                 latched_qos())
        self.cmd_pub = self.create_publisher(Twist, f'/{self.name_}/cmd_vel',
                                             10)
        self.location_pub = self.create_publisher(
            Pose, f'/{self.name_}_location', 10)
        self.create_timer(0.05, self.control)
        self.get_logger().info(f'{self.name_} hunting (lead={self.lead})')

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
        vx = self.prey.linear_velocity * math.cos(self.prey.theta)
        vy = self.prey.linear_velocity * math.sin(self.prey.theta)
        linear, angular = predator_command(
            self.pose.x, self.pose.y, self.pose.theta, self.prey.x,
            self.prey.y, vx, vy, list(self.mates.values()), self.lead,
            self.max_linear)
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
