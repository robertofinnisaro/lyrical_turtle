"""Referee: ends the game on capture or when the time limit expires.

It reads ground-truth poses straight from turtlesim, starts the clock once
every robot exists, and publishes the outcome on the latched /game_status
topic ("capture:<predator>" or "timeout").  All other nodes stop when they
see it, and the referee also sends zero velocities to every turtle.
"""

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from .common import Pose, distance, latched_qos


class Referee(Node):

    def __init__(self):
        super().__init__('referee')
        self.prey_name = self.declare_parameter('prey', 'prey').value
        self.predator_names = list(self.declare_parameter(
            'predators', ['predator']).value)
        self.capture_distance = self.declare_parameter(
            'capture_distance', 0.5).value
        self.time_limit = self.declare_parameter('time_limit', 60.0).value

        self.poses = {}
        self.start = None
        self.last_announce = 0
        self.done = False
        for name in [self.prey_name] + self.predator_names:
            self.create_subscription(
                Pose, f'/{name}/pose',
                lambda msg, n=name: self.poses.__setitem__(n, msg), 10)
        self.cmd_pubs = {
            name: self.create_publisher(Twist, f'/{name}/cmd_vel', 10)
            for name in [self.prey_name] + self.predator_names}
        self.status_pub = self.create_publisher(String, '/game_status',
                                                latched_qos())
        self.create_timer(0.05, self.check)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def check(self):
        if self.done:
            return
        if len(self.poses) < len(self.cmd_pubs):
            return
        if self.start is None:
            self.start = self.now()
            self.get_logger().info(
                f'all robots present - {self.time_limit:.0f} s clock started')
        elapsed = self.now() - self.start

        prey = self.poses[self.prey_name]
        for name in self.predator_names:
            p = self.poses[name]
            if distance(p.x, p.y, prey.x, prey.y) < self.capture_distance:
                self.finish(f'capture:{name}',
                            f'{name} CAPTURED the prey after {elapsed:.1f} s')
                return
        if elapsed >= self.time_limit:
            self.finish('timeout',
                        f'prey ESCAPED - {self.time_limit:.0f} s elapsed')
            return
        if int(elapsed) // 10 > self.last_announce:
            self.last_announce = int(elapsed) // 10
            self.get_logger().info(f'{elapsed:.0f} s elapsed')

    def finish(self, status, text):
        self.done = True
        self.get_logger().info(text)
        self.status_pub.publish(String(data=status))
        for pub in self.cmd_pubs.values():
            pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = Referee()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        # Keep stopping the turtles briefly so the final frame is static.
        end = node.now() + 1.0
        while rclpy.ok() and node.now() < end:
            for pub in node.cmd_pubs.values():
                pub.publish(Twist())
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
