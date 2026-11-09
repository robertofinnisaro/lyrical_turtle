"""Task 3 predator: pursuit guided by a learnt knowledge base.

* At start-up it loads ``predator-kb.xml`` if it exists.
* Every turn (``turn_period`` s) it records the predator and prey positions.
* With no knowledge it chases the prey's live position (trial 1).
* With knowledge it chases a *virtual target*: the prey position predicted
  by ``KnowledgeBase.predict`` for the moment it expects to arrive.
* It steers round the obstacles with ``avoidance_heading``.
* When /game_status arrives it appends this trial (moves + result) to the
  XML file so the next run can learn from it.
"""

import os

import rclpy
from geometry_msgs.msg import Point, Twist
from rclpy.node import Node
from std_msgs.msg import String

from .common import (Pose, SetPen, Spawn, TeleportAbsolute, avoidance_heading,
                     distance, heading_controller, latched_qos)
from .knowledge_base import KnowledgeBase
from .obstacles import OBSTACLES


class LearningPredator(Node):

    def __init__(self):
        super().__init__('predator')
        self.max_linear = self.declare_parameter('max_linear', 1.7).value
        self.turn_period = self.declare_parameter('turn_period', 0.5).value
        self.use_kb = self.declare_parameter('use_kb', True).value
        self.show_target = self.declare_parameter(
            'show_virtual_target', True).value
        kb_path = os.path.expanduser(
            self.declare_parameter('kb_path', 'predator-kb.xml').value)

        self.kb = KnowledgeBase(kb_path, OBSTACLES)
        n = self.kb.load()
        if n:
            self.get_logger().info(
                f'loaded {n} previous trial(s), {self.kb.num_samples} '
                f'learnt situations from {os.path.abspath(kb_path)}')
            for t in self.kb.trials:
                secs = len(t['moves']) * self.turn_period
                self.get_logger().info(
                    f'  trial {t["number"]}: {t["result"]} after '
                    f'~{secs:.1f} s ({len(t["moves"])} moves)')
        else:
            self.get_logger().info(
                f'no knowledge base at {os.path.abspath(kb_path)} - '
                'trial 1 uses plain pursuit')
        self.learning = self.use_kb and self.kb.num_samples > 0

        self.pose = None
        self.prey = None
        self.moves = []
        self.target = None
        self.done = False
        self.start = None
        self.marker_ready = False

        self.create_subscription(Pose, '/predator/pose', self.on_pose, 10)
        self.create_subscription(Pose, '/prey_location', self.on_prey, 10)
        self.create_subscription(String, '/game_status', self.on_status,
                                 latched_qos())
        self.cmd_pub = self.create_publisher(Twist, '/predator/cmd_vel', 10)
        self.location_pub = self.create_publisher(Pose, '/predator_location',
                                                  10)
        self.target_pub = self.create_publisher(
            Point, '/predator/virtual_target', 10)
        self.create_timer(self.turn_period, self.record_turn)
        self.create_timer(0.25, self.update_target)
        self.create_timer(0.05, self.control)

        if self.learning and self.show_target:
            self.spawn_cli = self.create_client(Spawn, '/spawn')
            self.pen_cli = self.create_client(SetPen,
                                              '/virtual_target/set_pen')
            self.tp_cli = self.create_client(
                TeleportAbsolute, '/virtual_target/teleport_absolute')
            self.marker_requested = False

    # ------------------------------------------------------------ callbacks
    def on_pose(self, msg):
        self.pose = msg
        self.location_pub.publish(msg)

    def on_prey(self, msg):
        self.prey = msg

    def on_status(self, msg):
        if self.done:
            return
        self.done = True
        self.cmd_pub.publish(Twist())
        self.record_turn(force=True)
        result = 'capture' if msg.data.startswith('capture') else 'escape'
        number = self.kb.append_trial(self.moves, result)
        elapsed = self.now() - self.start if self.start else 0.0
        self.get_logger().info(
            f'trial {number}: {result} after {elapsed:.1f} s, '
            f'{len(self.moves)} moves saved to {self.kb.path}')

    # ------------------------------------------------------------- learning
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def record_turn(self, force=False):
        if (self.done and not force) or self.pose is None or self.prey is None:
            return
        if self.start is None:
            self.start = self.now()
        self.moves.append((self.pose.x, self.pose.y, self.prey.x, self.prey.y))

    def update_target(self):
        if self.done or self.pose is None or self.prey is None:
            return
        if not self.learning or not self.moves:
            self.target = (self.prey.x, self.prey.y)
            return
        last = self.moves[-1]
        tx, ty, conf, horizon = self.kb.predict(
            self.pose.x, self.pose.y, self.prey.x, self.prey.y,
            self.prey.x - last[2], self.prey.y - last[3],
            self.max_linear * self.turn_period)
        # Once very close, the live position is the best target.
        if distance(self.pose.x, self.pose.y, self.prey.x, self.prey.y) < 1.0:
            tx, ty = self.prey.x, self.prey.y
        self.target = (tx, ty)
        self.target_pub.publish(Point(x=tx, y=ty, z=conf))
        self.get_logger().debug(
            f'virtual target ({tx:.2f},{ty:.2f}) conf={conf:.2f} '
            f'horizon={horizon} turns')
        self.show_marker(tx, ty)

    def show_marker(self, x, y):
        """Optional turtle drawn at the virtual target (pen up)."""
        if not (self.learning and self.show_target):
            return
        if not self.marker_requested:
            if not self.spawn_cli.service_is_ready():
                return
            self.marker_requested = True
            fut = self.spawn_cli.call_async(
                Spawn.Request(x=x, y=y, theta=0.0, name='virtual_target'))
            fut.add_done_callback(lambda _: self.pen_cli.call_async(
                SetPen.Request(r=0, g=0, b=0, width=1, off=1)
            ).add_done_callback(self._marker_ready))
            return
        if self.marker_ready:
            self.tp_cli.call_async(
                TeleportAbsolute.Request(x=x, y=y, theta=0.0))

    def _marker_ready(self, _):
        self.marker_ready = True

    # -------------------------------------------------------------- control
    def control(self):
        if self.done or self.pose is None or self.target is None:
            return
        tx, ty = self.target
        heading = avoidance_heading(self.pose.x, self.pose.y, tx, ty,
                                    OBSTACLES)
        linear, angular = heading_controller(self.pose.theta, heading,
                                             self.max_linear)
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = LearningPredator()
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
