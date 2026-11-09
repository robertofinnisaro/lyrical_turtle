"""Task 4 prey: learns to evade three predators with Q-learning.

* Loads its Q-table from ``prey_kb_path`` (default ``prey-kb.json``) if it
  exists, so it keeps improving from run to run.  Otherwise it starts from
  the pre-trained table shipped in the package (``use_pretrained:=false``
  starts from scratch instead).  Pre-train it quickly with
  ``ros2 run task4 train_prey`` (same code, offline simulator).
* Every ``DECISION_PERIOD`` seconds: encode the state from all three
  predator positions, mask unsafe actions, pick an escape heading
  (epsilon-greedy) and update the Q-table with the reward of the last move.
* At game over it applies the terminal reward (-10 capture / +10 escape)
  and saves the table.
"""

import os

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from .common import Pose, cruise_controller, distance, latched_qos
from .core import (CAPTURE_REWARD, DECISION_PERIOD, PREDATOR_NAMES,
                   PREY_SPEED, SURVIVE_REWARD, PreyAgent, step_reward)


def _pretrained_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('task4')
    except Exception:  # noqa: BLE001 - package not installed
        return ''
    return os.path.join(share, 'config', 'prey-kb-pretrained.json')


class LearningPrey(Node):

    def __init__(self):
        super().__init__('prey')
        self.max_linear = self.declare_parameter('max_linear', PREY_SPEED).value
        self.learn = self.declare_parameter('learn', True).value
        epsilon = self.declare_parameter('epsilon', 0.05).value
        self.path = os.path.expanduser(
            self.declare_parameter('prey_kb_path', 'prey-kb.json').value)

        self.agent = PreyAgent(epsilon=epsilon)
        use_pretrained = self.declare_parameter('use_pretrained', True).value
        if self.agent.load(self.path):
            self.get_logger().info(
                f'loaded Q-table: {len(self.agent.q)} states, '
                f'{self.agent.episodes} episodes of experience '
                f'({os.path.abspath(self.path)})')
        elif use_pretrained and self.agent.load(_pretrained_path()):
            self.get_logger().info(
                f'loaded pre-trained Q-table ({self.agent.episodes} offline '
                f'episodes); will save to {os.path.abspath(self.path)}')
        else:
            self.agent = PreyAgent(epsilon=epsilon)
            self.get_logger().info(
                'no Q-table yet - starting from "run away" default '
                f'(will save to {os.path.abspath(self.path)})')
        self.agent.epsilon = epsilon if self.learn else 0.0

        self.pose = None
        self.predators = {}
        self.heading = None
        self.state = self.action = self.old_min = None
        self.done = False

        self.create_subscription(Pose, '/prey/pose', self.on_pose, 10)
        for name in PREDATOR_NAMES:
            self.create_subscription(
                Pose, f'/{name}_location',
                lambda m, n=name: self.predators.__setitem__(n, (m.x, m.y)),
                10)
        self.create_subscription(String, '/game_status', self.on_status,
                                 latched_qos())
        self.cmd_pub = self.create_publisher(Twist, '/prey/cmd_vel', 10)
        self.location_pub = self.create_publisher(Pose, '/prey_location', 10)
        self.create_timer(DECISION_PERIOD, self.decide)
        self.create_timer(0.05, self.control)

    def on_pose(self, msg):
        self.pose = msg
        self.location_pub.publish(msg)

    def decide(self):
        if (self.done or self.pose is None
                or len(self.predators) < len(PREDATOR_NAMES)):
            return
        prey_xy = (self.pose.x, self.pose.y)
        preds = list(self.predators.values())
        d_min = min(distance(*prey_xy, *p) for p in preds)
        away = self.agent.away_angle(prey_xy, preds)
        state = self.agent.encode(prey_xy, preds)
        legal = self.agent.legal_actions(prey_xy, away)
        if self.learn and self.state is not None:
            self.agent.update(self.state, self.action,
                              step_reward(self.old_min, d_min), state, legal,
                              False)
        self.action = self.agent.choose(state, legal, explore=self.learn)
        self.state, self.old_min = state, d_min
        self.heading = self.agent.action_heading(self.action, away)

    def control(self):
        if self.done or self.pose is None or self.heading is None:
            return
        linear, angular = cruise_controller(self.pose.theta, self.heading,
                                            self.max_linear)
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)

    def on_status(self, msg):
        if self.done:
            return
        self.done = True
        self.cmd_pub.publish(Twist())
        captured = msg.data.startswith('capture')
        if self.learn and self.state is not None:
            self.agent.update(self.state, self.action,
                              CAPTURE_REWARD if captured else SURVIVE_REWARD,
                              None, None, True)
            self.agent.end_episode()
            self.agent.save(self.path)
            self.get_logger().info(
                f'{"caught" if captured else "ESCAPED"} - Q-table saved '
                f'({len(self.agent.q)} states, {self.agent.episodes} '
                'episodes)')
        else:
            self.get_logger().info(f'game over: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    node = LearningPrey()
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
