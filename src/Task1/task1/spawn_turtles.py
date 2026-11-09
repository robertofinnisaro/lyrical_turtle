"""Task 1 world setup: remove turtle1, spawn 'predator' and 'prey' randomly."""

import math

import rclpy
from rclpy.node import Node

from .common import WorldBuilder, make_rng, random_free_point


def main(args=None):
    rclpy.init(args=args)
    node = Node('task1_spawner')
    seed = node.declare_parameter('seed', -1).value
    rng = make_rng(seed)
    world = WorldBuilder(node)

    world.kill('turtle1')
    pred_xy = random_free_point(rng)
    prey_xy = random_free_point(rng, others=[pred_xy], min_separation=4.0)
    world.spawn('predator', *pred_xy, rng.uniform(-math.pi, math.pi))
    world.spawn('prey', *prey_xy, rng.uniform(-math.pi, math.pi))
    world.set_pen('predator', 255, 140, 0, width=3)
    world.set_pen('prey', 255, 255, 255, width=2, off=True)

    node.get_logger().info('world ready')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
