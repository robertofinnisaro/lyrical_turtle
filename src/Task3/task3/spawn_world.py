"""Task 3 world setup: paint the obstacles, then spawn predator and prey."""

import math

import rclpy
from rclpy.node import Node

from .common import WorldBuilder, make_rng, random_free_point
from .obstacles import OBSTACLES


def main(args=None):
    rclpy.init(args=args)
    node = Node('task3_spawner')
    seed = node.declare_parameter('seed', -1).value
    rng = make_rng(seed)
    world = WorldBuilder(node)

    world.kill('turtle1')
    world.draw_obstacles(OBSTACLES)
    pred_xy = random_free_point(rng, OBSTACLES)
    prey_xy = random_free_point(rng, OBSTACLES, others=[pred_xy],
                                min_separation=4.0)
    world.spawn('predator', *pred_xy, rng.uniform(-math.pi, math.pi))
    world.spawn('prey', *prey_xy, rng.uniform(-math.pi, math.pi))
    world.set_pen('predator', 255, 140, 0, width=3)
    world.set_pen('prey', 255, 255, 255, width=2)

    node.get_logger().info(
        f'world ready (seed={seed}) - predator {pred_xy[0]:.2f},'
        f'{pred_xy[1]:.2f}  prey {prey_xy[0]:.2f},{prey_xy[1]:.2f}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
