"""Task 4 world setup: paint obstacles, spawn prey and three predators."""

import math

import rclpy
from rclpy.node import Node

from .common import WorldBuilder, make_rng, random_free_point
from .core import PREDATOR_NAMES
from .obstacles import OBSTACLES

PEN = {'predator1': (255, 80, 80), 'predator2': (255, 170, 0),
       'predator3': (255, 0, 200)}


def main(args=None):
    rclpy.init(args=args)
    node = Node('task4_spawner')
    seed = node.declare_parameter('seed', -1).value
    rng = make_rng(seed)
    world = WorldBuilder(node)

    world.kill('turtle1')
    world.draw_obstacles(OBSTACLES)
    prey_xy = random_free_point(rng, OBSTACLES)
    taken = [prey_xy]
    world.spawn('prey', *prey_xy, rng.uniform(-math.pi, math.pi))
    world.set_pen('prey', 255, 255, 255, width=2)
    for name in PREDATOR_NAMES:
        xy = random_free_point(rng, OBSTACLES, others=taken,
                               min_separation=3.0)
        taken.append(xy)
        world.spawn(name, *xy, rng.uniform(-math.pi, math.pi))
        world.set_pen(name, *PEN[name], width=2)

    node.get_logger().info(f'world ready (seed={seed})')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
