"""Task 1 - single robot, static target."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    seed = LaunchConfiguration('seed')
    return LaunchDescription([
        DeclareLaunchArgument('seed', default_value='-1',
                              description='spawn RNG seed (-1 = random)'),
        Node(package='turtlesim', executable='turtlesim_node', name='sim'),
        Node(package='task1', executable='spawn_turtles', output='screen',
             parameters=[{'seed': seed}]),
        Node(package='task1', executable='prey', output='screen'),
        Node(package='task1', executable='predator', output='screen'),
    ])
