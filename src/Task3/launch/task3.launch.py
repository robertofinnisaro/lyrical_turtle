"""Task 3 - single robot with learning (predator-kb.xml), plus obstacles.

Run it several times: each run appends a trial to predator-kb.xml and the
predator uses all previous trials to predict where the prey is going.
Use the same ``seed`` on every run to compare trials from identical starts.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    seed = LaunchConfiguration('seed')
    kb_path = LaunchConfiguration('kb_path')
    use_kb = LaunchConfiguration('use_kb')
    show_target = LaunchConfiguration('show_virtual_target')
    return LaunchDescription([
        DeclareLaunchArgument('seed', default_value='-1',
                              description='spawn RNG seed (-1 = random)'),
        DeclareLaunchArgument('kb_path', default_value='predator-kb.xml',
                              description='knowledge base file'),
        DeclareLaunchArgument('use_kb', default_value='true',
                              description='false = plain pursuit baseline'),
        DeclareLaunchArgument('show_virtual_target', default_value='true',
                              description='show predicted target as a turtle'),
        Node(package='turtlesim', executable='turtlesim_node', name='sim'),
        Node(package='task3', executable='spawn_world', output='screen',
             parameters=[{'seed': seed}]),
        Node(package='task3', executable='referee', output='screen',
             parameters=[{'predators': ['predator'], 'time_limit': 60.0}]),
        Node(package='task3', executable='prey', output='screen'),
        Node(package='task3', executable='predator', output='screen',
             parameters=[{'kb_path': kb_path, 'use_kb': use_kb,
                          'show_virtual_target': show_target}]),
    ])
