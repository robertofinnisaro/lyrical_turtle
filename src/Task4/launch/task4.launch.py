"""Task 4 - three independent predators vs a learning prey, with obstacles."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PREDATORS = ['predator1', 'predator2', 'predator3']


def generate_launch_description():
    seed = LaunchConfiguration('seed')
    kb = LaunchConfiguration('prey_kb_path')
    learn = LaunchConfiguration('learn')
    nodes = [
        DeclareLaunchArgument('seed', default_value='-1',
                              description='spawn RNG seed (-1 = random)'),
        DeclareLaunchArgument('prey_kb_path', default_value='prey-kb.json',
                              description='prey Q-table file'),
        DeclareLaunchArgument('learn', default_value='true',
                              description='prey keeps learning online'),
        Node(package='turtlesim', executable='turtlesim_node', name='sim'),
        Node(package='task4', executable='spawn_world', output='screen',
             parameters=[{'seed': seed}]),
        Node(package='task4', executable='referee', output='screen',
             parameters=[{'predators': PREDATORS, 'time_limit': 60.0}]),
        Node(package='task4', executable='prey', output='screen',
             parameters=[{'prey_kb_path': kb, 'learn': learn}]),
    ]
    for name in PREDATORS:
        nodes.append(Node(package='task4', executable='predator',
                          name=name, output='screen',
                          parameters=[{'robot_name': name}]))
    return LaunchDescription(nodes)
