import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'task1'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='student@example.com',
    description='Task 1: single predator chasing a static prey in turtlesim',
    license='MIT',
    entry_points={
        'console_scripts': [
            'spawn_turtles = task1.spawn_turtles:main',
            'prey = task1.prey:main',
            'predator = task1.predator:main',
        ],
    },
)
