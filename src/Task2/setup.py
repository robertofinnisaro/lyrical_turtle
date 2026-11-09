import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'task2'

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
    description='Task 2: single predator chasing a fleeing prey with a 60 s time limit',
    license='MIT',
    entry_points={
        'console_scripts': [
            'spawn_turtles = task2.spawn_turtles:main',
            'prey = task2.prey:main',
            'predator = task2.predator:main',
            'referee = task2.referee:main',
        ],
    },
)
