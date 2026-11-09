import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'task3'

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
    description='Task 3: predator learning from an XML knowledge base, with obstacles',
    license='MIT',
    entry_points={
        'console_scripts': [
            'spawn_world = task3.spawn_world:main',
            'prey = task3.prey:main',
            'predator = task3.predator:main',
            'referee = task3.referee:main',
        ],
    },
)
