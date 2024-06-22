from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'yolo8_realsense'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*'))),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'model'), glob(os.path.join('model', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mickey',
    maintainer_email='mickey.li@ucl.ac.uk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = yolo8_realsense.detector:main'
        ],
    },
)
