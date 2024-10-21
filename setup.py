import setuptools
# from setuptools import setup

setuptools.setup(
    name="cranavgym",
    version='0.1.0',
    description="A gym for navigation in dynamic, noisy, interactable environments. Ran on ROS/Gazebo with image and lidar observaitons. Built on top of DRL-robot-navigation (https://github.com/reiniscimurs/DRL-robot-navigation)",
    url='https://github.com/mazqtpopx/cranfield-navigation-gym',
    author='Mariusz Wisniewski, Paris Chatzithanos',
    author_email='m.wisniewski@cranfield.ac.uk',
    license='MIT',
    packages=setuptools.find_packages(),
)