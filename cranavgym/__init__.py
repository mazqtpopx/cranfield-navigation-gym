__version__ = "0.1.0"
__author__ = "Mariusz Wisniewski, Paris Chatzithanos"
__credits__ = "Cranfield University, AI Lab"

from gymnasium.envs.registration import register

from cranavgym.envs.drl_robot_navigation import DRLRobotNavigation


register(
    id="DRLRobotNavigation-v0",
    entry_point="cranavgym:DRLRobotNavigation",
    # launchfile = "",
    # lidar_dim = 20,
    # time_delta = 0.001,
    # img_width = 160,
    # img_height = 160,
    # obs_space = "lidar",
    # camera_noise = False,
    # camera_noise_area_size = (4,4),
    # camera_noise_type = "gaussian",
    # lidar_noise = False,
    # lidar_noise_area_size = (4,4),
    # ros_port = 11311,
    # max_episode_steps=1200,
    # reward_threshold=100.0
)
