import gymnasium as gym
from cranavgym.envs.drl_robot_navigation import DRLRobotNavigation

import os
import subprocess
import numpy as np

# load config
import yaml
from munch import Munch


def load_ros_config():
    filepath = "../configs/ros_interface_config.yaml"
    with open(filepath) as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    munch_config = Munch.fromDict(config_dict)
    return munch_config


def load_env_config():
    filepath = "../configs/env_config.yaml"
    with open(filepath) as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    munch_config = Munch.fromDict(config_dict)
    return munch_config


def main():
    ros_config = load_ros_config()
    env_config = load_env_config()
    env = gym.make(
        "DRLRobotNavigation-v0",
        ros_interface_config=ros_config,
        max_episode_steps=env_config.scenario_settings.max_steps,
        obs_space=env_config.scenario_settings.obs_space,
        camera_noise=env_config.drl_robot_navigation.camera_noise,
        camera_noise_area_size=env_config.drl_robot_navigation.camera_noise_area_size,
        random_camera_noise_area=env_config.drl_robot_navigation.random_camera_noise_area,
        static_camera_noise_area_pos=env_config.drl_robot_navigation.static_camera_noise_area_pos,
        camera_noise_type=env_config.drl_robot_navigation.camera_noise_type,
        lidar_noise=env_config.drl_robot_navigation.lidar_noise,
        lidar_noise_area_size=env_config.drl_robot_navigation.lidar_noise_area_size,
        random_lidar_noise_area=env_config.drl_robot_navigation.random_lidar_noise_area,
        static_lidar_noise_area_pos=env_config.drl_robot_navigation.static_lidar_noise_area_pos,
        static_goal=env_config.scenario_settings.static_goal,
        static_goal_xy=env_config.scenario_settings.static_goal_xy,
        static_spawn=env_config.scenario_settings.static_spawn,
        static_spawn_xy=env_config.scenario_settings.static_spawn_xy,
    )

    obs, _ = env.reset()

    for i in range(10000):
        action = env.action_space.sample()
        # print(f"step: {action=}")
        # action = np.array([1,1])
        obs, rew, done, _, _ = env.step(action)

        if done:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
