import gymnasium as gym
from cranavgym.envs.drl_robot_navigation import DRLRobotNavigation

import os
import subprocess
import numpy as np
import yaml
from munch import Munch

import pygame

from xbox_controller import XboxController


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


import time

import rospy
from std_srvs.srv import Empty


def filter_deadzone(input, deadzone):
    if abs(input) < deadzone:
        return 0
    else:
        return input


def register_input_joystick(joy):
    INVERT = True
    DEADZONE = 0.1

    joy_outputs = joy.read()
    for i in range(len(joy_outputs)):
        joy_outputs[i] = filter_deadzone(joy_outputs[i], DEADZONE)

    if INVERT:
        for i in range(len(joy_outputs)):
            joy_outputs[i] = -joy_outputs[i]

    return joy_outputs


def main():
    unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
    pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)

    pygame.init()
    screen = pygame.display.set_mode((160, 160))

    ros_config = load_ros_config()
    env_config = load_env_config()
    env = gym.make(
        "DRLRobotNavigation-v0",
        ros_interface_config=ros_config,
        max_episode_steps=int(env_config.scenario_settings.max_steps),
        obs_space="camera",
        reward_type="alternative",  # should put in config
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

    print(env)

    joy = XboxController()

    for i in range(10000):
        # action = env.action_space.sample()
        # print(f"step: {action=}")
        # action = np.array([1,1])
        action = 0
        unpause()

        joy_output = register_input_joystick(joy)

        action = [0.0, 0.0, 0.0]
        action[0] = joy_output[0] / 10
        action[1] = joy_output[1] / 10
        action[2] = -joy_output[2] / 10
        # for event in pygame.event.get():
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_UP:
        #             action = 1
        #         elif event.key == pygame.K_LEFT:
        #             action = 2
        #         elif event.key == pygame.K_RIGHT:
        #             action = 3

        print(f"step: {action=}")

        obs, rew, done, trunc, _ = env.step(action)

        obs = np.rot90(obs)
        frame_surf = pygame.surfarray.make_surface(obs)

        screen.blit(frame_surf, (0, 0))
        pygame.display.flip()
        time.sleep(0.1)

        if done or trunc:
            pause()
            obs, _ = env.reset()
            unpause()


if __name__ == "__main__":
    main()
