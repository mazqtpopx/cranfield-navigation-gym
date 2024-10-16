import gymnasium as gym
from cranavgym.envs.drl_robot_navigation import DRLRobotNavigation

import sys
import os
import subprocess
import numpy as np
from datetime import datetime


from stable_baselines3 import TD3, PPO
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from sb3_contrib.ppo_recurrent import RecurrentPPO

# load config
import yaml
from munch import Munch

import argparse

from customcnn import CustomCNN
from saveonbesttrainingcallback import SaveOnBestTrainingRewardCallback

import torch as th


def main(env_config, ros_config):
    # get all the necessary dirs
    # log_dir, tensorboard_dir, configs_dir, model_dir, video_dir = get_dirs(
    #     rl_config, run_name=run_name
    # )

    # env = DummyVecEnv([lambda: make_vec_env(env_config, ros_config, log_dir)])
    env = make_env_test(env_config, ros_config)
    model_path = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip"
    model = PPO.load(model_path, env, learning_rate=0.0003, verbose=1, device="cuda")

    n_eval_episodes = 50
    print(f"Finished training. Starting evaluation")
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        callback=None,
        reward_threshold=None,
        return_episode_rewards=True,
        warn=True,
    )
    print(
        f"Evaluation results across {n_eval_episodes} episodes: {np.mean(episode_rewards)=} {np.std(episode_rewards)=}"
    )

    with open("evaluation_results_raw.yaml", "w") as file:
        yaml.dump(f"{episode_rewards=}, {episode_lengths=}", file)

    # model = PPO.load(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=float(rl_config.lr),
    #     # policy_kwargs=policy_kwargs,
    #     n_steps=256,
    #     batch_size=256,
    #     verbose=1,
    #     tensorboard_log=tensorboard_dir,
    #     device="cuda",
    # )


# -------------------------------ENV---------------------------------
def make_env(env_config, ros_config):
    env = gym.make(
        "DRLRobotNavigation-v0",
        ros_interface_config=ros_config,
        max_episode_steps=int(env_config.scenario_settings.max_episode_steps),
        obs_space=env_config.scenario_settings.obs_space,
        reward_type="alternative",  # should put in config
        camera_noise=env_config.drl_robot_navigation.camera_noise,
        camera_noise_area_size=env_config.drl_robot_navigation.camera_noise_area_size,
        camera_noise_type=env_config.drl_robot_navigation.camera_noise_type,
        lidar_noise=env_config.drl_robot_navigation.lidar_noise,
        lidar_noise_area_size=env_config.drl_robot_navigation.lidar_noise_area_size,
        static_goal=env_config.scenario_settings.static_goal,
        static_goal_xy=env_config.scenario_settings.static_goal_xy,
        static_spawn=env_config.scenario_settings.static_spawn,
        static_spawn_xy=env_config.scenario_settings.static_spawn_xy,
    )

    # env = Monitor(env)
    return env


def make_env_test(env_config, ros_config):
    env = gym.make(
        "DRLRobotNavigation-v0",
        ros_interface_config=ros_config,
        max_episode_steps=int(env_config.scenario_settings.max_episode_steps),
        obs_space=env_config.scenario_settings.obs_space,
        reward_type="alternative",  # should put in config
        camera_noise=env_config.drl_robot_navigation.camera_noise,
        camera_noise_area_size=env_config.drl_robot_navigation.camera_noise_area_size,
        camera_noise_type=env_config.drl_robot_navigation.camera_noise_type,
        lidar_noise=env_config.drl_robot_navigation.lidar_noise,
        lidar_noise_area_size=env_config.drl_robot_navigation.lidar_noise_area_size,
        static_goal=env_config.scenario_settings.static_goal,
        static_goal_xy=env_config.scenario_settings.static_goal_xy,
        static_spawn=env_config.scenario_settings.static_spawn,
        static_spawn_xy=env_config.scenario_settings.static_spawn_xy,
    )

    # env = Monitor(env)
    return env


def load_configs():
    return load_env_config(), load_ros_config(), load_RL_config()


# -------------------------------CONFIGS---------------------------------
def load_ros_config():
    # filename = "env_config.yaml"
    # filepath = "~/cranfield-navigation-gym/cranavgym/configs/ros_interface_config.yaml"
    filepath = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/configs/ros_interface_config.yaml"
    # filepath = "../configs/ros_interface_config.yaml"
    filepath = os.path.abspath(os.path.expanduser(filepath))
    with open(filepath) as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    munch_config = Munch.fromDict(config_dict)
    return munch_config


def load_env_config():
    # filename = "env_config.yaml"
    # filepath = "~/cranfield-navigation-gym/cranavgym/configs/env_config.yaml"
    filepath = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/configs/env_config.yaml"
    # filepath = "../configs/env_config.yaml"
    filepath = os.path.abspath(os.path.expanduser(filepath))
    with open(filepath) as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    munch_config = Munch.fromDict(config_dict)
    return munch_config


def load_RL_config():
    filepath = "~/cranfield-navigation-gym/cranavgym/configs/rl_config.yaml"
    # filepath = "../configs/rl_config.yaml"
    filepath = os.path.abspath(os.path.expanduser(filepath))
    with open(filepath) as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    munch_config = Munch.fromDict(config_dict)
    return munch_config


if __name__ == "__main__":
    env_config, ros_config, rl_config = load_configs()
    main(env_config, ros_config)
