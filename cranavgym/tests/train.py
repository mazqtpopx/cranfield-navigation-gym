import gymnasium as gym
from cranavgym.envs.drl_robot_navigation import DRLRobotNavigation

import sys
import os
import subprocess
import numpy as np
from datetime import datetime

import stable_baselines3
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

# from stable_baselines3.common.evaluation import evaluate_policy #NB: we're using a local modified version instead!
from evaluate_policy import (
    evaluate_policy,
)  # NB: this is a local modified version of the SB3 evaluate policy!
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

th.autograd.set_detect_anomaly(True)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# -------------------------------CONFIGS---------------------------------
# def load_ros_config():
#     # filename = "env_config.yaml"
#     filepath = "~/cranfield-navigation-gym/cranavgym/configs/ros_interface_config.yaml"
#     # filepath = "../configs/ros_interface_config.yaml"
#     filepath = os.path.abspath(os.path.expanduser(filepath))
#     with open(filepath) as stream:
#         try:
#             config_dict = yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#     munch_config = Munch.fromDict(config_dict)
#     return munch_config


# def load_env_config():
#     # filename = "env_config.yaml"
#     filepath = "~/cranfield-navigation-gym/cranavgym/configs/env_config.yaml"
#     # filepath = "../configs/env_config.yaml"
#     filepath = os.path.abspath(os.path.expanduser(filepath))
#     with open(filepath) as stream:
#         try:
#             config_dict = yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#     munch_config = Munch.fromDict(config_dict)
#     return munch_config


# def load_RL_config():
#     filepath = "~/cranfield-navigation-gym/cranavgym/configs/rl_config.yaml"
#     # filepath = "../configs/rl_config.yaml"
#     filepath = os.path.abspath(os.path.expanduser(filepath))
#     with open(filepath) as stream:
#         try:
#             config_dict = yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#     munch_config = Munch.fromDict(config_dict)
#     return munch_config


def load_config(config_dir_path, config_name):
    """
    Loads config and outputs in munch format
    config_dir_path: path to the config directory
    config_name: name of the config file (including extension)
    """
    filepath = os.path.abspath(
        os.path.expanduser(os.path.join(config_dir_path, config_name))
    )
    with open(filepath) as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    munch_config = Munch.fromDict(config_dict)
    return munch_config


def dump_configs(configs_dir, env_config, ros_config, rl_config):
    # dump the configs!
    with open(os.path.join(configs_dir, "env_config.yaml"), "w") as file:
        yaml.dump(env_config, file)

    with open(os.path.join(configs_dir, "ros_interface_config.yaml"), "w") as file:
        yaml.dump(ros_config, file)

    with open(os.path.join(configs_dir, "rl_config.yaml"), "w") as file:
        yaml.dump(rl_config, file)

    print("\n\n\n--------------------------------------------------")
    print(
        f"{bcolors.HEADER}{bcolors.OKGREEN}Starting training with params:\n {bcolors.BOLD}RL Algorithm: {rl_config.algorithm}\n {bcolors.BOLD}Observation Space: {env_config.scenario_settings.obs_space}"
    )
    print(f"{bcolors.ENDC}Good Luck!")
    print("--------------------------------------------------\n\n\n")


def main(env_config, ros_config, rl_config, run_name):
    # get all the necessary dirs
    log_dir, tensorboard_dir, configs_dir, model_dir, video_dir = get_dirs(
        rl_config, run_name=run_name
    )

    # Becuase we might have modified the configs by passing in args, dump the configs as they are for the run
    dump_configs(configs_dir, env_config, ros_config, rl_config)

    # env = setup_env(env_config, ros_config)

    # env = Monitor(env, log_dir)

    # env = stable_baselines3.common.atari_wrappers.MaxAndSkipEnv(env, skip=4)
    env = make_vec_env(env_config, ros_config, log_dir)
    env = gym.wrappers.RecordVideo(
        env, f"{log_dir}/videos/{run_name}", video_length=1000
    )
    # env = stable_baselines3.common.atari_wrappers.MaxAndSkipEnv(env, skip=4)
    # env = gym.wrappers.GrayScaleObservation(env)
    # env = gym.wrappers.FrameStack(env, 4)

    if rl_config.frame_stack:
        print(f"{bcolors.WARNING}---------------------FRAME STACK IS ENABLED-------------------------")
        print(f"{bcolors.WARNING}Only enable this option if you know what you are doing{bcolors.ENDC}")
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        new_obs_space = gym.spaces.Box(low=0, high=255, shape=(4, 160, 160), dtype=np.uint8)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.array(obs), observation_space=new_obs_space
        )
        # env = gym.wrappers.FrameStack(env, 4) #NB this line can work in earlier versions of SB3
        env = DummyVecEnv([lambda: env])
    else:
        print(f"{bcolors.OKGREEN}---------------------FRAME STACK IS DISABLED-------------------------{bcolors.ENDC}")



    if env_config.scenario_settings.obs_space == "lidar":
        if rl_config.algorithm == "TD3":
            model = setup_TD3(env, rl_config, tensorboard_dir)
        elif rl_config.algorithm == "PPO":
            model = setup_PPO(env, rl_config, tensorboard_dir)
        elif rl_config.algorithm == "PPO_LSTM":
            model = setup_PPO_LSTM(env, rl_config, tensorboard_dir)
        elif rl_config.algorithm == "DreamerV3":
            return
        else:
            raise ValueError(
                "Incorrect algorithm selected. Please select from TD3, PPO, PPO_LSTM or DreamerV3"
            )
    elif env_config.scenario_settings.obs_space == "camera":
        if rl_config.algorithm == "TD3":
            model = setup_TD3_camera(env, rl_config, tensorboard_dir)
        elif rl_config.algorithm == "PPO":
            model = setup_PPO_camera(env, rl_config, tensorboard_dir)
        elif rl_config.algorithm == "PPO_LSTM":
            model = setup_PPO_LSTM_camera(env, rl_config, tensorboard_dir)
        elif rl_config.algorithm == "DreamerV3":
            return
        else:
            raise ValueError(
                "Incorrect algorithm selected. Please select from TD3, PPO, PPO_LSTM or DreamerV3"
            )

    _ = env.reset()

    action = env.action_space.sample()

    # -----------DEBUG-----------
    # obs_sample = env.observation_space.sample()
    # print(f"{action=}")
    # print(f"{obs_sample=}")
    # print(f"{obs_sample.shape=}")
    # state, reward, done, _ = env.step([action])

    # print(f"{state=}")
    # print(f"{state.shape=}")
    # print(f"{reward=}")

    # raise Exception("Ending")

    if not rl_config.evaluate_only:
        callback = SaveOnBestTrainingRewardCallback(10000, log_dir, model_dir, 1)
        model.learn(
            total_timesteps=int(rl_config.max_training_steps),
            progress_bar=True,
            callback=callback,
        )

    evaluate(model, env, log_dir)

    print("Exiting the program...")
    sys.exit(0)
    # dump mean reward and std reward into a yaml file


def evaluate(model, env, log_dir):
    n_eval_episodes = 400
    print(f"Finished training. Starting evaluation")
    (
        episode_rewards,
        episode_lengths,
        episode_infos,
        episode_values,
        episode_features,
    ) = evaluate_policy(
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

    import pandas as pd

    rows = []
    i = 0
    # make dir to store the latent features
    os.makedirs(os.path.join(log_dir, "lf"), exist_ok=True)
    for ep in range(n_eval_episodes):
        for step in range(episode_lengths[ep]):
            lf_name = f"latent_features{i}.npy"
            lf = episode_features[ep][step].detach().cpu().numpy()
            latent_features_name = np.save(os.path.join(log_dir, f"lf/{lf_name}"), lf)
            data_step = {
                "episode": ep,
                "x_position": episode_infos[ep][step][0]["x_position"],
                "y_position": episode_infos[ep][step][0]["y_position"],
                "x_velocity": episode_infos[ep][step][0]["x_velocity"],
                "y_velocity": episode_infos[ep][step][0]["y_velocity"],
                "step_reward": episode_infos[ep][step][0]["reward"],
                "dist_to_target": episode_infos[ep][step][0]["dist_to_target"],
                "angle_to_goal": episode_infos[ep][step][0]["angle_to_goal"],
                "value": episode_values[ep][step],
                "latent_features_name": latent_features_name,
            }
            # df.loc[i] = pd.concat([df, pd.DataFrame([data_step])])
            rows.append(data_step)
            i += 1
    df = pd.DataFrame(
        rows,
        columns=[
            "episode",
            "x_position",
            "y_position",
            "x_velocity",
            "y_velocity",
            "step_reward",
            "dist_to_target",
            "angle_to_goal",
            "value",
            "latent_features_name",
        ],
    )
    df.to_pickle(os.path.join(log_dir, "evaluation_results_raw.pkl"))

    # if reward is greater than 1, assume success. Multiply by 100 for percentage
    success_rate = (sum([1 for i in episode_rewards if i > 0]) / n_eval_episodes) * 100

    dump_str = f"""
    Evaluation results across {n_eval_episodes} episodes: {np.mean(episode_rewards)=} {np.std(episode_rewards)=}\n
    {episode_rewards=}\n
    {episode_lengths=}
    Success rate:
    {success_rate=}
    # Episode infos:
    # {episode_infos=}
    """  #

    with open(os.path.join(log_dir, "evaluation_results_general.yaml"), "w") as file:
        yaml.dump(dump_str, file)


# -------------------------------ENV---------------------------------
def make_vec_env(env_config, ros_config, log_dir):
    env = gym.make(
        "DRLRobotNavigation-v0",
        ros_interface_config=ros_config,
        max_episode_steps=int(env_config.scenario_settings.max_steps),
        obs_space=env_config.scenario_settings.obs_space,
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

    env = Monitor(env, log_dir)
    return env


def setup_env(env_config, ros_config):

    env = gym.make(
        "DRLRobotNavigation-v0",
        ros_interface_config=ros_config,
        max_episode_steps=int(env_config.scenario_settings.max_steps),
        obs_space=env_config.scenario_settings.obs_space,
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

    return env


# -------------------------------ALGOS---------------------------------
def setup_TD3(env, rl_config, tensorboard_dir):
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions)
    )

    if rl_config.load_model_path is not (None or ""):
        model = TD3.load(
            rl_config.load_model_path,
            env,
            learning_rate=float(rl_config.lr),
            verbose=1,
            device="cuda",
        )
    else:
        model = TD3(
            "MlpPolicy",  # rl_config.TD3.policy_type,
            env,
            learning_rate=float(rl_config.lr),
            buffer_size=1000000,
            learning_starts=5000,
            tau=float(rl_config.TD3.tau),
            batch_size=int(rl_config.batch_size),
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=tensorboard_dir,
            device="cuda",
        )

    return model


def setup_PPO(env, rl_config, tensorboard_dir):

    if rl_config.load_model_path is not (None or ""):
        model = PPO.load(
            rl_config.load_model_path,
            env,
            learning_rate=float(rl_config.lr),
            verbose=1,
            device="cuda",
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=float(rl_config.lr),
            # policy_kwargs=policy_kwargs,
            n_steps=int(rl_config.PPO.n_steps),
            batch_size=int(rl_config.PPO.batch_size),
            verbose=1,
            tensorboard_log=tensorboard_dir,
            device="cuda",
        )

    return model


def setup_PPO_LSTM(env, rl_config, tensorboard_dir):
    if rl_config.load_model_path is not (None or ""):
        model = RecurrentPPO.load(
            rl_config.load_model_path,
            env,
            learning_rate=float(rl_config.lr),
            verbose=1,
            device="cuda",
        )
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=float(rl_config.lr),
            # policy_kwargs=policy_kwargs,
            n_steps=int(rl_config.PPO_LSTM.n_steps),
            batch_size=int(rl_config.PPO_LSTM.batch_size),
            verbose=1,
            tensorboard_log=tensorboard_dir,
            device="cuda",
        )

    return model


def setup_DreamerV3(env, rl_config, tensorboard_dir):
    return NotImplementedError()


def setup_TD3_camera(env, rl_config, tensorboard_dir):
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions)
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    if rl_config.load_model_path is not (None or ""):
        model = TD3.load(
            rl_config.load_model_path,
            env,
            learning_rate=float(rl_config.lr),
            verbose=1,
            device="cuda",
        )
    else:
        model = TD3(
            "CnnPolicy",
            env,
            learning_rate=float(rl_config.lr),
            policy_kwargs=policy_kwargs,
            buffer_size=10000,
            learning_starts=float(rl_config.TD3.learning_starts),
            tau=float(rl_config.TD3.tau),
            batch_size=int(rl_config.TD3.batch_size),
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=tensorboard_dir,
            device="cuda",
        )

    return model


def setup_PPO_camera(env, rl_config, tensorboard_dir):
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )
    print(f"{rl_config=}")
    if rl_config.load_model_path is not (None or ""):
        print(f"Loading PPO, camera model, from {rl_config.load_model_path}")
        model = PPO.load(
            rl_config.load_model_path,
            env,
            learning_rate=float(rl_config.lr),
            verbose=1,
            device="cuda",
        )
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=float(rl_config.lr),
            policy_kwargs=policy_kwargs,
            n_steps=int(rl_config.PPO.n_steps),
            batch_size=int(rl_config.PPO.batch_size),
            verbose=1,
            tensorboard_log=tensorboard_dir,
            device="cuda",
        )
    return model


def setup_PPO_LSTM_camera(env, rl_config, tensorboard_dir):
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    if rl_config.load_model_path is not (None or ""):
        model = RecurrentPPO.load(
            rl_config.load_model_path,
            env,
            learning_rate=float(rl_config.lr),
            verbose=1,
            device="cuda",
        )
    else:
        model = RecurrentPPO(
            "CnnLstmPolicy",
            env,
            learning_rate=float(rl_config.lr),
            policy_kwargs=policy_kwargs,
            n_steps=int(rl_config.PPO_LSTM.n_steps),
            batch_size=int(rl_config.PPO_LSTM.batch_size),
            verbose=1,
            tensorboard_log=tensorboard_dir,
            device="cuda",
        )

    return model


# -------------------------------logging dirs---------------------------------
def get_log_dir(log_dir_basepath, algorithm, evaluate_only, run_name="default"):
    now = (
        str(datetime.now())
        .replace(" ", "_")
        .replace(":", "")
        .replace("-", "")
        .split(".")[0]
    )
    if evaluate_only:
        log_dir = os.path.abspath(
            os.path.expanduser(
                os.path.join(
                    log_dir_basepath,
                    "evaluation",
                    "camera_repeatability",
                    algorithm + "_" + str(now) + "_" + run_name,
                )
            )
        )
    else:
        log_dir = os.path.abspath(
            os.path.expanduser(
                os.path.join(
                    log_dir_basepath, algorithm + "_" + str(now) + "_" + run_name
                )
            )
        )
    return log_dir


# dump tensorboard stuff here
def get_tensorboard_dir(log_dir):
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    return tensorboard_dir


# dump configs here
def get_configs_dir(log_dir):
    configs_dir = os.path.join(log_dir, "configs")
    return configs_dir


# dump models here
def get_model_dir(log_dir):
    model_dir = os.path.join(log_dir, "model")
    return model_dir


def get_video_dir(log_dir):
    video_dir = os.path.join(log_dir, "videos")
    return video_dir


def get_dirs(rl_config, run_name="default"):
    # get the log_dir basepath (as defined in config)
    log_dir = get_log_dir(
        rl_config.log_dir, rl_config.algorithm, rl_config.evaluate_only, run_name
    )
    os.makedirs(log_dir, exist_ok=True)

    tensorboard_dir = get_tensorboard_dir(log_dir)
    configs_dir = get_configs_dir(log_dir)
    model_dir = get_model_dir(log_dir)
    video_dir = get_video_dir(log_dir)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    return log_dir, tensorboard_dir, configs_dir, model_dir, video_dir


def load_configs():
    config_path = "~/cranfield-navigation-gym/cranavgym/configs/"
    env_config = load_config(config_path, "env_config.yaml")
    ros_config = load_config(config_path, "ros_interface_config.yaml")
    rl_config = load_config(config_path, "rl_config.yaml")

    return env_config, ros_config, rl_config


def load_evaluation_configs():
    config_path = "~/cranfield-navigation-gym/cranavgym/configs/evaluation_configs/"
    env_config = load_config(config_path, "env_config.yaml")
    ros_config = load_config(config_path, "ros_interface_config.yaml")
    rl_config = load_config(config_path, "rl_config.yaml")

    return env_config, ros_config, rl_config


if __name__ == "__main__":
    env_config, ros_config, rl_config = load_configs()

    parser = argparse.ArgumentParser(description="Process input arguments")
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument("-test-name", "--test_name", type=str, default="test_name")

    parser.add_argument("--lidar-noise", action="store_true")
    parser.add_argument("--no-lidar-noise", dest="lidar_noise", action="store_false")

    parser.add_argument("--camera-noise", action="store_true")
    parser.add_argument("--no-camera-noise", dest="camera_noise", action="store_false")

    parser.add_argument("--static-spawn", action="store_true")
    parser.add_argument("--no-static-spawn", dest="static_spawn", action="store_false")

    parser.add_argument("--static-goal", action="store_true")
    parser.add_argument("--no-static-goal", dest="static_goal", action="store_false")

    parser.add_argument("--frame-stack", action="store_true")
    parser.add_argument("--no-frame-stack", dest="frame_stack", action="store_false")

    parser.add_argument("-camera-noise-size", "--camera_noise_size", type=int)
    parser.add_argument("-lidar-noise-size", "--lidar_noise_size", type=int)
    parser.add_argument("-max-training-steps", "--max_training_steps", type=int)

    parser.add_argument("--evaluate-only", dest="evaluate_only", action="store_true")
    parser.add_argument(
        "-load-model-path",
        "--load_model_path",
        type=str,
        default="",
        help="Full path to the model",
    )

    parser.add_argument("-map-bounds", "--map_bounds", type=float)

    parser.add_argument(
        "-obs",
        "--observation_space",
        type=str,
        default="lidar",
        help='Options: "lidar", "camera", or "dict"',
    )

    # TD3, PPO, PPO_LSTM, DreamerV3
    parser.add_argument(
        "-algo",
        "--rl_algorithm",
        type=str,
        default="TD3",
        help='Options: "TD3", "PPO", "PPO_LSTM", or "DreamerV3"',
    )

    parser.add_argument(
        "-algo-notes",
        "--algo_notes",
        type=str,
        default="N/A",
        help="(Optional) Use to write notes about the experiment (Used for logging only)",
    )
    parser.add_argument(
        "-env-notes",
        "--env_notes",
        type=str,
        default="N/A",
        help="(Optional) Use to write notes about the experiment (Used for logging only)",
    )

    parser.add_argument(
        "-run-name",
        "--run_name",
        type=str,
        default="default",
        help="(Optional) Used to name the folder where the logs, weights, etc. are saved",
    )

    # parser.add_argument("-test_name", "--test_name", type=str, default="test_name")

    # parser.add_argument("--camera-noise", action="store_true")
    # parser.add_argument("--no-camera-noise", dest="camera_noise", action="store_false")

    args = parser.parse_args()

    # make sure this is first!!!!
    if args.evaluate_only is not None:
        # For evaluation, load evaluation configs - they should be in different dir!
        env_config, ros_config, rl_config = load_evaluation_configs()
        rl_config.evaluate_only = args.evaluate_only

    # print(f"1{rl_config=}")
    # overwrite the configs
    if args.learning_rate is not None:
        rl_config.lr = args.learning_rate

    if args.lidar_noise is not None:
        env_config.drl_robot_navigation.lidar_noise = args.lidar_noise
    if args.lidar_noise_size is not None:
        env_config.drl_robot_navigation.lidar_noise_area_size[0] = args.lidar_noise_size
        env_config.drl_robot_navigation.lidar_noise_area_size[1] = args.lidar_noise_size

    if args.camera_noise is not None:
        env_config.drl_robot_navigation.camera_noise = args.camera_noise
    if args.camera_noise_size is not None:
        env_config.drl_robot_navigation.camera_noise_area_size[0] = (
            args.camera_noise_size
        )
        env_config.drl_robot_navigation.camera_noise_area_size[1] = (
            args.camera_noise_size
        )

    if args.static_spawn is not None:
        env_config.scenario_settings.static_spawn = args.static_spawn

    print(f"Static goal = {args.static_goal}")
    if args.static_goal is not None:
        print(f"Static goal = {args.static_goal}")
        env_config.scenario_settings.static_goal = args.static_goal
        print(f"Static goal = {env_config.scenario_settings.static_goal}")

    if args.map_bounds is not None:
        ros_config.map.min_xy[0] = -args.map_bounds
        ros_config.map.min_xy[1] = -args.map_bounds
        ros_config.map.max_xy[0] = args.map_bounds
        ros_config.map.max_xy[1] = args.map_bounds

    if args.max_training_steps is not None:
        rl_config.max_training_steps = args.max_training_steps

    if args.observation_space is not None:
        env_config.scenario_settings.obs_space = args.observation_space

    if args.rl_algorithm is not None:
        rl_config.algorithm = args.rl_algorithm

    if args.algo_notes is not None:
        rl_config.experimental_notes = args.algo_notes

    if args.env_notes is not None:
        env_config.experimental_notes = args.env_notes

    if args.load_model_path is not None:
        rl_config.load_model_path = args.load_model_path

    if args.frame_stack is not None:
        rl_config.frame_stack = args.frame_stack

    run_name = ""
    if args.run_name:
        run_name = args.run_name

    main(env_config, ros_config, rl_config, run_name)
