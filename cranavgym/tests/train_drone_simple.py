import gymnasium as gym
import cranavgym
import sys
import os
import numpy as np
from datetime import datetime

from stable_baselines3 import TD3, PPO
from stable_baselines3.common.noise import (
    NormalActionNoise,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.ppo_recurrent import RecurrentPPO
import yaml
from munch import Munch
import argparse
from customcnn import CustomCNN
from saveonbesttrainingcallback import SaveOnBestTrainingRewardCallback
import torch as th
import os
import rospy
import roslaunch
import subprocess
import time

th.autograd.set_detect_anomaly(True)

# switch between these for dev/prod repos
repo = "cranfield-navigation-gym"


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


from evaluate_policy import (
    evaluate_policy,
)  # NB: this is a local modified version of the SB3 evaluate policy!


def launch_ROS(launchfile, ros_port):
    """
    Initialize and launch the ROS core and Gazebo simulation.

    Args:
        launchfile (str): The path of the launchfile to be used for the simulation.
        ros_port (int): The port number for the ROS core.
    """
    print(f"{ros_port=}")

    # Start roscore with subprocess
    subprocess.Popen(["roscore", "-p", str(ros_port)])

    rospy.loginfo("Roscore launched!")

    # Give roscore some time to initialize
    time.sleep(2)

    # Initialize ROS node
    rospy.init_node("gym", anonymous=True, log_level=rospy.ERROR)

    # Expand launch file path
    fullpath = os.path.abspath(os.path.expanduser(launchfile))

    # Create a launch parent to manage roslaunch processes
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    # Setup the roslaunch arguments with custom parameters (gui, rviz, robot_name)
    cli_args = [
        fullpath,
        "gui:=true",
        "rviz:=true",
        "robot_name:=uav1",
    ]

    roslaunch_file = [
        (roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], cli_args[1:])
    ]

    # Create roslaunch parent process
    launch_parent = roslaunch.parent.ROSLaunchParent(
        uuid,
        roslaunch_file,
        is_core=False,
    )

    # Start roslaunch
    launch_parent.start()
    time.sleep(3)
    rospy.loginfo("Gazebo and RViz launched!")
    return


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
        f"{bcolors.HEADER}{bcolors.OKGREEN}Starting training with params:\n {bcolors.BOLD}RL Algorithm: {rl_config.algorithm}\n {bcolors.BOLD}Observation Space: {env_config.scenario_settings.observations.obs_space_type}"
    )
    print(f"{bcolors.ENDC}Good Luck!")
    print("--------------------------------------------------\n\n\n")


def main(env_config, ros_config, rl_config, run_name):
    env_name = env_config.gym_env_id

    # get all the necessary dirs
    log_dir, tensorboard_dir, configs_dir, model_dir, video_dir = get_dirs(
        rl_config, env_name, run_name=run_name
    )

    # Because we might have modified the configs by passing in args, dump the configs as they are for the run
    dump_configs(configs_dir, env_config, ros_config, rl_config)

    env = DummyVecEnv([lambda: make_vec_env(env_name, env_config, ros_config, log_dir)])

    if env_config.scenario_settings.observations.obs_space_type == "lidar":
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
    elif env_config.scenario_settings.observations.obs_space_type == "camera":
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

    # -------------------------------TRAINING---------------------------------
    if not rl_config.evaluate_only:
        callback = SaveOnBestTrainingRewardCallback(10000, log_dir, model_dir, 1)
        model.learn(
            total_timesteps=int(rl_config.max_training_steps),
            progress_bar=True,
            callback=callback,
        )

    evaluate(model, env, log_dir)

    env.close()

    print("Exiting the program...")
    sys.exit(0)
    # dump mean reward and std reward into a yaml file


def evaluate(model, env, log_dir):
    n_eval_episodes = 100
    print(f"Finished training. Starting evaluation")
    episode_rewards, episode_lengths, episode_infos, episode_values = evaluate_policy(
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
    for ep in range(n_eval_episodes):
        for step in range(episode_lengths[ep]):
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
def make_vec_env(env_name, env_config, ros_config, log_dir):
    env = gym.make(
        env_name,
        ros_interface_config=ros_config,
        max_ep_steps=int(env_config.scenario_settings.max_episode_steps),
        obs_space_type=env_config.scenario_settings.observations.obs_space_type,
        reward_type=env_config.scenario_settings.reward_type,
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


# -------------------------------ALGOS---------------------------------
def setup_TD3(env, rl_config, tensorboard_dir):
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.3 * np.ones(n_actions),
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
        print(f"Using TD3 Camer training policy -MW!!!")
        model = TD3(
            "CnnPolicy",  # rl_config.TD3.policy_type,
            env,
            learning_rate=float(rl_config.lr),
            policy_kwargs=policy_kwargs,
            buffer_size=4000,
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
        features_extractor_kwargs=dict(features_dim=256),
    )
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
def get_log_dir(
    log_dir_basepath, env_name, algorithm, evaluate_only, run_name="default"
):
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
                    env_name + "_" + algorithm + "_" + str(now) + "_" + run_name,
                    "dev_runs",
                    algorithm + "_" + str(now) + "_" + run_name,
                )
            )
        )
    else:
        log_dir = os.path.abspath(
            os.path.expanduser(
                os.path.join(
                    log_dir_basepath,
                    env_name + "_" + algorithm + "_" + str(now) + "_" + run_name,
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


def get_dirs(rl_config, env_name, run_name="default"):
    # get the log_dir basepath (as defined in config)
    log_dir = get_log_dir(
        rl_config.log_dir,
        env_name,
        rl_config.algorithm,
        rl_config.evaluate_only,
        run_name,
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
    config_path = f"~/{repo}/cranavgym/configs/"
    env_config = load_config(config_path, "env_config_drone.yaml")
    ros_config = load_config(config_path, "ros_interface_config_drone.yaml")
    rl_config = load_config(config_path, "rl_config_drone.yaml")

    return env_config, ros_config, rl_config


def load_evaluation_configs():
    config_path = f"~/{repo}/cranavgym/configs/"
    env_config = load_config(config_path, "env_config_drone.yaml")
    ros_config = load_config(config_path, "ros_interface_config_drone.yaml")
    rl_config = load_config(config_path, "rl_config_drone.yaml")

    return env_config, ros_config, rl_config


# -------------------------------MAIN---------------------------------
if __name__ == "__main__":
    env_config, ros_config, rl_config = load_configs()

    parser = argparse.ArgumentParser(description="Process input arguments")
    parser.add_argument("-gym-env-id", "--gym_env_id", type=str)
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

    parser.add_argument("-camera-noise-size", "--camera_noise_size", type=int)
    parser.add_argument("-lidar-noise-size", "--lidar_noise_size", type=int)
    parser.add_argument("-max-training-steps", "--max_training_steps", type=int)

    parser.add_argument("--evaluate-only", dest="evaluate_only", action="store_true")
    parser.add_argument(
        "-load-model-path",
        "--load_model_path",
        type=str,
        help="Full path to the model",
    )

    parser.add_argument("-map-bounds", "--map_bounds", type=float)

    parser.add_argument(
        "-obs",
        "--observation_space",
        type=str,
        default="",
        help='Options: "lidar", "camera", or "dict"',
    )

    # TD3, PPO, PPO_LSTM, DreamerV3
    parser.add_argument(
        "-algo",
        "--rl_algorithm",
        type=str,
        default="",
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

    if args.gym_env_id:
        env_config.gym_env_id = args.gym_env_id

    # make sure this is first!!!!
    if args.evaluate_only:
        # For evaluation, load evaluation configs - they should be in different dir!
        env_config, ros_config, rl_config = load_evaluation_configs()
        rl_config.evaluate_only = args.evaluate_only

    # print(f"1{rl_config=}")
    # overwrite the configs
    if args.learning_rate:
        rl_config.lr = args.learning_rate

    if args.lidar_noise:
        env_config.drl_robot_navigation.lidar_noise = args.lidar_noise
    if args.lidar_noise_size:
        env_config.drl_robot_navigation.lidar_noise_area_size[0] = args.lidar_noise_size
        env_config.drl_robot_navigation.lidar_noise_area_size[1] = args.lidar_noise_size

    if args.camera_noise:
        env_config.drl_robot_navigation.camera_noise = args.camera_noise
    if args.camera_noise_size:
        env_config.drl_robot_navigation.camera_noise_area_size[0] = (
            args.camera_noise_size
        )
        env_config.drl_robot_navigation.camera_noise_area_size[1] = (
            args.camera_noise_size
        )

    if args.static_spawn:
        env_config.scenario_settings.static_spawn = args.static_spawn

    if args.static_goal:
        env_config.scenario_settings.static_goal = args.static_goal

    if args.map_bounds:
        ros_config.map.min_xy[0] = -args.map_bounds
        ros_config.map.min_xy[1] = -args.map_bounds
        ros_config.map.max_xy[0] = args.map_bounds
        ros_config.map.max_xy[1] = args.map_bounds

    if args.max_training_steps:
        rl_config.max_training_steps = args.max_training_steps

    if args.observation_space:
        env_config.scenario_settings.observations.obs_space_type = (
            args.observation_space
        )

    if args.rl_algorithm:
        rl_config.algorithm = args.rl_algorithm

    if args.algo_notes:
        rl_config.experimental_notes = args.algo_notes

    if args.env_notes:
        env_config.experimental_notes = args.env_notes

    if args.load_model_path:
        rl_config.load_model_path = args.load_model_path

    run_name = ""
    if args.run_name:
        run_name = args.run_name

    launchfile = f"/home/{os.getenv('USER')}/{repo}/cranavgym/ros_interface/DroneNavSimple.launch"
    launch_ROS(launchfile, 11311)
    main(env_config, ros_config, rl_config, run_name)
