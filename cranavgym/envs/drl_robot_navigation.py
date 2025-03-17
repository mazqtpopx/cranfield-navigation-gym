import math
import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from typing import Union

from cranavgym.ros_interface.ros_interface import ROSInterface

import random
import threading

DISCRETE_ACTIONS = False


# import time
from scipy.spatial.transform import Rotation
from squaternion import Quaternion


# import pandas as pd
# from numba import njit

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# @njit
# def quaternion_to_euler(q):
#     x, y, z, w = q[0], q[1], q[2], q[3]

#     # Compute roll (x-axis rotation)
#     t0 = 2.0 * (w * x + y * z)
#     t1 = 1.0 - 2.0 * (x * x + y * y)
#     roll = np.arctan2(t0, t1)

#     # Compute pitch (y-axis rotation)
#     t2 = 2.0 * (w * y - z * x)
#     t2 = max(min(t2, 1.0), -1.0)  # Clamp to [-1,1]
#     pitch = np.arcsin(t2)

#     # Compute yaw (z-axis rotation)
#     t3 = 2.0 * (w * z + x * y)
#     t4 = 1.0 - 2.0 * (y * y + z * z)
#     yaw = np.arctan2(t3, t4)

#     return np.array([roll, pitch, yaw])

# @njit
# def euler_to_quaternion(roll, pitch, yaw):
#     cr, cp, cy = np.cos(np.array([roll, pitch, yaw]) / 2)
#     sr, sp, sy = np.sin(np.array([roll, pitch, yaw]) / 2)

#     w = cr * cp * cy + sr * sp * sy
#     x = sr * cp * cy - cr * sp * sy
#     y = cr * sp * cy + sr * cp * sy
#     z = cr * cp * sy - sr * sp * cy

#     return np.array([x, y, z, w])


# class StepProfilerPandas:
#     def __init__(self):
#         self.data = pd.DataFrame(columns=["step", "function", "duration"])

#     def record_time(self, step, function, duration):
#         new_row = pd.DataFrame(
#             {"step": [step], "function": [function], "duration": [duration]}
#         )
#         self.data = pd.concat([self.data, new_row], ignore_index=True)

#     def get_stats(self):
#         return self.data.groupby("function")["duration"].agg(["mean", "std", "count"])

#     def plot(self):
#         import matplotlib.pyplot as plt
#         import seaborn as sns

#         plt.figure(figsize=(10, 5))
#         sns.lineplot(data=self.data, x="step", y="duration", hue="function")
#         plt.xlabel("Step")
#         plt.ylabel("Duration (s)")
#         plt.title("Execution Time per Function Over Steps")
#         plt.legend(loc="upper right")
#         plt.show()


def move_forward(x, y, yaw, distance):
    x_new = x + distance * np.cos(yaw)
    y_new = y + distance * np.sin(yaw)

    return (x_new, y_new)


class DRLRobotNavigation(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    """The random room environment as presented in DRL robot navigation
    (https://github.com/reiniscimurs/DRL-robot-navigation)
    Use this for ....
    (NB: collisions - fixed!
    In the original, the collision system was based on the lidar return.
    This caused issues when adding noise to lidar values. 
    Here we change it so that it is based on the environment collision flags.)
    (NB2: we have a added a mechanism to output the camera image from the environment
    - in the dict gymnasium format)

    UPDATE THE DESCRIPTION!

    Modified the reward to be a sparse reward! 
    No more gradients the closer you get to the goal


    ## Action Space - Update the action space

    The action is a `ndarray` with shape `(2,)` which can take values `{0, 1}` indicating the direction
    of the fixed force the cart is pushed with.

    - 0: ..
    - 1: ...

    ## Observation Space - Update the obs space

    The observation is a `ndarray` with shape `(...,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |


    ## Rewards
    +100 for reaching the goal
    -100 for collision with wall

    
    ## Episode End
    The episode ends if any one of the following occurs:

    1. Termination: Collision with wall
    2. Termination: Robot reaches the goal
    3. Truncation: Episode length is greater than 1200

    
    inputs:
        lidar_dim: dim of lidar points (i.e. how many lidar points are outputted.)
        The entire lidar field is discretized into this amount of points.
        Note: this affects the output of the state size!

        launchfile: name of the launchfile as it appears in assets/{launchfile}.launch

        obs_space: output observation space options: 
        "lidar", "camera", "dict"
        lidar: outputs the lidar, distance to goal, and angle to goal
        img: outputs image from the camera onboard the robot only
        dict: outputs lidar, img, distance to goal, angle, etc. as a dictionary

        camera_noise: Boolean flag to enable/disable camera noise areas

        lidar_noise: Boolean flag to enable/disable lidar noise areas

    """

    def __init__(
        self,
        ros_interface_config,
        max_episode_steps=500,
        obs_space="lidar",
        reward_type="alternative",
        camera_noise=False,
        camera_noise_area_size=[4, 4],
        random_camera_noise_area=True,
        static_camera_noise_area_pos=[0, 0],
        camera_noise_type="gaussian",
        lidar_noise=False,
        lidar_noise_area_size=[4, 4],
        random_lidar_noise_area=True,
        static_lidar_noise_area_pos=[0, 0],
        static_goal=False,
        static_goal_xy=[3, 3],
        static_spawn=False,
        static_spawn_xy=[0, 0],
        # reward_threshold=100.0,
    ):
        conf = ros_interface_config
        self.ros = ROSInterface(
            launchfile=conf.ros.launchfile,
            ros_port=conf.ros.port,
            time_delta=conf.ros.step_pause_time_delta,
            map_min_xy=conf.map.min_xy,
            map_max_xy=conf.map.max_xy,
            img_width=conf.camera.img_width,
            img_height=conf.camera.img_height,
            camera_noise_type=camera_noise_type,
            lidar_dim=conf.lidar.lidar_dim,
            static_goal=static_goal,
            static_goal_xy=static_goal_xy,
            static_spawn=static_spawn,
            static_spawn_xy=static_spawn_xy,
        )

        self.current_step = 0
        self.max_episode_steps = max_episode_steps

        self.obs_space = obs_space
        self.camera_noise = camera_noise
        self.camera_noise_type = camera_noise_type
        self.lidar_noise = lidar_noise

        self._camera_noise_area_size = camera_noise_area_size
        self._random_camera_noise_area = random_camera_noise_area
        self._static_camera_noise_area_pos = static_camera_noise_area_pos
        self._lidar_noise_area_size = lidar_noise_area_size  # moved to ROS interface
        self._random_lidar_noise_area = random_lidar_noise_area
        self._static_lidar_noise_area_pos = static_lidar_noise_area_pos

        # ------------------------------------------REWARD--------------------------------------------
        self.reward_type = reward_type

        # -----------------------------------ACTION/OBS SPACE--------------------------------------------
        if DISCRETE_ACTIONS:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Box(
                np.array([0, -1]).astype(np.float32),
                np.array([+1, +1]).astype(np.float32),
            )  # PAN, TILT, ZOOM

        image = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(
                3,
                ros_interface_config.camera.img_width,
                ros_interface_config.camera.img_height,
            ),
            dtype=np.float32,
        )
        lidar = spaces.Box(low=0.0, high=1.0, shape=(20,), dtype=np.float32)
        robot_position = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        goal_position = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        dist_to_goal = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        angle_to_goal = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        goal_polar_coordinate = spaces.Box(
            low=np.array([0, -np.pi]), high=np.array([20, np.pi]), dtype=np.float32
        )
        actions = spaces.Box(
            low=np.array([0.0, -1.0]),  # Lower bounds for linear and angular velocities
            high=np.array([1.0, 1.0]),  # Upper bounds for linear and angular velocities
            dtype=np.float32,
        )

        if self.obs_space == "lidar":
            # self.observation_space = spaces.Box(
            #     low=0, high=20, shape=(24,), dtype=np.float32
            # )
            # self.observation_space = gym.spaces.Dict(
            #     {
            #         "lidar": lidar,
            #         "dist_to_goal": dist_to_goal,
            #         "angle_to_goal": angle_to_goal,
            #         "actions": actions,
            #     }
            # )
            # self.observation_space = spaces.Box(
            #     low=0.0, high=1.0, shape=(24,), dtype=np.float32
            # )
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(24,), dtype=np.float32
            )
        elif self.obs_space == "camera":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(
                    ros_interface_config.camera.img_width,
                    ros_interface_config.camera.img_height,
                    3,
                ),
                dtype=np.uint8,
            )
            # self.observation_space = spaces.Box(
            #     low=0.0, high=1.0, shape=(64,64,3), dtype=np.float32
            # )
            # self.observation_space = gym.spaces.Dict(
            #     {
            #         "camera": image,
            #         "robot_position": robot_position,
            #         "goal_position": goal_position,
            #         "goal_polar_coordinate": goal_polar_coordinate,
            #         "actions": actions,
            #     }
            # )
        elif self.obs_space == "dict":
            self.observation_space = gym.spaces.Dict(
                {
                    "camera": image,
                    "lidar": lidar,
                    "robot_position": robot_position,
                    "goal_position": goal_position,
                    "goal_polar_coordinate": goal_polar_coordinate,
                    "actions": actions,
                }
            )

        # self.collision_detection = False

        # -----------------------------------INIT VARIABLES--------------------------------------------------
        # init odom and goal positions
        # position of the robot
        # self.pos_x, self.pos_y = 0, 0
        # position of the goal
        # The init goal cannot be 0 because it cannot be the same as the robot init
        # (it causes issues when calculating distance/angle between the two)
        self.last_odom = None

        self.render_mode = "rgb_array"

        # Add to init!
        self.goal_reached_threshold = 0.5

        # self.profiler = StepProfilerPandas()
        # self.global_step = 0

        # self.max_episode_steps = 500  # again add to init

    """
    inputs: 
        action: action performed by the agent. The action format depends on whether the discrete_action flag was 
        called true during the initialization of the flag.
        For discrete actions the actions are:
        0:
        1:
        2:
        For continuous actions the actions are in a range:
        [0]: range [0,1]; 0 is stop, 1 is go forward at full speed
        [1]: range [-1,1]; -1 is move to the left, 1 is move to the right

    """

    # Perform an action and read a new state
    def step(self, action):
        # self.ros.set_robot_velocity(0.0, 0.0, 0.0)
        # threading.Thread(target=self.ros.unpause(), daemon=True).start()
        # self.ros.unpause()
        # perform actions (set the action to be the velocity of the robot)
        self._perform_action(action)

        # self._pause_ROS()

        state = self._get_state(self.obs_space, action)

        # get reward:
        # -100 if collision
        # +100 if reached reward
        # .... otherwise
        # First, get the collison status
        collided = self.ros.get_collision_status()

        dist_to_goal, angle_to_goal = self._get_dist_and_angle_to_goal()

        # dist_to_goal = 4
        # angle_to_goal = 0

        #     # Set the reached goal flag (true if distance to goal is below the threshold)
        #     # move to get_state_observation
        #     # NB: make sure to use the unnormalized version of dist_to_goal!
        reached_goal = dist_to_goal < self.goal_reached_threshold

        if self.reward_type == "original":
            reward = self._get_reward_original(
                reached_goal, collided, action, min(self.ros.get_velodyne_data)
            )
        elif self.reward_type == "alternative":
            reward = self._get_reward_alternative(reached_goal, collided)
        # check if scenario is done
        terminated, truncated = self._is_done(
            collided, reached_goal, self.current_step, self.max_episode_steps
        )

        x_vel, y_vel = self.ros.get_robot_velocity()
        info = {
            "x_position": self.ros.robot_position[0],
            "y_position": self.ros.robot_position[1],
            "x_velocity": x_vel,
            "y_velocity": y_vel,
            "dist_to_target": dist_to_goal,
            "angle_to_goal": angle_to_goal,
            "reward": reward,
        }
        if terminated:
            info["terminal_observation"] = state

        self.current_step = self.current_step + 1

        # self.ros.pause()
        return state, reward, terminated, truncated, info

    # def step(self, action):

    #     self.global_step += 1

    #     start_time = time.perf_counter()

    #     # threading.Thread(target=self.ros.unpause(), daemon=True).start()
    #     # self.ros.unpause()

    #     sub_start = time.perf_counter()
    #     self._perform_action(action)
    #     self.profiler.record_time(self.global_step, "_perform_action", time.perf_counter() - sub_start)

    #     sub_start = time.perf_counter()
    #     state = self._get_state(
    #         self.obs_space, action
    #     )
    #     self.profiler.record_time(self.global_step, "_get_state", time.perf_counter() - sub_start)

    #     sub_start = time.perf_counter()
    #     collided = self.ros.get_collision_status()
    #     self.profiler.record_time(self.global_step, "get_collision_status", time.perf_counter() - sub_start)

    #     sub_start = time.perf_counter()
    #     dist_to_goal, angle_to_goal = self._get_dist_and_angle_to_goal()
    #     # Set the reached goal flag (true if distance to goal is below the threshold)
    #     # move to get_state_observation
    #     # NB: make sure to use the unnormalized version of dist_to_goal!
    #     reached_goal = dist_to_goal < self.goal_reached_threshold
    #     self.profiler.record_time(self.global_step, "_get_dist_and_angle_to_goal", time.perf_counter() - sub_start)

    #     sub_start = time.perf_counter()
    #     if self.reward_type == "original":
    #         reward = self._get_reward_original(
    #             reached_goal, collided, action, min(self.ros.get_velodyne_data)
    #         )
    #     elif self.reward_type == "alternative":
    #         reward = self._get_reward_alternative(reached_goal, collided)
    #     self.profiler.record_time(self.global_step, "reward_computation", time.perf_counter() - sub_start)

    #     sub_start = time.perf_counter()
    #     terminated, truncated = self._is_done(
    #         collided, reached_goal, self.current_step, self.max_episode_steps
    #     )
    #     self.profiler.record_time(self.global_step, "_is_done", time.perf_counter() - sub_start)

    #     sub_start = time.perf_counter()
    #     x_vel, y_vel = self.ros.get_robot_velocity()
    #     self.profiler.record_time(self.global_step, "get_robot_velocity", time.perf_counter() - sub_start)

    #     sub_start = time.perf_counter()
    #     info = {
    #         "x_position": self.ros.robot_position[0],
    #         "y_position": self.ros.robot_position[1],
    #         "x_velocity": x_vel,
    #         "y_velocity": y_vel,
    #         "dist_to_target": dist_to_goal,
    #         "angle_to_goal": angle_to_goal,
    #         "reward": reward,
    #     }
    #     if terminated:
    #         info["terminal_observation"] = state
    #     self.profiler.record_time(self.global_step, "info_creation", time.perf_counter() - sub_start)

    #     self.current_step = self.current_step + 1

    #     # self.ros.pause()
    #     total_time = time.perf_counter() - start_time
    #     # self.profiler.record_time(self.global_step, "total_step", total_time)
    #     # print(f"\n\n")

    #     return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, **kwargs):
        super().reset(seed=seed)

        # self.ros.set_robot_velocity(0.0, 0.0, 0.0)
        self._reset_ROS()
        self.ros.pause()
        self._respawn_robot()
        self._reset_goal()

        self._reset_noise_areas()

        self.ros.reset_collision_status()

        # Move these inputs to the init
        state = self._get_state(self.obs_space, [0, 0])

        self.current_step = 0

        x_vel, y_vel = self.ros.get_robot_velocity()
        dist_to_goal, angle_to_goal = self._get_dist_and_angle_to_goal()
        info = {
            "x_position": self.ros.robot_position[0],
            "y_position": self.ros.robot_position[1],
            "x_velocity": x_vel,
            "y_velocity": y_vel,
            "dist_to_target": dist_to_goal,
            "angle_to_goal": angle_to_goal,
            # "terminal_observation": False,
        }
        self.ros.unpause()
        # time.sleep(0.1)
        # self._pause_ROS()

        # print(self.profiler.get_stats())  # Mean and std for each function
        # self.profiler.plot()
        return state, info

    def render(self):
        return self.ros.get_camera_data()

    def _get_dist_and_angle_to_goal(self):
        # should move funct of dist to goal/angle to goal to ros interface
        goal_x, goal_y = self.ros.get_goal_position()
        quaternion = self.ros.get_robot_quaternion()

        dist_to_goal, angle_to_goal = self._convert_quaternion_to_angles(
            quaternion,
            self.ros.robot_position[0],
            self.ros.robot_position[1],
            goal_x,
            goal_y,
        )
        # dist in meters, angle in degrees
        return dist_to_goal, angle_to_goal

    # ----------------------STEP FUNCTIONS-------------------------
    # @profile
    def _perform_action(self, action):
        # Publish the robot action
        if DISCRETE_ACTIONS:
            pos = self.ros.robot_position
            x = pos[0]
            y = pos[1]
            quat = self.ros.get_robot_quaternion()
            # euler = quat.as_euler('xyz', degrees=False)

            # euler = quaternion_to_euler(quat)
            euler = quat.to_euler(degrees=False)

            yaw = euler[2]
            if action == 0:
                # no action
                # self.ros.set_robot_velocity(0.0, 0.0, 0.0)
                return
                # move forward
            elif action == 1:
                # default for map
                x_new, y_new = move_forward(x, y, euler[2], 0.1)

                # for flight arena
                # x_new, y_new = move_forward(x, y, euler[2], 0.3)

                # x_new, y_new = move_forward(x, y, euler[2], 1.0)
                self.ros.set_robot_position(x_new, y_new, quat)

                # self.ros.set_robot_velocity(x_new, y_new, 0.0)

                # good for actual training
                # self.ros.set_robot_velocity(1.0, 0.0)
                # self.ros.set_robot_velocity(1.0, 0.0)
                # works well for humans (keyboard)
                # self.ros.set_robot_velocity(0.02, 0.0)

                # self.ros.set_robot_velocity(2.5, 0.0)
                return
                # move left
            elif action == 2:
                # x_new, y_new = move_forward(x, y, quat, 0.5)
                yaw = yaw - 0.3
                # quat_new = Rotation.from_euler('xyz', [euler[0], euler[1], yaw], degrees=False)
                # quat_new = euler_to_quaternion(euler[0], euler[1], yaw)
                quat_new = Quaternion.from_euler(euler[0], euler[1], yaw)

                self.ros.set_robot_position(x, y, quat_new)
                return
            elif action == 3:
                # x_new, y_new = move_forward(x, y, quat, 0.5)
                yaw = yaw + 0.3
                # quat_new = Rotation.from_euler('xyz', [euler[0], euler[1], yaw], degrees=False)
                # quat_new = euler_to_quaternion(euler[0], euler[1], yaw)

                quat_new = Quaternion.from_euler(euler[0], euler[1], yaw)
                self.ros.set_robot_position(x, y, quat_new)
                return
                # move right
        else:
            # CONTINUOUS ACTIONS
            # 7/3/25 - this is broken MW special, linx/y will need to be a vector
            # self.ros.set_robot_velocity(action[0], action[1], 0.0)

            # convert (local) forward velocity to global x/y velocity
            # first get the robot pose vector (quat) and convert to euler and scale by the forward velocity action
            quat = self.ros.get_robot_quaternion()
            euler = quat.to_euler(degrees=False)
            yaw = euler[2]

            # scale actions
            action[0] *= 2
            action[1] *= 6

            x = math.cos(yaw) * action[0]
            y = math.sin(yaw) * action[0]

            self.ros.set_robot_velocity(x, y, action[1])  # 15/03changes interface

            # Move to ros interface...
            # Publish visualization markers for debugging or monitoring
            self.ros.publish_velocity(action)

        # self.ros.publish_goal()
        return

    """Pauses the current python script to let the ROS/gazebo
    simulation execute the action
    inputs:
        time_delta: the amount of time to sleep for
        in order to let the simulation execute the actions
    """

    def unpause(self):
        self.ros.unpause()

    def _pause_ROS(self):
        self.ros.pause_ros()

    def _reset_ROS(self):
        self.ros.reset_ros()

    def _close_ROS(self):
        self.ros.close_ros()

    def _get_state(self, obs_space, action):
        if obs_space == "lidar":

            # should move out of _get state and instead set these as inputs to
            # increase lidar proc speed
            robot_x, robot_y = self.ros.robot_position
            goal_x, goal_y = self.ros.get_goal_position()

            quaternion = self.ros.get_robot_quaternion()

            dist_to_goal, angle_to_goal = self._convert_quaternion_to_angles(
                quaternion, robot_x, robot_y, goal_x, goal_y
            )

            # get the lidar data
            lidar_state = np.array([self.ros.get_velodyne_data()])
            # NB: moved to obs_space lidar so that these are not called for camera obs space

            # Calculate robot heading from odometry data

            robot_state = [
                robot_x,
                robot_y,
                goal_x,
                goal_y,
                dist_to_goal,
                angle_to_goal,
                action[0],
                action[1],
            ]

            # normalize values between 0-1
            lidar_state_normalized = self._normalize_lidar(lidar_state)
            dist_to_goal_normalized = self._normalize_dist_to_goal(dist_to_goal)
            angle_to_goal_normalized = self._normalize_angle_rad(angle_to_goal)
            # state = self._pack_state_lidar_dict(
            #     dist_to_goal, angle_to_goal, lidar_state, action
            # )
            state = self._pack_state_lidar(
                dist_to_goal_normalized,
                angle_to_goal_normalized,
                lidar_state_normalized,
                action,
            )
        elif obs_space == "camera":
            camera_state = self.ros.get_camera_data()
            state = self._pack_state_img(camera_state)
            # state = np.transpose(camera_state, (2, 1, 0))
        elif obs_space == "dict":
            state = self._pack_state_dict(robot_state, lidar_state)

        return state

    def _convert_quaternion_to_angles(self, quaternion, pos_x, pos_y, goal_x, goal_y):
        # euler = quaternion.as_euler('xyz', degrees=False)
        euler = quaternion.to_euler(degrees=False)
        # euler = quaternion_to_euler(quaternion)
        yaw = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        dist_to_goal = np.linalg.norm([pos_x - goal_x, pos_y - goal_y])

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = goal_x - pos_x
        skew_y = goal_y - pos_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        total_mag = mag1 * mag2
        if total_mag != 0:
            beta = math.acos(dot / (mag1 * mag2))
        else:
            beta = 0
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - yaw
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        angle_to_goal = theta

        return dist_to_goal, angle_to_goal

    # Because the lidar readings are clipped at 10.0, it is the
    # absolute max value of the state.
    def _normalize_lidar(self, lidar_state):
        return np.divide(lidar_state, 10.0)

    def _normalize_dist_to_goal(self, dist_to_goal):
        # clip when dist is above 10.0
        if dist_to_goal > 10.0:
            dist_to_goal = 10.0
        return dist_to_goal / 10.0

    # normalize between 0-1
    def _normalize_angle_rad(self, angle):
        angle += np.pi
        return angle / (2 * np.pi)

    # define the state
    def _pack_state_lidar(self, dist_to_goal, angle_to_goal, lidar_state, action):
        lidar_state = np.array(lidar_state, dtype=np.float32)
        robot_state = np.array(
            [dist_to_goal, angle_to_goal, action[0], action[1]], dtype=np.float32
        )
        state = np.append(robot_state, lidar_state)
        return state

    def _pack_state_lidar_dict(self, dist_to_goal, angle_to_goal, lidar_state, actions):
        print(f"{actions=}")
        # print(f"{actions.shape=}")
        actions = np.array(actions)
        print(f"{actions=}")
        print(f"{actions.shape=}")
        state = gym.spaces.Dict(
            {
                "lidar": np.array(lidar_state, dtype=np.float32),
                "dist_to_goal": np.array(dist_to_goal, dtype=np.float32),
                "angle_to_goal": np.array(angle_to_goal, dtype=np.float32),
                # "actions": np.array(actions, dtype=np.float32),
            }
        )
        return state

    # def _pack_state_lidar_dict_debug(self, lidar_state):
    #     # print(f"1{lidar_state=}")
    #     state = np.array(lidar_state, dtype=np.float32)
    #     # print(f"2{state=}")
    #     # print(f"3{state[0]=}")
    #     return state[0]

    #!!!!! NB !!!!!!
    # we have to take the first to match the observation state specified
    # if we don't do this, we return a np.array([[]])
    # instead, we simply want np.array([])
    #!!!!! NB !!!!!!

    # def _pack_state_lidar_dict(self, dist_to_goal, angle_to_goal, lidar_state, action):
    #     state = {
    #         "lidar": lidar_state,
    #         "dist_to_goal": dist_to_goal,
    #         "angle_to_goal": angle_to_goal,
    #         "actions": action,
    #     }
    #     return state

    def _pack_state_img(self, camera_state):
        # swap from (w,h,c) to (c,w,h)
        # if camera_state is None:
        #     return np.zeros((3,64,64))

        # state = np.transpose(camera_state, (2, 1, 0))
        # divide by 255 to conver from uint8 to float [0,1]
        # camera_state = camera_state / 255.0

        # return np.array(camera_state, dtype=np.float32)
        return np.array(camera_state, dtype=np.uint8)

    def _pack_state_dict(self, robot_state, laser_state, camera_state):
        state = spaces.Dict(
            {
                "camera": camera_state,
                "lidar": np.array(laser_state, dtype=np.float32),
                "robot_position": np.array(robot_state[0:2], dtype=np.float32),
                "goal_position": np.array(robot_state[2:4], dtype=np.float32),
                "goal_polar_coordinate": np.array(robot_state[4:6], dtype=np.float32),
                "actions": np.array(robot_state[6:], dtype=np.float32),
            }
        )
        return state

    """Computes whether the scenario is done.
    Conditions for being done are: 
    collision detected
    reached goal
    """

    def _is_done(
        self, collision_detected, reached_goal, current_step, max_episode_steps
    ):
        """
        returns teminated and truncated
        terminated means: the robot hit a wall or found the goal
        truncated means: max time steps
        """
        if collision_detected:
            return True, False
        elif reached_goal:
            return True, False
        elif current_step >= max_episode_steps:
            return False, True
        else:
            return False, False

    # def __get_reward(self, target, collision):
    #     if target:
    #         print("------------------Reached the goal-------------------")
    #         return 1.0
    #     elif collision:
    #         return -1.0
    #     else:
    #         return 1 / self.max_episode_steps

    def _get_reward_alternative(self, target, collision):
        if target:
            print("------------------Reached the goal-------------------")
            return 1.0
        elif collision:
            return -0.5
        else:
            return -(1 / self.max_episode_steps)

    def _get_reward_original(self, target, collision, action, min_laser):
        if target:
            print("------------------Reached the goal-------------------")
            return 1.0
        elif collision:
            return -1.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

    # move to ROS interface
    def _respawn_robot(self):
        self.ros.respawn_robot()

    def _reset_goal(self):
        self.ros.reset_goal()

    def _reset_noise_areas(self):
        if self.camera_noise:
            self._reset_sensor_noise(
                self._camera_noise_area_size,
                self._random_camera_noise_area,
                self._static_camera_noise_area_pos,
                "camera",
            )

        if self.lidar_noise:
            self._reset_sensor_noise(
                self._lidar_noise_area_size,
                self._random_lidar_noise_area,
                self._static_lidar_noise_area_pos,
                "lidar",
            )

    def _reset_sensor_noise(
        self, size, random_area_spawn, static_position_centre, sensor
    ):
        """
        Resets camera/lidar noise area.
        If random camera noise is enabled, random xy pos is generated.
        Otherwise, the specified xy (from config) pos is selected.

        size: (xy tuple) size of the area in meters
        random_area_spawn: (bool) is the position of the area random?
        static_position_centre: (xy tuple) static centre  of the noise area (only used if random is True!)
        sensor: "lidar" or "camera"
        """
        w = size[0]
        h = size[1]

        if random_area_spawn:
            # centre of the rectangle xy pos
            x = random.uniform(-5 + (w / 2), 5 - (w / 2))
            y = random.uniform(-5 + (h / 2), 5 - (h / 2))
        else:
            x = static_position_centre[0]
            y = static_position_centre[1]

        if sensor == "camera":
            self.ros.reset_camera_noise_area(x, y, w, h)
        elif sensor == "lidar":
            self.ros.reset_lidar_noise_area(x, y, w, h)
