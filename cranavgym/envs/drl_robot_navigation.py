import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Union
from cranavgym.ros_interface.ros_interface import ROSInterface
import random

# Flag to switch between discrete and continuous actions
DISCRETE_ACTIONS = False

class DRLRobotNavigation(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    Environment for Deep Reinforcement Learning (DRL) based robot navigation.
    
    This environment simulates a robot navigating through a room, 
    equipped with lidar and camera sensors. The robot's task is to reach a 
    goal while avoiding collisions with the walls. It supports discrete and 
    continuous action spaces, and various observation space configurations.

    Attributes:
        ros: Interface to communicate with ROS for controlling the robot.
        obs_space: The type of observation space (lidar, camera, or dict).
        reward_type: The reward system to be used ('original' or 'alternative').
        camera_noise: Boolean indicating if camera noise is enabled.
        lidar_noise: Boolean indicating if lidar noise is enabled.
        max_episode_steps: The maximum number of steps allowed per episode.
        render_mode: Defines the rendering mode (e.g., 'human', 'rgb_array').
        current_step: Counter to track the number of steps in the current episode.
    """
    """
    ### Description
    A reinforcement learning environment for robot navigation using ROS.

    This environment is based on the [DRL robot navigation framework](https://github.com/reiniscimurs/DRL-robot-navigation).

    The environment simulates a robot navigating towards a goal while avoiding obstacles.
    It supports both discrete and continuous action spaces and provides observations in various configurations, such as LiDAR data, camera images, or a combination in a dictionary.

    There are three configuration files that can be used to configure the environment:

    - `ros_interface_config`: Configuration object for the ROS interface.
    - `rl_interface_config`: Configuration object for the RL interface.
    - `env_config`: Configuration object for the environment.

    **Features:**

    - **Collision Handling:** Accurate collision detection based on environment collision flags, ensuring reliable detection even with sensor noise.
    - **Sensor Noise Areas:** Ability to simulate areas with camera and LiDAR noise to test the robustness of navigation algorithms under adverse conditions.
    - **Sparse Rewards:** A reward function designed to encourage efficient navigation by providing rewards primarily when the robot reaches the goal or collides.
    # TODO add more features in the future:
        - **Moving Goal:** Option to enable a moving goal scenario where the goal moves along a predefined path.
        - **Soft Body Obstacles:** Obstacles that can be pushed out of the way by the robot.
        - **Dynamic Obstacles:** Moving obstacles that the robot must avoid.

    ### Action Space

    The action space can be either **continuous** or **discrete**, determined by the `action_space_type` parameter.

    - **Continuous Actions (`action_space_type='continuous'`):**

        The action is a `numpy.ndarray` of shape defined by `action_space_shape` with values in the range `action_space_low` to `action_space_high`.

        For example, with:

        - `action_space_shape=2`
        - `action_space_low=-1`
        - `action_space_high=1`

        Each element in the action array represents a control input to the robot:

        | Index | Control Input                  | Min  | Max |
        |-------|--------------------------------|------|-----|
        | 0     | Linear velocity (forward/back) |  0.0 | 1.0 |
        | 1     | Angular velocity (rotation)    | -1.0 | 1.0 |


    - **Discrete Actions (`action_space_type='discrete'`):**

        The action is an integer representing predefined movements:

        | Action | Description     |
        |--------|-----------------|
        | 0      | Move forward    |
        | 1      | Turn left       |
        | 2      | Turn right      |

        ```python
        self.action_space = gym.spaces.Discrete(3)
        ```

    ### Observation Space

    - **LiDAR Observations :**

        The observation is a `numpy.ndarray` containing:

        - **LiDAR Readings:** Normalized distances from LiDAR sensors.
        - **Distance to Goal:** Normalized distance between the robot and the goal.
        - **Angle to Goal:** Normalized angle between the robot's orientation and the goal direction.
        - **Last Action Taken:** Normalized representation of the last action.
        - **Shape:** `(lidar_dim + 2 + last_actions,)`
            - `lidar_dim`: Number of LiDAR readings (e.g., 20).

    - **Camera Observations :**

        The observation is an RGB image from the robot's onboard camera.

        ```python
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(img_height, img_width, 3),
            dtype=np.uint8
        )
        ```

    - **Dictionary Observations :** NOT TESTED

    The observation is a dictionary containing:

      - **'camera'**: Camera images.
      - **'lidar'**: LiDAR data.
      - **'position'**: Robot's current position.
      - **'goal_position'**: Goal position.
      - **'goal_polar_coords'**: Distance and angle to the goal.
      - **'last_action'**: Last action taken.

        ```python
        self.observation_space = gym.spaces.Dict({
            'camera': gym.spaces.Box(...),
            'lidar': gym.spaces.Box(...),
            'position': gym.spaces.Box(...),
            'goal_position': gym.spaces.Box(...),
            'goal_polar_coords': gym.spaces.Box(...),
            'last_action': gym.spaces.Discrete(...) or gym.spaces.Box(...)
        })
        ```

    ### Rewards

    The environment provides sparse rewards to encourage efficient navigation:

    - **+1.0** when the robot reaches the goal.
    - **-1.0** when the robot collides with an obstacle.
    - **-0.01** per time step to encourage faster goal-reaching.

    ### Starting State

    - The robot starts at a random position within the map boundaries unless `static_spawn` is set to `True`, in which case it starts at `static_spawn_xy`.
    - The goal position is random within the map boundaries unless `static_goal` is set to `True`, in which case it is set to `static_goal_xy`.

    ### Episode Termination

    An episode ends when any of the following occurs:

    1. **Collision:** The robot collides with an obstacle.
    2. **Goal Reached:** The robot reaches the goal position within a specified threshold distance.
    3. **Max Steps Exceeded:** The episode length exceeds `max_ep_steps`.

    ### Solved Requirements

    The environment is considered solved when the agent consistently reaches the goal without collisions over a significant number of episodes. Specific thresholds can be defined based on the application's requirements.

    ### Parameters

    - **Environment Configuration:**
        - `ros_interface_config` (object): Configuration object for the ROS interface.
        - `rl_interface_config` (object): Configuration object for the RL interface.
        - `env_config` (object): Configuration object for the environment.
        - `max_ep_steps` (int): Maximum number of steps per episode.
        - `lidar_dim` (int): Number of LiDAR readings (e.g., `20`).

    - **Action Space Configuration:**
        - `action_space_type` (str): Type of action space (`'continuous'` or `'discrete'`).
        - `action_space_shape` (int or tuple): Shape of the action space for continuous actions (e.g., `3`).
        - `action_space_low` (float or array-like): Lower bounds for continuous action space (e.g., `-1`).
        - `action_space_high` (float or array-like): Upper bounds for continuous action space (e.g., `1`).

    - **Reward Configuration:**
        - `reward_type` (str): Type of reward function (`'alternative'` or `'original'`).

    - **Noise Configuration:**
        - `camera_noise` (bool): Enable or disable camera noise areas.
        - `camera_noise_area_size` (list of float): Size `[width, height]` of the camera noise area.
        - `random_camera_noise_area` (bool): Randomize the position of the camera noise area.
        - `static_camera_noise_area_pos` (list of float): Static position `[x, y]` of the camera noise area.
        - `camera_noise_type` (str): Type of camera noise (e.g., `'gaussian'`).
        - `lidar_noise` (bool): Enable or disable LiDAR noise areas.
        - `lidar_noise_area_size` (list of float): Size `[width, height]` of the LiDAR noise area.
        - `random_lidar_noise_area` (bool): Randomize the position of the LiDAR noise area.
        - `static_lidar_noise_area_pos` (list of float): Static position `[x, y]` of the LiDAR noise area.

    - **Goal and Spawn Configuration:**
        - `static_goal` (bool): Use a static goal position.
        - `static_goal_xy` (list of float): Static goal position `[x, y]`.
        - `static_spawn` (bool): Spawn the robot at a static position.
        - `static_spawn_xy` (list of float): Static spawn position `[x, y]`.

    ### Version History

    - **v1:** First version.

    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        ros_interface_config,
        max_episode_steps=50,
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
    ):
        """
        Initialize the DRLRobotNavigation environment.
        
        Args:
            ros_interface_config: Configuration for the ROS interface.
            max_episode_steps: Maximum number of steps per episode.
            obs_space: Observation space type ('lidar', 'camera', 'dict').
            reward_type: Reward system ('original', 'alternative').
            camera_noise: Enable or disable camera noise.
            camera_noise_area_size: Size of the camera noise area.
            random_camera_noise_area: Randomize camera noise area position.
            static_camera_noise_area_pos: Fixed position for camera noise.
            camera_noise_type: Type of noise to apply on the camera ('gaussian').
            lidar_noise: Enable or disable lidar noise.
            lidar_noise_area_size: Size of the lidar noise area.
            random_lidar_noise_area: Randomize lidar noise area position.
            static_lidar_noise_area_pos: Fixed position for lidar noise.
            static_goal: Use static goal or not.
            static_goal_xy: Static goal position.
            static_spawn: Use static spawn or not.
            static_spawn_xy: Static spawn position.
        """
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
        self.lidar_noise = lidar_noise

        # Store sensor noise configurations
        self._camera_noise_area_size = camera_noise_area_size
        self._random_camera_noise_area = random_camera_noise_area
        self._static_camera_noise_area_pos = static_camera_noise_area_pos
        self._lidar_noise_area_size = lidar_noise_area_size
        self._random_lidar_noise_area = random_lidar_noise_area
        self._static_lidar_noise_area_pos = static_lidar_noise_area_pos

        # Reward type setting
        self.reward_type = reward_type

        # Define action and observation spaces
        if DISCRETE_ACTIONS:
            self.action_space = spaces.Discrete(3)  # Example: Forward, Left, Right
        else:
            self.action_space = spaces.Box(
                np.array([0, -1]).astype(np.float32),
                np.array([+1, +1]).astype(np.float32),
            )  # Example for continuous action: (linear velocity, angular velocity)

        # Define observation spaces based on observation type
        if self.obs_space == "lidar":
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(24,), dtype=np.float32
            )
        elif self.obs_space == "camera":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(
                    conf.camera.img_width,
                    conf.camera.img_height,
                    3,
                ),
                dtype=np.uint8,
            )
        elif self.obs_space == "dict":
            self.observation_space = spaces.Dict(
                {
                    "camera": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(3, conf.camera.img_width, conf.camera.img_height),
                        dtype=np.float32,
                    ),
                    "lidar": spaces.Box(low=0.0, high=1.0, shape=(20,), dtype=np.float32),
                    "robot_position": spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
                    "goal_position": spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
                    "goal_polar_coordinate": spaces.Box(
                        low=np.array([0, -np.pi]), high=np.array([20, np.pi]), dtype=np.float32
                    ),
                    "actions": spaces.Box(
                        low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
                    ),
                }
            )

        self.goal_reached_threshold = 0.5
        self.render_mode = "rgb_array"

    def step(self, action):
        """
        Perform a single step in the environment based on the action taken.

        Args:
            action: The action taken by the agent. It can be discrete or continuous depending on the configuration.

        Returns:
            state: The new state of the environment after the action.
            reward: The reward received after taking the action.
            done: Boolean indicating if the episode is finished.
            info: Additional information about the environment state.
        """
        self._perform_action(action)
        self._pause_ROS()
        state, reached_goal = self._get_state(self.obs_space, self.goal_reached_threshold, action)

        # Calculate reward based on collision or reaching the goal
        collided = self.ros.get_collision_status()
        if self.reward_type == "original":
            reward = self._get_reward_original(
                reached_goal, collided, action, min(self.ros.get_velodyne_data)
            )
        else:
            reward = self._get_reward_alternative(reached_goal, collided)

        # Check if the episode is done
        done = self._is_done(collided, reached_goal, self.current_step, self.max_episode_steps)

        dist_to_goal, angle_to_goal = self._get_dist_and_angle_to_goal()

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

        self.current_step += 1

        return state, reward, done, False, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state at the beginning of a new episode.

        Args:
            seed: Seed for random number generation.
            options: Additional options for the reset process.

        Returns:
            state: The initial state of the environment.
            info: Additional information about the initial state.
        """
        super().reset(seed=seed)
        self._reset_ROS()
        self._respawn_robot()
        self._reset_goal()
        self._reset_noise_areas()
        self._pause_ROS()

        state, _ = self._get_state(self.obs_space, self.goal_reached_threshold, [0, 0])
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
        }

        return state, info

    def render(self):
        """
        Render the current environment state.

        Returns:
            The camera image data from the robot's onboard camera.
        """
        return self.ros.get_camera_data()

    def _get_dist_and_angle_to_goal(self):
        """
        Calculate the distance and angle to the goal.

        Returns:
            dist_to_goal: The distance to the goal in meters.
            angle_to_goal: The angle to the goal in radians.
        """
        goal_x, goal_y = self.ros.get_goal_position()
        quaternion = self.ros.get_robot_quaternion()

        dist_to_goal, angle_to_goal = self._convert_quaternion_to_angles(
            quaternion,
            self.ros.robot_position[0],
            self.ros.robot_position[1],
            goal_x,
            goal_y,
        )
        return dist_to_goal, angle_to_goal

    def _perform_action(self, action):
        """
        Execute the action in the environment.

        Args:
            action: The action to perform, controlling robot movement.
        """
        if DISCRETE_ACTIONS:
            if action == 0:
                # Move forward
                pass
            elif action == 1:
                # Move left
                pass
            elif action == 2:
                # Move right
                pass
        else:
            # Set continuous action (linear and angular velocities)
            self.ros.set_robot_velocity(action[0], action[1])
            self.ros.publish_velocity(action)

        self.ros.publish_goal()

    def _pause_ROS(self):
        """
        Pause the ROS simulation to let the environment process the actions.
        """
        self.ros.pause_ros()

    def _reset_ROS(self):
        """
        Reset the ROS environment to its initial state.
        """
        self.ros.reset_ros()

    def _is_done(self, collision_detected, reached_goal, current_step, max_episode_steps):
        """
        Check if the episode has ended.

        Args:
            collision_detected: Whether the robot has collided.
            reached_goal: Whether the robot has reached its goal.
            current_step: The current step number in the episode.
            max_episode_steps: The maximum number of allowed steps in the episode.

        Returns:
            True if the episode has ended, otherwise False.
        """
        if collision_detected or reached_goal or current_step >= max_episode_steps:
            return True
        return False

    def _get_reward_alternative(self, target, collision):
        """
        Compute the reward using the alternative reward system.

        Args:
            target: Boolean indicating if the goal has been reached.
            collision: Boolean indicating if a collision has occurred.

        Returns:
            The computed reward.
        """
        if target:
            return 1.0
        elif collision:
            return -1.0
        return -(1 / self.max_episode_steps)

    def _get_reward_original(self, target, collision, action, min_laser):
        """
        Compute the reward using the original reward system.

        Args:
            target: Boolean indicating if the goal has been reached.
            collision: Boolean indicating if a collision has occurred.
            action: The action taken by the agent.
            min_laser: The minimum lidar reading.

        Returns:
            The computed reward.
        """
        if target:
            return 1.0
        elif collision:
            return -1.0

        r3 = lambda x: 1 - x if x < 1 else 0.0
        return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

    def _get_state(self, obs_space, goal_reached_threshold, action):
        """
        Get the current state of the environment.

        Args:
            obs_space: The type of observation space ('lidar', 'camera', 'dict').
            goal_reached_threshold: Distance threshold to determine if the goal is reached.
            action: The action taken by the agent.

        Returns:
            state: The current state of the environment.
            reached_goal: Boolean indicating if the goal is reached.
        """
        lidar_state = np.array([self.ros.get_velodyne_data()])
        robot_x, robot_y = self.ros.robot_position
        goal_x, goal_y = self.ros.get_goal_position()
        quaternion = self.ros.get_robot_quaternion()

        dist_to_goal, angle_to_goal = self._convert_quaternion_to_angles(
            quaternion, robot_x, robot_y, goal_x, goal_y
        )
        camera_state = self.ros.get_camera_data()

        # Normalize state values between 0 and 1
        lidar_state_normalized = self._normalize_lidar(lidar_state)
        dist_to_goal_normalized = self._normalize_dist_to_goal(dist_to_goal)
        angle_to_goal_normalized = self._normalize_angle_rad(angle_to_goal)

        if obs_space == "lidar":
            state = self._pack_state_lidar(
                dist_to_goal_normalized,
                angle_to_goal_normalized,
                lidar_state_normalized,
                action,
            )
        elif obs_space == "camera":
            state = self._pack_state_img(camera_state)
        elif obs_space == "dict":
            robot_state = [
                robot_x, robot_y, goal_x, goal_y, dist_to_goal, angle_to_goal, action[0], action[1]
            ]
            state = self._pack_state_dict(robot_state, lidar_state)

        reached_goal = dist_to_goal < goal_reached_threshold
        return state, reached_goal

    def _convert_quaternion_to_angles(self, quaternion, pos_x, pos_y, goal_x, goal_y):
        """
        Convert quaternion to Euler angles to calculate the distance and angle to the goal.

        Args:
            quaternion: The quaternion representing robot orientation.
            pos_x: X position of the robot.
            pos_y: Y position of the robot.
            goal_x: X position of the goal.
            goal_y: Y position of the goal.

        Returns:
            dist_to_goal: Distance to the goal.
            angle_to_goal: Angle to the goal in radians.
        """
        euler = quaternion.to_euler(degrees=False)
        yaw = round(euler[2], 4)

        dist_to_goal = np.linalg.norm([pos_x - goal_x, pos_y - goal_y])
        skew_x = goal_x - pos_x
        skew_y = goal_y - pos_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        beta = math.acos(dot / (mag1 * 1.0 + 1e-6))

        if skew_y < 0:
            beta = -beta if skew_x < 0 else 0 - beta

        theta = beta - yaw
        theta = np.clip(theta, -np.pi, np.pi)

        return dist_to_goal, theta

    def _normalize_lidar(self, lidar_state):
        """
        Normalize lidar readings.

        Args:
            lidar_state: Lidar data to normalize.

        Returns:
            Normalized lidar data.
        """
        return np.divide(lidar_state, 10.0)

    def _normalize_dist_to_goal(self, dist_to_goal):
        """
        Normalize the distance to the goal.

        Args:
            dist_to_goal: The distance to the goal.

        Returns:
            Normalized distance.
        """
        return min(dist_to_goal / 10.0, 1.0)

    def _normalize_angle_rad(self, angle):
        """
        Normalize an angle in radians to a value between 0 and 1.

        Args:
            angle: The angle in radians.

        Returns:
            Normalized angle value.
        """
        angle += np.pi
        return angle / (2 * np.pi)

    def _pack_state_lidar(self, dist_to_goal, angle_to_goal, lidar_state, action):
        """
        Pack the lidar state into a single state array.

        Args:
            dist_to_goal: Normalized distance to the goal.
            angle_to_goal: Normalized angle to the goal.
            lidar_state: Normalized lidar readings.
            action: The action taken.

        Returns:
            Packed state array.
        """
        robot_state = np.array([dist_to_goal, angle_to_goal, action[0], action[1]], dtype=np.float32)
        return np.append(robot_state, lidar_state)

    def _pack_state_img(self, camera_state):
        """
        Prepare camera state for observation.

        Args:
            camera_state: The image data from the camera.

        Returns:
            Camera state as a numpy array.
        """
        return np.array(camera_state, dtype=np.uint8)

    def _pack_state_dict(self, robot_state, lidar_state):
        """
        Pack robot, goal, and lidar states into a dictionary.

        Args:
            robot_state: The current robot state (position, goal, etc.).
            lidar_state: The current lidar readings.

        Returns:
            Dictionary with robot, goal, and lidar state information.
        """
        return {
            "camera": self.ros.get_camera_data(),
            "lidar": np.array(lidar_state, dtype=np.float32),
            "robot_position": np.array(robot_state[0:2], dtype=np.float32),
            "goal_position": np.array(robot_state[2:4], dtype=np.float32),
            "goal_polar_coordinate": np.array(robot_state[4:6], dtype=np.float32),
            "actions": np.array(robot_state[6:], dtype=np.float32),
        }

    def _reset_goal(self):
        """
        Reset the goal position in the environment.
        """
        self.ros.reset_goal()

    def _respawn_robot(self):
        """
        Respawn the robot at its starting position.
        """
        self.ros.respawn_robot()

    def _reset_noise_areas(self):
        """
        Reset noise areas for lidar and camera, if enabled.
        """
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

    def _reset_sensor_noise(self, size, random_area_spawn, static_position_centre, sensor):
        """
        Reset sensor noise area.

        Args:
            size: Tuple specifying the size of the noise area.
            random_area_spawn: Whether to randomly spawn the noise area.
            static_position_centre: The fixed center position for static noise.
            sensor: The sensor type ('lidar' or 'camera').
        """
        w, h = size

        if random_area_spawn:
            x = random.uniform(-5 + (w / 2), 5 - (w / 2))
            y = random.uniform(-5 + (h / 2), 5 - (h / 2))
        else:
            x = static_position_centre[0]
            y = static_position_centre[1]

        if sensor == "camera":
            self.ros.reset_camera_noise_area(x, y, w, h)
        elif sensor == "lidar":
            self.ros.reset_lidar_noise_area(x, y, w, h)
