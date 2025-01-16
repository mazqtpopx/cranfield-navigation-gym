import math
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from cranavgym.ros_interface.ros_interface_drone import ROSInterface

import random

DISCRETE_ACTIONS = False


class DroneNavigationSimple(gym.Env):
    """
    ### Description
    A reinforcement learning environment for robot navigation using ROS.

    This environment is based on the [DRL robot navigation framework](https://github.com/reiniscimurs/DRL-robot-navigation).

    The environment simulates a robot navigating towards a goal while avoiding obstacles.
    It supports continuous action spaces and provides observations in various configurations, such as LiDAR data, camera images, or a combination in a dictionary.

    In addition to paring arguments while executing the script, there are three configuration files that can be used to configure the environment:

    - `ros_interface_config`: Configuration object for the ROS interface.
    - `rl_interface_config`: Configuration object for the RL interface.
    - `env_config`: Configuration object for the environment.

    **Features:**

    - **Collision Handling:** Accurate collision detection based on environment collision flags, ensuring reliable detection even with sensor noise.
    - **Sensor Noise Areas:** Ability to simulate areas with camera and LiDAR noise to test the robustness of navigation algorithms under adverse conditions.
    - **Sparse Rewards:** A reward function designed to encourage efficient navigation by providing rewards primarily when the robot reaches the goal or collides.

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
        | 0     | Linear velocity (forward/back) | -1.0 | 1.0 |
        | 1     | Angular velocity (rotation)    | -1.0 | 1.0 |


        ```python
        self.action_space = gym.spaces.Box(
            low=action_space_low,
            high=action_space_high,
            shape=(action_space_shape,),
            dtype=np.float32
        )
        ```
        ### Observation Space

    The observation space can be configured to provide different types of observations, determined by the `obs_space_type` parameter:

    - **LiDAR Observations (`obs_space_type='lidar'`):**

        The observation is a `numpy.ndarray` containing:

        - **LiDAR Readings:** Normalized distances from LiDAR sensors.
        - **Distance to Goal:** Normalized distance between the robot and the goal.
        - **Angle to Goal:** Normalized angle between the robot's orientation and the goal direction.
        - **Last Action Taken:** Normalized representation of the last action.
        - **Shape:** `(lidar_dim + 2 + last_actions,)`
            - `lidar_dim`: Number of LiDAR readings (e.g., 20).

    - **Camera Observations (`obs_space_type='camera'`):**

        The observation is an RGB image from the robot's onboard camera.

        ```python
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(img_height, img_width, 3),
            dtype=np.uint8
        )
        ```

    - **Dictionary Observations (`obs_space_type='dict'`):** NOT TESTED TODO

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

    ### Parameters
    - **Environment Configuration:**
        - `ros_interface_config` (object): Configuration object for the ROS interface.
        - `rl_interface_config` (object): Configuration object for the RL interface.
        - `env_config` (object): Configuration object for the environment.
        - `max_ep_steps` (int): Maximum number of steps per episode.
        - `obs_space_type` (str): Type of observation space (`'lidar'`, `'camera'`, or `'dict'`).
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

    - Code version for GDP.
    """

    def __init__(
        self,
        ros_interface_config,
        max_ep_steps,
        obs_space_type="lidar",
        obs_space_shape=25,
        obs_space_low=0,
        obs_space_high=1,
        action_space_type="continuous",
        action_space_shape=3,
        action_space_low=[0.0, -1.0, -1.0],
        action_space_high=[1, 1, 1],
        reward_type="alternative",
        camera_noise=False,
        camera_noise_area_size=[0, 0],
        random_camera_noise_area=False,
        static_camera_noise_area_pos=[0, 0],
        camera_noise_type="gaussian",
        lidar_noise=False,
        lidar_noise_area_size=[0, 0],
        random_lidar_noise_area=False,
        static_lidar_noise_area_pos=[0, 0],
        static_goal=False,
        static_goal_xy=[3, 3],
        static_spawn=False,
        static_spawn_xy=[0, 0],
    ):
        """
        Initializes the AgentNavigation environment.

        Parameters:
            ros_interface_config: Configuration object for the ROS interface.
            max_ep_steps (int): Maximum number of steps per episode.
            obs_space_type (str, optional): Type of observation space ('lidar', 'camera', or 'dict'). Defaults to 'lidar'.
            action_space_type (str, optional): Type of action space ('discrete' or 'continuous'). Defaults to 'continuous'.
            action_space_shape (int or tuple, optional): Shape or size of the action space. Defaults to 3.
            action_space_low (float or list, optional): Lower bounds for continuous action space. Defaults to -1.
            action_space_high (float or list, optional): Upper bounds for continuous action space. Defaults to 1.
            reward_type (str, optional): Type of reward function ('alternative' or 'original'). Defaults to 'alternative'.
            camera_noise (bool, optional): Enable camera noise areas. Defaults to False.
            camera_noise_area_size (list, optional): Size [width, height] of the camera noise area. Defaults to [4, 4].
            random_camera_noise_area (bool, optional): Randomize camera noise area position. Defaults to True.
            static_camera_noise_area_pos (list, optional): Static position [x, y] of the camera noise area. Defaults to [0, 0].
            camera_noise_type (str, optional): Type of camera noise. Defaults to 'gaussian'.
            lidar_noise (bool, optional): Enable lidar noise areas. Defaults to False.
            lidar_noise_area_size (list, optional): Size [width, height] of the lidar noise area. Defaults to [4, 4].
            random_lidar_noise_area (bool, optional): Randomize lidar noise area position. Defaults to True.
            static_lidar_noise_area_pos (list, optional): Static position [x, y] of the lidar noise area. Defaults to [0, 0].
            static_goal (bool, optional): Use a static goal position. Defaults to False.
            static_goal_xy (list, optional): Static goal position [x, y]. Defaults to [3, 3].
            static_spawn (bool, optional): Spawn the robot at a static position. Defaults to False.
            static_spawn_xy (list, optional): Static spawn position [x, y]. Defaults to [0, 0].
        """

        # ros_interface_config.ros.launchfile = "~/cranfield-navigation-gym-dev/cranavgym/ros_interface/launchfiles/DroneNav.launch"
        # ros_interface_config.interface.robot_name_arg = "robot_name:=uav1"

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
            gui_arg=conf.interface.gui_arg,
            rviz_arg=conf.interface.rviz_arg,
            robot_name=conf.interface.robot_name_arg,
        )

        self.current_step = 0
        self.max_episode_steps = max_ep_steps

        self.action_space_shape = action_space_shape

        self.obs_space_type = obs_space_type
        self.camera_noise = camera_noise
        self.camera_noise_type = camera_noise_type
        self.lidar_noise = lidar_noise

        self._camera_noise_area_size = camera_noise_area_size
        self._random_camera_noise_area = random_camera_noise_area
        self._static_camera_noise_area_pos = static_camera_noise_area_pos
        self._lidar_noise_area_size = lidar_noise_area_size
        self._random_lidar_noise_area = random_lidar_noise_area
        self._static_lidar_noise_area_pos = static_lidar_noise_area_pos

        # Reward configuration
        self.reward_type = reward_type

        # Action and observation spaces
        self._setup_action_space(
            action_space_type, action_space_shape, action_space_low, action_space_high
        )
        self._setup_observation_space(
            obs_space_shape,
            conf.lidar.lidar_dim,
            obs_space_low,
            obs_space_high,
            action_space_shape,
            action_space_low,
            action_space_high,
            conf.camera.camera_dim,
        )

        # Initialize variables
        self.last_odom = None
        self.render_mode = "rgb_array"
        self.config_map_max_xy = conf.map.max_xy
        self.goal_reached_threshold = 0.5

        # Moving goal setup
        self.moving_goal = False
        if self.moving_goal:
            self.goal_positions = self._generate_rectangle_perimeter()
            self.current_position_index = 0

    def _create_bound_array(self, value, shape, name):
        """
        Creates a NumPy array representing the bounds of an action space.

        This function generates an array of the specified shape to be used as the
        lower or upper bounds of an action space in a reinforcement learning environment.

        Parameters:
            value (float, int, list, or np.ndarray): The value(s) to fill the array with.
                - If a scalar (float or int) is provided, the array is filled with that value.
                - If a list or np.ndarray is provided, it is converted to a NumPy array.
            shape (tuple or int): The desired shape of the output array.
                - For a one-dimensional action space, this can be an integer.
                - For multi-dimensional spaces, provide a tuple representing the shape.
            name (str): The name of the bound (used for error messages), e.g., "action_space_low" or "action_space_high".

        Returns:
            np.ndarray: A NumPy array of the specified shape and dtype `np.float32`.

        Raises:
            ValueError: If the provided array does not match the specified shape.
            TypeError: If the input `value` is not of an expected type (float, int, list, or np.ndarray).

        Examples:
            >>> _create_bound_array(-1, (3,), "action_space_low")
            array([-1., -1., -1.], dtype=float32)

            >>> _create_bound_array([0.0, -0.5, -1.0], (3,), "action_space_low")
            array([ 0. , -0.5, -1. ], dtype=float32)

        """
        if isinstance(value, (float, int)):
            # If 'value' is a scalar, create an array filled with that scalar value
            arr = np.full(shape, value, dtype=np.float32)
        elif isinstance(value, (list, np.ndarray)):
            # If 'value' is a list or ndarray, convert it to a NumPy array with dtype float32
            arr = np.array(value, dtype=np.float32)
            # Check if the shape of the array matches the expected 'shape'
            if arr.shape != shape:
                raise ValueError(
                    f"{name} shape {arr.shape} does not match action space shape {shape}."
                )
        else:
            # Raise a TypeError if 'value' is not a supported type
            raise TypeError(f"{name} must be a float, int, list, or ndarray.")
        # Return the created NumPy array
        return arr

    def _setup_action_space(
        self, action_space_type, action_space_shape, action_space_low, action_space_high
    ):
        """
        Sets up the action space based on the configuration provided.

        Parameters:
            action_space_type (str): 'discrete' or 'continuous'.
            action_space_shape (int or tuple): Shape or size of the action space.
            action_space_low (float or list): Lower bounds for continuous action space.
            action_space_high (float or list): Upper bounds for continuous action space.
        """
        if action_space_type == "discrete":
            # For discrete action space, action_space_shape should be an integer representing the number of actions
            self.action_space = spaces.Discrete(action_space_shape)
        elif action_space_type == "continuous":
            # Ensure that action_space_shape is a tuple
            if isinstance(action_space_shape, int):
                action_space_shape = (action_space_shape,)
            else:
                action_space_shape = tuple(action_space_shape)

            # Create low and high arrays matching the action space shape
            low = self._create_bound_array(
                action_space_low, action_space_shape, "action_space_low"
            )
            high = self._create_bound_array(
                action_space_high, action_space_shape, "action_space_high"
            )
            # Define the action space
            self.action_space = spaces.Box(
                low=low,
                high=high,
                shape=action_space_shape,
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Invalid action_space_type: {action_space_type}")

    def _setup_observation_space(
        self,
        obs_space_shape,
        lidar_dim,
        obs_space_low,
        obs_space_high,
        action_space_shape,
        action_space_low,
        action_space_high,
        camera_dim,
    ):
        """
        Sets up the observation space based on the selected observation type.

        Parameters:
            obs_space_shape (tuple or int): The expected shape of the observation space.
            lidar_dim (int): Number of LiDAR readings.
            obs_space_low (float or array-like): Lower bounds for the observation space (distance and angle to goal).
            obs_space_high (float or array-like): Upper bounds for the observation space (distance and angle to goal).
            action_space_shape (int): Shape of the action space (number of action dimensions).
            action_space_low (float or array-like): Lower bounds for the action space.
            action_space_high (float or array-like): Upper bounds for the action space.
            camera_dim (tuple or list): Dimensions of the camera image (height, width, channels).

        Raises:
            ValueError: If the computed obs_shape does not match obs_space_shape.
        """
        if self.obs_space_type == "lidar":
            # Observation: [dist_to_goal, angle_to_goal, action[0], action[1], action[2], lidar readings...]

            obs_shape = (2 + action_space_shape + lidar_dim,)

            # Helper function to ensure a variable is a tuple
            def ensure_tuple(var):
                return (var,) if isinstance(var, int) else tuple(var)

            # Ensure that action_space_shape, obs_space_shape, and lidar_dim are tuples
            action_space_shape = ensure_tuple(action_space_shape)
            obs_space_shape = ensure_tuple(obs_space_shape)
            lidar_dim = ensure_tuple(lidar_dim)

            # Check if obs_shape matches obs_space_shape
            if obs_shape != obs_space_shape:
                raise ValueError(
                    f"obs_shape {obs_shape} does not match obs_space_shape {obs_space_shape}. "
                    "Please check the configuration files, specifically the 'obs_space_shape' in the env_config, "
                    "the 'lidar_dim' in the ros_interface_config, and the '_get_state' function of the "
                    "AgentNavigation class to make sure the state dimensions are correct."
                )

            low_obs = self._create_bound_array(obs_space_low, (2,), "obs_space_low")
            high_obs = self._create_bound_array(obs_space_high, (2,), "obs_space_high")

            low_actions = self._create_bound_array(
                action_space_low, action_space_shape, "action_space_low"
            )
            high_actions = self._create_bound_array(
                action_space_high, action_space_shape, "action_space_high"
            )

            low_lidar = self._create_bound_array(
                obs_space_low, lidar_dim, "lidar_space_low"
            )
            high_lidar = self._create_bound_array(
                obs_space_high, lidar_dim, "lidar_space_high"
            )

            # Concatenate the low and high arrays
            low = np.concatenate([low_obs, low_actions, low_lidar])
            high = np.concatenate([high_obs, high_actions, high_lidar])

            # Define the observation space
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        elif self.obs_space_type == "camera":
            # Define the camera image space
            image_space = spaces.Box(
                low=0,
                high=255,
                shape=(camera_dim[0], camera_dim[1], camera_dim[2]),
                dtype=np.uint8,
            )
            self.observation_space = image_space
        else:
            raise ValueError(f"Invalid observation space type: {self.obs_space_type}")

    def step(self, action):
        """
        Executes one time step within the environment based on the given action.

        Parameters:
            action (int or np.ndarray): The action to be executed.
                - For discrete actions, an integer representing the action index.
                - For continuous actions, an array-like object with action values.

        Returns:
            observation: The next observation of the environment, format depends on `obs_space_type`.
            reward (float): The reward obtained after executing the action.
            terminated (bool): Whether the episode has ended due to a terminal state (goal reached or collision).
            truncated (bool): Whether the episode was truncated due to exceeding `max_episode_steps`.
            info (dict): Additional information about the environment's state.
        """
        # Perform the action by setting the robot's velocity
        self._perform_action(action)
        self._pause_ROS()

        # Move the goal if it's a moving goal scenario
        if self.moving_goal:
            self.move_goal_in_rectangle()

        # Retrieve the new state and check if the goal has been reached
        state, reached_goal = self._get_state(
            self.obs_space_type, self.goal_reached_threshold, action
        )

        # Check for collisions
        collided = self.ros.get_collision_status()

        # Compute the reward based on the selected reward type
        if self.reward_type == "original":
            # Ensure that get_lidar_data() is called properly to retrieve the data
            lidar_data = self.ros.get_lidar_data()
            min_laser = min(lidar_data) if lidar_data.size > 0 else float("inf")
            reward = self._get_reward_original(
                reached_goal, collided, action, min_laser
            )
        elif self.reward_type == "alternative":
            reward = self._get_reward_alternative(reached_goal, collided)
        else:
            # Default to alternative reward if an unknown reward type is specified
            reward = self._get_reward_alternative(reached_goal, collided)

        # Determine if the episode has terminated or been truncated
        terminated, truncated = self._is_done(
            collision_detected=collided,
            reached_goal=reached_goal,
            current_step=self.current_step,
            max_episode_steps=self.max_episode_steps,
        )

        # print(terminated, truncated, reached_goal, collided)

        # Get the distance and angle to the goal for info purposes
        dist_to_goal, angle_to_goal = self._get_dist_and_angle_to_goal()

        # Retrieve the robot's current velocities
        x_vel, y_vel, z_vel = self.ros.get_robot_velocity()

        # Collect additional information about the environment's state
        info = {
            "x_position": self.ros.robot_position[0],
            "y_position": self.ros.robot_position[1],
            "z_position": self.ros.robot_position[2],
            "x_velocity": x_vel,
            "y_velocity": y_vel,
            "z_velocity": z_vel,
            "dist_to_target": dist_to_goal,
            "angle_to_goal": angle_to_goal,
            "reward": reward,
        }

        # Increment the current step counter
        self.current_step += 1

        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.

        Parameters:
            seed (int, optional): A seed for the random number generator.
            options (dict, optional): Additional options for the reset.

        Returns:
            tuple:
                state: The initial state of the environment.
                info (dict): Additional information, including positions, velocities, and distance to the goal.
        """
        # Reset the environment's random number generator
        super().reset(seed=seed)

        # Reset the ROS simulation
        self._reset_ROS()
        # Respawn the robot at the starting position
        self._respawn_robot()
        # Reset the goal position
        self._reset_goal()
        # Reset any noise areas for sensors
        self._reset_noise_areas()
        # Pause the ROS simulation to ensure all reset actions have taken effect
        self._pause_ROS()

        # Initialize the action with zeros matching the length of the action space
        zero_action = [0] * self.action_space.shape[0]

        # Initialize the state
        state, _ = self._get_state(
            self.obs_space_type, self.goal_reached_threshold, zero_action
        )

        # Reset the current step counter
        self.current_step = 0

        # Retrieve the robot's current velocities
        x_vel, y_vel, z_vel = self.ros.get_robot_velocity()

        # Calculate the distance and angle to the goal
        dist_to_goal, angle_to_goal = self._get_dist_and_angle_to_goal()

        # Collect additional information about the environment's state
        info = {
            "x_position": self.ros.robot_position[0],
            "y_position": self.ros.robot_position[1],
            "z_position": self.ros.robot_position[2],
            "x_velocity": x_vel,
            "y_velocity": y_vel,
            "z_velocity": z_vel,
            "dist_to_target": dist_to_goal,
            "angle_to_goal": angle_to_goal,
        }

        return state, info

    def close(self):
        """
        Performs any necessary cleanup when closing the environment.
        """
        self._close_ROS()

    def render(
        self,
    ):  # TODO the name render is reserved in gym for render stuff and not camera stuff
        """
        Retrieves the current camera image from the robot.

        Returns:
            numpy.ndarray: The image data captured by the robot's camera.
        """
        return self.ros.get_camera_data()

    # ----------------------MOVE GOAL FUNCTIONS-------------------------

    def _generate_rectangle_perimeter(
        self, center=(0, 0), width=6, height=6, num_points=15
    ):
        """
        Generates a list of (x, y) tuples representing points along the perimeter of a rectangle.

        Parameters:
            center (tuple): The (x, y) coordinates of the rectangle's center. Defaults to (0, 0).
            width (float): The width of the rectangle. Defaults to 6.
            height (float): The height of the rectangle. Defaults to 6.
            num_points (int): The number of points to generate along each edge. Defaults to 15.

        Returns:
            list of tuples: The perimeter points of the rectangle.
        """
        cx, cy = center
        half_width = width / 2
        half_height = height / 2

        # Define the corners of the rectangle (clockwise order)
        corners = [
            (cx - half_width, cy - half_height),  # Bottom-left
            (cx + half_width, cy - half_height),  # Bottom-right
            (cx + half_width, cy + half_height),  # Top-right
            (cx - half_width, cy + half_height),  # Top-left
        ]

        perimeter = []

        # Generate points along each edge using list comprehensions
        # Bottom edge (from bottom-left to bottom-right)
        bottom_edge = [
            (x, corners[0][1])
            for x in np.linspace(
                corners[0][0], corners[1][0], num=num_points, endpoint=False
            )
        ]
        perimeter.extend(bottom_edge)

        # Right edge (from bottom-right to top-right)
        right_edge = [
            (corners[1][0], y)
            for y in np.linspace(
                corners[1][1], corners[2][1], num=num_points, endpoint=False
            )
        ]
        perimeter.extend(right_edge)

        # Top edge (from top-right to top-left)
        top_edge = [
            (x, corners[2][1])
            for x in np.linspace(
                corners[2][0], corners[3][0], num=num_points, endpoint=False
            )
        ]
        perimeter.extend(top_edge)

        # Left edge (from top-left to bottom-left)
        left_edge = [
            (corners[3][0], y)
            for y in np.linspace(
                corners[3][1], corners[0][1], num=num_points, endpoint=False
            )
        ]
        perimeter.extend(left_edge)

        return perimeter

    def move_goal_in_rectangle(self):
        """
        Moves the goal to the next position along the rectangle perimeter.

        The goal positions are precomputed and stored in self.goal_positions.
        This method updates the goal position to the next point in the list, looping back to the start if necessary.
        """
        # Get the current goal position from the precomputed list
        goal_x, goal_y = self.goal_positions[self.current_position_index]

        # Use a fixed z-height or a random value within a desired range
        goal_z = np.random.uniform(0.75, 1.0)

        # Move the goal to the calculated position
        self._move_goal(goal_x, goal_y, goal_z)

        # Update the position index for the next call, looping back if at the end
        self.current_position_index = (self.current_position_index + 1) % len(
            self.goal_positions
        )

    # ----------------------STEP FUNCTIONS-------------------------
    def _perform_action(self, action):
        """
        Executes the given action by publishing it to the robot.

        Parameters:
            action (int or array-like): The action to perform.
                - If using discrete actions (DISCRETE_ACTIONS is True), action is an integer representing the action index.
                - If using continuous actions, action is an array-like object containing velocity components.

        Notes:
            - For discrete actions, the actual movement commands need to be implemented where indicated.
            - For continuous actions, the robot's velocity is set directly using the provided components.
        """
        # Publish the robot action
        if DISCRETE_ACTIONS:
            if action == 0:
                # Move forward
                pass  # Implementation needed
            elif action == 1:
                # Move left
                pass  # Implementation needed
            elif action == 2:
                # Move right
                pass  # Implementation needed
        else:
            # Continuous actions: set robot velocity directly
            self.ros.set_robot_velocity(*action)

        # Publish the goal marker for visualization
        self.ros.publish_goal_marker()

    def _get_dist_and_angle_to_goal(self):
        """
        Calculates the distance and angle from the robot to the goal position.

        Returns:
            tuple:
                dist_to_goal (float): The Euclidean distance to the goal in meters.
                angle_to_goal (float): The relative angle to the goal in radians.
        """
        # Retrieve the goal position from the ROS interface
        goal_x, goal_y, goal_z = self.ros.get_goal_position()

        # Get the robot's current orientation as a quaternion
        quaternion = self.ros.get_robot_quaternion()

        # Calculate the distance and heading to the goal
        dist_to_goal, angle_to_goal = self._calculate_distance_and_heading_to_goal(
            quaternion,
            self.ros.robot_position[0],
            self.ros.robot_position[1],
            self.ros.robot_position[2],
            goal_x,
            goal_y,
            goal_z,
        )

        # dist_to_goal is in meters, angle_to_goal is in radians
        return dist_to_goal, angle_to_goal

    def _pause_ROS(self):
        """
        Pauses the ROS simulation.

        This method calls the ROS service responsible for pausing the simulation,
        which can be useful during resets or when the simulation needs to be temporarily halted.
        """
        self.ros.pause_ros()

    def _reset_ROS(self):
        """
        Resets the ROS simulation to its initial state.

        This method resets the simulation environment, including the robot's position,
        sensor readings, and any other stateful components.
        """
        self.ros.reset_ros()

    def _close_ROS(self):
        """
        Closes the ROS simulation and performs any necessary cleanup.

        This method should be called when the simulation is no longer needed,
        to ensure that all ROS nodes and services are properly shut down.
        """
        self.ros.close_ros()

    def _get_state(self, obs_space_type, goal_reached_threshold, action):
        """
        Retrieves the current state of the environment based on the observation space type.

        This method gathers sensor data from the robot and environment, processes it by normalizing
        the values, and packages it into the appropriate observation format required by the environment.
        It also determines whether the goal has been reached based on the current distance to the goal.

        Parameters:
            obs_space_type (str): The type of observation space ('lidar', 'camera', or 'dict').
            goal_reached_threshold (float): The distance threshold within which the goal is considered reached.
            action (np.ndarray): The last action taken by the agent.

        Returns:
            tuple:
                - state (np.ndarray or dict): The current state of the environment formatted according to the observation space.
                    - If `obs_space_type` is 'lidar', `state` is a NumPy array containing:
                        `[distance_to_goal_normalized, angle_to_goal_normalized, last_action(s), lidar_readings...]`
                    - If `obs_space_type` is 'camera', `state` is a NumPy array representing the camera image.
                    - If `obs_space_type` is 'dict', `state` is a dictionary containing various sensor data.
                - reached_goal (bool): True if the goal has been reached, False otherwise.
        """

        # Get the latest LiDAR data from the ROS interface
        lidar_state = np.array([self.ros.get_lidar_data()])

        # Compute the distance and angle to the goal
        dist_to_goal, angle_to_goal = self._get_dist_and_angle_to_goal()

        # Normalize the sensor data and goal information
        lidar_state_normalized = self._normalize_lidar(lidar_state)
        dist_to_goal_normalized = self._normalize_dist_to_goal(dist_to_goal)
        angle_to_goal_normalized = self._normalize_angle_rad(angle_to_goal)

        # Pack the state based on the observation space type
        if obs_space_type == "lidar":
            # For 'lidar' observation space:
            # Pack normalized distance to goal, angle to goal, last action, and LiDAR data
            state = self._pack_state_lidar(
                dist_to_goal_normalized,
                angle_to_goal_normalized,
                lidar_state_normalized,
                action,
            )
        elif obs_space_type == "camera":
            # For 'camera' observation space:
            # Retrieve and pack the camera image data
            camera_state = self.ros.get_camera_data()
            state = self._pack_state_img(camera_state)
        elif obs_space_type == "dict":
            # For 'dict' observation space:
            # Combine robot state information and LiDAR data into a dictionary
            robot_state = [
                dist_to_goal_normalized,
                angle_to_goal_normalized,
                action,
            ]
            state = self._pack_state_dict(robot_state, lidar_state)
        else:
            # Raise an error if the observation space type is invalid
            raise ValueError(f"Invalid observation space type: {obs_space_type}")

        # TODO maybe reached_goal can be staying inside the goal_reached_threshold for a few steps?
        # Determine if the goal has been reached based on the distance threshold
        reached_goal = dist_to_goal <= goal_reached_threshold

        # Return the current state and goal status
        return state, reached_goal

    """
    # Legacy code, not used
    def _convert_quaternion_to_angles_legacy(self, quaternion, pos_x, pos_y, goal_x, goal_y):
        euler = quaternion.to_euler(degrees=False)
        yaw = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        dist_to_goal = np.linalg.norm([pos_x - goal_x, pos_y - goal_y])

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = goal_x - pos_x
        skew_y = goal_y - pos_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
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
    """

    def _calculate_distance_and_heading_to_goal(
        self, quaternion, pos_x, pos_y, pos_z, goal_x, goal_y, goal_z
    ):
        """
        Calculates the 3D distance to the goal and the relative heading angle from the robot's current orientation.

        Parameters:
            quaternion: Quaternion object
                Represents the robot's current orientation.
            pos_x (float): The robot's current x-coordinate.
            pos_y (float): The robot's current y-coordinate.
            pos_z (float): The robot's current z-coordinate.
            goal_x (float): The goal's x-coordinate.
            goal_y (float): The goal's y-coordinate.
            goal_z (float): The goal's z-coordinate.

        Returns:
            tuple:
                dist_to_goal (float): The Euclidean distance to the goal position.
                angle_to_goal (float): The relative heading angle to the goal in radians.
        """
        # Calculate 3D distance to the goal from the robot
        dist_to_goal = np.linalg.norm([pos_x - goal_x, pos_y - goal_y, pos_z - goal_z])

        # Extract yaw (rotation around the Z-axis) from the quaternion
        euler = quaternion.to_euler(degrees=False)
        yaw = round(euler[2], 4)

        # Compute the vector from the robot to the goal in the XY-plane
        skew_x = goal_x - pos_x
        skew_y = goal_y - pos_y

        # Calculate the angle (beta) between the robot's forward direction and the goal direction
        # Using the dot product between the robot's forward vector (assumed to be along the X-axis) and the goal vector
        dot = skew_x * 1 + skew_y * 0  # Robot's forward vector is (1, 0)
        mag1 = math.sqrt(skew_x**2 + skew_y**2)  # Magnitude of the goal vector
        mag2 = 1  # Magnitude of the robot's forward vector

        # Avoid division by zero when the robot is exactly at the goal position
        if mag1 == 0:
            beta = 0.0
        else:
            beta = math.acos(dot / (mag1 * mag2))

        # Adjust beta based on the quadrant to get the correct sign
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = -beta  # Simplify to beta = -beta

        # Compute the relative angle to the goal
        theta = beta - yaw

        # Normalize theta to be within [-π, π]
        if theta > np.pi:
            theta = theta - 2 * np.pi
        if theta < -np.pi:
            theta = theta + 2 * np.pi

        angle_to_goal = theta

        return dist_to_goal, angle_to_goal

    def _normalize_lidar(self, lidar_state, max_range=30.0):
        """
        Normalizes lidar readings by handling infinite values and scaling the data.

        Parameters:
            lidar_state (numpy.ndarray): Array of lidar readings.
            max_range (float, optional): Maximum possible lidar range. Defaults to 30.0.

        Returns:
            numpy.ndarray: Normalized lidar data with values between 0 and 1.
        """
        # Replace infinite values with max_range
        lidar_state = np.where(np.isinf(lidar_state), max_range, lidar_state)

        # Replace NaN values with max_range
        lidar_state = np.where(np.isnan(lidar_state), max_range, lidar_state)

        # Clip values to [0, max_range]
        lidar_state = np.clip(lidar_state, 0, max_range)

        # Normalize to [0, 1]
        normalized_lidar = lidar_state / max_range

        return normalized_lidar

    """
    # Legacy code, not used
    def _normalize_dist_to_goal_legacy(self, dist_to_goal):
        if dist_to_goal > 10.0:
            dist_to_goal = 10.0
        return dist_to_goal / 10.0
    """

    def _normalize_dist_to_goal(self, dist_to_goal):
        """
        Normalizes the distance to the goal based on the maximum possible distance within the map.

        Parameters:
            dist_to_goal (float): The current distance to the goal.

        Returns:
            float: The normalized distance to the goal, scaled between 0 and 1.
        """
        # Calculate the maximum possible distance using the map's maximum X and Y coordinates
        max_distance = np.hypot(self.config_map_max_xy[0], self.config_map_max_xy[1])

        # Cap the distance to the maximum possible distance to prevent values greater than 1 after normalization
        if dist_to_goal > max_distance:
            dist_to_goal = max_distance

        # Normalize the distance to a value between 0 and 1
        normalized_distance = dist_to_goal / max_distance

        return normalized_distance

    def _normalize_angle_rad(self, angle):
        # normalize between 0-1
        angle += np.pi
        return angle / (2 * np.pi)

    def _pack_state_lidar(self, dist_to_goal, angle_to_goal, lidar_state, action):
        """
        Combines the robot's state information and lidar readings into a single array.

        Parameters:
            dist_to_goal (float): The distance from the robot to the goal.
            angle_to_goal (float): The angle from the robot's heading to the goal.
            lidar_state (array-like): The lidar sensor readings.
            action (array-like): The last action taken by the robot, containing three elements.

        Returns:
            numpy.ndarray: A combined array of robot state and lidar readings.
        """
        # Convert lidar_state to a NumPy array of type float32 for consistency
        lidar_state = np.array(lidar_state, dtype=np.float32)

        # Create an array containing the robot's state information and last action
        robot_state = np.array(
            [dist_to_goal, angle_to_goal] + list(action),
            dtype=np.float32,
        )

        # Concatenate robot_state and lidar_state to form the full state representation
        state = np.append(robot_state, lidar_state)

        return state

    """
    # Legacy code, not used
    def _pack_state_lidar_dict_legacy(self, dist_to_goal, angle_to_goal, lidar_state, actions):
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


    def _pack_state_lidar_dict_debug(self, lidar_state):
        # print(f"1{lidar_state=}")
        state = np.array(lidar_state, dtype=np.float32)
        # print(f"2{state=}")
        # print(f"3{state[0]=}")
        return state[0]

    #!!!!! NB !!!!!!
    # we have to take the first to match the observation state specified
    # if we don't do this, we return a np.array([[]])
    # instead, we simply want np.array([])
    #!!!!! NB !!!!!!

    def _pack_state_lidar_dict(self, dist_to_goal, angle_to_goal, lidar_state, action):
        state = {
            "lidar": lidar_state,
            "dist_to_goal": dist_to_goal,
            "angle_to_goal": angle_to_goal,
            "actions": action,
        }
        return state
    """

    def _pack_state_img(self, camera_state):
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

    """
    # Legacy code, not used
    def _is_done_legacy(
        self, collision_detected, reached_goal, current_step, max_episode_steps
    ):
        if collision_detected:
            return True
        elif reached_goal:
            return True
        # Do the bellow if training or evaluating
        elif current_step >= max_episode_steps:
            if self.moving_goal:
                return False
            else:
                return True
        else:
            return False
    """

    def _is_done(
        self, collision_detected, reached_goal, current_step, max_episode_steps
    ):
        """
        Determines whether the episode has terminated or been truncated.

        Parameters:
            collision_detected (bool): Indicates if a collision has occurred.
            reached_goal (bool): Indicates if the goal has been reached.
            current_step (int): The current time step of the episode.
            max_episode_steps (int): The maximum number of steps allowed per episode.

        Returns:
            tuple:
                terminated (bool): True if the episode should terminate due to a terminal condition.
                truncated (bool): True if the episode should end due to truncation (e.g., max steps reached).
        """
        terminated = collision_detected or reached_goal
        truncated = current_step >= max_episode_steps and not terminated
        return terminated, truncated

    def _get_reward_alternative(self, target, collision):
        """
        Calculates the reward based on whether the goal is reached or a collision has occurred.

        Parameters:
            target (bool): Indicates if the robot has reached the goal.
            collision (bool): Indicates if the robot has collided with an obstacle.

        Returns:
            float: The calculated reward.
                - Returns 1.0 if the goal is reached.
                - Returns -1.0 if a collision has occurred.
                - Returns a small negative value to penalize time spent otherwise.
        """
        if target:
            # The robot has reached the goal
            print("------------------Reached the goal-------------------")
            return 1.0
        elif collision:
            # The robot has collided with an obstacle
            return -1.0
        else:
            # Penalize the robot slightly for each time step to encourage faster goal reaching
            return -(1 / self.max_episode_steps)

    """
    # Legacy code, not used
    def _get_reward_original_legacy(self, target, collision, action, min_laser):
        if target:
            # print("------------------Reached the goal-------------------")
            return 1.0
        elif collision:
            return -1.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
    """

    def _get_reward_original(self, target, collision, action, min_laser):
        """
        Calculates the reward based on the robot's actions, proximity to obstacles, and whether the goal is reached or a collision has occurred.

        Parameters:
            target (bool): Indicates if the robot has reached the goal.
            collision (bool): Indicates if the robot has collided with an obstacle.
            action (tuple): The robot's action represented as (linear_velocity, angular_velocity).
            min_laser (float): The minimum distance detected by the laser sensor to the nearest obstacle.

        Returns:
            float: The calculated reward.
                - Returns 1.0 if the goal is reached.
                - Returns -1.0 if a collision has occurred.
                - Returns a computed value based on actions and proximity to obstacles otherwise.
        """
        if target:
            # The robot has reached the goal
            # print("------------------Reached the goal-------------------")
            return 1.0
        elif collision:
            # The robot has collided with an obstacle
            return -1.0
        else:
            # Penalize being too close to obstacles
            obstacle_penalty = 1 - min_laser if min_laser < 1 else 0.0

            # Compute the reward based on actions and obstacle proximity
            reward = (
                (action[0] / 2) - (abs(action[1]) + 0.0001 / 2) - (obstacle_penalty / 2)
            )
            return reward

    def _respawn_robot(self):
        """
        Respawns the robot at its initial position.

        This method calls the ROS (Robot Operating System) service responsible for respawning
        the robot within the simulation environment.
        """
        self.ros.respawn_robot(top_height=0.5)

    def _reset_goal(self, goal_z_max=2):
        """
        Resets the goal position within the simulation.

        This method calls the ROS service to reset the goal to a new position,
        preparing for the next episode or trial.
        """
        self.ros.reset_goal(goal_z_max)

    def _move_goal(self, x, y, z):
        """
        Moves the goal to a specified position in the simulation.

        Parameters:
            x (float): The x-coordinate of the new goal position.
            y (float): The y-coordinate of the new goal position.
            z (float): The z-coordinate of the new goal position.
        """
        self.ros.move_goal(x, y, z)

    def _reset_noise_areas(self):
        """
        Resets the noise areas for sensors in the simulation.

        Depending on the configuration, this method resets the noise areas for the camera and
        lidar sensors by calling the appropriate functions.
        """
        if self.camera_noise:
            self._reset_sensor_noise(
                size=self._camera_noise_area_size,
                random_area_spawn=self._random_camera_noise_area,
                static_position_centre=self._static_camera_noise_area_pos,
                sensor="camera",
            )

        if self.lidar_noise:
            self._reset_sensor_noise(
                size=self._lidar_noise_area_size,
                random_area_spawn=self._random_lidar_noise_area,
                static_position_centre=self._static_lidar_noise_area_pos,
                sensor="lidar",
            )

    def _reset_sensor_noise(
        self, size, random_area_spawn, static_position_centre, sensor
    ):
        """
        Resets the noise area for a specific sensor.

        If random noise area positionin0000000 g is enabled, it generates a random position within the
        map boundaries for the noise area. Otherwise, it uses the specified static position.

        Parameters:
            size (tuple of float): Size of the noise area as (width, height) in meters.
            random_area_spawn (bool): If True, the noise area's position is randomized.
            static_position_centre (tuple of float): The static center (x, y) of the noise area.
                Used only if random_area_spawn is False.
            sensor (str): The sensor type for which to reset the noise area; either "lidar" or "camera".
        """
        w, h = size  # Width and height of the noise area

        if random_area_spawn:
            # Generate random center position for the noise area within the map bounds
            # Assuming the map ranges from -5 to 5 in both x and y directions
            x = random.uniform(-5 + (w / 2), 5 - (w / 2))
            y = random.uniform(-5 + (h / 2), 5 - (h / 2))
        else:
            # Use the static center position provided
            x, y = static_position_centre

        # Reset the noise area for the specified sensor
        if sensor == "camera":
            self.ros.reset_camera_noise_area(x, y, w, h)
        elif sensor == "lidar":
            self.ros.reset_lidar_noise_area(x, y, w, h)
