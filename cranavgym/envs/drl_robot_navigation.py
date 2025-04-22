import math
import numpy as np
import gymnasium as gym
from gymnasium import logger, spaces
from typing import Union
import random
from squaternion import Quaternion

from cranavgym.ros_interface.ros_interface import ROSInterface

#Discrete actions generally only used for debugging. 
#TODO: Move to class init/config
DISCRETE_ACTIONS = False
#Fast: if true uses a kinematic model of the robot with only the camera sensor (so it doenst work for lidar/dict currently)
#If false uses the original model with all the sensors (slower)
#TODO: Move to config
FAST = True


def move_forward(x, y, yaw, distance):
    x_new = x + distance * np.cos(yaw)
    y_new = y + distance * np.sin(yaw)

    return (x_new, y_new)

class DRLRobotNavigation(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30} #TODO: remove

    """
    The random room environment as presented in DRL robot navigation
    (https://github.com/reiniscimurs/DRL-robot-navigation)
    We have made significant updates to this including:
    -inheriting it as a gymnasium environment (hence, we can now interface with common RL libs such as Stable-baselines3)
    -adding camera observation space (previously the observation was lidar only even tho the cam was in the environment)
    -making it faster (we can run at 100x speed, processing at 400 fps using stable-baselines for the camera observation space)
    -we have fixed collisions (previously, lidar - which is a sensor - was used to calculate collisions. Instead we simply observe gazebo collisions)
    -simplified the reward (+1 reached goal, -0.5 collision, -1/max_steps otherwise)
    
    However, as Gazebo classic has now reached end of life we are planning on moving this to ROS2 - as it presents a useful benchmarking environment - but as we have performed all of the simulations on this (in gazebo classic) we will leave this as is (it's still somewhat useful as a reference if nothing else)

    ## Action Space - Update the action space
    The action is a `ndarray` with shape `(2,)` in range [0,1] for first action (forward), and [-1,1] for second action (yaw)
    
    ## Observation Space - Update the obs space
    The observation is a `ndarray` with three options:
    - Lidar
    - Image
    - Dict (experimental, not fully implemented)

    Lidar:
    - (24,),

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0-20| Lidar                 | 0.0                 | 1.0 (normalized)  |
    |21-22| previous actions      | [0.0, -1.0]         | [1.0, 1.0]        |
    | 23  | Dist to goal          | 0.0                 | 1.0 (normalized)  |
    | 24  | Angle to goal         | -1.0                | +1.0 (normalized) |
    
    Image:
    (160,160,3,) np.uint8 [0,255] - The resolution can be updated in the ros_interface_config.yaml, but for the FAST environment it is 
    hardcoded in gazebo. 
    
    Dict:
    Lidar and Image

    ## Reward
    +1 for reaching the goal
    -0.5 for collision with wall
    -1/max_episode_steps otherwise
    
    ## Episode End
    The episode ends if any one of the following occurs:

    1. Termination: Collision with wall
    2. Termination: Robot reaches the goal
    3. Truncation: Episode length is greater than max_episode_steps

    
    inputs:
        lidar_dim: dim of lidar points (i.e. how many lidar points are outputte 20 by default.)
        The entire lidar field is discretized into this amount of points.
        Note: this affects the output of the state size!
        launchfile: name of the launchfile as it appears in assets/{launchfile}.launch
        obs_space: output observation space options: 
        "lidar", "camera", "dict"
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

        # -----------------------------------INIT OTHER VARIABLES-------------------------------------
        self.current_step = 0
        self.render_mode = "rgb_array" #NB: currently unused
        #Distance to the goal before goal is achieved
        self.goal_reached_threshold = 0.5

        # -----------------------------------ACTION/OBS SPACE-----------------------------------------
        if DISCRETE_ACTIONS:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Box(
                np.array([0, -1]).astype(np.float32),
                np.array([+1, +1]).astype(np.float32),
            )  # PAN, TILT, ZOOM

        if self.obs_space == "lidar":
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
        elif self.obs_space == "dict": #NB: untested
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

    """
    inputs: 
        action: action performed by the agent. The action format depends on whether the discrete_action flag was 
        called true during the initialization of the flag.
        For discrete actions the actions are:
        0: No action
        1: Forward 
        2: Left
        3: Right
        For continuous actions the actions are:
        [0]: range [0,1]; 0 is stop, 1 is go forward at full speed
        [1]: range [-1,1]; -1 is move to the left, 1 is move to the right
    """
    # Perform an action and read a new state
    def step(self, action):
        #If not fast: unpause the sim
        #If fast, this line needs to be within the collect rollouts code. (i.e. the forked stable baselines3 version)
        if not FAST:
            self.ros.unpause()

        # perform actions depending on configuration: discrete, continuous (fast) or continuous (slow)
        if DISCRETE_ACTIONS:
            self._perform_action_discrete(action)
        elif FAST:
            self._perform_action_continuous_fast(action)
        else:
            self._perform_action_continuous_slow(action)

        #If not fast: pause
        #If fast, this line needs to be within the collect rollouts code. (i.e. the forked stable baselines3 version)
        if not FAST:
            self.ros.pause()

        #Get the state from the environment depending on the obs space
        if self.obs_space == "lidar":
            state = self._get_state_lidar(action)
        elif self.obs_space == "camera":
            state = self._get_state_camera()
        elif self.obs_space == "dict":
            state = self._get_state_dict(action)

        # First, get the collison status
        collided = self.ros.collision_status

        #get the distance and angle to goal
        dist_to_goal, angle_to_goal = self._get_dist_and_angle_to_goal()

        # Set the reached goal flag (true if distance to goal is below the threshold)
        # move to get_state_observation
        # NB: make sure to use the unnormalized version of dist_to_goal!
        # As the state contains the normalized version, we have to get the dist again
        reached_goal = dist_to_goal < self.goal_reached_threshold

        #Get the reward - based on the reward type
        #Original - original reward as presented in the original Cimurs et al. repo
        #Alternative - the new reward as updated by us in the Benchmarking DRL paper
        # alternative reward (recommended) is as follows:
        # -0.5 if collision
        # +1 if reached reward
        # -1/max_steps otherwise
        if self.reward_type == "original":
            reward = self._get_reward_original(
                reached_goal, collided, action, min(self.ros.velodyne_data)
            )
        elif self.reward_type == "alternative":
            reward = self._get_reward_alternative(reached_goal, collided)

        # check if scenario is done
        terminated, truncated = self._is_done(
            collided, reached_goal, self.current_step, self.max_episode_steps
        )

        info = {
            "x_position": self.ros.robot_position[0],
            "y_position": self.ros.robot_position[1],
            "x_velocity": self.ros.robot_velocity[0],
            "y_velocity": self.ros.robot_velocity[1],
            "dist_to_target": dist_to_goal,
            "angle_to_goal": angle_to_goal,
            "reward": reward,
        }
        if terminated:
            #If terminated, add the terminal observation to info
            info["terminal_observation"] = state

        #Update the step
        self.current_step = self.current_step + 1

        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, **kwargs):
        """
        Reset the simulation. Respawns the robot and goal. 
        """
        super().reset(seed=seed)
        #Reset the ros world
        self.ros.reset_ros()
        #Pause the simulation
        self.ros.pause()
        #Respawn the robot - either in static position or randomly around the map (specified in the config)
        self.ros.respawn_robot()
        #Respawn the goal - either in static position or randomly around the map (specified in the config)
        self.ros.reset_goal()
        #reset the noise areas (if specified in the config)
        self._reset_noise_areas()
        #Reset the collision status (toggle off the collided flag)
        self.ros.reset_collision_status()
        #Get the state from the environment depending on the obs space
        if self.obs_space == "lidar":
            state = self._get_state_lidar([0,0])
        elif self.obs_space == "camera":
            state = self._get_state_camera()
        elif self.obs_space == "dict":
            state = self._get_state_dict([0,0])
        #Reset the current step
        self.current_step = 0
        #Get the dist/angle (for info)
        dist_to_goal, angle_to_goal = self._get_dist_and_angle_to_goal()
        #Pack the info
        info = {
            "x_position": self.ros.robot_position[0],
            "y_position": self.ros.robot_position[1],
            "x_velocity": self.ros.robot_velocity[0],
            "y_velocity": self.ros.robot_velocity[1],
            "dist_to_target": dist_to_goal,
            "angle_to_goal": angle_to_goal,
            # "terminal_observation": False,
        }
        #unpause the ros sim
        self.ros.unpause()
        return state, info

    def render(self):
        return self.ros.camera_data


    def _get_dist_and_angle_to_goal(self):
        """
        Returns distance and angle to goal (meters, radians)
        """
        # should move funct of dist to goal/angle to goal to ros interface
        goal_x, goal_y = self.ros.goal_position
        quaternion = self.ros.robot_quaternion

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

    def _perform_action_discrete(self, action):
        """Performs action in discrete space.
        \naction -- 
        0: No action, 
        1: Forward, 
        2: Left, 
        3: Right
        """
        pos = self.ros.robot_position
        x = pos[0]
        y = pos[1]
        quat = self.ros.robot_quaternion
        # euler = quat.as_euler('xyz', degrees=False)

        # euler = quaternion_to_euler(quat)
        euler = quat.to_euler(degrees=False)

        yaw = euler[2]
        if action == 0: #do nothing
            return
        elif action == 1: # move forward
            x_new, y_new = move_forward(x, y, euler[2], 0.1)
            self.ros.set_robot_position(x_new, y_new, quat)
            return
        elif action == 2: # move left
            yaw = yaw - 0.3
            quat_new = Quaternion.from_euler(euler[0], euler[1], yaw)
            self.ros.set_robot_position(x, y, quat_new)
            return
        elif action == 3: # move right
            yaw = yaw + 0.3
            quat_new = Quaternion.from_euler(euler[0], euler[1], yaw)
            self.ros.set_robot_position(x, y, quat_new)
            return
           
        return
    
    def _perform_action_continuous_fast(self, action):
        """Performs action in continuous space (fast by applying a velocity to the robot, rather than torque to motors).
        \naction -- list of size 2
        [0]: range [0,1]; 0 is stop, 1 is go forward at full speed
        [1]: range [-1,1]; -1 is move to the left, 1 is move to the right
        """

        # convert (local) forward velocity to global x/y velocity
        # first get the robot pose vector (quat) and convert to euler and scale by the forward velocity action
        quat = self.ros.robot_quaternion
        euler = quat.to_euler(degrees=False)
        yaw = euler[2]

        # scale actions
        action[0] *= 2
        action[1] *= 4

        x = math.cos(yaw) * action[0]
        y = math.sin(yaw) * action[0]

        self.ros.set_robot_velocity(x, y, action[1])  # 15/03changes interface

        # Move to ros interface...
        # Publish visualization markers for debugging or monitoring
        self.ros.publish_velocity(action)
        return
    
    def _perform_action_continuous_slow(self, action):
        """Performs action in continuous space (slow, by applying torque to the motors - this does not work with the sim sped up)
        \naction -- list of size 2
        [0]: range [0,1]; 0 is stop, 1 is go forward at full speed
        [1]: range [-1,1]; -1 is move to the left, 1 is move to the right
        """
        return


    """Pauses the current python script to let the ROS/gazebo
    simulation execute the action
    inputs:
        time_delta: the amount of time to sleep for
        in order to let the simulation execute the actions
    """
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

    def _get_state_lidar(self, action):
        """
        Get the lidar state 
        """
        # should move out of _get state and instead set these as inputs to
        # increase lidar proc speed
        robot_x, robot_y = self.ros.robot_position
        goal_x, goal_y = self.ros.goal_position

        quaternion = self.ros.robot_quaternion

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
        return state
    
    def _get_state_camera(self):
        """
        Get the camera state 
        """
        camera_state = self.ros.camera_data
        #Leaving this in but there is no interface to trigger this at the moment 
        #if set to true, converts from uint8 to [0,1] floating point range
        #and flips channels from (w,h,c) to (c,w,h)
        NORMALIZE_AND_FLIP_CHANNELS = False
        if NORMALIZE_AND_FLIP_CHANNELS:
            state = self._pack_state_img(camera_state)
        else:
            state = np.array(camera_state, dtype=np.uint8)
        return state
    
    def _get_state_dict(self, action):
        """
        Get the dict state 
        """
        robot_x, robot_y = self.ros.robot_position
        goal_x, goal_y = self.ros.goal_position

        quaternion = self.ros.robot_quaternion

        dist_to_goal, angle_to_goal = self._convert_quaternion_to_angles(
            quaternion, robot_x, robot_y, goal_x, goal_y
        )

        # get the lidar data
        lidar_state = np.array([self.ros.velodyne_data])
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
        
        state = self._pack_state_dict(robot_state=robot_state, laser_state=lidar_state, camera_state=self._get_state_camera())
        return state

    def _pack_state_img(self, camera_state):
        # swap from (w,h,c) to (c,w,h)
        # if camera_state is None:
        #     return np.zeros((3,64,64))

        state = np.transpose(camera_state, (2, 1, 0))
        # divide by 255 to convert from uint8 to float [0,1]
        state = state / 255.0
        return np.array(camera_state, dtype=np.float32)

    def _pack_state_dict(self, robot_state, laser_state, camera_state):
        """
        Packs the state in a dictionary format containing: camera, lidar, robot_position, goal_position, goal_polar_coordinate, (previous) actions
        """
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

    def _get_reward_alternative(self, target, collision):
        """
        (Recommended reward) returns the (updated) reward as described https://arxiv.org/pdf/2410.14616
        """
        if target:
            print("------------------Reached the goal-------------------")
            return 1.0
        elif collision:
            return -0.5
        else:
            return -(1 / self.max_episode_steps)

    def _get_reward_original(self, target, collision, action, min_laser):
        """
        Original reward as presented by Cimurs et al. https://github.com/reiniscimurs/DRL-robot-navigation 
        """
        if target:
            print("------------------Reached the goal-------------------")
            return 1.0
        elif collision:
            return -1.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

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
