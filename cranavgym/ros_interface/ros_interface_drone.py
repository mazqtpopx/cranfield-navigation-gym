import os
import random
import subprocess
import cv2
import time
import numpy as np
import rospy
from squaternion import Quaternion
from skimage.util import random_noise
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, LaserScan
from gazebo_msgs.msg import ModelState, ContactsState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from std_msgs.msg import Float64

from cranavgym.ros_interface.utils.map_drone import Map
from cranavgym.ros_interface.utils.rectangle import Rectangle
from cranavgym.ros_interface.utils.marker_publisher_drone import MarkerPublisher


from cranavgym.ros_interface.models.goal_drone import Goal
from cranavgym.ros_interface.models.robot_drone import Robot

import warnings
import roslaunch
import rospy
import rosnode
import subprocess
import os
import psutil
import time
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose

"""
Python conventions
Identifiers:
-  Contain only (A-z, 0-9, and _ )
-  Start with a lowercase letter or _.
-  Single leading _ :  private
-  Double leading __ :  strong private
-  Start & End  __ : Language defined Special Name of Object/ Method
-  Class names start with an uppercase letter.
-
"""


class ROSInterface:
    """
    Interface class to handle the launching of ROS,
    initializing publishers and subscribers,
    and handling all of the callbacks.

    This class provides methods to initialize and manage a ROS (Robot Operating System) interface for a robotics simulation.
    It handles launching ROS nodes, setting up communication via topics and services, and processing sensor data and control commands.
    """

    def __init__(
        self,
        launchfile,
        ros_port="11311",
        time_delta=0.1,
        map_min_xy=[-5, -5],
        map_max_xy=[5, 5],
        img_width=160,
        img_height=160,
        camera_noise_type="gaussian",
        lidar_dim=20,
        static_goal=False,
        static_goal_xy=[3, 3],
        static_spawn=False,
        static_spawn_xy=[0, 0],
        gui_arg="gui:=false",
        rviz_arg="rviz:=true",
        robot_name="robot_name:=uav1",
    ):
        """
        Initializes the ROSInterface object.

        Args:
            launchfile (str): The path to the ROS launch file to start the simulation environment.
            ros_port (str, optional): The port number for ROS communication (ROS Master URI). Defaults to "11311".
            time_delta (float, optional): The time interval between simulation steps in seconds. Defaults to 0.1.
            map_min_xy (list, optional): The minimum x and y coordinates defining the simulation map boundaries.
                Defaults to [-5, -5].
            map_max_xy (list, optional): The maximum x and y coordinates defining the simulation map boundaries.
                Defaults to [5, 5].
            img_width (int, optional): The width of the camera images captured from the robot's camera, in pixels.
                Defaults to 160.
            img_height (int, optional): The height of the camera images captured from the robot's camera, in pixels.
                Defaults to 160.
            camera_noise_type (str, optional): The type of noise to apply to the camera images. Defaults to "gaussian".
            lidar_dim (int, optional): The number of beams or readings in the LiDAR sensor. Defaults to 20.
            static_goal (bool, optional): If True, the goal position is fixed at `static_goal_xy`. If False, the goal
                position may be dynamic or randomized. Defaults to False.
            static_goal_xy (list, optional): The x and y coordinates of the static goal position, used if
                `static_goal` is True. Defaults to [3, 3].
            static_spawn (bool, optional): If True, the robot's starting position is fixed at `static_spawn_xy`. If False,
                the spawn position may be dynamic or randomized. Defaults to False.
            static_spawn_xy (list, optional): The x and y coordinates of the robot's static spawn position, used
                if `static_spawn` is True. Defaults to [0, 0].
            gui_arg (str, optional): ROS launch file argument to control the display of the Gazebo GUI. Set to "gui:=true"
                to display the GUI, or "gui:=false" to hide it. Defaults to "gui:=false".
            rviz_arg (str, optional): ROS launch file argument to control the use of RViz visualization. Set to
                "rviz:=true" to launch RViz, or "rviz:=false" to skip it. Defaults to "rviz:=true".
            robot_name (str, optional): The ROS parameter name for the robot in the simulation. Defaults to "robot_name:=r1".
        """

        # -----------------------------------LAUNCH ROS--------------------------------------------------
        self.gui_arg = gui_arg
        self.rviz_arg = rviz_arg
        self.robot_name_arg = robot_name
        self.robot_name = robot_name.split("=")[1]

        self.ros_port = ros_port
        # self.__init_launch_ROS(launchfile, ros_port) # moving the launching of ros outside...
        self.__init_ROS_pubs_and_subs()
        self.interaction_flag = False

        # ---------------------------------------MAP--------------------------------------------------
        self.__map = Map(map_min_xy[0], map_max_xy[0], map_min_xy[1], map_max_xy[1])

        self.__static_goal = static_goal
        if self.__static_goal:
            self.__static_goal_x = static_goal_xy[0]
            self.__static_goal_y = static_goal_xy[1]
            self.__static_goal_z = 0.25
        else:
            self.__static_goal_x = None
            self.__static_goal_y = None
            self.__static_goal_z = None

        self.__static_spawn = static_spawn
        if self.__static_spawn:
            self.__static_spawn_x = static_spawn_xy[0]
            self.__static_spawn_y = static_spawn_xy[1]
            self.__static_spawn_z = 0.0
        else:
            self.__static_spawn_x = None
            self.__static_spawn_y = None
            self.__static_spawn_z = None

        # -----------------------------------NOISE - CAMERA AND LIDAR-------------------------------------
        self.__camera_noise_type = camera_noise_type
        # Init camera/lidar noise area as a rectangle with no width or height
        # If it's enabled it will be overwritten to have width and height
        self._camera_noise_area = Rectangle(10, 10, 0, 0)
        self._lidar_noise_area = Rectangle(10, 10, 0, 0)
        self._camera_noise_deviation = None
        self._lidar_noise_deviation = None

        # Create a rectangle at a random position for lidar
        # self.__lidar_noise = lidar_noise
        # self.__init_lidar_noise_area() - do we actually need to init this? just call during reset!

        # -----------------------------------INIT VARIABLES--------------------------------------------------
        # init odom and goal positions
        self.__goal_x, self.__goal_y, self.__goal_z = 0.0, 0.0, 0.0
        self.__last_odom = None
        self.__camera_data = np.zeros((img_width, img_height, 3))
        self.__time_delta = time_delta

        self._current_position = None
        self._current_velocity = None

        self._img_width = img_width
        self._img_height = img_height

        self._zoom_state = 0

        # -----------------------------------INIT LIDAR--------------------------------------------------
        # lidar: init the self.lidar_discretized_points
        # self.lidar_discretized_points = self.__init_lidar(lidar_dim)
        self.lidar_dim = lidar_dim
        self.__lidar_data = []

        # -----------------------------------VIZUALIZATION--------------------------------------------------
        # markers
        self.__marker_publisher = MarkerPublisher()

        # -----------------------------------GAZEBO MODELS--------------------------------------------------
        self.__goal = Goal(
            "goal_obj",
            self.__goal_x,
            self.__goal_y,
            self.__goal_z,
            os.path.abspath(
                os.path.expanduser(
                    "~/ros-rl-env/catkin_ws/src/multi_robot_scenario/models/goal.sdf"
                )
            ),
        )

        self.__robot = Robot(self.robot_name, 0, 0, 0)

        # Init this  properly
        self.__collision_detection = False

    # -----------------------------------Getters + Setters-------------------------------------------------
    @property
    def robot_position(self):
        if self.__last_odom is None:
            return 0, 0, 0
        return (
            self.__last_odom.pose.pose.position.x,
            self.__last_odom.pose.pose.position.y,
            self.__last_odom.pose.pose.position.z,
        )

    # @property
    def get_robot_velocity(self):
        if self._current_velocity is None:
            return 0.0, 0.0, 0.0
        # Otherwise, _current_velocity is a Twist, so we can query linear/angular velocity (forward, turn)
        return (
            self._current_velocity.linear.x,
            self._current_velocity.angular.z,
            self._current_velocity.linear.z,
        )

    # @robot_velocity.setter - work out how to do getter/setters?
    def set_robot_velocity(self, linear_x, angular_z=0.0, linear_z=0.0):
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_x
        vel_cmd.angular.z = angular_z
        vel_cmd.linear.z = linear_z
        self.vel_pub.publish(vel_cmd)
        self._current_velocity = vel_cmd
        self.publish_velocity([linear_x, angular_z, linear_z])

    def set_drone_velocity(self, linear_x, linear_y, linear_z, angular_z):
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_x * 2.5
        vel_cmd.linear.y = linear_y * 2.5
        vel_cmd.linear.z = linear_z * 2.5
        vel_cmd.angular.z = angular_z * 2.5

        self.vel_pub.publish(vel_cmd)
        self._current_velocity = vel_cmd
        self.publish_velocity([linear_x, linear_y, linear_z, angular_z])

    def set_ptz_velocity(self, pan, tilt, zoom):
        pan_msg = Float64(data=float(pan))
        self.pan_pub.publish(pan_msg)
        tilt_msg = Float64(data=float(tilt))
        self.tilt_pub.publish(tilt_msg)

        if zoom == 1:
            self._zoom_state = min(self._zoom_state + 1, 3)
        elif zoom == -1:
            self._zoom_state = max(self._zoom_state - 1, 1)

        # print(f"{self._zoom_state=}")
        return

    def get_robot_quaternion(self):
        if self.__last_odom is None:
            return Quaternion(0, 0, 0, 0)
        quaternion = Quaternion(
            self.__last_odom.pose.pose.orientation.w,
            self.__last_odom.pose.pose.orientation.x,
            self.__last_odom.pose.pose.orientation.y,
            self.__last_odom.pose.pose.orientation.z,
        )
        return quaternion

    def get_goal_position(self):
        return self.__goal_x, self.__goal_y, self.__goal_z

    def get_lidar_data(self):  # Ex get_velodyne_data()
        return self.__lidar_data

    def get_camera_data(self):
        return cv2.resize(self.__camera_data, (self._img_width, self._img_height))

    def get_collision_status(self):
        return self.__collision_detection

    def get_ptz_segmentation_mask(self):
        # not implemented yet. Return the segmentation mask of the goal
        # from the PTZ camera
        return 0

    # -----------------------------------Public Functs-------------------------------------------------
    def reset_camera_noise_area(self, x, y, w, h):
        """
        Resets the camera noise area.
        x: x centre of the area
        y: y centre of the area
        w: width
        h: height
        """
        # xy is the centre of the rectangle
        self._camera_noise_area = Rectangle(
            x,
            y,
            w,
            h,
        )
        self.__marker_publisher.publish_marker_rec(self._camera_noise_area, "world")

    def reset_lidar_noise_area(self, x, y, w, h):
        """
        Resets the lidar noise area.
        x: x centre of the area
        y: y centre of the area
        w: width
        h: height
        """
        # xy is the centre of the rectangle
        self._lidar_noise_area = Rectangle(
            x,
            y,
            w,
            h,
        )
        self.__marker_publisher.publish_marker_lidar(self._lidar_noise_area, "world")

    def respawn_robot(self, top_height=0.5):
        """
        Respawn the robot in the environment.

        If the `static_spawn` flag is set to True, the robot will be respawned at a static position.
        Otherwise, the robot will be respawned at a random position.
        """
        if self.__static_spawn:
            self.__reset_robot_position_static()
        else:
            self.__reset_robot_position_random(top_height)

    def move_goal(self, x, y, z=0.25):
        self.__move_goal(x, y, z)
        self.__goal_x, self.__goal_y, self.__goal_z = x, y, z

    def reset_goal(self, goal_z_max):
        """
        Resets the goal position for the navigation task.

        If the goal is static, the goal position is reset to a predefined static position.
        If the goal is not static, the goal position is reset to a random position.
        """
        if self.__static_goal:
            self.__reset_goal_position_static(goal_z_max)
        else:
            self.__reset_goal_position_random(goal_z_max)

    def publish_velocity(self, action):
        """
        Publishes the velocity markers for the given action.

        Args:
            action (str): The action to be performed.
        """
        self.__marker_publisher.publish_velocity_markers(action, "uav1/base_link")

    def publish_goal_marker(self):
        """
        Publishes the goal position as goal markers.

        Retrieves the goal position using the `get_goal_position` method and publishes it as goal markers
        using the `marker_publisher.publish_goal_markers` method.
        """
        goal_x, goal_y, goal_z = self.get_goal_position()
        self.__marker_publisher.publish_goal_markers("world", goal_x, goal_y, goal_z)

    # -----------------------------------Private Functs-------------------------------------------------
    def __reset_robot_position_static(self):
        """
        Resets the position of the robot to the static spawn position.

        This method sets the x, y, and angle values of the robot's position to the static spawn position
        and then moves the robot to that position.
        """
        x, y, z, angle = (
            self.__static_spawn_x,
            self.__static_spawn_y,
            self.__static_spawn_z,
            0.0,
        )
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        self.__move_robot(x, y, z, quaternion)

    def __reset_robot_position_random(self, top_height=0.05):
        """
        Resets the position of the robot to a random point on the map.

        This method generates random x and y coordinates using the `get_random_point` method of the map object.
        It also generates a random angle between -pi and pi, and converts it to a quaternion using the `from_euler` method of the Quaternion class.
        Finally, it calls the `__move_robot` method to move the robot to the generated position and orientation.
        """
        # TODO give as inputs the bounds of the spawn area
        x, y, z = self.__map.get_random_point(top_height)
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        self.__move_robot(x, y, z, quaternion)

    def __reset_goal_position_static(self, goal_z_max=0.5):
        """
        Resets the goal position to the static goal coordinates.

        This method sets the goal position to the static goal coordinates
        stored in the variables __static_goal_x and __static_goal_y. It then
        updates the current goal position (__goal_x and __goal_y) to match
        the new coordinates.
        """
        goal_x, goal_y = self.__static_goal_x, self.__static_goal_y
        self.__move_goal(goal_x, goal_y, goal_z_max)

        self.__goal_x, self.__goal_y, self.__goal_z = goal_x, goal_y, goal_z_max

    def __reset_goal_position_random(self, goal_z_max):
        """
        Resets the goal position to a random point on the map.

        This method selects a random goal position that is at least a minimum distance
        away from the robot's current position. Once a suitable goal position is found,
        it moves the goal to that position and updates the internal goal coordinates.
        """
        MIN_DISTANCE = 1.5  # Minimum required distance between robot and goal
        MAX_ATTEMPTS = 100  # Maximum number of attempts to find a valid goal position

        robot_position = np.array(self.robot_position)

        for _ in range(MAX_ATTEMPTS):
            # Get a random point on the map
            goal_position = np.array(self.__map.get_random_point(top_height=goal_z_max))

            distance_to_robot = np.linalg.norm(robot_position - goal_position)

            if distance_to_robot >= MIN_DISTANCE:
                # Valid goal position found
                self.__move_goal(*goal_position)
                self.__goal_x, self.__goal_y, self.__goal_z = goal_position
                return

        # If no valid position is found after MAX_ATTEMPTS, raise an exception
        raise ValueError(
            f"Unable to find a suitable goal position at least {MIN_DISTANCE} units away from the robot."
        )

    def __move_goal(self, x, y, z=0.25):
        """
        Moves the goal to the specified coordinates.

        Args:
            x (float): The x-coordinate of the goal.
            y (float): The y-coordinate of the goal.
        """
        self.__goal.move(x, y, z)

    def __move_robot(self, x, y, z, quaternion):
        """
        Moves the robot to the specified position and orientation.

        Args:
            x (float): The x-coordinate of the target position.
            y (float): The y-coordinate of the target position.
            quaternion (Quaternion): The target orientation represented as a quaternion.
        """
        self.__robot.move(x, y, z, quaternion)

    # -----------------------------------Standard functs-------------------------------------------------
    # ---------------------------------(pause, close, etc)-------------------------------------------------
    # NB: THESE FUNCTS CANT BE CALLED just 'pause'! because, self.pause is a reserved
    def pause_ros(self):
        """
        Pauses the ROS interface by calling the appropriate Gazebo services.

        This method waits for the "/gazebo/unpause_physics" service before trying to unpause the physics simulation.
        If the service call fails, it logs an error message.

        After unpausing the physics simulation, it sleeps for a specified time delta.

        Then, it waits for the "/gazebo/pause_physics" service before trying to pause the physics simulation.
        If the service call fails, it logs an error message.
        """

        rospy.wait_for_service(
            "/gazebo/unpause_physics"
        )  # Wait for service before you try
        try:
            self.unpause()
        except rospy.ServiceException as e:
            rospy.loginfo("/gazebo/unpause_physics service call failed")

        rospy.sleep(self.__time_delta)  # propagate state for TIME_DELTA seconds

        rospy.wait_for_service(
            "/gazebo/pause_physics"
        )  # Wait for service before you try
        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.loginfo("/gazebo/pause_physics service call failed")

    def reset_ros(self):
        """
        Resets the ROS environment by calling the "/gazebo/reset_world" service.

        This method waits for the service to be available before making the service call.
        If the service call fails, an error message is logged.
        """

        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to pause Gazebo simulation: {e}")
        time.sleep(1)

        # Wait for service before you try
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            rospy.loginfo("/gazebo/reset_world service call failed")
        rospy.wait_for_service("/gazebo/reset_simulation")
        try:
            self.reset_proxy_sim()
        except rospy.ServiceException as e:
            rospy.loginfo("/gazebo/reset_simulation service call failed")

    def close_ros(self):
        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to pause Gazebo simulation: {e}")
        # 1. Signal shutdown to ROS
        try:
            rospy.signal_shutdown("Environment closed")
            print("Signaled ROS shutdown.")
        except Exception as e:
            print(f"Error signaling ROS shutdown: {e}")
        # 2. Kill all running ROS nodes
        try:
            node_names = rosnode.get_node_names()
            # Exclude '/rosout' if you prefer not to kill it
            nodes_to_kill = [node for node in node_names if node != "/rosout"]
            if nodes_to_kill:
                rosnode.kill_nodes(nodes_to_kill)
                print(f"Killed ROS nodes: {nodes_to_kill}")
            else:
                print("No ROS nodes to kill.")
        except Exception as e:
            print(f"Error killing ROS nodes: {e}")
        # 3. Terminate ROS-related processes using psutil
        ros_process_names = [
            "roscore",
            "rosout",
            "gzserver",
            "gzclient",
            "rviz",
            "python3",
            "python",
        ]
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                process_name = proc.info["name"]
                if process_name in ros_process_names:
                    proc.terminate()
                    print(f"Terminated process: {process_name} (PID: {proc.pid})")
            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess,
            ) as e:
                print(f"Error terminating process {proc.pid}: {e}")
        # 4. Wait for processes to terminate
        time.sleep(2)
        # 5. Force kill any remaining ROS-related processes
        for process_name in ros_process_names:
            os.system(f"killall -9 {process_name} >/dev/null 2>&1")
            print(f"Force killed any remaining {process_name} processes.")
        print("ROS environment cleanup complete.")

    # -----------------------------------Inits-------------------------------------------------

    def __init_launch_ROS(self, launchfile, ros_port):
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
            self.gui_arg,
            self.rviz_arg,
            self.robot_name_arg,
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

    def __init_ROS_pubs_and_subs(self):
        # TODO
        # Check that ROS is launches - if it fails throw an exception
        # ROS should be launched from outside the class

        # TODO put the 4 main topics in the launch file. This way we can avoid remaps in the launch files.
        # Set up the ROS publishers and subscribers
        # self.vel_pub = rospy.Publisher("/uav1/cmd_vel", Twist, queue_size=1)
        self.vel_pub = rospy.Publisher(
            "/" + self.robot_name + "/cmd_vel", Twist, queue_size=1
        )
        # NB: PTZ only exists on the UAV.
        self.pan_pub = rospy.Publisher(
            "/uav1/ptz_cam/ptz_pan_vel/command", Float64, queue_size=1
        )
        self.tilt_pub = rospy.Publisher(
            "/uav1/ptz_cam/ptz_tilt_vel/command", Float64, queue_size=1
        )
        # self.tilt_pub = rospy.Publisher(
        #     "/uav1/ptz_cam/ptz_tilt_vel/command", Float64, queue_size=1
        # )

        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=1
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.reset_proxy_sim = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)

        # We should use the laser /scan from now on
        # self.velodyne = rospy.Subscriber(
        #     "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        # )
        # self.noisy_velodyne_pub = rospy.Publisher(
        #     "/velodyne_points_noisy", PointCloud2, queue_size=10
        # )
        self.scan_sub = rospy.Subscriber(
            "/" + self.robot_name + "/scan",
            LaserScan,
            self.laserscan_callback,
            queue_size=1,
        )
        self.odom = rospy.Subscriber(
            "/" + self.robot_name + "/ground_truth/state",
            Odometry,
            self.odom_callback,
            queue_size=1,
        )

        # Collision stuff
        self.collision_subscriber = rospy.Subscriber(
            "/gazebo/robot_collisions",
            ContactsState,
            self.collision_callback_default,
            queue_size=1,
        )

        # Set up a Publisher for the noisy data
        self.noisy_image_pub = rospy.Publisher(
            "/" + self.robot_name + "/front_cam/camera/image_noisy", Image, queue_size=1
        )
        self.noisy_laserscan_pub = rospy.Publisher(
            "/" + self.robot_name + "/scan_noisy", LaserScan, queue_size=1
        )
        self.scan_msg = LaserScan()

        # Camera stuff
        self.bridge = CvBridge()
        self.camera = rospy.Subscriber(
            # "/" + self.robot_name + "/camera/camera/image_raw",
            "/" + self.robot_name + "/front_cam/camera/image",
            Image,
            self.camera_callback,
            queue_size=1,
        )
        # /uav1/front_cam/camera/image

        # Wait for spawn service to start before spawning models
        rospy.wait_for_service("/gazebo/spawn_sdf_model")

    # -----------------------------------Callbacks-------------------------------------------------

    def laserscan_callback(self, scan_msg):
        ranges = np.array(scan_msg.ranges)
        total_points = len(ranges)

        # Replace inf and NaN values with range_max to avoid invalid ranges
        invalid_indices = np.isnan(ranges) | np.isinf(ranges)
        ranges[invalid_indices] = scan_msg.range_max

        # Downsample the data to the desired number of points
        original_angles = np.linspace(
            scan_msg.angle_min, scan_msg.angle_max, total_points
        )

        # Calculate the angle increment for downsampled points
        downsampled_angle_increment = (
            scan_msg.angle_max - scan_msg.angle_min
        ) / self.lidar_dim

        # Create an array for the downsampled ranges
        downsampled_ranges = np.zeros(self.lidar_dim)

        # For each downsampled point, find the closest original scan point based on angle
        for i in range(self.lidar_dim):
            downsampled_angle = scan_msg.angle_min + i * downsampled_angle_increment
            closest_index = np.argmin(np.abs(original_angles - downsampled_angle))
            downsampled_ranges[i] = ranges[closest_index]

        robot_x, robot_y, _ = self.robot_position

        # Check if the robot is inside the rectangle
        if self._lidar_noise_area.contains((robot_x, robot_y)):
            proximity = self._lidar_noise_area.proximity_to_centre((robot_x, robot_y))
            noise = np.random.normal(0, proximity, downsampled_ranges.shape)
            downsampled_ranges = np.clip(
                downsampled_ranges + noise, scan_msg.range_min, scan_msg.range_max
            )
            self.__lidar_data = downsampled_ranges
        else:
            self.__lidar_data = downsampled_ranges

        self.scan_msg.header = scan_msg.header
        self.scan_msg.angle_min = scan_msg.angle_min
        self.scan_msg.angle_max = scan_msg.angle_max
        self.scan_msg.angle_increment = downsampled_angle_increment
        self.scan_msg.time_increment = scan_msg.time_increment
        self.scan_msg.scan_time = scan_msg.scan_time
        self.scan_msg.range_min = scan_msg.range_min
        self.scan_msg.range_max = scan_msg.range_max
        self.scan_msg.ranges = downsampled_ranges.tolist()

        # Publish the downsampled LaserScan message
        self.noisy_laserscan_pub.publish(self.scan_msg)

    def odom_callback(self, od_data):
        """
        Callback function for the odometry data.

        Args:
            od_data: The odometry data received from the ROS topic.
        """
        self.__last_odom = od_data

    def collision_callback_default(self, contact_states):
        """Collisions Callback

        This method is called when there are collision states detected.
        # NOTE that this way whatever collides within the env counts as detection. It doens't have to be the robot.

        Args:
            contact_states (ContactStates): The contact states object containing collision information.

        """
        if contact_states.states:
            self.__collision_detection = True
        else:
            self.__collision_detection = False

    def camera_callback(self, img):
        """
        Callback function for the camera topic.
        It converts the ROS image message to a NumPy array, checks the robot's
        position relative to a predefined noisy rectangle,
        and applies noise to the image if the robot is inside this rectangle.
        The function then converts the image back to a ROS message and publishes it.

        Args:
            img: The ROS image message received from the camera topic.
        """
        w = img.width
        h = img.height
        # Convert the ROS image to a NumPy array
        original_header, original_connection_header = img.header, img._connection_header
        img_cv = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")

        # get robot positions
        robot_x, robot_y, _ = self.robot_position
        # Calculate proximity to the centre of the rectangle
        proximity = self._camera_noise_area.proximity_to_centre((robot_x, robot_y))

        # Check if the robot is inside the noisy rectangle
        if self._camera_noise_area.contains((robot_x, robot_y)):
            noisy_img = self.__generate_noisy_img(img_cv, noise_str=proximity)

            # Publish noisy image if inside the rectangle
            self.__camera_data = cv2.resize(noisy_img, (w, h))
            img_to_publish = noisy_img
        else:
            # Publish normal image if outside the rectangle
            self.__camera_data = cv2.resize(img_cv, (w, h))
            img_to_publish = img_cv

        # Convert back to ROS msg - need to add original header!

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted_img = self.bridge.cv2_to_imgmsg(img_to_publish, "bgr8")

        converted_img.header, converted_img._connection_header = (
            original_header,
            original_connection_header,
        )

        # Publish the noisy image (for debugging)
        self.noisy_image_pub.publish(converted_img)

    def __generate_noisy_img(self, img, noise_str):
        """
        Generate a noisy image based on the selected noise type.

        Args:
            img (numpy.ndarray): The input image.
            noise_str (float): The strength of the noise.

        Returns:
            noisy_img (numpy.ndarray): The noisy image.
        """
        if self.__camera_noise_type == "gaussian":
            # Add Gaussian noise to the image
            noisy_img = (
                255 * random_noise(img, mode="gaussian", var=noise_str**2, clip=True)
            ).astype(np.uint8)
        elif self.__camera_noise_type == "fail":
            # Make the image all black
            noisy_img = np.zeros_like(img)
        elif self.__camera_noise_type == "random_erase":
            # Randomly erase a rectangle from the image and make it black
            rectangle_pos = (
                random.randint(0, img.width),
                random.randint(0, img.height),
            )
            rectangle_size = (
                random.randint(0, img.width),
                random.randint(0, img.height),
            )
            noisy_img = cv2.rectangle(
                img.copy(), rectangle_pos, rectangle_size, (0, 0, 0), -1
            )
        return noisy_img

    def __check_collision_probability(object_name):
        """Check collision probability based on object name"""
        obj_name = object_name.split("_")[1]
        probability = random.random()
        if obj_name == "1":  # foliage number 1
            return probability >= 1
        elif obj_name == "2":  # foliage number 2
            return probability >= 0
        elif obj_name == "3":  # foliage number 3
            return probability >= 0
        else:
            return False  # Default case if foliage number is not recognized
