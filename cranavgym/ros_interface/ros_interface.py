import math
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

from sensor_msgs.msg import Image, PointField, PointCloud2, LaserScan
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState, ContactsState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import std_msgs.msg
from std_srvs.srv import Empty


from cranavgym.ros_interface.utils.map import Map
from cranavgym.ros_interface.utils.rectangle import Rectangle
from cranavgym.ros_interface.utils.marker_publisher import MarkerPublisher


from cranavgym.ros_interface.models.goal import Goal
from cranavgym.ros_interface.models.robot import Robot
import time

import warnings

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

DISCRETE_ACTIONS = False
MULTIPLE_CAMERA_NOISE_AREAS = True


class ROSInterface:
    """
    Interface class to handle the launching of ROS,
    initializing publishers and subscribers,
    and handing all of the callbacks
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
    ):
        """
        Initialize the ROS Interface object.

        Args:
            launchfile (str): The path to the ROS launch file.
            ros_port (str, optional): The port number for the ROS communication. Defaults to "11311".
            time_delta (float, optional): The time delay between each step in the simulation. Defaults to 0.1.
            map_min_xy (list, optional): The minimum x and y coordinates of the map. Defaults to [-5, -5].
            map_max_xy (list, optional): The maximum x and y coordinates of the map. Defaults to [5, 5].
            img_width (int, optional): The width of the camera image. Defaults to 160.
            img_height (int, optional): The height of the camera image. Defaults to 160.
            camera_noise_type (str, optional): The type of noise to apply to the camera. Defaults to "gaussian".
            18/09 - moving to function input # camera_noise_area_size (list, optional): The size of the area for camera noise. Defaults to [4, 4].
            18/09 - moving to function input # lidar_noise_area_size (list, optional): The size of the area for lidar noise. Defaults to [4, 4].
            lidar_dim (int, optional): The dimension of the lidar. Defaults to 20.
            static_goal (bool, optional): Flag indicating whether the goal position is static. Defaults to False.
            static_goal_xy (list, optional): The x and y coordinates of the static goal position. Defaults to [3, 3].
            static_spawn (bool, optional): Flag indicating whether the spawn position is static. Defaults to False.
            static_spawn_xy (list, optional): The x and y coordinates of the static spawn position. Defaults to [0, 0].
        """

        # -----------------------------------LAUNCH ROS--------------------------------------------------
        self.ros_port = ros_port
        self.__init_launch_ROS(launchfile, ros_port)
        self.__init_ROS_pubs_and_subs()
        self.interaction_flag = False

        # ---------------------------------------MAP--------------------------------------------------
        self.__map = Map(map_min_xy[0], map_max_xy[0], map_min_xy[1], map_max_xy[1])

        # NB: for the default ROS map the goal is always random,
        # and the spawn is always random - hence static goal/spawn
        # are set to false. But I included the flags to include for
        # other maps!
        self.__static_goal = static_goal
        if self.__static_goal:
            self.__static_goal_x = static_goal_xy[0]
            self.__static_goal_y = static_goal_xy[1]
        else:
            self.__static_goal_x = None
            self.__static_goal_y = None

        self.__static_spawn = static_spawn
        if self.__static_spawn:
            self.__static_spawn_x = static_spawn_xy[0]
            self.__static_spawn_y = static_spawn_xy[1]
        else:
            self.__static_spawn_x = None
            self.__static_spawn_y = None

        # -----------------------------------NOISE - CAMERA AND LIDAR-------------------------------------
        self.__camera_noise_type = camera_noise_type
        # Init camera/lidar noise area as a rectangle with no width or height
        # If it's enabled it will be overwritten to have width and height
        self._camera_noise_area = Rectangle(10, 10, 0, 0)
        self._lidar_noise_area = Rectangle(10, 10, 0, 0)

        # Create a rectangle at a random position for lidar
        # self.__lidar_noise = lidar_noise
        # self.__init_lidar_noise_area() - do we actually need to init this? just call during reset!

        # -----------------------------------INIT VARIABLES--------------------------------------------------
        # init odom and goal positions
        self.__goal_x, self.__goal_y = 0.0, 0.0
        self.__last_odom = None
        self.__camera_data = np.zeros((img_width, img_height, 3))
        self.__time_delta = time_delta

        self._current_position = None
        self._current_velocity = None

        self._img_width = img_width
        self._img_height = img_height

        # -----------------------------------INIT LIDAR--------------------------------------------------
        # lidar: init the self.lidar_discretized_points
        self.lidar_discretized_points = self.__init_lidar(lidar_dim)
        self.lidar_dim = lidar_dim

        # -----------------------------------VIZUALIZATION--------------------------------------------------
        # markers
        self.__marker_publisher = MarkerPublisher()

        # -----------------------------------GAZEBO MODELS--------------------------------------------------
        self.__goal = Goal(
            "goal_obj",
            self.__goal_x,
            self.__goal_y,
            os.path.abspath(
                os.path.expanduser(
                    "~/ros-rl-env/catkin_ws/src/multi_robot_scenario/models/goal.sdf"
                )
            ),
        )
        self.__robot = Robot("r1", 0, 0)

        self.__soft_body_name = "foliage"  # This should be passed into the funct!

        # Init this  properly
        self.__collision_detection = False
        time.sleep(4)

    # -----------------------------------Getters + Setters-------------------------------------------------
    @property
    def robot_position(self):
        if self.__last_odom is None:
            return 0, 0
        return (
            self.__last_odom.pose.pose.position.x,
            self.__last_odom.pose.pose.position.y,
        )

    # not actually used
    @robot_position.setter
    def robot_position(self, x, y, quaternion):
        self.__move_robot(x, y, quaternion)

    def set_robot_position(self, x, y, quaternion):
        self.__move_robot(x, y, quaternion)

    # @property
    def get_robot_velocity(self):
        if self._current_velocity is None:
            return 0.0, 0.0
        # Otherwise, _current_velocity is a Twist, so we can query linear/angular velocity (forward, turn)
        return self._current_velocity.linear.x, self._current_velocity.angular.z

    # not actually used
    # @robot_velocity.setter - work out how to do getter/setters?
    def set_robot_velocity(self, linear_x, angular_z):
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_x
        vel_cmd.angular.z = angular_z
        self.vel_pub.publish(vel_cmd)
        self._current_velocity = vel_cmd

    # archive
    # def set_robot_velocities(self, linear_x, angular_z):
    #     vel_cmd = Twist()
    #     vel_cmd.linear.x = linear_x
    #     vel_cmd.angular.z = angular_z
    #     self.vel_pub.publish(vel_cmd)
    #     self._current_velocity = vel_cmd

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
        return self.__goal_x, self.__goal_y

    def get_velodyne_data(self):
        return self.__velodyne_data

    def get_camera_data(self):
        # return cv2.resize(self.__camera_data, (self._img_width, self._img_height))
        # get rid of resizes: these slow the sim down!
        # Instead define correct size in the xacro
        return self.__camera_data

    def get_collision_status(self):
        return self.__collision_detection
    
    def reset_collision_status(self):
        self.__collision_detection = False
        return

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
        self.__marker_publisher.publish_marker_rec(self._camera_noise_area, "odom")

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
        self.__marker_publisher.publish_marker_lidar(self._lidar_noise_area, "odom")

    def respawn_robot(self):
        """
        Respawn the robot in the environment.

        If the `static_spawn` flag is set to True, the robot will be respawned at a static position.
        Otherwise, the robot will be respawned at a random position.
        """
        if self.__static_spawn:
            self.__reset_robot_position_static()
        else:
            self.__reset_robot_position_random()

    def reset_goal(self):
        """
        Resets the goal position for the navigation task.

        If the goal is static, the goal position is reset to a predefined static position.
        If the goal is not static, the goal position is reset to a random position.
        """
        if self.__static_goal:
            self.__reset_goal_position_static()
        else:
            self.__reset_goal_position_random()

    def publish_velocity(self, action):
        """
        Publishes the velocity markers for the given action.

        Args:
            action (str): The action to be performed.
        """
        goal_x, goal_y = (
            self.get_goal_position()
        )  # apparently were showing goal position here for some reason?
        self.__marker_publisher.publish_velocity_markers(
            action, "base_link", goal_x, goal_y
        )

    def publish_goal(self):
        """
        Publishes the goal position as goal markers.

        Retrieves the goal position using the `get_goal_position` method and publishes it as goal markers
        using the `marker_publisher.publish_goal_markers` method.
        """
        goal_x, goal_y = self.get_goal_position()
        self.__marker_publisher.publish_goal_markers("odom", goal_x, goal_y)

    # -----------------------------------Private Functs-------------------------------------------------
    def __reset_robot_position_static(self):
        """
        Resets the position of the robot to the static spawn position.

        This method sets the x, y, and angle values of the robot's position to the static spawn position
        and then moves the robot to that position.
        """
        x, y, angle = self.__static_spawn_x, self.__static_spawn_y, 0.0
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        self.__move_robot(x, y, quaternion)

    def __reset_robot_position_random(self):
        """
        Resets the position of the robot to a random point on the map.

        This method generates random x and y coordinates using the `get_random_point` method of the map object.
        It also generates a random angle between -pi and pi, and converts it to a quaternion using the `from_euler` method of the Quaternion class.
        Finally, it calls the `__move_robot` method to move the robot to the generated position and orientation.
        """
        x, y = self.__map.get_random_point()
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        self.__move_robot(x, y, quaternion)

    def __reset_goal_position_static(self):
        """
        Resets the goal position to the static goal coordinates.

        This method sets the goal position to the static goal coordinates
        stored in the variables __static_goal_x and __static_goal_y. It then
        updates the current goal position (__goal_x and __goal_y) to match
        the new coordinates.
        """
        goal_x, goal_y = self.__static_goal_x, self.__static_goal_y
        self.__move_goal(goal_x, goal_y)
        self.__goal_x, self.__goal_y = goal_x, goal_y

    def __reset_goal_position_random(self):
        """
        Resets the goal position to a random point on the map.

        This method calculates the distance between the robot and the goal position.
        If the distance is less than 4, it queries different points until a suitable
        goal position is found. Once a suitable goal position is found, it moves the
        goal to that position and updates the internal goal coordinates.
        """
        distance_to_robot = 0
        # while distance between the robot and the goal is less than 4,
        # query different points
        while distance_to_robot < 4:
            robot_x, robot_y = self.robot_position
            goal_x, goal_y = self.__map.get_random_point()
            distance_to_robot = math.sqrt(
                (robot_x - goal_x) ** 2 + (robot_y - goal_y) ** 2
            )

        self.__move_goal(goal_x, goal_y)
        self.__goal_x, self.__goal_y = goal_x, goal_y

    def __move_goal(self, x, y):
        """
        Moves the goal to the specified coordinates.

        Args:
            x (float): The x-coordinate of the goal.
            y (float): The y-coordinate of the goal.
        """
        self.__goal.move(x, y)

    def __move_robot(self, x, y, quaternion):
        """
        Moves the robot to the specified position and orientation.

        Args:
            x (float): The x-coordinate of the target position.
            y (float): The y-coordinate of the target position.
            quaternion (Quaternion): The target orientation represented as a quaternion.
        """
        self.__robot.move(x, y, quaternion)

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

        # rospy.wait_for_service(
        #     "/gazebo/unpause_physics"
        # )  # Wait for service before you try
        try:
            self.unpause()
        except rospy.ServiceException as e:
            rospy.loginfo("/gazebo/unpause_physics service call failed")

        # time.sleep(self.__time_delta)  # propagate state for TIME_DELTA seconds

        # rospy.wait_for_service(
        #     "/gazebo/pause_physics"
        # )  # Wait for service before you try
        # try:
        #     self.pause()
        # except rospy.ServiceException as e:
        #     rospy.loginfo("/gazebo/pause_physics service call failed")

    def reset_ros(self):
        """
        Resets the ROS environment by calling the "/gazebo/reset_world" service.

        This method waits for the service to be available before making the service call.
        If the service call fails, an error message is logged.
        """
        # Wait for service before you try
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            rospy.loginfo("/gazebo/reset_simulation service call failed")

    def close_ros(self):
        """Terminate everything.

        This method shuts down the ROS node if it's running and closes various ROS-related processes.
        It kills the following processes:
        - rosout
        - roslaunch
        - rosmaster
        - gzserver
        - nodelet
        - robot_state_publisher
        - gzclient
        - python
        - python3
        - rviz
        """
        # Shutdown ROS node if it's running
        if rospy.is_shutdown() == False:
            rospy.signal_shutdown("Closing GazeboEnv")

        # Close various ROS-related processes
        subprocess.Popen(["killall", "rosout"])
        subprocess.Popen(["killall", "roslaunch"])
        subprocess.Popen(["killall", "rosmaster"])
        subprocess.Popen(["killall", "gzserver"])
        subprocess.Popen(["killall", "nodelet"])
        subprocess.Popen(["killall", "robot_state_publisher"])
        subprocess.Popen(["killall", "gzclient"])
        subprocess.Popen(["killall", "python"])
        subprocess.Popen(["killall", "python3"])
        subprocess.Popen(["killall", "rviz"])

    # -----------------------------------Inits-------------------------------------------------
    def __init_launch_ROS(self, launchfile, ros_port):
        """
        Initialize and launch the ROS core and Gazebo simulation.

        Args:
            launchfile (str): The path of the launchfile to be used for the simulation.
            ros_port (int): The port number for the ROS core.
        """
        print(f"{ros_port=}")
        subprocess.Popen(["roscore", "-p", ros_port])

        rospy.loginfo("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True, log_level=rospy.WARN)
        fullpath = os.path.abspath(os.path.expanduser(launchfile))

        # TODO put them in the config file and remove them from DRLNav.launch
        # gui_arg = "gui:=true"
        # rviz_arg = "rviz:=false"
        # world_arg = "world_name:=TD3.world"
        subprocess.Popen(
            ["roslaunch", "-p", ros_port, fullpath]  # , gui_arg, rviz_arg, world_arg]
        )

        rospy.loginfo("Gazebo launched!")

    def __init_ROS_pubs_and_subs(self):
        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
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
            "/r1/front_camera/image_raw_noisy", Image, queue_size=10
        )
        self.noisy_velodyne_pub = rospy.Publisher(
            "/velodyne_points_noisy", PointCloud2, queue_size=10
        )
        self.noisy_laserscan_pub = rospy.Publisher(
            "/r1/front_laser/scan_noise", LaserScan, queue_size=10
        )
        self.scan_msg = LaserScan()

        # Camera stuff
        self.bridge = CvBridge()
        self.camera = rospy.Subscriber(
            "/r1/front_camera/image_raw", Image, self.camera_callback, queue_size=1
        )

        # Wait for spawn service to start before spawning models
        rospy.wait_for_service("/gazebo/spawn_sdf_model")

    def __init_lidar(self, lidar_dim):
        """
        Initializes the lidar with the given dimensions.
        Discretizes lidar points to the size of lidar_dim
        by taking half of a circle and dividing it into the discrete points

        Args:
            lidar_dim (int): The number of lidar dimensions.

        Returns:
            list: A list of lidar discretized points.

        """
        self.__velodyne_data = np.ones(lidar_dim) * 10
        # lidar gaps - discretize the lidar to the lidar_dim
        lidar_discretized_points = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / lidar_dim]]
        for m in range(lidar_dim - 1):
            lidar_discretized_points.append(
                [
                    lidar_discretized_points[m][1],
                    lidar_discretized_points[m][1] + np.pi / lidar_dim,
                ]
            )
        lidar_discretized_points[-1][-1] += 0.03
        return lidar_discretized_points

    # -----------------------------------Callbacks-------------------------------------------------

    def velodyne_callback(self, v):
        # TODO introduce noise to specific Z in the pointcloud
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        # get robot positions
        robot_x, robot_y = self.robot_position
        # Check if the robot is inside the rectangle
        if self._lidar_noise_area.contains((robot_x, robot_y)):
            # Convert the PointCloud2 data to a numpy array
            np_data = pc2.read_points(v, field_names=("x", "y", "z"), skip_nans=False)
            np_data = np.array(list(np_data))

            # Calculate proximity to the centre of the rectangle
            proximity = self._lidar_noise_area.proximity_to_centre((robot_x, robot_y))

            # Add Gaussian noise to the data
            # noise = np.random.normal(
            #     0, 0.1 * proximity, np_data.shape
            # )  # Adjust the mean and stddev as needed
            noise = np.random.normal(
                0, 3.0, np_data.shape
            )  # MW update - just make it consistent across the area
            noisy_points = np_data + noise

            # Convert the noisy data back to a PointCloud2 message
            header = std_msgs.msg.Header(
                stamp=v.header.stamp, frame_id=v.header.frame_id
            )
            fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
            ]
            noisy_data = pc2.create_cloud(header, fields, noisy_points)
        else:
            noisy_data = v
            np_data = pc2.read_points(v, field_names=("x", "y", "z"), skip_nans=False)
            np_data = np.array(list(np_data))
            noisy_points = np_data

        # Publish the noisy data
        self.noisy_velodyne_pub.publish(noisy_data)

        data = noisy_points
        self.__velodyne_data = np.ones(self.lidar_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                for j in range(len(self.lidar_discretized_points)):
                    if (
                        self.lidar_discretized_points[j][0]
                        <= beta
                        < self.lidar_discretized_points[j][1]
                    ):
                        self.__velodyne_data[j] = min(self.__velodyne_data[j], dist)
                        break

        self.scan_msg.header = std_msgs.msg.Header(
            stamp=v.header.stamp, frame_id=v.header.frame_id
        )
        self.scan_msg.angle_min = -np.pi / 2  # Set the minimum angle of the laser scan
        self.scan_msg.angle_max = np.pi / 2  # Set the maximum angle of the laser scan
        self.scan_msg.angle_increment = np.pi / len(
            self.__velodyne_data
        )  # Set the angle increment
        self.scan_msg.range_min = 0.0  # Set the minimum range value
        self.scan_msg.range_max = 100.0  # Set the maximum range value
        self.scan_msg.ranges = self.__velodyne_data.tolist()  # Set the range values
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

        Args:
            contact_states (ContactStates): The contact states object containing collision information.

        """
        if contact_states.states:
            self.__collision_detection = True
        else:
            self.__collision_detection = False

    def __collision_callback_i_room(self, contact_states):
        """
        Collision callback which checks for collisions with soft bodies
        Only used in the I room!
        """
        if contact_states.states:
            for contact_state in contact_states.states:
                # Extracting collision names
                collision1_name = contact_state.collision1_name
                collision2_name = contact_state.collision2_name
                if (
                    collision2_name.split("_")[0] == self.__soft_body_name
                ):  # If collision is a softbody
                    collision = self.__check_collision_probability(collision2_name)
                    if collision:
                        self.__collision_detection = True
                    else:
                        self.model_handler.move_model(
                            collision2_name.split(":")[0], 20, 20, 0, 0, 0, 0, 1
                        )
                        self.interaction_flag = True
                    return
                elif (
                    collision1_name.split("_")[0] == self.__soft_body_name
                ):  # If collision is a softbody
                    collision = self.__check_collision_probability(collision1_name)
                    if collision:
                        self.__collision_detection = True
                    else:
                        self.model_handler.move_model(
                            collision1_name.split(":")[0], 20, 20, 0, 0, 0, 0, 1
                        )
                        self.interaction_flag = True
                    return
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
        robot_x, robot_y = self.robot_position
        # Calculate proximity to the centre of the rectangle
        proximity = self._camera_noise_area.proximity_to_centre((robot_x, robot_y))

        # Check if the robot is inside the noisy rectangle
        if self._camera_noise_area.contains((robot_x, robot_y)):
            noisy_img = self.__generate_noisy_img(img_cv, noise_str=proximity)

            # Publish noisy image if inside the rectangle
            # self.__camera_data = cv2.resize(noisy_img, (w, h))
            # get rid of resizes - define correct w/h in the camera xacro
            self.__camera_data = noisy_img
            img_to_publish = noisy_img
        else:
            # Publish normal image if outside the rectangle
            # self.__camera_data = cv2.resize(img_cv, (w, h))
            # get rid of resizes - define correct w/h in the camera xacro
            self.__camera_data = img_cv
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
        # self.noisy_image_pub.publish(converted_img)

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
