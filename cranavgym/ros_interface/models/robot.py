from cranavgym.ros_interface.models.gazebo_model import GazeboModel

# from squaternion import Quaternion
# from scipy.spatial.transform import Rotation

# import rospy
# from geometry_msgs.msg import Pose

class Robot(GazeboModel):
    """Superclass for all Gazebo environments."""

    def __init__(self, model_name, init_x, init_y):
        super().__init__()  # Call the initializer of the ModelHandler

        self.__x_pos, self.__y_pos = init_x, init_y
        self.__model_name = model_name
        self.pose = None
        self.twist = None
        # self.state_subscriber = rospy.Subscriber(
        #     "/robot_pose", Pose, self.model_states_callback
        # )
        self.msg_idx = None

    def move(self, x, y, qx, qy, qz, qw):
        self.move_model(
            self.__model_name,
            x,
            y,
            0.0,
            qx,
            qy,
            qz,
            qw,
        )
        self.__x_pos, self.__y_pos = x, y

    def set_velocity(self, current_stare, x, y, z, ax, ay, az):
        self.set_model_velocity(self.__model_name, current_stare, x, y, z, ax, ay, az)


    # def model_states_callback(self, msg):
    #     self.pose = msg

    # def get_position(self):
    #     return self.pose

    # def get_rotation(self):
    #     return self.twist
        # return self.get_model_state(self.__model_name)
