from cranavgym.ros_interface.models.gazebo_model import GazeboModel

# from squaternion import Quaternion
# from scipy.spatial.transform import Rotation


class Robot(GazeboModel):
    """Superclass for all Gazebo environments."""

    def __init__(self, model_name, init_x, init_y):
        super().__init__()  # Call the initializer of the ModelHandler

        self.__x_pos, self.__y_pos = init_x, init_y
        self.__model_name = model_name

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

    def set_velocity(self, x, y, z, ax, ay, az):
        self.set_model_velocity(self.__model_name, x, y, z, ax, ay, az)

    # def get_position(self):

    #     return self.__x_pos, self.__y_pos

    def get_position(self):
        return self.get_model_state(self.__model_name)
