from cranavgym.ros_interface.models.gazebo_model import GazeboModel
from squaternion import Quaternion


class Robot(GazeboModel):
    """Superclass for all Gazebo environments."""

    def __init__(self, model_name, init_x, init_y, init_z):
        super().__init__()  # Call the initializer of the ModelHandler

        self.__x_pos, self.__y_pos, self.__z_pos = init_x, init_y, init_z
        self.__model_name = model_name

    def move(self, x, y, z, quaternion):
        self.move_model(
            self.__model_name,
            x,
            y,
            z,
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        )
        self.__x_pos, self.__y_pos, self.__z_pos = x, y, z

    def get_position(self):
        return self.__x_pos, self.__y_pos
