from cranavgym.ros_interface.models.gazebo_model import GazeboModel
from squaternion import Quaternion


class Robot(GazeboModel):
    """Superclass for all Gazebo environments."""

    def __init__(self, model_name, init_x, init_y):
        super().__init__()  # Call the initializer of the ModelHandler

        self.__x_pos, self.__y_pos = init_x, init_y
        self.__model_name = model_name

    def move(self, x, y, quaternion):
        self.move_model(
            self.__model_name,
            x,
            y,
            0.0,
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        )
        self.__x_pos, self.__y_pos = x, y

    def get_position(self):
        return self.__x_pos, self.__y_pos
