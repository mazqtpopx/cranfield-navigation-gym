from cranavgym.ros_interface.models.gazebo_model import GazeboModel


class Goal(GazeboModel):
    """Superclass for all Gazebo environments."""

    def __init__(self, model_name, init_x, init_y, sdf_file_path):
        super().__init__()  # Call the initializer of the ModelHandler

        self.spawn_model_sdf(model_name, init_x, init_y, 0, sdf_file_path)

        self.__x_pos, self.__y_pos = init_x, init_y
        self.__model_name = model_name
        self.__sdf_file_path = sdf_file_path

    def move(self, x, y):
        # goal_ok, goal_x, goal_y = self.__get_new_goal_post()
        self.move_model(self.__model_name, x, y, 0.25, 0, 0, 0, 1)

        self.x_pos, self.y_pos = x, y

    def delete(self):
        self.delete_model(self.__model_name)
