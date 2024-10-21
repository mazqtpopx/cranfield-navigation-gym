import rospy
from geometry_msgs.msg import Pose, Point
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel, DeleteModel

from squaternion import Quaternion


class GazeboModel:
    """
    A class that handles the spawning, moving, and deleting of models in the Gazebo simulation environment.

    Attributes:
        spawn_model_service (rospy.ServiceProxy): The service proxy for spawning a model.
        delete_model_service (rospy.ServiceProxy): The service proxy for deleting a model.
        set_state (rospy.Publisher): The publisher for setting the state of a model.
        spawned_models (list): A list of the names of the models that have been spawned.

    Methods:
        spawn_model_sdf: Spawns a model in the Gazebo simulation environment using an SDF file.
        move_model: Moves the specified model to the given position and orientation.
        delete_model: Deletes a model from Gazebo.
    """

    def __init__(self):
        self.__spawn_model_service = rospy.ServiceProxy(
            "/gazebo/spawn_sdf_model", SpawnModel
        )
        self.__delete_model_service = rospy.ServiceProxy(
            "/gazebo/delete_model", DeleteModel
        )
        self.__set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.__spawned_models = []

    """
    This needs to be ... 
    """

    def move(self):
        raise NotImplementedError

    def delete(self):
        raise NotImplementedError

    def spawn_model_sdf(self, model_name, x, y, z, sdf_file_path):
        """
        Spawns a model in the Gazebo simulation environment using an SDF file.

        Args:
            model_name (str): The name of the model to be spawned.
            x (float): The x-coordinate of the model's position.
            y (float): The y-coordinate of the model's position.
            z (float): The z-coordinate of the model's position.
            sdf_file_path (str): The file path to the SDF file.

        Returns:
            None

        Raises:
            rospy.ServiceException: If the service call to spawn the model fails.
        """
        with open(sdf_file_path, "r") as sdf_file:
            model_xml = sdf_file.read()

        model_pose = Pose(
            position=Point(x, y, z), orientation=Quaternion(0.0, 0.0, 0.0, 1.0)
        )

        # Call the service to spawn the model
        rospy.loginfo(f"Trying to spawn again the model {model_name}")
        try:
            self.__spawn_model_service(model_name, model_xml, "", model_pose, "world")
            rospy.sleep(0.2)

            # Add the spawned model to the list
            self.__spawned_models.append(model_name)
            rospy.loginfo(f"Added model {model_name} to the list")

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def move_model(self, model_name, x, y, z, qx, qy, qz, qw):
        """
        Moves the specified model to the given position and orientation.

        Args:
            model_name (str): The name of the model to be moved.
            x (float): The x-coordinate of the desired position.
            y (float): The y-coordinate of the desired position.
            z (float): The z-coordinate of the desired position.
            qx (float): The x-component of the desired orientation quaternion.
            qy (float): The y-component of the desired orientation quaternion.
            qz (float): The z-component of the desired orientation quaternion.
            qw (float): The w-component of the desired orientation quaternion.
        """
        state = ModelState()
        state.model_name = model_name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw
        self.__set_state.publish(state)
        # rospy.sleep(0.1)

    def delete_model(self, model_name):
        """
        Deletes a model from Gazebo.

        Args:
            model_name (str): The name of the model to be deleted.

        Returns:
            None

        Raises:
            rospy.ServiceException: If the service call fails.
        """
        try:
            response = self.delete_model_service(model_name)
            rospy.sleep(0.2)  # Time for models to despawn
            if response.success:
                rospy.loginfo(f"Model '{model_name}' successfully deleted from Gazebo.")
            else:
                rospy.logwarn(f"Failed to delete model '{model_name}' from Gazebo.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
