import numpy as np
import rospy
import math
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial import transform
from squaternion import Quaternion
from std_msgs.msg import ColorRGBA


class MarkerPublisher:
    def __init__(self):
        # Set up a Publisher for the noise areas
        self.marker_pub_rec = rospy.Publisher('camera_marker', Marker, queue_size=10)
        self.marker_pub_lidar = rospy.Publisher('lidar_marker', Marker, queue_size=10)
        self.goal_point_publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.velocity_publisher = rospy.Publisher("action_velocity", MarkerArray, queue_size=1)
        self.subgoal_point_1_publisher = rospy.Publisher("subgoal_point_1", MarkerArray, queue_size=3)
        self.subgoal_point_2_publisher = rospy.Publisher("subgoal_point_2", MarkerArray, queue_size=3)

    def publish_goal_markers(self, frame_id, x, y):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)
        self.goal_point_publisher.publish(markerArray)

    def publish_velocity_markers(self, action,frame_id, x, y):
        markerArray2 = MarkerArray()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.scale.x = abs(action[0]) * 1.0 + 0.0001
        marker.scale.y = 0.1
        marker.scale.z = 0.001
        marker.color.a = 1.0
        marker.color.r = 0.6
        marker.color.g = 0.2
        marker.color.b = 1.0

        if action[0] >= 0:
            quaternion = transform.Rotation.from_euler('z', -action[1], degrees=False).as_quat()
        else:
            quaternion = transform.Rotation.from_euler('z', -action[1]+ math.pi, degrees=False).as_quat()

        normalized_quaternion = quaternion / np.linalg.norm(quaternion)

        marker.pose.orientation.x = normalized_quaternion[0]
        marker.pose.orientation.y = normalized_quaternion[1]
        marker.pose.orientation.z = normalized_quaternion[2]
        marker.pose.orientation.w = normalized_quaternion[3]

        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0.1

        markerArray2.markers.append(marker)
        self.velocity_publisher.publish(markerArray2)
            
    def publish_marker_rec(self,rec,frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id  # Change this to the frame in which your rectangle is defined
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = rec.x + rec.width / 2
        marker.pose.position.y = rec.y + rec.height / 2
        marker.pose.position.z = 0
        marker.pose.orientation = Quaternion(0, 0, 0, 1)  # Identity quaternion
        marker.scale.x = rec.width
        marker.scale.y = rec.height
        marker.scale.z = 0.1  # Change this to the desired height of the rectangle
        marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.3)  # RGBA color (blue in this case)
        marker.lifetime = rospy.Duration()  # The marker will last until it is removed

        self.marker_pub_rec.publish(marker)

    def publish_marker_lidar(self,rec,frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id  # Change this to the frame in which your rectangle is defined
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = rec.x + rec.width / 2
        marker.pose.position.y = rec.y + rec.height / 2
        marker.pose.position.z = 0
        marker.pose.orientation = Quaternion(0, 0, 0, 1)  # Identity quaternion
        marker.scale.x = rec.width
        marker.scale.y = rec.height
        marker.scale.z = 0.1  # Change this to the desired height of the rectangle
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.3)  # RGBA color (red in this case)
        marker.lifetime = rospy.Duration()  # The marker will last until it is removed

        self.marker_pub_lidar.publish(marker)


    def render_subgoal(self, subgoal_1, subgoal_2):
        #render subgoal

        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = subgoal_1[0]
        marker.pose.position.y = subgoal_1[1]
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.subgoal_point_1_publisher.publish(markerArray)

        # Publish visual data in Rviz
        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CYLINDER
        marker2.action = marker.ADD
        marker2.scale.x = 0.25
        marker2.scale.y = 0.25
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 0.0
        marker2.color.g = 1.0
        marker2.color.b = 0.5
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = subgoal_2[0]
        marker2.pose.position.y = subgoal_2[1]
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)

        self.subgoal_point_2_publisher.publish(markerArray2)
        return