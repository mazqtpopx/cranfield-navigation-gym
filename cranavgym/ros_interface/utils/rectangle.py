from scipy.spatial import distance


class Rectangle:
    """
    A class representing a rectangle.

    Attributes:
        centre_x (float): The x-coordinate of the centre of the rectangle.
        centre_y (float): The y-coordinate of the centre of the rectangle.
        width (float): The width of the rectangle.
        height (float): The height of the rectangle.

        x (float): The x-coordinate of the bottom-right corner of the rectangle.
        y (float): The y-coordinate of the bottom-right corner of the rectangle.
        centre (tuple): A tuple representing the coordinates of the centre of the rectangle.

    Methods:
        contains(point): Check if a given point is inside the rectangle.
        distance_to_centre(point): Calculate the Euclidean distance between a given point and the centre of the rectangle.
        proximity_to_centre(point): Calculate the proximity of a given point to the centre of the rectangle.

    """

    def __init__(self, centre_x, centre_y, width, height):
        """
        Initialize a Rectangle object.

        Args:
            centre_x (float): The x-coordinate of the centre of the rectangle.
            centre_y (float): The y-coordinate of the centre of the rectangle.
            width (float): The width of the rectangle.
            height (float): The height of the rectangle.
        """
        self.centre_x = centre_x
        self.centre_y = centre_y

        self.x = self.centre_x - width / 2
        self.y = self.centre_y - height / 2
        self.width = width
        self.height = height

        self.centre = (self.centre_x, self.centre_y)

    def contains(self, point):
        """
        Check if a given point is inside the rectangle.

        Args:
            point (tuple): A tuple representing the coordinates of the point.

        Returns:
            bool: True if the point is inside the rectangle, False otherwise.
        """
        x, y = point
        return (
            self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height
        )

    def distance_to_centre(self, point):
        """
        Calculate the Euclidean distance between a given point and the centre of the rectangle.

        Args:
            point (tuple): A tuple representing the coordinates of the point.

        Returns:
            float: The Euclidean distance between the point and the centre of the rectangle.
        """
        return distance.euclidean(self.centre, point)

    def proximity_to_centre(self, point):
        """
        Calculate the proximity of a given point to the centre of the rectangle.

        Args:
            point (tuple): A tuple representing the coordinates of the point.

        Returns:
            float: The proximity of the point to the centre of the rectangle, ranging from 0 to 1.
        """
        max_distance = distance.euclidean(self.centre, (self.x, self.y)) + 1e-6
        current_distance = self.distance_to_centre(point)
        return 1 - current_distance / max_distance
