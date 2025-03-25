import random
import numpy as np


class Map:
    def __init__(self, min_x, max_x, min_y, max_y):
        self.__min_x = min_x
        self.__max_x = max_x
        self.__min_y = min_y
        self.__max_y = max_y

    """
    Returns a random point within the bounds of the map.
    The point is checked to make sure it does not fall
    within the position of the walls or the objects in the map.
    """

    def get_random_point(self):
        pos_ok = False
        while not pos_ok:
            x = np.random.uniform(self.__min_x, self.__max_x)
            y = np.random.uniform(self.__min_y, self.__max_y)
            pos_ok = self.__check_pos(x, y)
        return x, y

    """
    Check that the position of the point does not fall within 
    areas occupied by walls, or objects, that cannot be 
    accessed by the agent
    """

    def __check_pos(self, x, y):
        # Define the conditions as a list of tuples
        # NB: uncomment to integrate with map I shape
        # if MAP_SHAPE_I:
        #     conditions = [
        #     (-2.5 < x < 2.5 and -2.5<y<2.5),
        #     (3.5 < x < 6.5 and -6.5<y<6.5),
        #     (-6.5 < x < -3.5 and -6.5<y<6.5)
        #     ]
        #     # Check conditions in a loop
        #     goal_ok = any(conditions)

        # else:
        conditions = [
            (-3.8 > x > -6.2 and 6.2 > y > 3.8),
            (-1.3 > x > -2.7 and 4.7 > y > -0.2),
            (-0.3 > x > -4.2 and 2.7 > y > 1.3),
            (-0.8 > x > -4.2 and -2.3 > y > -4.2),
            (-1.3 > x > -3.7 and -0.8 > y > -2.7),
            (4.2 > x > 0.8 and -1.8 > y > -3.2),
            (4 > x > 2.5 and 0.7 > y > -3.2),
            (6.2 > x > 3.8 and -3.3 > y > -4.2),
            (4.2 > x > 1.3 and 3.7 > y > 1.5),
            (-3.0 > x > -7.2 and 0.5 > y > -1.5),
            (x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5),
            ((x < -1 or x > 9) and (y > 6 or y < -4)),
        ]

        # Check conditions in a loop
        pos_ok = not any(conditions)

        return pos_ok
