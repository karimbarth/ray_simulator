import numpy as np


class PointCloud:
    def __init__(self, lidar_origin, lidar_orientation):
        self.__lidar_origin = lidar_origin
        self.__lidar_orientation = 0
        self.__points = list()
        self.__lidar_frame_array = None
        self.__map_frame_array = None

    def add_point(self, point):
        self.__points += point
        self.__lidar_frame_array = None
        self.__map_frame_array = None

    @property
    def lidar_origin(self):
        return self.__lidar_origin

    @property
    def lidar_orientation(self):
        return self.__lidar_orientation

    @property
    def count(self):
        return len(self.__points)

    @property
    def lidar_frame_array(self):
        if self.__map_frame_array is None:
            cloud_array = self.map_frame_array
            return cloud_array - np.array([[self.__lidar_origin[0]], [self.__lidar_origin[1]]])
        else:
            return self.__map_frame_array

    @property
    def map_frame_array(self):
        if self.__map_frame_array is None:
            return np.stack(self.__points, axis=1)
        else:
            return self.__map_frame_array

    @property
    def map_frame_list(self):
        return self.__points

    def transform_from_lidar_frame_to(self, translation_x, translation_y, theta):
        points = self.lidar_frame_array

        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        translation = np.array([[translation_x], [translation_y]])

        return np.matmul(rotation, points) + translation


