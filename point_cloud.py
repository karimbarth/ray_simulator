import numpy as np


class PointCloud:
    def __init__(self, lidar_origin, lidar_orientation):
        self.lidar_origin_ = lidar_origin
        self.lidar_orientation_ = 0
        self.points_ = []
        self.lidar_frame_array_ = None
        self.map_frame_array_ = None

    def add_point(self, point):
        self.points_ += point
        self.lidar_frame_array_ = None
        self.map_frame_array_ = None

    @property
    def count(self):
        return len(self.points_)

    @property
    def lidar_frame_array(self):
        if self.map_frame_array_ is None:
            cloud_array = self.map_frame_array
            return cloud_array - np.array([[self.lidar_origin_[0]], [self.lidar_origin_[1]]])
        else:
            return self.map_frame_array_

    @property
    def map_frame_array(self):
        if self.map_frame_array_ is None:
            return np.stack(self.points_, axis=1)
        else:
            return self.map_frame_array_

    @property
    def map_frame_list(self):
        return self.points_

    def transform_from_lidar_frame_to(self, translation_x, translation_y, theta):
        points = self.lidar_frame_array

        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        translation = np.array([[translation_x], [translation_y]])

        return np.matmul(rotation, points) + translation


