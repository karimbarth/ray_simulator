import numpy as np


def calc_angle(vec):
    x = vec[0]
    y = vec[1]

    if x > 0 and y >= 0:
        return np.arctan(y/x)
    elif x > 0 and y < 0:
        return np.arctan(y/x) + 2*np.pi
    elif x < 0:
        return np.arctan(y/x) + np.pi
    elif x == 0 and y > 0:
        return np.pi/2
    else:
        return 3/2*np.pi


def __pca(points):
    np_points = np.array(points).transpose()
    cov = np.cov(np_points)
    eigen_values = np.linalg.eigvals(cov)
    min_index = np.argmax(eigen_values)
    eigen_vectors = np.linalg.eig(cov)

    normal_candidate = eigen_vectors[min_index]

    if normal_candidate.shape == (2, 2):
        normal_candidate = normal_candidate[:, 1]

    normal = np.array([normal_candidate[1], -normal_candidate[0]])

    return normal


def normalize(vec):
    return vec / np.linalg.norm(vec)


def select_neighborhood(point, points, ball_radius):
    neighbors = []
    min_radius = ball_radius
    while len(neighbors) < 3:
        neighbors = []
        for single_point in points:
            distance = np.linalg.norm(point - single_point)
            if distance <= min_radius:
                neighbors.append(single_point)
        min_radius = min_radius + 0.5 * ball_radius

    return neighbors


def estimate_normal(point, neighbors, viewpoint):
    normal_candidate = __pca(neighbors)
    normalized_candidate = normalize(normal_candidate)
    viewpoint_vec = np.array(viewpoint)
    temp = viewpoint_vec - point
    dir = np.dot(normalized_candidate, temp)
    return normalized_candidate if dir > 0 else -normalized_candidate


class PointCloud:
    def __init__(self, lidar_origin, lidar_orientation):
        self.__lidar_origin = lidar_origin
        self.__lidar_orientation = 0
        self.__points = list()
        self.__lidar_frame_array = None
        self.__map_frame_array = None
        self.__descriptors = dict()

    def add_point(self, point):
        self.__points += point
        self.__lidar_frame_array = None
        self.__map_frame_array = None

    def add_descriptor(self, name, value):
        if not (name in self.__descriptors):
            self.__descriptors[name] = list()

        self.__descriptors[name].append(value)

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

    def calc_normals(self):
        normals = dict()
        normals_direction = dict()
        for n in range(len(self.map_frame_list)):
            point = self.map_frame_list[n]
            normal = estimate_normal(point, select_neighborhood(point, self.map_frame_list, 0.25),
                                     self.lidar_origin)
            normals[n] = normal
            normals_direction[n] = calc_angle(normal)

        self.__descriptors["normals"] = normals
        self.__descriptors["normals_direction"] = normals_direction

    def get_descriptor(self, name, index):
        return (self.__descriptors[name])[index]

