import math
import numpy as np

import shapely
from shapely.geometry import LineString
from point_cloud import PointCloud


class Rangefinder:
    def __init__(self, max_num_points, range_variance=0.02, angular_variance=0.02):
        self.min_angle = -math.pi
        self.max_angle = math.pi
        self.angular_resultion = (self.max_angle - self.min_angle)/max_num_points  # 0.025
        self.max_range = 30
        self.range_variance = range_variance
        self.angular_variance = angular_variance
        np.random.seed(43)

    def scan(self, environment, origin):
        point_cloud = PointCloud(origin, 0)
        for ray_angle in np.arange(self.min_angle, self.max_angle, self.angular_resultion):
            ray_angle_noise = np.random.normal(0, self.angular_variance, 1)
            ray_angle += ray_angle_noise
            dx = np.sin(ray_angle)
            dy = np.cos(ray_angle)
            ray = (origin[0] + self.max_range * dx, origin[1] + self.max_range * dy)
            ray_line = LineString([origin, ray])
            # x, y = ray_line.xy
            # ax.plot(x, y, color='black',  linewidth=1)
            intersect = ray_line.intersection(environment)
            if not intersect.is_empty:
                if (isinstance(intersect, shapely.geometry.point.Point)):
                    np_ray = np.array([intersect.coords[0][0] - origin[0], intersect.coords[0][1] - origin[1]])
                    ray_length = np.linalg.norm(np_ray)
                    range_noise = np.random.normal(0, self.range_variance, 1)
                    point_cloud.add_point([origin + np_ray * ((ray_length + range_noise) / ray_length)])
                elif (isinstance(intersect, shapely.geometry.multipoint.MultiPoint)):
                    # get observation closest to the sensor
                    # print(intersect)
                    nearest_observation_distance = np.inf
                    nearest_observation = None
                    for p in intersect:
                        candidate_distance = np.linalg.norm(np.array(origin) - np.array(p))
                        if candidate_distance < nearest_observation_distance:
                            nearest_observation = p
                            nearest_observation_distance = candidate_distance
                    np_ray = np.array([nearest_observation.x - origin[0], nearest_observation.y - origin[1]])
                    ray_length = np.linalg.norm(np_ray)
                    range_noise = np.random.normal(0, self.range_variance, 1)
                    point_cloud.add_point([origin + np_ray * ((ray_length + range_noise) / ray_length)])
                else:
                    print('Unhandled intersection type', type(intersect))
        return point_cloud
