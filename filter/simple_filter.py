from point_cloud import PointCloud
import copy
from random import shuffle
from filter.point_cloud_filter import PointCloudFilter


class IdentityFilter(PointCloudFilter):

    def __init__(self):
        super()
        self.set_name("identity_filter")

    def apply(self, point_cloud):
        return point_cloud


class RandomFilter(PointCloudFilter):
    def __init__(self):
        super()
        self.set_name("random_filter")

    def apply(self, point_cloud):
        result = PointCloud(point_cloud.lidar_origin, point_cloud.lidar_orientation)
        copy_list = copy.deepcopy(point_cloud.map_frame_list)
        shuffle(copy_list)
        list_of_points = copy_list[0:self.wished_size]
        for point in list_of_points:
            result.add_point([point])

        return result



