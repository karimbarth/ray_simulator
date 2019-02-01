import numpy as np
from point_cloud import PointCloud
from filter.point_cloud_filter import PointCloudFilter


class VoxelFilter:
    """
        Voxel filter for point clouds. For each voxel, the assembled point cloud
        contains the first point that fell into it from any of the inserted point
        clouds.
    """
    def __init__(self, size):
        self.resolution_ = size
        self.voxel_set_ = set()

    def filter(self, point_cloud):
        result = PointCloud(point_cloud.lidar_origin, point_cloud.lidar_orientation)
        for point in point_cloud.map_frame_list:
            element = self.index_to_key_(self.get_cell_index_(point))
            if not(element in self.voxel_set_):
                self.voxel_set_.add(element)
                result.add_point([point])

        return result

    """
        Helper methode
    """
    def index_to_key_(self, index):
        return tuple(index)

    """
        Helper methode
    """
    def get_cell_index_(self, point):
        index = point / self.resolution_
        return np.rint(index)


class AdaptivelyVoxelFilter(PointCloudFilter):

    def __init__(self, max_length):
        super()
        self.__max_length = max_length
        self.set_name("voxel_filter")

    def apply(self, point_cloud):
        if point_cloud.count <= self.wished_size:
            return point_cloud

        result = VoxelFilter(self.__max_length).filter(point_cloud)
        if result.count >= self.wished_size:
            return result

        high_length = self.__max_length
        while high_length > 1e-2:
            low_length = high_length / 2.
            result = VoxelFilter(low_length).filter(point_cloud)

            if result.count >= self.wished_size:
                '''
                    Binary search to find the right amount of filtering. 'low_length' gave
                    a sufficiently dense 'result', 'high_length' did not. We stop when the
                    edge length is at most 10% off.
                '''
                while (high_length - low_length) / low_length > 1e-1:
                    mid_length = (low_length + high_length) / 2.
                    point_cloud_candidate = VoxelFilter(mid_length).filter(point_cloud)
                    if point_cloud_candidate.count >= self.wished_size:
                        low_length = mid_length
                        result = point_cloud_candidate
                    else:
                        high_length = mid_length
                return result
            high_length /= 2.

        return point_cloud


def adaptively_voxel_filtered(point_cloud, min_number_of_points, max_length):
    if point_cloud.count <= min_number_of_points:
        return point_cloud

    result = VoxelFilter(max_length).filter(point_cloud)
    if result.count >= min_number_of_points:
        return result

    high_length = max_length
    while high_length > 1e-2:
        low_length = high_length / 2.
        result = VoxelFilter(low_length).filter(point_cloud)

        if result.count >= min_number_of_points:
            '''
                Binary search to find the right amount of filtering. 'low_length' gave
                a sufficiently dense 'result', 'high_length' did not. We stop when the
                edge length is at most 10% off.
            '''
            while (high_length - low_length) / low_length > 1e-1:
                mid_length = (low_length + high_length) / 2.
                point_cloud_candidate = VoxelFilter(mid_length).filter(point_cloud)
                if point_cloud_candidate.count >= min_number_of_points:
                    low_length = mid_length
                    result = point_cloud_candidate
                else:
                    high_length = mid_length
            return result
        high_length /= 2.

    return point_cloud


