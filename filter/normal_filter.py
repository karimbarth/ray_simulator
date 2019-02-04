import numpy as np
from point_cloud import PointCloud

from filter.point_cloud_filter import PointCloudFilter


class MaxEntropyNormalAngleFilter(PointCloudFilter):

    def __init__(self, number_of_bins):
        super()
        self.__number_of_bins = number_of_bins
        self.set_name("max_entropy_normal_filter")

    def apply(self, point_cloud):
        bin_size = 2 * np.pi / self.__number_of_bins
        histogram = [list() for _ in range(self.__number_of_bins)]
        result = PointCloud(point_cloud.lidar_origin, point_cloud.lidar_orientation)
        list_of_points = point_cloud.map_frame_list

        # generate histogram
        for index in range(len(list_of_points)):
            angle = point_cloud.get_descriptor("normals_direction", index)
            bin_index = int(np.floor(angle/bin_size))
            histogram[bin_index].append(index)

        # remove points of the largest remaining bin to maximize entropy of resulting normal angle distribution
        for _ in range(point_cloud.count - self.wished_size):
            lengths = list(map(lambda histogram_bin: len(histogram_bin), histogram))
            index = lengths.index(max(lengths))
            histogram[index].pop(0)

        # create result point cloud
        for histogram_bin in histogram:
            for index in histogram_bin:
                result.add_point([list_of_points[index]])
                result.add_descriptor("normals", point_cloud.get_descriptor("normals", index))
                result.add_descriptor("normals_direction", point_cloud.get_descriptor("normals_direction", index))

        return result





