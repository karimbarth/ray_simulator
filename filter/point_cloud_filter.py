import abc


class PointCloudFilter:

    def __init__(self):
        self.__name = None
        self.__wished_size = None

    def set_name(self, name):
        self.__name = name

    def set_wished_size(self, size):
        """
        set the wished resulting number of points after applying the filter
        Note:  number_of_points is only a lower bound, some filter can't archive an exact count!!!
        :param size: the wished size
        :return:
        """
        self.__wished_size = size

    @property
    def wished_size(self):
        """
        get the wished resulting point cloud size
        :return: the size
        """
        return self.__wished_size

    @property
    def name(self):
        """
        get the name of the current filter
        :return: the name of the filter
        """
        return self.__name

    @abc.abstractmethod
    def apply(self, point_cloud):
        """Apply the filter. Note:  number_of_points is only a lower bound, some filter can't archive an exact count!!!
            :param point_cloud: the original point cloud
            :return: the filtered point cloud
        """
        return
