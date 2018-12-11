import numpy as np
from shapely.geometry import Point


class GridMap:
    def __init__(self, size, resolution):
        self.size_ = size
        self.resolution_ = resolution
        self.data_ = []

    def load_environment(self, environment):
        num_cells = int(self.size_ / self.resolution_)
        self.data_ = np.zeros((num_cells, num_cells))
        for i in range(num_cells):
            for j in range(num_cells):
                self.data_[i, j] = environment.distance(Point(i * self.resolution_, j * self.resolution_)) < self.resolution_

    @property
    def obstacle_indices(self):
        indices = np.where(self.data_ == 1)
        return indices[0] * self.resolution_, indices[1] * self.resolution_

    @property
    def data(self):
        return self.data_

    @property
    def size(self):
        return self.size_

    @property
    def resolution(self):
        return self.resolution_

