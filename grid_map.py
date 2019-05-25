import numpy as np
import math
from shapely.geometry import Point, MultiLineString
import svgpathtools


def load_svg_environment(svg_file, size=10, offset=1):
    paths = svgpathtools.svg2paths(svg_file, return_svg_attributes=False)[0]

    points = []

    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for path in paths:
        for line in path:
            if line.start.real < min_x:
                min_x = line.start.real
            if line.start.imag < min_y:
                min_y = line.start.imag
            if line.start.real > max_x:
                max_x = line.start.real
            if line.start.imag > max_y:
                max_y = line.start.imag

    scale = size - 2*offset
    for path in paths:
        for line in path:
            p_start_x = offset + scale * (line.start.real - min_x) / (max_x - min_x)
            p_start_y = offset + scale * (line.start.imag - min_y) / (max_y - min_y)
            p_start = [p_start_x, p_start_y]
            p_end_x = offset + scale * (line.end.real - min_x) / (max_x - min_x)
            p_end_y = offset + scale * (line.end.imag - min_y) / (max_y - min_y)
            p_end = [p_end_x, p_end_y]
            points += [[p_start, p_end]]
    obstacle = MultiLineString(points)
    return obstacle


class GridMap:
    def __init__(self, size, resolution):
        self.__size = size
        self.__resolution = resolution
        self.__data = []

    def load_environment(self, environment):
        num_cells = int(self.__size / self.__resolution)
        self.__data = np.zeros((num_cells, num_cells))
        for i in range(num_cells):
            for j in range(num_cells):
                self.__data[i, j] = environment.distance(Point(i * self.__resolution, j * self.__resolution)) < self.__resolution

    @property
    def obstacle_indices(self):
        indices = np.where(self.__data == 1)
        return indices[0] * self.__resolution, indices[1] * self.__resolution

    @property
    def data(self):
        return self.__data

    @property
    def size(self):
        return self.__size

    @property
    def resolution(self):
        return self.__resolution

