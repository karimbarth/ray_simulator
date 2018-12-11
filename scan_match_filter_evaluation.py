import math

import matplotlib.pyplot as plt
import nativ.scan_matcher as scan_matcher
import numpy as np
import svgpathtools
from shapely.geometry import MultiLineString

from grid_map import GridMap
from rangefinder import Rangefinder
import result_plotter

DEFAULT_MAP = "floorplan_simplified.svg"
DEFAULT_FILTER = "voxel_filter"


def svg_to_environment(svg_file, size=10):
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

    i = 0
    offset = 1
    scale = size - 2
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


def input_user_selection():
    map_name = input("Please enter the name of the world map (default .....): ")
    if not map_name:
        map_name = DEFAULT_MAP

    filter_name = input("Please select the filter (default voxel_filter):")

    if not filter_name:
        filter_name = DEFAULT_FILTER

    return map_name, filter_name


def evaluate():
    map_name, filter_name = input_user_selection()
    sensor_origin = (4, 3)
    initial_pose = np.array([0, 0, 0])
    map_size = 10
    map_resolution = 0.1

    print("Estimate with initial pose: ", initial_pose)

    environment = svg_to_environment("./geometries/" + map_name, map_size)
    grid_map = GridMap(map_size, map_resolution)
    grid_map.load_environment(environment)

    rangefinder = Rangefinder()
    point_cloud = rangefinder.scan(environment, sensor_origin)

    estimate = scan_matcher.match(point_cloud.lidar_frame_array, grid_map.data, grid_map.resolution, initial_pose)
    print("Result: ", estimate)

    result_plotter.plot_scan(environment, sensor_origin, estimate, point_cloud, "No Filter")
    result_plotter.plot_grid_map(grid_map, sensor_origin, estimate, point_cloud)


if __name__ == '__main__':
    evaluate()
    plt.show()
