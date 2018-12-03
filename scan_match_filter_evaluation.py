import svgpathtools
import math
import numpy as np
from shapely.geometry import MultiLineString
import matplotlib.pyplot as plt

from rangefinder import Rangefinder
import scan_matcher

DEFAULT_MAP = "floorplan_simplified.svg"
DEFAULT_FILTER = "voxel_filter"


def svg_to_environment(svg_file):
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
    scale = 8
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


def plotScan(sensor_origin, estimate, hits, environment, title):
    plt.figure()
    plt.title(title)
    for e in environment:
        label = "environment" if "environment" not in plt.gca().get_legend_handles_labels()[1] else ""
        environment_x, environment_y = e.xy
        plt.plot(environment_x, environment_y, color='black', linewidth=2, label=label)

    plt.plot(environment_x, environment_y, color='black', linewidth=1)
    for hit in hits:
        label = "hit" if "hit" not in plt.gca().get_legend_handles_labels()[1] else ""
        hit_ray = np.transpose(np.array([hit, sensor_origin]))
        plt.plot(hit_ray[0], hit_ray[1], color='green', linewidth=1, linestyle=':', label=label)

    plt.scatter([sensor_origin[0]], [sensor_origin[1]], s=np.array([100, 100]), marker='x', c='blue', label="ground "
                                                                                                            "truth")
    plt.scatter([estimate[0]], [estimate[1]], s=np.array([100, 100]), marker='x', c='red', label="estimate")
    plt.legend(loc='lower right')


def input_user_selection():
    map_name = input("Please enter the name of the world map (default .....): ")
    if not map_name:
        map_name = DEFAULT_MAP

    filter_name = input("Please select the filter (default voxel_filter):")

    if not filter_name:
        filter_name = DEFAULT_FILTER

    return map_name, filter_name


def make_scan(sensor_origin, obstacle):
    rangefinder = Rangefinder()
    return rangefinder.scan(obstacle, sensor_origin)


if __name__ == '__main__':

    map_name, filter_name = input_user_selection()

    # TODO generate Gridmap
    environment = svg_to_environment("./geometries/" + map_name)

    sensor_origin = (4, 3)
    hits = make_scan(sensor_origin, environment)
    plotScan(sensor_origin, (1, 5), hits, environment, "Test")




