import math

import matplotlib.pyplot as plt
import nativ.scan_matcher as scan_matcher
import numpy as np
import svgpathtools
from shapely.geometry import MultiLineString

from grid_map import GridMap
from rangefinder import Rangefinder

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


def plot_scan(environment, sensor_origin, estimate, point_cloud, title):
    plt.figure()
    plt.title(title)
    for e in environment:
        label = "environment" if "environment" not in plt.gca().get_legend_handles_labels()[1] else ""
        environment_x, environment_y = e.xy
        plt.plot(environment_x, environment_y, color='black', linewidth=2, label=label)

    #plt.plot(environment_x, environment_y, color='black', linewidth=1)
    hits = point_cloud.map_frame_list
    for hit in hits:
        label = "hit" if "hit" not in plt.gca().get_legend_handles_labels()[1] else ""
        hit_ray = np.transpose(np.array([hit, sensor_origin]))
        plt.plot(hit_ray[0], hit_ray[1], color='green', linewidth=1, linestyle=':', label=label)

    plt.scatter([sensor_origin[0]], [sensor_origin[1]], s=np.array([100, 100]), marker='x', c='blue', label="ground "
                                                                                                            "truth")
    plt.scatter([estimate[0]], [estimate[1]], s=np.array([100, 100]), marker='x', c='red', label="estimate")
    plt.legend(loc='lower right')


def plot_grid_map(grid_map, sensor_origin, estimate, point_cloud):
    fig = plt.figure()
    map_size = grid_map.size
    plt.title("Grid Map")
    index_array = grid_map.obstacle_indices
    plt.scatter(index_array[0], index_array[1], s=grid_map.resolution * np.array([120, 120]), marker='x',
                c='black', label="map obstacle")

    plt.scatter([sensor_origin[0]], [sensor_origin[1]], s=np.array([100, 100]), marker='o', c='green',
                label="sensor origin")

    ground_truth_point_cloud = point_cloud.map_frame_array
    plt.scatter(ground_truth_point_cloud[0], ground_truth_point_cloud[1], s=np.array([120, 120]), marker='x', c='green',
                label="original point cloud")

    plt.scatter([estimate[0]], [estimate[1]], s=np.array([100, 100]), marker='o', c='red',
                label="estimate")

    estimate_transformed_point_cloud = point_cloud.transform_from_lidar_frame_to(*estimate)
    plt.scatter(estimate_transformed_point_cloud[0], estimate_transformed_point_cloud[1], s=np.array([120, 120]), marker='x', c='red',
                label="estimated point cloud")

    ax = fig.axes[0]
    major_ticks = np.arange(0, map_size, 1)
    minor_ticks = np.arange(0, map_size, grid_map.resolution)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    plt.legend(loc='lower right')


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
    print("Result: ", estimate[0], ", ", estimate[1], ", ", estimate[2])

    plot_scan(environment, sensor_origin, estimate, point_cloud, "No Filter")

    plot_grid_map(grid_map, sensor_origin, estimate, point_cloud)


if __name__ == '__main__':
    evaluate()
    plt.show()
