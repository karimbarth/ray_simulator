import svgpathtools
import math
import numpy as np
from shapely.geometry import MultiLineString, Point
import matplotlib.pyplot as plt

from rangefinder import Rangefinder
import nativ.scan_matcher as scan_matcher

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


def create_grid_map(environment, size, resolution):
    num_cells = int(size/resolution)
    grid_map = np.zeros((num_cells, num_cells))

    for i in range(num_cells):
        for j in range(num_cells):
            grid_map[i, j] = environment.distance(Point(i*resolution, j*resolution)) < resolution

    return grid_map


def plot_scan(sensor_origin, estimate, hits, environment, title):
    plt.figure()
    plt.title(title)
    for e in environment:
        label = "environment" if "environment" not in plt.gca().get_legend_handles_labels()[1] else ""
        environment_x, environment_y = e.xy
        plt.plot(environment_x, environment_y, color='black', linewidth=2, label=label)

    plt.plot(environment_x, environment_y, color='black', linewidth=1)
    for i in range(hits.shape[1]):
        hit = hits[:, i]
        label = "hit" if "hit" not in plt.gca().get_legend_handles_labels()[1] else ""
        hit_ray = np.transpose(np.array([hit, sensor_origin]))
        plt.plot(hit_ray[0], hit_ray[1], color='green', linewidth=1, linestyle=':', label=label)

    plt.scatter([sensor_origin[0]], [sensor_origin[1]], s=np.array([100, 100]), marker='x', c='blue', label="ground "
                                                                                                            "truth")
    plt.scatter([estimate[0]], [estimate[1]], s=np.array([100, 100]), marker='x', c='red', label="estimate")
    plt.legend(loc='lower right')


def plot_grid_map(grid_map, map_resolution):
    fig = plt.figure()
    map_size = grid_map.shape[0] * map_resolution
    plt.title("Grid Map")
    index_array = np.where(grid_map == 1)
    plt.scatter(index_array[0] * map_resolution, index_array[1] * map_resolution,
               s=np.array([120 * map_resolution, 120 * map_resolution]), marker='x', c='black')

    ax = fig.axes[0]
    major_ticks = np.arange(0, map_size, 1)
    minor_ticks = np.arange(0, map_size, map_resolution)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)



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
    # use 2d numpy array instead list of 1d numpy arrays
    map_coords = np.stack(rangefinder.scan(obstacle, sensor_origin), axis=1)
    # transform point cloud in lidar coordinate frame (currently without rotation)
    sensor_coords = map_coords - np.array([[sensor_origin[0]], [sensor_origin[1]]])
    return map_coords, sensor_coords


def evaluate():
    map_name, filter_name = input_user_selection()
    sensor_origin = (4, 3)
    initial_pose = np.array([4, 3, 0]);
    map_size = 10
    map_resolution = 0.25

    environment = svg_to_environment("./geometries/" + map_name, map_size)
    grid_map = create_grid_map(environment, map_size, map_resolution)

    hits, hits_array = make_scan(sensor_origin, environment)

    x, y, theta = scan_matcher.match(hits_array, grid_map, map_resolution, initial_pose)
    print("Result: ", x, ", ", y, ", ", theta);

    plot_scan(sensor_origin, (x, y), hits, environment, "No Filter")

    plot_grid_map(grid_map, map_resolution)


if __name__ == '__main__':
    evaluate()
    plt.show()





