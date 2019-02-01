import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import native.scan_matcher as scan_matcher
import numpy as np


from grid_map import GridMap, load_svg_environment
from rangefinder import Rangefinder
from filter.voxel_filter import AdaptivelyVoxelFilter
from filter.simple_filter import RandomFilter
import result_plotter

DEFAULT_MAP = "floorplan_simplified.svg"  # "test.svg"
DEFAULT_FILTER = "voxel_filter"


def evaluate_position(grid_map, point_cloud, point_cloud_filter, sample_count=200, evaluation_radius=2):
    filtered_point_cloud = point_cloud_filter.apply(point_cloud)
    sensor_origin = filtered_point_cloud.lidar_origin
    map_resolution = grid_map.resolution
    errors = np.zeros((sample_count, 2))  # init abs translation error, result abs translation error
    estimates = []
    for n in range(sample_count):
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(0, evaluation_radius * map_resolution)
        errors[n, 0] = length
        x = np.cos(angle) * length + sensor_origin[0]
        y = np.sin(angle) * length + sensor_origin[1]
        initial_pose = np.array([x, y, 0])  # for visualization reasons, set angular to 0

        estimate = scan_matcher.match(filtered_point_cloud.lidar_frame_array, grid_map.data, grid_map.resolution, initial_pose,
                                      False)
        estimates += estimate
        errors[n, 1] = np.linalg.norm(np.asarray(sensor_origin) - np.array([estimate[0], estimate[1]]))

    return errors


def evaluate_point_cloud_sizes(grid_map, point_cloud, point_cloud_filter, sample_count=200, min_size=5):
    path = "data/evaluate_" + point_cloud_filter.name + "_data_test.npy"

    means = dict()
    std = dict()
    data = dict()

    if os.path.isfile(path):
        data = np.load(path).item()
        for key in data.keys():
            post_opti_error = (data[key])[:, 1]
            means[key] = np.mean(post_opti_error)
            std[key] = np.std(post_opti_error)
    else:
        start = point_cloud.count

        for n in tqdm(range(start, min_size-1, -1)):
            point_cloud_filter.set_wished_size(n)
            errors = evaluate_position(grid_map, point_cloud, point_cloud_filter, sample_count=sample_count)
            means[n] = np.mean(errors[:, 1])
            std[n] = np.std(errors[:, 1])
            data[n] = errors

        np.save(path, data)

    return means, std


def evaluate_filters(grid_map, point_cloud, point_cloud_filters, sample_count=200):
    filter_results = dict()
    for pc_filter in point_cloud_filters:
        print("Evaluate " + pc_filter.name)
        mean, std = evaluate_point_cloud_sizes(grid_map, point_cloud, pc_filter, sample_count=sample_count)
        filter_results[pc_filter.name] = (mean, std)

    result_plotter.plot_filter_statistics(filter_results, grid_map.resolution)


def evaluate_map(grid_map, point_cloud):
    sample_count = 200
    point_cloud_filters = [RandomFilter(), AdaptivelyVoxelFilter(2 * grid_map.size)]
    evaluate_filters(grid_map, point_cloud, point_cloud_filters, sample_count)


def evaluate_radius_of_convergence(grid_map, point_cloud, sample_count=200):
    random_filter = RandomFilter()
    errors = evaluate_position(grid_map, point_cloud, random_filter,
                               sample_count=sample_count, evaluation_radius=10)

    result_plotter.plot_radius_of_convergence(errors, grid_map.resolution, point_cloud.count)


def voxel_filter_visualization(environment, point_cloud, map_size, min_number_of_points=10):
    voxel_filter = AdaptivelyVoxelFilter(2*map_size)
    voxel_filter.set_wished_size(min_number_of_points)

    filtered_pc = voxel_filter.apply(point_cloud)
    result_plotter.plot_filtered_point_cloud(environment, point_cloud, filtered_pc, "Voxel filter (original size: " +
                                             str(point_cloud.count) + ", filtered size: "
                                             + str(filtered_pc.count) + ")")


def scan_matching_visualization(grid_map, point_cloud):
    voxel_filter = AdaptivelyVoxelFilter(2 * grid_map.size)
    voxel_filter.set_wished_size(25)
    filtered_pc = voxel_filter.apply(point_cloud)
    result_plotter.plot_scan_matching(grid_map, point_cloud=filtered_pc, estimate=[4., 7., 0.], step=1)
    result_plotter.plot_scan_matching(grid_map, point_cloud=filtered_pc, estimate=[4., 7., 0.], step=2)
    result_plotter.plot_scan_matching(grid_map, point_cloud=filtered_pc, estimate=[4., 7., 0.], step=3)


def normal_filter_visualization(environment, point_cloud):
    result_plotter.plot_normal_filter_visualization(environment, point_cloud, step=1)
    result_plotter.plot_normal_filter_visualization(environment, point_cloud, step=2)


def evaluate():
    # init params

    sensor_origin = (4, 8)
    map_resolution = 0.1
    cloud_size = 80 #25 #80  # 80  # 5 - 80 const
    map_size = 10
    sample_count = 200
    # end init

    environment = load_svg_environment("./geometries/" + DEFAULT_MAP, map_size)
    grid_map = GridMap(map_size, map_resolution)
    grid_map.load_environment(environment)

    rangefinder = Rangefinder(cloud_size, range_variance=0.012)
    point_cloud = rangefinder.scan(environment, sensor_origin)

    evaluate_map(grid_map, point_cloud)
    #evaluate_radius_of_convergence(grid_map, point_cloud, sample_count=200)
    #voxel_filter_visualization(environment, point_cloud, map_size)
    #evaluate_voxel_filter(grid_map, point_cloud, sample_count)
    #scan_matching_visualization(grid_map, point_cloud)
    #normal_filter_visualization(environment, point_cloud)
    '''
    errors = np.zeros((sample_count, 2))  # init abs translation error, result abs translation error
    estimates = []
    for n in tqdm(range(sample_count)):
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(0, 10 * map_resolution)
        errors[n, 0] = length
        x = np.cos(angle) * length + sensor_origin[0]
        y = np.sin(angle) * length + sensor_origin[1]
        initial_pose = np.array([x, y, 0])  # for visualization reasons, set angular to 0

        estimate = scan_matcher.match(point_cloud.lidar_frame_array, grid_map.data, grid_map.resolution, initial_pose,
                                      False)
        estimates += estimate
        errors[n, 1] = np.linalg.norm(np.asarray(sensor_origin) - np.array([estimate[0], estimate[1]]))

    # plot result
    result_plotter.plot_radius_of_convergence(errors, grid_map, cloud_size)
    result_plotter.plot_scan(environment, sensor_origin, estimates, point_cloud, "No Filter")
    result_plotter.plot_grid_map(grid_map, sensor_origin, estimates, point_cloud)

    result_plotter.plot_cost_function(
        lambda x, y: scan_matcher.evaluate_cost_function(point_cloud.lidar_frame_array, grid_map.data,
                                                         grid_map.resolution, np.array([x, y, 0])),
        (sensor_origin[0] - 2 * map_resolution, sensor_origin[0] + 2 * map_resolution),
        (sensor_origin[1] - 2 * map_resolution, sensor_origin[1] + 2 * map_resolution),
        map_resolution * 0.1,
        (sensor_origin[0] + map_resolution, sensor_origin[1] + map_resolution),
        (sensor_origin[0] + 2 * map_resolution, sensor_origin[1] + 2 * map_resolution)
    )
    '''


if __name__ == '__main__':
    evaluate()
    plt.show()
