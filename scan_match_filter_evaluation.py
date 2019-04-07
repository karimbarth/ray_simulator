from tqdm import tqdm
import matplotlib.pyplot as plt
import native.scan_matcher as scan_matcher
import numpy as np

from data_manager import DataManager
from grid_map import GridMap, load_svg_environment
from rangefinder import Rangefinder
from filter.voxel_filter import AdaptivelyVoxelFilter
from filter.simple_filter import RandomFilter
from filter.normal_filter import MaxEntropyNormalAngleFilter
import result_plotter

DEFAULT_MAP = "floorplan_simplified"  # "normal_test.svg"  #"floorplan_simplified.svg"  # "test.svg"
DEFAULT_FILTER = "voxel_filter"


def evaluate_position(grid_map, point_cloud, point_cloud_filter, sample_count=200, evaluation_radius=2):
    filtered_point_cloud = point_cloud_filter.apply(point_cloud)
    true_size = filtered_point_cloud.count
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

    return errors, true_size


def evaluate_point_cloud_sizes(grid_map, point_cloud, point_cloud_filter, data_manager, sample_count=200, min_size=5):
    data_manager.load_data(point_cloud_filter.name, "translation")
    means = dict()
    std = dict()
    data = dict()

    start = point_cloud.count

    for n in tqdm(range(start, min_size-1, -1)):
        point_cloud_filter.set_wished_size(n)
        filtered_point_cloud = point_cloud_filter.apply(point_cloud)
        true_size = filtered_point_cloud.count
        initial_pose = np.array([filtered_point_cloud.lidar_origin[0], filtered_point_cloud.lidar_origin[1], 0])
        collected_data = scan_matcher.evaluate_position_with_samples(filtered_point_cloud.lidar_frame_array,
                                                                     grid_map.data, grid_map.resolution, initial_pose,
                                                                     sample_count, 2.0)
        means[true_size] = np.mean(collected_data[1, :])
        std[true_size] = np.std(collected_data[1, :])
        data[true_size] = collected_data

    data_manager.add_data(data, point_cloud.lidar_origin)
    data_manager.safe_and_close()

    return means, std


def evaluate_filters(grid_map, point_cloud, point_cloud_filters, data_manager, sample_count=200):
    filter_results = dict()
    for pc_filter in point_cloud_filters:
        print("Evaluate " + pc_filter.name)
        mean, std = evaluate_point_cloud_sizes(grid_map, point_cloud, pc_filter, data_manager, sample_count=sample_count)
        filter_results[pc_filter.name] = (mean, std)


def generate_map_data(grid_map, environment):
    sample_count = 100
    cloud_size = 80
    rangefinder_noise = 0.1
    data_manager = DataManager(DEFAULT_MAP, "perfect")
    point_cloud_filters = [MaxEntropyNormalAngleFilter(number_of_bins=20), RandomFilter(),
                           AdaptivelyVoxelFilter(2 * grid_map.size)]
    points = [(1.5, 1.5), (3, 1.5), (4.5, 1.5),
              (1.5, 2.5), (3, 3), (5, 2.4),
              (3, 3), (5, 3.5),
              (3.5, 5),
              (1.5, 7), (3.5, 7),
              (2.5, 8), (4, 8), (5, 8.5),
              (6.5, 8.5), (8, 8)]
    for point in points:
        print("Process point: ", point)
        rangefinder = Rangefinder(cloud_size, range_variance=rangefinder_noise, angular_variance=rangefinder_noise)
        point_cloud = rangefinder.scan(environment, point)
        point_cloud.calc_normals()
        evaluate_filters(grid_map, point_cloud, point_cloud_filters, data_manager, sample_count)


def evaluate_map_data(map_resolution, environment):
    data_manager = DataManager(DEFAULT_MAP, "perfect")
    filter_types = ["random_filter", "voxel_filter", "normal_filter"]
    filter_translation_results = dict()
    filter_orientation_results = dict()
    filter_iter_results = dict()
    point_map = dict()
    for filter_name in filter_types:
        data_manager.load_data(filter_name, "translation")

        trans_mean = data_manager.data.post_translation_means()
        trans_std = data_manager.data.post_translation_std()
        filter_translation_results[filter_name] = (trans_mean, trans_std)

        orientation_mean = data_manager.data.post_rotation_means()
        orientation_std = data_manager.data.post_rotation_std()
        filter_orientation_results[filter_name] = (orientation_mean, orientation_std)

        iter_mean = data_manager.data.iter_means()
        iter_std = data_manager.data.iter_std()
        filter_iter_results[filter_name] = (iter_mean, iter_std)

        point_map[filter_name] = data_manager.data.positions
        data_manager.readonly_close()

    result_plotter.plot_filter_statistics_translation(filter_translation_results, map_resolution)
    result_plotter.plot_filter_statistics_orientation(filter_orientation_results, map_resolution)
    result_plotter.plot_filter_statistics_iter(filter_iter_results, map_resolution)
    result_plotter.plot_points_on_map(environment, point_map)


def evaluate_radius_of_convergence(grid_map, point_cloud, sample_count=200):
    random_filter = RandomFilter()
    errors, true_size = evaluate_position(grid_map, point_cloud, random_filter,
                               sample_count=sample_count, evaluation_radius=10)

    result_plotter.plot_radius_of_convergence(errors, grid_map.resolution, point_cloud.count)


def voxel_filter_visualization(environment, point_cloud, map_size, min_number_of_points=30):
    point_cloud.calc_normals()
    voxel_filter = MaxEntropyNormalAngleFilter(20)
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
    point_cloud.calc_normals()
    result_plotter.plot_normals(environment, point_cloud)


def evaluate():
    # init params

    sensor_origin = (2, 2) #(5, 5)#(4, 4)#(4, 8)
    map_resolution = 0.1
    cloud_size = 80 #25 #80  # 80  # 5 - 80 const
    map_size = 10
    sample_count = 200
    # end init

    environment = load_svg_environment("./geometries/" + DEFAULT_MAP + ".svg", map_size)
    grid_map = GridMap(map_size, map_resolution)
    grid_map.load_environment(environment)

    #test_function(point_cloud, grid_map)
    #generate_map_data(grid_map, environment)
    evaluate_map_data(map_resolution, environment)
    #evaluate_radius_of_convergence(grid_map, point_cloud, sample_count=200)
    #voxel_filter_visualization(environment, point_cloud, map_size)
    #evaluate_voxel_filter(grid_map, point_cloud, sample_count)
    #scan_matching_visualization(grid_map, point_cloud)
    #normal_filter_visualization(environment, point_cloud)


if __name__ == '__main__':
    evaluate()
    plt.show()
