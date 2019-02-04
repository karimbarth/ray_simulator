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

DEFAULT_MAP = "floorplan_simplified.svg"  # "normal_test.svg"  #"floorplan_simplified.svg"  # "test.svg"
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


def evaluate_point_cloud_sizes(grid_map, point_cloud, point_cloud_filter, data_manager, sample_count=200, min_size=5):
    data_manager.load_data(point_cloud_filter.name, "translation")
    means = dict()
    std = dict()
    data = dict()

    start = point_cloud.count

    for n in tqdm(range(start, min_size-1, -1)):
        point_cloud_filter.set_wished_size(n)
        errors = evaluate_position(grid_map, point_cloud, point_cloud_filter, sample_count=sample_count)
        means[n] = np.mean(errors[:, 1])
        std[n] = np.std(errors[:, 1])
        data[n] = errors

    data_manager.add_data(data, point_cloud.lidar_origin)
    data_manager.safe_and_close()

    return means, std


def evaluate_filters(grid_map, point_cloud, point_cloud_filters, data_manager, sample_count=200):
    filter_results = dict()
    for pc_filter in point_cloud_filters:
        print("Evaluate " + pc_filter.name)
        mean, std = evaluate_point_cloud_sizes(grid_map, point_cloud, pc_filter, data_manager, sample_count=sample_count)
        filter_results[pc_filter.name] = (mean, std)

    result_plotter.plot_filter_statistics(filter_results, grid_map.resolution)


def generate_map_data(grid_map, point_cloud):
    sample_count = 200
    point_cloud.calc_normals()
    data_manager = DataManager("floorplan_simplified", "perfect")
    point_cloud_filters = [MaxEntropyNormalAngleFilter(number_of_bins=20), RandomFilter(), AdaptivelyVoxelFilter(2 * grid_map.size)]
    evaluate_filters(grid_map, point_cloud, point_cloud_filters, data_manager, sample_count)


def evaluate_map_data(map_resolution, environment):
    data_manager = DataManager("floorplan_simplified", "perfect")
    filter_types = ["random_filter", "voxel_filter", "max_entropy_normal_filter"]
    filter_results = dict()
    point_map = dict()
    for filter_name in filter_types:
        data_manager.load_data(filter_name, "translation")
        mean = data_manager.data.post_means()
        std = data_manager.data.post_std()
        filter_results[filter_name] = (mean, std)
        point_map[filter_name] = data_manager.data.positions
        data_manager.readonly_close()

    result_plotter.plot_filter_statistics(filter_results, map_resolution)
    result_plotter.plot_points_on_map(environment, point_map)


def evaluate_radius_of_convergence(grid_map, point_cloud, sample_count=200):
    random_filter = RandomFilter()
    errors = evaluate_position(grid_map, point_cloud, random_filter,
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

    sensor_origin = (1.5, 7.) #(5, 5)#(4, 4)#(4, 8)
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

    #generate_map_data(grid_map, point_cloud)
    evaluate_map_data(map_resolution, environment)
    #evaluate_radius_of_convergence(grid_map, point_cloud, sample_count=200)
    #voxel_filter_visualization(environment, point_cloud, map_size)
    #evaluate_voxel_filter(grid_map, point_cloud, sample_count)
    #scan_matching_visualization(grid_map, point_cloud)
    #normal_filter_visualization(environment, point_cloud)



if __name__ == '__main__':
    evaluate()
    plt.show()
