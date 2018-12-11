import numpy as np
import matplotlib.pyplot as plt


def plot_scan(environment, sensor_origin, estimate, point_cloud, title):
    plt.figure()
    plt.title(title)
    for e in environment:
        label = "environment" if "environment" not in plt.gca().get_legend_handles_labels()[1] else ""
        environment_x, environment_y = e.xy
        plt.plot(environment_x, environment_y, color='black', linewidth=2, label=label)

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