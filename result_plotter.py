import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_radius_of_convergence(error, grid_map, number_of_points):
    plt.figure()
    plt.title(
        'Radius of convergence (PointCloud size = ' + str(number_of_points) + ')\n' + r'Sensor origin: $o$, Initial pose $x_0$, Estimate $\xi_{xy}$, map resolution r=' + str(
            grid_map.resolution))

    plt.scatter(error[:, 0], error[:, 1], marker='x', c='blue', label="ground truth")
    plt.plot([2 * grid_map.resolution, 2 * grid_map.resolution], [0, 1.0], c='red', linestyle=':')
    plt.plot([0, 1], [0.2 * grid_map.resolution, 0.2 * grid_map.resolution], c='green', linestyle=':')
    plt.text(2 * grid_map.resolution + 0.01, 0.9, r'$2r$', color='r')
    plt.text(1.02, 0.2 * grid_map.resolution, r'$0.2r$', color='g')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel(r"$ \Vert o - x_0 \Vert_2 $")
    plt.ylabel(r"$ \Vert o - \xi_{xy} \Vert_2 $")


def plot_scan(environment, point_cloud, step):
    sensor_origin = point_cloud.lidar_origin
    plt.figure()
    # plot environment
    for e in environment:
        label = "environment" if "environment" not in plt.gca().get_legend_handles_labels()[1] else ""
        environment_x, environment_y = e.xy
        plt.plot(environment_x, environment_y, color='black', linewidth=2, label=label)

    # plot rays from filtered pc
    hits = point_cloud.map_frame_list
    for hit in hits:
        label = "hit filtered point cloud" if "hit filtered point cloud" not in plt.gca().get_legend_handles_labels()[
            1] else ""
        hit_ray = np.transpose(np.array([hit, sensor_origin]))
        plt.plot(hit_ray[0], hit_ray[1], color='blue', linewidth=1, linestyle=':', label=label)

    # plot hits of original pc
    point_cloud_points = point_cloud.map_frame_array
    plt.scatter(point_cloud_points[0], point_cloud_points[1], s=np.array([100, 100]), marker='x', c='red',
                label="original point cloud")

    # plot sensor
    plt.scatter([sensor_origin[0]], [sensor_origin[1]], s=np.array([100, 100]), marker='x', c='blue', label="sensor")
    plt.axis('equal')
    plt.axis([0, 7, 0, 7])
    if step == 2:
        for hit in hits:
            print(hit)
            if hit[0] < 1.4:
                plt.arrow(hit[0], hit[1], -0.75, 0, head_width=0.1, head_length=0.15,
                          length_includes_head=True, color="green")
            elif hit[0] < 5.5:
                plt.arrow(hit[0], hit[1], 0, -0.75, head_width=0.1, head_length=0.15,
                          length_includes_head=True, color="green")
            else:

                plt.arrow(hit[0], hit[1], 0.67271, 0.3316, head_width=0.1, head_length=0.15,
                          length_includes_head=True, color="green")


def plot_filtered_point_cloud(environment, point_cloud, filtered_point_cloud, filter_name):
    sensor_origin = point_cloud.lidar_origin

    plt.figure()
    plt.title(filter_name)

    # plot environment
    for e in environment:
        label = "environment" if "environment" not in plt.gca().get_legend_handles_labels()[1] else ""
        environment_x, environment_y = e.xy
        plt.plot(environment_x, environment_y, color='black', linewidth=2, label=label)

    # plot rays from filtered pc
    hits = filtered_point_cloud.map_frame_list
    for hit in hits:
        label = "hit filtered point cloud" if "hit filtered point cloud" not in plt.gca().get_legend_handles_labels()[1] else ""
        hit_ray = np.transpose(np.array([hit, sensor_origin]))
        plt.plot(hit_ray[0], hit_ray[1], color='blue', linewidth=1, linestyle=':', label=label)

    # plot filtered point cloud
    filtered_point_cloud_points = filtered_point_cloud.map_frame_array
    plt.scatter(filtered_point_cloud_points[0], filtered_point_cloud_points[1], s=np.array([100, 100]), marker='o',
                edgecolors='blue', facecolors='none', label="filtered point cloud")

    # plot hits of original pc
    point_cloud_points = point_cloud.map_frame_array
    plt.scatter(point_cloud_points[0], point_cloud_points[1], s=np.array([100, 100]), marker='x', c='red',
                label="original point cloud")

    # plot sensor
    plt.scatter([sensor_origin[0]], [sensor_origin[1]], s=np.array([100, 100]), marker='x', c='blue', label="sensor")

    plt.legend(loc='lower right')


def plot_filter_statistics(mean, std, grid_map, filter_type):
    plt.figure()
    plt.title(filter_type + ' with different resolutions')
    X = np.array(list(mean.keys()))
    Y = np.array(list(map(lambda m: mean[m], X)))
    Std = np.array(list(map(lambda m: std[m], X)))
    plt.plot(X, Y, color='red', linewidth=2, label="mean error")
    plt.fill_between(X, Y - Std, Y + Std, facecolor='red', alpha=0.4, label="standard derivation")
    plt.plot([5, 80], [0.2 * grid_map.resolution, 0.2 * grid_map.resolution], c='green', linestyle=':')
    plt.text(81, 0.2 * grid_map.resolution, r'$0.2r$', color='g')
    plt.ylim(0, 0.15)
    plt.xlim(5, 80)
    plt.xlabel("Point cloud size")
    plt.ylabel(r"$ \Vert o - \xi_{xy} \Vert_2 $")
    plt.legend(loc='upper right')


def plot_scan_matching(grid_map, point_cloud, estimate, step=3):
    fig = plt.figure()
    map_size = grid_map.size
    index_array = grid_map.obstacle_indices
    plt.scatter(index_array[0], index_array[1], s=np.array([80, 80]), marker='x',
                c='black', label="map obstacle")

    plt.scatter([point_cloud.lidar_origin[0]], [point_cloud.lidar_origin[1]], s=np.array([60, 60]), marker='o', c='green',
                label="ground truth")

    plt.scatter([estimate[0]], [estimate[1]], s=np.array([60, 60]), marker='o', c='red',
                label="estimate")

    # plot grid
    ax = fig.axes[0]
    major_ticks = np.arange(0, map_size, 1)
    minor_ticks = np.arange(0, map_size, grid_map.resolution)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    plt.ylim(0, 10)
    plt.xlim(0, 10)

    if step >= 2:
        estimate_transformed_point_cloud = point_cloud.transform_from_lidar_frame_to(*estimate)
        plt.scatter(estimate_transformed_point_cloud[0], estimate_transformed_point_cloud[1], s=np.array([100, 100]),
                    marker='x', c='red',
                    label="estimated point cloud")
        if step >= 3:
            # plot transformation arrows
            plt.arrow(estimate[0], estimate[1], point_cloud.lidar_origin[0]-estimate[0],
                      point_cloud.lidar_origin[1]-estimate[1], head_width=0.2, head_length=0.3,
                      length_includes_head=True, color="blue")

            plt.arrow(estimate[0], estimate[1], point_cloud.lidar_origin[0] - estimate[0], 0, length_includes_head=True,
                      color="blue", linestyle=":")
            plt.text((estimate[0] + point_cloud.lidar_origin[0]) / 2, estimate[1] - 0.3,
                     r'$\Delta\xi_x$', color='b', fontsize=20)
            plt.arrow(estimate[0], estimate[1], 0, point_cloud.lidar_origin[1] - estimate[1], length_includes_head=True,
                      color="blue", linestyle=":")
            plt.text(estimate[0] + 0.05, (estimate[1] + point_cloud.lidar_origin[1])/2,
                     r'$\Delta\xi_y$', color='b', fontsize=20)
            # plot point cloud arrow
            for n in range(point_cloud.count):
                plt.arrow(estimate_transformed_point_cloud[0][n], estimate_transformed_point_cloud[1][n],
                          point_cloud.lidar_origin[0] - estimate[0], point_cloud.lidar_origin[1]-estimate[1],
                          head_width=0.1, head_length=0.15, length_includes_head=True, color="blue", linestyle=":")

'''
def plot_scan(environment, sensor_origin, estimates, point_cloud, title):
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
    # plt.scatter([estimates[0][0]], [estimates[0][1]], s=np.array([100, 100]), marker='x', c='red', label="estimate")
    plt.legend(loc='lower right')
'''

def plot_cost_function(cost_function, x_range, y_range, resolution, point1, point2):
    x = np.arange(x_range[0], x_range[1], resolution)
    y = np.arange(y_range[0], y_range[1], resolution)

    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(cost_function)(X, Y)

    fig = plt.figure()
    fig.suptitle(r'Cost function with $\theta = 0$')
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.RdBu, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(plt.LinearLocator(10))
    ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))

    ax.set_xlabel('x-position')
    ax.set_ylabel('y-position')
    ax.set_zlabel('cost function')
    ax.view_init(elev=25, azim=-120)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Plot 2D
    fig = plt.figure()
    im = plt.imshow(Z, cmap=plt.cm.RdBu)
    cset = plt.contour(Z, [cost_function(point1[0], point1[1]), cost_function(point2[0], point2[1])],
                       linewidths=2, cmap=plt.cm.Set2)
    plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    plt.colorbar(im)
