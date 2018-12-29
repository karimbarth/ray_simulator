import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_radius_of_convergence(error, grid_map):
    plt.figure()
    plt.title(
        'Radius of convergence\n' + r'Sensor origin: $o$, Initial pose $x_0$, Estimate $\xi_{xy}$, map resolution r=' + str(
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


def plot_grid_map(grid_map, sensor_origin, estimates, point_cloud):
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
    '''
    plt.scatter([estimates[0][0]], [estimates[0][1]], s=np.array([100, 100]), marker='o', c='red',
                label="estimate")

    
    estimate_transformed_point_cloud = point_cloud.transform_from_lidar_frame_to(*estimates[0])
    plt.scatter(estimate_transformed_point_cloud[0], estimate_transformed_point_cloud[1], s=np.array([120, 120]),
                marker='x', c='red',
                label="estimated point cloud")
    '''
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
