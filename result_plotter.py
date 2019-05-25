import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_radius_of_convergence(error, resolution, number_of_points):
    """ Plots the translation errors at initialization of the optimization
        and after it. visualize the radius of convergence

        error: (nx2) shape numpy array - for n samples the translation error at initialization ([:0]) and after
        optimization ([0:1])

        resolution: the resolution of the grid map

        number_of_points: The number of points used for estimating the error data

        Note: Its not a scientific proof!
    """

    plt.figure()
    plt.title(
        'Radius of convergence (PointCloud size = ' + str(
            number_of_points) + ')\n' + r'Sensor origin: $o$, Initial pose $x_0$, Estimate $\xi_{xy}$, map resolution r=' + str(
            resolution))

    plt.scatter(error[:, 0], error[:, 1], marker='x', c='blue', label="ground truth")
    plt.plot([2 * resolution, 2 * resolution], [0, 1.0], c='red', linestyle=':')
    plt.plot([0, 1], [0.2 * resolution, 0.2 * resolution], c='green', linestyle=':')
    plt.text(2 * resolution + 0.01, 0.9, r'$2r$', color='r')
    plt.text(1.02, 0.2 * resolution, r'$0.2r$', color='g')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel(r"$ \Vert o - x_0 \Vert_2 $")
    plt.ylabel(r"$ \Vert o - \xi_{xy} \Vert_2 $")


def plot_filtered_point_cloud(environment, point_cloud, filtered_point_cloud, filter_name):
    """ Plot the environment and both point clouds to represents the different between them

            environment: the svg map of the scenario

            point_cloud: the original point cloud

            filtered_point_cloud: the filtered point cloud

            filter_name: the name of the filter which was used.

            Note: Use this method only for small point clouds to prevent confusing
    """

    sensor_origin = point_cloud.lidar_origin

    plt.figure()
    plt.title(filter_name)

    __plot_environment(environment)

    # plot rays from filtered pc
    hits = filtered_point_cloud.map_frame_list
    __plot_point_cloud_rays(hits, sensor_origin)

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


def plot_filter_statistics_translation(filter_results, resolution):
    """ Plot the mean and std of the error using a specific filter with different point cloud sizes

                mean: dict(number_points -> mean) the mean error of the specific filter resolution

                std: dict(number_points -> mean) the std of the error of the specific filter resolution

                grid_map:

                filter_type: the name of the filter which was used.
    """
    plt.figure()
    plt.title("Translation Error")

    for key in filter_results.keys():
        __plot_mean_std(key, filter_results[key])

    plt.plot([5, 80], [0.2 * resolution, 0.2 * resolution], c='green', linestyle=':')
    plt.text(81, 0.2 * resolution, r'$0.2r$', color='g')
    plt.ylim(0, 0.2)
    plt.xlim(5, 80)
    plt.xlabel("Point cloud size")
    plt.ylabel(r"$ \Vert o - \xi_{xy} \Vert_2 $")
    plt.legend(loc='upper right')


def plot_filter_statistics_orientation(filter_results, resolution):
    """ Plot the mean and std of the error using a specific filter with different point cloud sizes

                mean: dict(number_points -> mean) the mean error of the specific filter resolution

                std: dict(number_points -> mean) the std of the error of the specific filter resolution

                grid_map:

                filter_type: the name of the filter which was used.
    """
    plt.figure()
    plt.title("Orientation Error")

    for key in filter_results.keys():
        __plot_mean_std(key, filter_results[key])

    plt.plot([5, 80], [0.2 * resolution, 0.2 * resolution], c='green', linestyle=':')
    plt.text(81, 0.2 * resolution, r'$0.2r$', color='g')
    plt.ylim(0, 0.2)
    plt.xlim(5, 80)
    plt.xlabel("Point cloud size")
    plt.ylabel(r"$ \Vert o - \xi_{\theta} \Vert_2 $")
    plt.legend(loc='upper right')


def plot_filter_statistics_iter(filter_results, resolution):
    """ Plot the mean and std of the error using a specific filter with different point cloud sizes

                mean: dict(number_points -> mean) the mean error of the specific filter resolution

                std: dict(number_points -> mean) the std of the error of the specific filter resolution

                grid_map:

                filter_type: the name of the filter which was used.
    """
    plt.figure()
    plt.title("Optimization Iteration")

    for key in filter_results.keys():
        __plot_mean_std(key, filter_results[key])

    plt.xlim(5, 80)
    plt.xlabel("Point cloud size")
    plt.ylabel("Iterations")
    plt.legend(loc='upper right')


def plot_points_on_map(environment, map_points):
    """
    plot the given points in the environment
    :param environment: the svg figure
    :param points: set of points
    :return:
    """
    plt.figure()
    plt.title('Points used for recorded data')
    __plot_environment(environment)

    scale = 1
    for key in map_points:
        points = map_points[key]
        data_in_array = np.array(list(points))
        plt.scatter(data_in_array[:, 0], data_in_array[:, 1], s=scale*np.array([100, 100]), marker='o', label=key)
        scale = scale/2

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


def plot_normals(environment, point_cloud):
    fig = plt.figure()
    plt.title('normals')

    __plot_environment(environment)

    # plot hits of original pc
    point_cloud_points = point_cloud.map_frame_array
    plt.scatter(point_cloud_points[0], point_cloud_points[1], s=np.array([100, 100]), marker='x', c='red',
                label="original point cloud")

    # Plot normals
    point_cloud_point_list = point_cloud.map_frame_list
    for index in range(len(point_cloud_point_list)):
        point = point_cloud_point_list[index]
        normal = point_cloud.get_descriptor("normals", index)
        print(normal)
        plt.arrow(point[0], point[1], normal[0], normal[1], head_width=0.2, head_length=0.3, length_includes_head=True,
                  color="blue")

# Helper Functions


def __plot_point_cloud_rays(hits, sensor_origin):
    """ plot the rays of the point cloud

        hits: list of numpy array[2] the hit points of the point cloud

        sensor_origin: the position of the sensor
    """
    for hit in hits:
        label = "hit filtered point cloud" if "hit filtered point cloud" not in plt.gca().get_legend_handles_labels()[
            1] else ""
        hit_ray = np.transpose(np.array([hit, sensor_origin]))
        plt.plot(hit_ray[0], hit_ray[1], color='blue', linewidth=1, linestyle=':', label=label)


def __plot_environment(environment):
    """ plot the svg environment

        environment: the svg environment
    """
    plt.grid(True)

    for e in environment:
        label = "environment" if "environment" not in plt.gca().get_legend_handles_labels()[1] else ""
        environment_x, environment_y = e.xy
        plt.plot(environment_x, environment_y, color='black', linewidth=2, label=label)


def __plot_grid_map(map):
    print("Hallo World")


def __plot_mean_std(filter_name, filter_result):
    mean = filter_result[0]
    std = filter_result[1]
    X = np.array(list(mean.keys()))
    Y = np.array(list(map(lambda m: mean[m], X)))
    Std = np.array(list(map(lambda m: std[m], X)))
    plt.plot(X, Y, linewidth=2, label=filter_name + " mean")
    plt.fill_between(X, Y - Std, Y + Std, alpha=0.4, label=filter_name + " std")






