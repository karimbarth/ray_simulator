
'''
def adaptively_voxel_filtered(point_cloud, min_number_of_points, max_length):
    if point_cloud.count <= min_number_of_points:
        return point_cloud

    result = VoxelFilter(max_length).filter(point_cloud)
    if result.count >= min_number_of_points:
        return result

    high_length = max_length
    while high_length > 1e-2:
        low_length = high_length / 2.
        result = VoxelFilter(low_length).filter(point_cloud)

        if result.count >= min_number_of_points:

            while (high_length - low_length) / low_length > 1e-1:
                mid_length = (low_length + high_length) / 2.
                point_cloud_candidate = VoxelFilter(mid_length).filter(point_cloud)
                if point_cloud_candidate.count >= min_number_of_points:
                    low_length = mid_length
                    result = point_cloud_candidate
                else:
                    high_length = mid_length
            return result
        high_length /= 2.

    return point_cloud

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

# def plot_scan_matching(grid_map, point_cloud, estimate, step=3):
#     fig = plt.figure()
#     map_size = grid_map.size
#     index_array = grid_map.obstacle_indices
#     plt.scatter(index_array[0], index_array[1], s=np.array([80, 80]), marker='x',
#                 c='black', label="map obstacle")
#
#     plt.scatter([point_cloud.lidar_origin[0]], [point_cloud.lidar_origin[1]], s=np.array([60, 60]), marker='o',
#                 c='green',
#                 label="ground truth")
#
#     plt.scatter([estimate[0]], [estimate[1]], s=np.array([60, 60]), marker='o', c='red',
#                 label="estimate")
#
#     # plot grid
#     ax = fig.axes[0]
#     major_ticks = np.arange(0, map_size, 1)
#     minor_ticks = np.arange(0, map_size, grid_map.resolution)
#
#     ax.set_xticks(major_ticks)
#     ax.set_xticks(minor_ticks, minor=True)
#     ax.set_yticks(major_ticks)
#     ax.set_yticks(minor_ticks, minor=True)
#     ax.grid(which='minor', alpha=0.2)
#     ax.grid(which='major', alpha=0.5)
#     plt.ylim(0, 10)
#     plt.xlim(0, 10)
#
#     if step >= 2:
#         estimate_transformed_point_cloud = point_cloud.transform_from_lidar_frame_to(*estimate)
#         plt.scatter(estimate_transformed_point_cloud[0], estimate_transformed_point_cloud[1], s=np.array([100, 100]),
#                     marker='x', c='red',
#                     label="estimated point cloud")
#         if step >= 3:
#             # plot transformation arrows
#             plt.arrow(estimate[0], estimate[1], point_cloud.lidar_origin[0] - estimate[0],
#                       point_cloud.lidar_origin[1] - estimate[1], head_width=0.2, head_length=0.3,
#                       length_includes_head=True, color="blue")
#
#             plt.arrow(estimate[0], estimate[1], point_cloud.lidar_origin[0] - estimate[0], 0, length_includes_head=True,
#                       color="blue", linestyle=":")
#             plt.text((estimate[0] + point_cloud.lidar_origin[0]) / 2, estimate[1] - 0.3,
#                      r'$\Delta\xi_x$', color='b', fontsize=20)
#             plt.arrow(estimate[0], estimate[1], 0, point_cloud.lidar_origin[1] - estimate[1], length_includes_head=True,
#                       color="blue", linestyle=":")
#             plt.text(estimate[0] + 0.05, (estimate[1] + point_cloud.lidar_origin[1]) / 2,
#                      r'$\Delta\xi_y$', color='b', fontsize=20)
#             # plot point cloud arrow
#             for n in range(point_cloud.count):
#                 plt.arrow(estimate_transformed_point_cloud[0][n], estimate_transformed_point_cloud[1][n],
#                           point_cloud.lidar_origin[0] - estimate[0], point_cloud.lidar_origin[1] - estimate[1],
#                           head_width=0.1, head_length=0.15, length_includes_head=True, color="blue", linestyle=":")
#
#
#
# def plot_scan(environment, sensor_origin, estimates, point_cloud, title):
#     plt.figure()
#     plt.title(title)
#     for e in environment:
#         label = "environment" if "environment" not in plt.gca().get_legend_handles_labels()[1] else ""
#         environment_x, environment_y = e.xy
#         plt.plot(environment_x, environment_y, color='black', linewidth=2, label=label)
#
#     hits = point_cloud.map_frame_list
#     for hit in hits:
#         label = "hit" if "hit" not in plt.gca().get_legend_handles_labels()[1] else ""
#         hit_ray = np.transpose(np.array([hit, sensor_origin]))
#         plt.plot(hit_ray[0], hit_ray[1], color='green', linewidth=1, linestyle=':', label=label)
#
#     plt.scatter([sensor_origin[0]], [sensor_origin[1]], s=np.array([100, 100]), marker='x', c='blue', label="ground "
#                                                                                                             "truth")
#     # plt.scatter([estimates[0][0]], [estimates[0][1]], s=np.array([100, 100]), marker='x', c='red', label="estimate")
#     plt.legend(loc='lower right')
#
#
# def plot_normal_filter_visualization(environment, point_cloud, step):
#
#     sensor_origin = point_cloud.lidar_origin
#     plt.figure()
#     # plot environment
#     for e in environment:
#         label = "environment" if "environment" not in plt.gca().get_legend_handles_labels()[1] else ""
#         environment_x, environment_y = e.xy
#         plt.plot(environment_x, environment_y, color='black', linewidth=2, label=label)
#
#     # plot rays from filtered pc
#     hits = point_cloud.map_frame_list
#     for hit in hits:
#         label = "hit filtered point cloud" if "hit filtered point cloud" not in plt.gca().get_legend_handles_labels()[
#             1] else ""
#         hit_ray = np.transpose(np.array([hit, sensor_origin]))
#         plt.plot(hit_ray[0], hit_ray[1], color='blue', linewidth=1, linestyle=':', label=label)
#
#     # plot hits of original pc
#     point_cloud_points = point_cloud.map_frame_array
#     plt.scatter(point_cloud_points[0], point_cloud_points[1], s=np.array([100, 100]), marker='x', c='red',
#                 label="original point cloud")
#
#     # plot sensor
#     plt.scatter([sensor_origin[0]], [sensor_origin[1]], s=np.array([100, 100]), marker='x', c='blue', label="sensor")
#     plt.axis('equal')
#     plt.axis([0, 7, 0, 7])
#     if step == 2:
#         for hit in hits:
#             print(hit)
#             if hit[0] < 1.4:
#                 plt.arrow(hit[0], hit[1], -0.75, 0, head_width=0.1, head_length=0.15,
#                           length_includes_head=True, color="green")
#             elif hit[0] < 5.5:
#                 plt.arrow(hit[0], hit[1], 0, -0.75, head_width=0.1, head_length=0.15,
#                           length_includes_head=True, color="green")
#             else:
#
#                 plt.arrow(hit[0], hit[1], 0.67271, 0.3316, head_width=0.1, head_length=0.15,
#                           length_includes_head=True, color="green")
#
