
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