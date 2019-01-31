import numpy as np
from point_cloud import PointCloud


def pca(points):
    cov = np.cov(points)
    eigen_values = np.linalg.eig(cov)
    min_index = np.argmin(eigen_values)
    eigen_vectors = np.linalg.eigvals(cov)
    return eigen_vectors[min_index]


def estimate_normal(point, neighbors, viewpoint):
    normal_candidate = pca(neighbors)
    dir = np.dot(normal_candidate, viewpoint - point)
    return normal_candidate if dir > 0 else  -normal_candidate


def select_neighborhood(point, points, ball_radius):
    neighbors = []
    for single_point in points:
        distance = np.linalg.norm(point - single_point)
        if distance <= ball_radius:
            neighbors.append(single_point)

    return neighbors


def estimate_normals(point_cloud, distance=0.02):
    normals = []
    for point in point_cloud.map_frame_list:
        normal = estimate_normal(point, select_neighborhood(point, point_cloud.map_frame_list, distance), point_cloud.lidar_origin)
        normals.append(normal)
    return normals


def filter(point_cloud, number_of_points):
    normals = estimate_normals(point_cloud)
    result = PointCloud(point_cloud.lidar_origin, point_cloud.lidar_orientation)
    return result



