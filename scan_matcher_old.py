""" Match the given (filtered) scan against a given map to estimate
    the position and orientation of the agent using ceres

Parameters:
    map (array_like): A (GridMap) map we try to match against. A 2D map containing 1 for obstacle and 0 for a free cell
    point_cloud (list of 3D vectors): representing a scan of the rangefinder
    param (list of param): parameters of the optimizer

Returns:
    (x, y, theta): representing the estimated position (x,y) and the orientation of the agent
"""


def match_scan(map, point_cloud, param):
    raise NotImplementedError
