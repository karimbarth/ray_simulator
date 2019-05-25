import os
import csv
import argparse
import numpy as np
import scan_match_filter_evaluation
from grid_map import GridMap, load_svg_environment


def print_param(args):

    print("######################## Used Parameters ########################")
    for param in args:
        print(param + ": " + str(args[param]))

    print("######################## Used Parameters ########################")


def store_param_list(args):
    dir_name = args["directory_name"]

    dir_path = "data/" + dir_name
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    w = csv.writer(open(dir_path + "/param.txt", "w"))
    for key, val in args.items():
        w.writerow([key, val])


def load_position_list(path, map_size, map_offset):
    normalized_data = np.genfromtxt(path, delimiter=",")
    transformed_data = normalized_data * map_size + map_offset

    res = list(map(lambda l: tuple(l), transformed_data.tolist()))
    return res


def main(args):
    map_name = args["map"]
    map_size = args["map_size"]
    map_offset = args["map_offset"]
    map_resolution = args["map_resolution"]

    position_list_path = args["position_list"]
    number_sample = args["number_sample"]
    point_cloud_size = args["point_cloud_size"]
    rangefinder_noise = args["rangefinder_noise"]

    prefix = args["file_prefix"]
    dir_name = args["directory_name"]

    points = load_position_list(position_list_path, map_size=map_size, map_offset=map_offset)

    environment = load_svg_environment("./geometries/" + map_name + ".svg", map_size, map_offset)
    grid_map = GridMap(map_size, map_resolution)
    grid_map.load_environment(environment)
    scan_match_filter_evaluation.generate_map_data(grid_map=grid_map, environment=environment, map_name=map_name,
                                                   sample_count=number_sample, cloud_size=point_cloud_size,
                                                   points=points, prefix=prefix, dir_name=dir_name,
                                                   rangefinder_noise=rangefinder_noise)
    store_param_list(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--map", required=False, default="floorplan_simplified", type=str)
    parser.add_argument('-ms', "--map_size", required=False, default=10, type=int)
    parser.add_argument('-mo', "--map_offset", required=False, default=1.0, type=float)
    parser.add_argument('-mr', "--map_resolution", required=False, default=0.1, type=float)
    parser.add_argument('-ns', "--number_sample", required=False, default=200, type=int)
    parser.add_argument('-pcs', "--point_cloud_size", required=False, default=80, type=int)
    parser.add_argument('-rn', "--rangefinder_noise", required=False, default=0.1, type=float)
    parser.add_argument('-pl', "--position_list", required=True)
    parser.add_argument('-fp', "--file_prefix", required=False, default="", type=str)
    parser.add_argument('-dir', "--directory_name", required=True, type=str)

    args = vars(parser.parse_args())
    print_param(args)
    main(args)
