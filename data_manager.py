import os
import numpy as np
import pickle


def sorted_key_list(key_list):
    result = list(key_list)
    result.sort()
    return result


class Datatype:
    def __init__(self, map_name, map_type, filter_name, transformation_type,
                 positions, pre_opti_error, post_opti_error, pre_orientation_error, post_orientation_error, num_iter):
        self.__map_name = map_name
        self.__map_type = map_type
        self.__filter_name = filter_name
        self.__transformation_type = transformation_type
        self.__positions = positions
        self.__pre_opti_error = pre_opti_error
        self.__post_opti_error = post_opti_error
        self.__pre_orientation_error = pre_orientation_error
        self.__post_orientation_error = post_orientation_error
        self.__num_iter = num_iter

    @property
    def filter_name(self):
        return self.__filter_name

    @property
    def transformation_type(self):
        return self.__transformation_type

    @property
    def positions(self):
        return self.__positions

    @property
    def pre_opti_error(self):
        return self.__pre_opti_error

    @property
    def post_opti_error(self):
        return self.__post_opti_error

    @property
    def pre_orientation_error(self):
        return self.__pre_orientation_error

    @property
    def post_orientation_error(self):
        return self.__post_orientation_error

    @property
    def num_iter(self):
        return self.__num_iter

    def add_position(self, position):
        self.__positions.add(position)

    def add_data(self, data):
        for key in data.keys():
            pre_translation_data = (data[key])[0, :]
            post_translation_data = (data[key])[1, :]
            pre_orientation_data = (data[key])[2, :]
            post_orientation_data = (data[key])[3, :]
            num_iter_data = (data[key])[4, :]

            # translation error before optimization
            if key in self.__pre_opti_error:
                self.__pre_opti_error[key] = np.append(self.__pre_opti_error[key], pre_translation_data)
            else:
                self.__pre_opti_error[key] = pre_translation_data
            # translation error after optimization
            if key in self.__post_opti_error:
                self.__post_opti_error[key] = np.append(self.__post_opti_error[key], post_translation_data)
            else:
                self.__post_opti_error[key] = post_translation_data

            # rotation error before optimization
            if key in self.__pre_orientation_error:
                self.__pre_orientation_error[key] = np.append(self.__pre_orientation_error[key], pre_orientation_data)
            else:
                self.__pre_orientation_error[key] = pre_orientation_data
            # rotation error after optimization
            if key in self.__post_orientation_error:
                self.__post_orientation_error[key] = np.append(self.__post_orientation_error[key], post_orientation_data)
            else:
                self.__post_orientation_error[key] = post_orientation_data

            # num iteration needed for optimization
            if key in self.__num_iter:
                self.__num_iter[key] = np.append(self.__num_iter[key], num_iter_data)
            else:
                self.__num_iter[key] = num_iter_data

    def post_translation_means(self):
        means = dict()

        for key in sorted_key_list(self.__post_opti_error.keys()):
            means[key] = np.mean(self.__post_opti_error[key])
        return means

    def post_translation_std(self):
        std = dict()
        for key in sorted_key_list(self.__post_opti_error.keys()):
            std[key] = np.std(self.__post_opti_error[key])
        return std

    def post_rotation_means(self):
        means = dict()
        for key in sorted_key_list(self.post_orientation_error.keys()):
            means[key] = np.mean(self.post_orientation_error[key])
        return means

    def post_rotation_std(self):
        std = dict()
        for key in sorted_key_list(self.post_orientation_error.keys()):
            std[key] = np.std(self.post_orientation_error[key])
        return std

    def iter_means(self):
        means = dict()
        for key in sorted_key_list(self.num_iter.keys()):
            means[key] = np.mean(self.num_iter[key])
        return means

    def iter_std(self):
        std = dict()
        for key in sorted_key_list(self.num_iter.keys()):
            std[key] = np.std(self.num_iter[key])
        return std


class DataManager:
    def __init__(self, map_name, map_type):
        self.__map_name = map_name
        self.__map_type = map_type
        self.__data = None

    @property
    def data(self):
        return self.__data

    def load_data(self, filter_name, transformation_type):

        if self.__data:
            print("DataManger already opens a file, please close it! All changes are lost!")

        path = "data/2d/evaluate_" + self.__map_name + "_" + self.__map_type \
               + "_" + filter_name + "_" + transformation_type + ".npy"

        if os.path.isfile(path):

            pkl_file = open(path, 'rb')
            root = pickle.load(pkl_file)
            pkl_file.close()

            positions = root["positions"]
            num_iter = root["num_iter"]
            error_data = root["data"]
            pre_opti_error = error_data["pre_trans"]
            post_opti_error = error_data["post_trans"]
            pre_orientation_error = error_data["pre_orientation"]
            post_orientation_error = error_data["post_orientation"]

            self.__data = Datatype(self.__map_name, self.__map_type, filter_name, transformation_type,
                                   positions, pre_opti_error, post_opti_error, pre_orientation_error,
                                   post_orientation_error, num_iter)
        else:
            positions = set()
            pre_trans = dict()
            post_trans = dict()
            pre_orientation = dict()
            post_orientation = dict()
            num_iter = dict()
            self.__data = Datatype(self.__map_name, self.__map_type, filter_name, transformation_type, positions,
                                   pre_trans, post_trans, pre_orientation, post_orientation, num_iter)

    def add_data(self, data, position):
        if self.__data is None:
            print("No file is open!")
            return
        else:
            self.__data.add_position(position)
            self.__data.add_data(data)

    def safe_and_close(self):
        if self.__data is None:
            print("No file is open!")
            return
        else:
            path = "data/2d/evaluate_" + self.__map_name + "_" + self.__map_type \
                   + "_" + self.__data.filter_name + "_" + self.__data.transformation_type + ".npy"

            root = dict()
            root["positions"] = self.__data.positions
            root["num_iter"] = self.__data.num_iter
            error_data = dict()
            error_data["pre_trans"] = self.__data.pre_opti_error
            error_data["post_trans"] = self.__data.post_opti_error
            error_data["pre_orientation"] = self.__data.pre_orientation_error
            error_data["post_orientation"] = self.__data.post_orientation_error
            root["data"] = error_data

            # np.save(root, path)

            output = open(path, 'wb')
            pickle.dump(root, output)
            output.close()
            self.__data = None

    def readonly_close(self):
        self.__data = None
