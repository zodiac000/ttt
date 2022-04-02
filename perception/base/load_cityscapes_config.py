import os
import numpy as np
from file.json_tool import load_json

class CityScapesConfig(object):
    def __init__(self):
        self._label_trainiD_map = {}
        self._label_name_map = {}
        self._name_map_map = {}
        self._skip_name_list = []
        self._keep_train_ids_list = []
        self._background_map = {}
        self._seg_cover_map = {}

        self.load_cityscapes_config()
        self.load_seg_cover_config()
        pass

    def is_keep_trainId(self, trainId):
        if trainId not in self._label_trainiD_map:
            return False
        if len(self._keep_train_ids_list) == 0:
            return True
        if trainId in self._keep_train_ids_list:
            return True
        return False

    def replace_name(self, name):
        if name in self._name_map_map:
            return self._name_map_map[name]
        else:
            return name

    def is_ignore_all_labels(self, name):
        if name in self._skip_name_list:
            return True
        else:
            return False
    def check_name(self, name):
        if name in self._label_name_map:
            return True
        return False

    def check_keep_train_ids(self, name):
        if name not in self._label_name_map:
            return False

        if len(self._keep_train_ids_list) == 0:
            return True
        trainId = self.get_trainId(name)
        if trainId in self._keep_train_ids_list:
            return True
        return False

    def check_trainId(self, trainId):
        if trainId in self._label_trainiD_map:
            return True
        return False

    def get_trainId(self, name):
        if name in self._label_name_map:
            return self._label_name_map[name]['trainId']
        return None

    def get_name(self, trainId):
        if trainId in self._label_trainiD_map:
            return self._label_trainiD_map[trainId]['name']
        return None

    def get_color(self, name):
        if name in self._label_name_map:
            return self._label_name_map[name]['color']
        return None

    def get_background_id(self):
        return self._background_map['trainId']

    def get_names(self):
        return list(self._label_name_map.keys())

    def get_color_map(self):
        color_map = np.zeros((256, 3), np.uint8)
        for train_id, label in self._label_trainiD_map.items():
            color_map[train_id] = label['color']
        return color_map

    def get_seg_cover(self, dataset_name):
        if dataset_name not in self._seg_cover_map:
            return []
        return self._seg_cover_map[dataset_name]

    def get_keep_trainIds(self):
        return self._keep_train_ids_list

    def load_cityscapes_config(self, label_config_file=None):
        if label_config_file is None:
            current_path = os.path.realpath(__file__)
            current_dir = os.path.split(current_path)[0]
            label_config_file = current_dir + '/../config/cityscapes_label.json'
        json_map = load_json(label_config_file)

        self._label_trainiD_map = {}
        self._label_name_map = {}
        self._name_map_map = json_map['name_map']
        self._skip_name_list = json_map['skip_name']
        self._keep_train_ids_list = json_map['keep_train_ids']
        self._background_map = json_map['background']
        for label in json_map['labels']:
            train_id = label['trainId']
            if train_id == 255 or train_id == -1:
                continue
            self._label_trainiD_map[train_id] = label

        for label in json_map['labels']:
            self._label_name_map[label['name']] = label

    def load_seg_cover_config(self, cover_config_file=None):
        if cover_config_file is None:
            current_path = os.path.realpath(__file__)
            current_dir = os.path.split(current_path)[0]
            cover_config_file = current_dir + '/../config/seg_cover.json'
        json_map = load_json(cover_config_file)

        self._seg_cover_map = json_map