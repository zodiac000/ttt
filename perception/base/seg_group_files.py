import os
import cv2
from pathlib import Path
from labelme_tool.labelme_tool import labelme_tool_main
from perception.base.seg_label_mask_convertor import seg_label_mask_convertor_main
# from perception.base.seg_label_mask_labelme_convertor import seg_label_mask_convertor_main

class SegGroupFiles(object):
    def __init__(self):
        self.num_last_folder = -1
        # 存储文件名配对，比如image_file, gt_mask_file, gt_labelme_file, pred_mask_file
        self.group_file_map = {}
        # 存储不同顺序的文件名，用于遍历，比如文件名排序，各add进入的顺序
        self.name_order = {
            'file name': []
        }
        pass

    def set_num_last_folder(self, num_last_folder):
        self.num_last_folder = num_last_folder
    def get_name(self, file, suffix=''):
        file_stem = file
        if '.' in suffix:
            file_stem = file.replace(suffix, '')+'.invalid'
        pathfile = Path(file_stem)
        name = str(pathfile.stem)
        if self.num_last_folder != -1:
            name = '/'.join(pathfile.parts[self.num_last_folder:-1]) + '$' + name
        return name

    def get_stem(self, name):
        return name.split('$')[-1]

    def get_folder_stem(self, name):
        if '$' in name:
            words = name.split('$')
            stem = words[-1]
            folder = words[0]
        else:
            stem = name
            folder = ''
        return folder, stem

    def get_parent_fullstem(self, root_dir, name):
        folder, stem = self.get_folder_stem(name)
        parent = os.path.join(root_dir, folder)
        fullstem = os.path.join(parent, stem)
        return parent, fullstem

    def _get_name_from_labelme_file(self, json_file):
        name = self.get_name(json_file, labelme_tool_main.labelme_suffix)
        return name

    def _get_filename_from_image_file(self, image_file):
        return str(Path(image_file).stem) + '.png'

    def add_train_pairs(self, pairs_file, image_main_root=''):
        with open(pairs_file, "r") as fid:
            image_pair_lines = fid.readlines()
            self.name_order['pairs'] = []
            for image_pair_line in image_pair_lines:
                words = image_pair_line.strip().split(' ')
                image_file = os.path.join(image_main_root, words[0].strip())
                name = self.get_name(image_file)
                gt_mask_file = None
                if len(words) >= 2:
                    gt_mask_file = os.path.join(image_main_root, words[1].strip())
                pairs = self._get_image_pair_map_extend(name)
                pairs["image_file"] = image_file
                pairs["gt_label_mask_file"] = gt_mask_file
                self.name_order['pairs'].append(name)
        print('pairs files has ', len(self.name_order['pairs']))

    def add_pred_mask(self, pred_mask_files):
        print('pred mask files has ', len(pred_mask_files))
        self.name_order['pred'] = []
        for pred_mask_file in pred_mask_files:
            name = self.get_name(pred_mask_file)
            pairs = self._get_image_pair_map_extend(name)
            pairs["pred_mask_file"] = pred_mask_file
            self.name_order['pred'].append(name)

    def add_image_origin(self, image_files):
        print('image origin files has ', len(image_files))
        self.name_order['image'] = []
        for image_file in image_files:
            name = self.get_name(image_file)
            pairs = self._get_image_pair_map_extend(name)
            pairs["image_file"] = image_file
            self.name_order['image'].append(name)

    def add_gt_mask(self, mask_files):
        print('gt mask files has ', len(mask_files))
        self.name_order['gt'] = []
        for mask_file in mask_files:
            name = self.get_name(mask_file)
            pairs = self._get_image_pair_map_extend(name)
            pairs["gt_label_mask_file"] = mask_file
            self.name_order['gt'].append(name)

    def add_gt_image_labelme(self, image_files):
        print('gt image labelme file has ', len(image_files))
        self.name_order['gt'] = []
        for image_file in image_files:
            name = self.get_name(image_file)
            pairs = self._get_image_pair_map_extend(name)
            pairs["image_file"] = image_file
            pairs["gt_labelme_file"] = labelme_tool_main.to_labelme_file(image_file)
            self.name_order['gt'].append(name)

    def add_pred_color(self, pred_color_files):
        print('pred color files has ', len(pred_color_files))
        self.name_order['pred'] = []
        for color_file in pred_color_files:
            name = self.get_name(color_file)
            pairs = self._get_image_pair_map_extend(name)
            pairs["pred_color_file"] = color_file
            self.name_order['pred'].append(name)

    def add_gt_labelme_json(self, json_files):
        print('gt labelme json files has ', len(json_files))
        self.name_order['gt'] = []
        for json_file in json_files:
            name = self._get_name_from_labelme_file(json_file)
            if name == 'DS_park_image_1#1637898080':
                name = self._get_name_from_labelme_file(json_file)
            pairs = self._get_image_pair_map_extend(name)
            pairs["gt_labelme_file"] = json_file
            self.name_order['gt'].append(name)
    def remove_dataset_name_diff_in_pairs(self):
        names = self._get_names('file name')
        group_file_map = {}
        for name in names:
            pairs = self._get_image_pairs(name)
            image_file = pairs['image_file']
            gt_label_mask_file = pairs['gt_label_mask_file']
            gt_labelme_file = pairs['gt_labelme_file']
            pred_mask_file = pairs['pred_mask_file']
            dataset_names = {}
            if image_file is not None:
                image_dataset_name = Path(image_file).parts[-2]
                dataset_names[image_dataset_name] = 0

            if gt_label_mask_file is not None:
                gt_label_mask_dataset_name = Path(gt_label_mask_file).parts[-2]
                dataset_names[gt_label_mask_dataset_name] = 0

            if gt_labelme_file is not None:
                gt_labelme_dataset_name = Path(gt_labelme_file).parts[-2]
                dataset_names[gt_labelme_dataset_name] = 0

            if pred_mask_file is not None:
                pred_mask_dataset_name = Path(pred_mask_file).parts[-2]
                dataset_names[pred_mask_dataset_name] = 0
            if len(dataset_names) == 1:
                group_file_map[name] = pairs
        self.group_file_map = group_file_map

    def _get_image_pair_map_extend(self, name, add_key=None):
        pairs = self._get_image_pairs(name, add_key)
        self.group_file_map[name] = pairs
        return pairs

    def _get_image_pairs(self, name, add_key=None):
        if name not in self.group_file_map:
            pairs = {
                "image_file": None,
                "gt_label_mask_file": None,
                "gt_labelme_file": None,
                "pred_mask_file": None,
                "prob_mask_file": None,
                "pred_color_file": None
            }
            if add_key is not None:
                pairs[add_key] = None
        else:
            pairs = self.group_file_map[name]
        return pairs

    def _get_names(self, key=''):
        if key == 'file name':
            names = sorted(list(self.group_file_map.keys()))
        elif key in self.name_order:
            names = self.name_order[key]
        elif 'pairs' in self.name_order:
            names = self.name_order['pairs']
        elif 'gt' in self.name_order:
            names = self.name_order['gt']
        elif 'image' in self.name_order:
            names = self.name_order['image']
        elif 'pred' in self.name_order:
            names = self.name_order['pred']
        else:
            return ValueError('no image file in _get_names')
        names = sorted(names)
        return names

    def _get_image_file(self, pairs):
        return pairs['image_file']

    def _read_pairs_map(self, name, num_classes):
        pairs = self.group_file_map[name]
        image_file = pairs["image_file"]
        pred_mask_file = pairs["pred_mask_file"]
        if image_file is None:
            return None, None, None, pairs
        image = cv2.imread(image_file)
        gt_label_mask, _ = self._read_gt_label_mask(pairs)

        if pred_mask_file is not None:
            pred_label_mask = cv2.imread(pred_mask_file, 0)
        else:
            pred_label_mask = None

        return image, gt_label_mask, pred_label_mask, pairs

    def _read_gt_label_mask(self, pairs):
        gt_mask_file = pairs["gt_label_mask_file"]
        gt_labelme_file = pairs["gt_labelme_file"]
        file = None
        if gt_mask_file is not None:
            gt_label_mask = cv2.imread(gt_mask_file, 0)
            file = gt_mask_file
        elif gt_labelme_file is not None:
            gt_label_mask = seg_label_mask_convertor_main.label_mask_from_labelme_base(
                labelme_file=gt_labelme_file)
            file = gt_labelme_file
        else:
            gt_label_mask = None
        return gt_label_mask, file

    def _read_pred_label_mask(self, pairs):
        pred_mask_file = pairs["pred_mask_file"]
        pred_mask = cv2.imread(pred_mask_file, 0)
        return pred_mask

    def _set_iou(self, pairs, iou):
        pairs['iou'] = iou

    def _get_iou(self, pairs):
        return pairs['iou']
