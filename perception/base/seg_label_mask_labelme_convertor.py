import cv2
import numpy as np
from file.json_tool import load_json
from labelme_tool.labelme_tool import labelme_tool_main

class SegLabelMaskConvertor(object):
    def __init__(self):
        self.background_trainId = 2
        self.ignore_frame_flag = ["ignore"]
        self.false_label_remap = {
            "ingore": "ignore",
            "road block": "roadblock",
            "road_block": "roadblock",
            "roadblockgroup": "roadblock",
            "road block group": "roadblock",
        }
        self.label_2_trainID = {
            "freespace": 0,
            "road": 0,
            "sidewalk": 1,
            "roadblock": 2,
            "parking": 255,
            "void": 255,
            "ground": 255}
        # 处理遮挡关系，大ID覆盖小ID
        self.sort_anno_names = ['__background__', 'freespace', 'road', 'sidewalk', 'parking', 'guard rail',
                                'road block', 'roadblock', 'road_block', 'roadblockgroup', 'road block group',
                                'void', 'ground', 'ignore',
                                'patch', 'road edge', 'road edge low', 'white  dash line',
                                'white solid line', 'yellow dash line', 'yellow solid line']
        self.sortAnno_2_labelId_map = self._gen_anno_labelId_map()
    pass

    def _gen_anno_labelId_map(self):
        self.lane_class_len = len(self.sort_anno_names)  # 种类
        self.lane_class_len_idx = np.arange(self.lane_class_len).tolist()
        cc = zip(self.sort_anno_names, self.lane_class_len_idx)
        anno_2_labelId = dict(cc)
        return anno_2_labelId

    def label_mask_from_labelme_base(self, image_file=None, labelme_file=None):
        if labelme_file is None:
            labelme_file = labelme_tool_main.to_labelme_file(image_file)
        json_info = load_json(labelme_file)
        self._remap_label(json_info)
        if (self._check_ignore(json_info)):
            return None
        seg_mask = self._fill_contour_seg(json_info)
        return seg_mask

    def _remap_label(self, json_info):
        for sub_shape in json_info["shapes"]:
            if sub_shape["label"] in self.false_label_remap:
                sub_shape["label"] = self.false_label_remap[sub_shape["label"]]

    def _check_ignore(self, json_info):
        for sub_shape in json_info["shapes"]:
            if sub_shape["label"] in self.ignore_frame_flag:
                return True
        return False

    def _sort_anno_bylabelId(self, shape_arr):
        labelIds = [self.sortAnno_2_labelId_map[sub_shape_arr["label"]] for sub_shape_arr in shape_arr]
        sortIdx  = sorted(range(len(labelIds)), key=lambda x: labelIds[x])
        shape_arr = [shape_arr[Idx] for Idx in sortIdx]
        return shape_arr

    def _fill_contour_seg(self, json_info):
        imgHeight = json_info['imageHeight']
        imgWidth = json_info['imageWidth']
        seg_label = np.zeros((imgHeight, imgWidth), np.uint8)
        label_mask = np.zeros((imgHeight, imgWidth), np.uint8)
        sub_shape_arr = []
        for sub_shape in json_info["shapes"]:
            lane_label = sub_shape["label"]
            if lane_label not in self.label_2_trainID.keys():
                continue
            sub_shape_arr.append(sub_shape)
        sub_shape_arr = self._sort_anno_bylabelId(sub_shape_arr)

        for sub_shape in sub_shape_arr:
            lane_label = sub_shape["label"]
            if lane_label not in self.label_2_trainID.keys():
                continue
            clsId = self.label_2_trainID[lane_label]
            points = np.round(np.array(sub_shape["points"])).astype(np.int32)
            cv2.fillPoly(seg_label, [points], clsId)
            cv2.fillPoly(label_mask, [points], 1)

        background_mask = (1 - label_mask).astype(np.bool)
        seg_label[background_mask] = self.background_trainId

        return seg_label

seg_label_mask_convertor_main = SegLabelMaskConvertor()