from pathlib import Path
import os
import json
from file.json_tool import load_json
from perception.base.seg_group_files import SegGroupFiles
from labelme_tool.labelme_tool import labelme_tool_main
from file.json_tool import save_json

class LabelmeToTrainFormat(SegGroupFiles):
    def __init__(self):
        super(LabelmeToTrainFormat, self).__init__()
        self.labelme_json_suffix = '.json'
        pass

    def gen_Tusimple(self):
        pass

    def gen_Curverlane(self, data_root, mode='train'):
        """
        生成 Curverlane 格式的数据集
        :return:
        """
        names = self._get_names()
        for name in names:
            pairs = self.group_file_map[name]
            image_file = pairs["image_file"]
            print(name)
            print(image_file)
        pass

    def gen_Culane(self):
        pass

    def gen_others(self):
        """
        to do
        :return:
        """
        pass

    # def add_labelme_files(self, json_files):
    #     print('json files has ', len(image_files))
    #     self.name_order['image'] = []
    #     for json_file in json_files:
    #         name = self.get_name(json_file, labelme_tool_main.labelme_suffix)
    #         pairs = self._get_image_pair_map_extend(name)
    #         pairs["gt_labelme_file"] = json_file
    #         self.name_order['image'].append(name)

labelme2train_format = LabelmeToTrainFormat()

if __name__ == '__main__':
    from file.file_list import file_list_main
    image_path = "/data4/tjk/project/lane_detection/auto_mark_data/src"
    json_path = "/data4/tjk/project/lane_detection/auto_mark_data/src"
    dataset_root = "/data4/tjk/project/lane_detection/auto_mark_data/test_data_root"
    image_files = file_list_main.find_files(image_path, ['png', 'jpg'], recursive=True)
    labelme_json_files = file_list_main.find_files(json_path, ['json'], recursive=True)


    labelme2train_format.add_image_origin(image_files)
    labelme2train_format.add_gt_labelme_json(labelme_json_files)
    # print(labelme2train_format.group_file_map)
    labelme2train_format.gen_Curverlane('dd')

    pass
