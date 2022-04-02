import sys
sys.path.insert(0, "../../")
import os
from file.file_utils import mkdirs, write_list
from perception.base.seg_group_files import SegGroupFiles
from labelme_tool.labelme_tool import labelme_tool_main
from file.json_tool import save_json, load_json

from pdb import set_trace

class CurveLane2LabelmeBatch(SegGroupFiles):
    def __init__(self):
        super(CurveLane2LabelmeBatch, self).__init__()
        self.labelme_list = []
        self.curvelane_suffix = '.lines.json'
        self.curvelane_txt_suffix = '.lines.json'


    def add_curvelane_json(self, json_files):
        print('gt curvelane json file has ', len(json_files))
        self.name_order['curvelane_json_file'] = []
        for json_file in json_files:
            name = self.get_name(json_file, self.curvelane_suffix)
            pairs = self._get_image_pair_map_extend(name, add_key='curvelane_json_file')
            pairs["curvelane_json_file"] = json_file
            self.name_order['curvelane_json_file'].append(name)

    def save_curvelane_txt(self, labelme_txt_dir=None):
        names = self._get_names('file name')
        labelme_list = []
        for name in names:
            pairs = self._get_image_pairs(name)
            curvelane_json_file = pairs['curvelane_json_file']
            
            json_map = load_json(curvelane_json_file)


    def save_labelme_jsons(self, labelme_json_dir=None):
        names = self._get_names('file name')
        labelme_list = []
        for name in names:
            pairs = self._get_image_pairs(name)
            image_file = pairs["image_file"]
            curvelane_json_file = pairs['curvelane_json_file']
            if image_file is None or curvelane_json_file is None:
                continue

            labelme_file = None
            if labelme_json_dir is not None:
                labelme_dir, labelme_fullstem = self.get_parent_fullstem(labelme_json_dir, name)
                labelme_file = labelme_fullstem+labelme_tool_main.labelme_suffix
                if not os.path.isdir(labelme_dir):
                    mkdirs(labelme_dir)
            labelme_file = self._curvelane_to_labelme(image_file, curvelane_json_file, labelme_file)
            labelme_list.append([image_file, labelme_file])
        self.labelme_list = labelme_list
        os.system('chmod -R 777 ' + labelme_json_dir)

    def save_curvelane_jsons(self, curvelane_json_path):
        names = self._get_names('file name')
        for name in names:
            pairs = self._get_image_pairs(name)
            gt_labelme_file = pairs['gt_labelme_file']
            if gt_labelme_file is None:
                continue
            curvelane_json_dir, json_curvelane_fullstem = self.get_parent_fullstem(curvelane_json_path, name)
            json_curvelane_file = json_curvelane_fullstem + self.curvelane_suffix

            if not os.path.isdir(curvelane_json_dir):
                os.makedirs(curvelane_json_dir)
            self._labelme_to_curvelane(gt_labelme_file, json_curvelane_file)
        os.system('chmod -R 777 ' + curvelane_json_path)

    def save_labelme_list(self, labelme_list_file):
        write_list(labelme_list_file, self.labelme_list)

    def _curvelane_to_labelme(self, image_file, curvelane_json_file, labelme_file=None):
        labelFile = labelme_tool_main.create_labelfile(image_file, labelme_file)
        if labelFile is None:
            return

        json_map = load_json(curvelane_json_file)
        if json_map is None:
            return
        for group_id, line in enumerate(json_map['Lines']):
            label = "line"
            shape_type = 'linestrip'
            points = []
            for point in line:
                points.append([float(point['x']), float(point['y'])])
            labelme_tool_main.add_shape(labelFile, label, shape_type, points, group_id)
        labelme_tool_main.save_label_file(labelFile)
        return labelFile.filename

    def _labelme_to_curvelane(self, labelme_file, curvelane_json_file):
        label_file = labelme_tool_main.load_labelfile(labelme_file)
        if label_file is None:
            return
        lines = []
        shapes = label_file.shapes
        for shape in shapes:
            if shape['label'] == "line":
                line = []
                for point in shape['points']:
                    line.append({"y": point[0], "x": point[1]})
                lines.append(line)
        json_map = {
            "Lines": lines
        }

        save_json(curvelane_json_file, json_map)
        os.chmod(curvelane_json_file, 0o777)

curvelane2labelme_main = CurveLane2LabelmeBatch()

if __name__ == "__main__":
    from file.file_list import file_list_main
    index = 12
    image_path = "/data2/OpenDataset/lane_det/train_image_label/train/images/curvelanes_{}".format(index)
    curvelane_json_path = "/data2/OpenDataset/lane_det/train_image_label/train/curvelane_json/curvelanes_{}".format(index)
    labelme_json_path = "/data2/OpenDataset/lane_det/train_image_label/train/labelme_json_v0.1/curvelanes_{}".format(index)
    # labelme_list_path = "/data4/wb/lane_detection/multi_data/json_labelme/1_labelme.list"



    image_files = file_list_main.find_files(image_path, ['png', 'jpg'], recursive=True)
    curvelane_json_files = file_list_main.find_files(curvelane_json_path, ['json'], recursive=True)
    labelme_json_files = file_list_main.find_files(labelme_json_path, ['json'], recursive=True)

    curvelane2labelme_main.set_num_last_folder(-1)
    curvelane2labelme_main.add_image_origin(image_files)
    # curvelane2labelme_main.add_gt_labelme_json(labelme_json_files)
    curvelane2labelme_main.add_curvelane_json(curvelane_json_files)
    curvelane2labelme_main.save_labelme_jsons(labelme_json_path)
    # curvelane2labelme_main.save_labelme_list(labelme_list_path)





