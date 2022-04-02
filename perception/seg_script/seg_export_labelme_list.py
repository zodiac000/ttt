import os
import cv2
from pathlib import Path
from perception.base.seg_group_files import SegGroupFiles
from tqdm import tqdm
from file.file_utils import write_list

class SegExportLabelmeList(SegGroupFiles):
    def __init__(self):
        super(SegExportLabelmeList, self).__init__()
        pass

    def save_labelme_list(self, labelme_list_file):
        names = self._get_names('image')
        lines = []
        for name in names:
            pairs = self._get_image_pairs(name)
            image_file = pairs["image_file"]
            gt_labelme_file = pairs["gt_labelme_file"]
            pred_color_file = pairs["pred_color_file"]
            line = []
            if image_file is None:
                continue
            line.append(image_file)
            if gt_labelme_file is None:
                continue
            line.append(gt_labelme_file)
            if pred_color_file is not None:
                line.append(pred_color_file)
            lines.append(line)
        write_list(labelme_list_file, lines)

seg_export_label_list_main = SegExportLabelmeList()

if __name__ == '__main__':
    from file.file_list import file_list_main
    from perception.config.seg_select_defects_label_config import untouch_road as Config

    image_origin_dir = "/data4/tjk/lane_detection/DataSet/RoadData/train_data/images"
    gt_labelme_dir = "/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train_image_label/labels-v1.0"
    pred_color_dir = "/data4/tjk/lane_detection/DataSet/RoadData/test_data/new/added_prediction"

    labelme_list_file = '/data4/wb/auto_diff/xx_labelme.list'

    image_origin_files = file_list_main.find_files(image_origin_dir, ['png', 'jpg'], recursive=True)
    gt_labelme_files = file_list_main.find_files(gt_labelme_dir, ['json'], recursive=True)
    pred_color_files = file_list_main.find_files(pred_color_dir, ['png', 'jpg'], recursive=True)

    image_origin_files = image_origin_files[:10]

    seg_export_label_list_main.add_image_origin(image_origin_files)
    seg_export_label_list_main.add_gt_labelme_json(gt_labelme_files)
    seg_export_label_list_main.add_pred_color(pred_color_files)

    seg_export_label_list_main.remove_dataset_name_diff_in_pairs()

    seg_export_label_list_main.save_labelme_list(labelme_list_file)



