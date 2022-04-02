import os
from pathlib import Path
from file.file_utils import copyfile, movefile, mkdirs
from perception.base.seg_group_files import SegGroupFiles
from perception.base.load_cityscapes_config import CityScapesConfig
from labelme_tool.labelme_tool import labelme_tool_main

class SegMoveIgnoreBatch(SegGroupFiles):
    def __init__(self):
        super(SegMoveIgnoreBatch, self).__init__()

        self.cityscapes_config = CityScapesConfig()
        pass

    def move_ignores(self, ambiguity_dir):
        ambiguity_image_dir = os.path.join(ambiguity_dir, 'images')
        ambiguity_label_dir = os.path.join(ambiguity_dir, 'labels')

        ignore_pairs_list = self._get_ignore_pairs()
        if len(ignore_pairs_list) == 0:
            print('no ignore file')
            return
        for pairs in ignore_pairs_list:
            image_file = pairs['image_file']
            gt_labelme_file = pairs['gt_labelme_file']

            dataset_name = Path(image_file).parts[-2]
            if dataset_name != Path(gt_labelme_file).parts[-2]:
                continue

            image_dataset_dir = os.path.join(ambiguity_image_dir, dataset_name)
            label_dataset_dir = os.path.join(ambiguity_label_dir, dataset_name)
            mkdirs(image_dataset_dir)
            mkdirs(label_dataset_dir)

            image_file_name = Path(image_file).parts[-1]
            labelme_file_name = Path(gt_labelme_file).parts[-1]
            movefile(image_file, os.path.join(image_dataset_dir, image_file_name))
            movefile(gt_labelme_file, os.path.join(label_dataset_dir, labelme_file_name))
        pass

    def _get_ignore_pairs(self):
        names = self._get_names('file name')
        ignore_pairs_list = []
        for name in names:
            pairs = self._get_image_pairs(name)
            image_file = pairs['image_file']
            gt_labelme_file = pairs['gt_labelme_file']
            if image_file is None or gt_labelme_file is None:
                continue
            labelFile = labelme_tool_main.load_labelfile(gt_labelme_file)
            for shape in labelFile.shapes:
                label = shape['label']
                label = self.cityscapes_config.replace_name(label)
                if self.cityscapes_config.is_ignore_all_labels(label):
                    ignore_pairs_list.append(pairs)
                    break
        return ignore_pairs_list
seg_move_ignore_batch_main = SegMoveIgnoreBatch()

if __name__ == '__main__':
    from file.file_list import file_list_main

    image_origin_dir = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train_label_rule_prev/images/'
    gt_labelme_json_dir = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train_label_rule_prev/labels-v1.0/'
    ambiguity_dir = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/ambiguity'

    image_origin_files = file_list_main.find_files(image_origin_dir, ['png', 'jpg'], recursive=True)
    gt_labelme_json_files = file_list_main.find_files(gt_labelme_json_dir, ['json'], recursive=True)
    seg_move_ignore_batch_main.add_gt_labelme_json(gt_labelme_json_files)
    seg_move_ignore_batch_main.add_image_origin(image_origin_files)

    seg_move_ignore_batch_main.move_ignores(ambiguity_dir)