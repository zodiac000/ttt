import numpy as np
import cv2
from pathlib import Path


class SegCityscapeRoIBatch(object):
    def __init__(self):
        self.key_label_trainIds = '_gtFine_labelTrainIds'
        self.key_gt_mask = '_leftImg8bit'
        self.group_file_map = {}
        pass

    def add_gtFine_label_trainIds(self, gt_fine_dir):
        label_trainIds_mask_files = file_list_main.find_files(gt_fine_dir, ['png', 'jpg'], recursive=True)
        label_trainIds_mask_files = file_list_main.keep_folders(label_trainIds_mask_files, ['cityscapes'])
        label_trainIds_mask_files = file_list_main.keep_key(label_trainIds_mask_files,
                                                            self.key_label_trainIds,
                                                            -2)

        for file in label_trainIds_mask_files:
            name = str(Path(file).stem)[:-len(self.key_label_trainIds)]
            if name not in self.group_file_map:
                self.group_file_map[name] = {
                    'label_trainIds_mask_file': None,
                    'gt_label_mask_file': None
                }
            self.group_file_map[name]['label_trainIds_mask_file'] = file

    def add_gt_mask(self, gt_mask_dir):
        gt_mask_files = file_list_main.find_files(gt_mask_dir, ['png', 'jpg'], recursive=True)
        gt_mask_files = file_list_main.keep_folders(gt_mask_files, ['cityscapes'])
        gt_mask_files = file_list_main.keep_key(gt_mask_files,
                                                self.key_gt_mask,
                                                -2)
        for file in gt_mask_files:
            name = str(Path(file).stem)[:-len(self.key_gt_mask)]
            if name not in self.group_file_map:
                self.group_file_map[name] = {
                    'label_trainIds_mask_file': None,
                    'gt_label_mask_file': None
                }
            self.group_file_map[name]['gt_label_mask_file'] = file
        pass

    def remove_trainIds255(self):
        names = self.group_file_map.keys()
        for name in names:
            gt_mask_file = self.group_file_map[name]['gt_label_mask_file']
            label_trainIds_mask_file = self.group_file_map[name]["label_trainIds_mask_file"]
            if gt_mask_file is None or label_trainIds_mask_file is None:
                continue
            gt_label_mask = cv2.imread(gt_mask_file, 0)
            label_trainIds_mask = cv2.imread(label_trainIds_mask_file, 0)

            pred_idxs = np.where(label_trainIds_mask == 255)
            gt_label_mask[pred_idxs[0], pred_idxs[1]] = 255

            cv2.imwrite(gt_mask_file, gt_label_mask)

seg_cityscapes_roi_batch_main = SegCityscapeRoIBatch()

if __name__ == '__main__':
    from file.file_list import file_list_main

    gt_fine_dir = '/nas2/untouch_data/srcData/auto_dirve/OpenData/cityscapes/cityscapes/gtFine/train/'
    gt_mask_dir = '/data8/ljj/dataset/RoadDataset/train/labels'

    seg_cityscapes_roi_batch_main.add_gtFine_label_trainIds(gt_fine_dir)
    seg_cityscapes_roi_batch_main.add_gt_mask(gt_mask_dir)

    seg_cityscapes_roi_batch_main.remove_trainIds255()
