import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from perception.base.seg_group_files import SegGroupFiles
from file.file_utils import remove_folder, create_last_folders
from perception.base.segmentation_evaluate import single_image_iou, eval_compare

class SegToLabelmeBatch(SegGroupFiles):
    def __init__(self):
        super(SegToLabelmeBatch, self).__init__()

    def save_tolabelme_json(self, label_dir=None):
        '''
        生成训练用的pairs,每一行，第一列为原图像地址，第二列是label_mask地址
        每个地址为相对image_main_root的地址
        为了功能单一，这里只处理label_mask存在的情况，通过labelme转换的功能由其他函数提供
        :param pairs_file:
        :param image_main_root:
        :return:
        '''
        from perception.base.seg_label_mask_convertor import seg_label_mask_convertor_main
        from labelme_tool.labelme_tool import labelme_tool_main
        names = self._get_names('file name')
        for name in names:
            pairs = self._get_image_pairs(name)
            if pairs['image_file'] is None:
                continue
            #gt_mask_file 不存在，就不保存了
            if pairs['gt_label_mask_file'] is None:
                continue
            image_file = pairs['image_file']
            label_mask_file = pairs['gt_label_mask_file']
            labelme_file = None
            if label_dir is not None:
                json_dir = os.path.join(label_dir, Path(image_file).parts[-2])
                labelme_file = labelme_tool_main.to_labelme_file(image_file, json_dir)
            seg_label_mask_convertor_main.label_mask_file_to_labelme(label_mask_file,
                                                                     image_file,
                                                                     labelme_file=labelme_file)
seg_tolabelme_batch_main = SegToLabelmeBatch()

if __name__ == '__main__':
    from file.file_list import file_list_main

    image_origin_dir = '/nas2/untouch_data/srcData/auto_dirve/OpenData_test/chehuo'
    gt_mask_dir = '/nas2/untouch_data/srcData/auto_dirve/OpenData_test/train_ljj/predict/image/OpenData_test/label_mask_prediction'
    save_gt_labelme_json_dir = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train_label_rule_prev/labels-v1.0/'
    # pred_mask_files = file_list_main.find_files(pred_mask_dir, ['png', 'jpg'], recursive=True)
    gt_mask_files = file_list_main.find_files(gt_mask_dir, ['png', 'jpg'], recursive=True)
    # gt_image_files = file_list_main.find_files(gt_labelme_dir, ['png', 'jpg'], recursive=True)
    image_origin_files_all = file_list_main.find_files(image_origin_dir, ['png', 'jpg'], recursive=True)
    image_origin_files_all = file_list_main.ignore_key(image_origin_files_all, 'pred', -1)
    image_origin_files_all = file_list_main.ignore_key(image_origin_files_all, 'mask', -1)
    image_origin_files = []
    image_origin_files = image_origin_files_all
    # image_origin_files += file_list_main.keep_key(image_origin_files_all, 'once_', -2)
    # image_origin_files = file_list_main.shuffle_and_sampling(image_origin_files, max_len=100)
    # image_origin_files += file_list_main.keep_key(image_origin_files_all, 'DS_', -2)
    # image_origin_files += file_list_main.keep_key(image_origin_files_all, 'unknown1_', -2)
    # gt_image_files = file_list_main.keep_key(gt_image_files, 'image_', -2)
    # pred_mask_files = pred_mask_files[3:]
    # seg_train_batch_main.add_train_pairs(pairs_file, image_main_root)
    # gt_image_files = ['/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train/DS_park_image_1/1637826786.780972004.png']
    seg_tolabelme_batch_main.add_image_origin(image_origin_files)
    # seg_train_batch_main.add_gt_labelme_json(gt_labelme_json_files)
    # seg_train_batch_main.add_gt_labelme(gt_image_files)
    seg_tolabelme_batch_main.add_gt_mask(gt_mask_files)
    # seg_train_batch_main.add_pred_mask(pred_mask_files)

    # seg_tolabelme_batch_main.remove_dataset_name_diff_in_pairs()

    seg_tolabelme_batch_main.save_tolabelme_json()

    # seg_train_batch_main.compare_label_mask()