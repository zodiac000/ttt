import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from perception.base.seg_group_files import SegGroupFiles
from file.file_utils import remove_folder, create_last_folders
from perception.base.segmentation_evaluate import single_image_iou, eval_compare

class SegTrainBatch(SegGroupFiles):
    def __init__(self):
        super(SegTrainBatch, self).__init__()

    def save_train_pairs(self, pairs_file, image_main_root):
        '''
        生成训练用的pairs,每一行，第一列为原图像地址，第二列是label_mask地址
        每个地址为相对image_main_root的地址
        为了功能单一，这里只处理label_mask存在的情况，通过labelme转换的功能由其他函数提供
        :param pairs_file:
        :param image_main_root:
        :return:
        '''
        from file.file_utils import write_list
        lines = []
        names = self._get_names('file name')

        if len(image_main_root) != 0 and image_main_root[-1] != '/':
            image_main_root += '/'
        for name in names:
            pairs = self._get_image_pairs(name)
            if pairs['image_file'] is None:
                continue
            #gt_mask_file 不存在，就不保存了
            if pairs['gt_label_mask_file'] is None:
                continue
            image_file = pairs['image_file'].replace(image_main_root, '')
            label_mask_file = pairs['gt_label_mask_file'].replace(image_main_root, '')
            lines.append([image_file, label_mask_file])
        write_list(pairs_file, lines)

    def save_gt_label_mask(self, gt_label_mask_dir, num_last_folder=1):
        '''
        保存gt的label mask数据
        :param gt_label_mask_dir:
        :return:
        '''
        names = self._get_names('file name')
        for name in tqdm(names):
            pairs = self._get_image_pairs(name)
            image_file = pairs['image_file']
            if image_file is None:
                print(pairs, ' is None')
                continue
            gt_label_mask, label_mask_file = self._read_gt_label_mask(pairs)

            if gt_label_mask is None:
                continue
            path = create_last_folders(gt_label_mask_dir, label_mask_file, num_last_folder)
            gt_mask_file_tmp = os.path.join(path, self._get_filename_from_image_file(image_file))
            cv2.imwrite(gt_mask_file_tmp, gt_label_mask)


    def eval_label_mask(self, threshold_classes):
        num_classes = len(threshold_classes)
        names = self._get_names('file name')
        IoUs = []
        for name in names:
            pairs = self._get_image_pairs(name)
            pred_mask_file = pairs["pred_mask_file"]
            if pred_mask_file is None:
                continue
            pred_label_mask = cv2.imread(pred_mask_file, 0)
            gt_label_mask, _ = self._read_gt_label_mask(pairs)
            iou = single_image_iou(pred_label_mask, gt_label_mask, num_classes)
            IoUs.append(iou)
        eval_compare(IoUs, threshold_classes)

    def compare_label_mask(self):
        names = self._get_names('file name')
        IoUs = []
        for name in names:
            pairs = self._get_image_pairs(name)
            pred_mask_file = pairs["pred_mask_file"]
            if pred_mask_file is None:
                continue
            pred_label_mask = cv2.imread(pred_mask_file, 0)
            gt_label_mask, _ = self._read_gt_label_mask(pairs)
            num_diff = np.sum(pred_label_mask != gt_label_mask)
            print(num_diff/(pred_label_mask.shape[0] * pred_label_mask.shape[1]))

seg_train_batch_main = SegTrainBatch()

if __name__ == '__main__':
    from file.file_list import file_list_main
    from perception.config.seg_view_batch_config import untouch_train_dataset as Config

    pairs_file = Config.get("pairs_file", "")
    image_main_root = Config.get("image_main_root", "")
    image_origin_dir = Config.get("image_origin_dir", "")
    pred_mask_dir = Config.get("pred_mask_dir", "")
    gt_labelme_dir = Config.get("gt_labelme_dir", "")
    gt_mask_dir = Config.get("gt_mask_dir", "")

    save_diff_dir = Config.get("save_diff_dir", "")
    save_label_mask_dir = Config.get("save_label_mask_dir", "")
    save_pairs_file = Config.get("save_pairs_file", "")
    # gt_labelme_dir = save_label_mask_dir

    num_pred_classes = Config.get("num_pred_classes", 3)
    pred_weight = Config.get("pred_weight", 0.5)
    num_gt_classes = Config.get("num_gt_classes", 3)
    gt_weight = Config.get("gt_weight", 0.5)

    threshold_classes = Config.get("threshold_classes", [1,1,1])

    save_pairs_file = '/data8/ljj/dataset/RoadDataset/list/train/0322_all_pairs.txt'

    # gt_labelme_dir = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train'
    save_label_mask_dir = '/data8/ljj/dataset/RoadDataset/train/labels'

    # image_origin_dir = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train_label_rule_prev/images/'
    image_origin_dir = '/data4/tjk/lane_detection/DataSet/RoadData/train_data/images'
    gt_mask_dir = '/data8/ljj/dataset/RoadDataset/train/labels'
    gt_labelme_json_dir = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train_label_rule_prev/labels-v1.0/'
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
    gt_labelme_json_files = file_list_main.find_files(gt_labelme_json_dir, ['json'], recursive=True)
    # gt_image_files = file_list_main.keep_key(gt_image_files, 'image_', -2)
    # pred_mask_files = pred_mask_files[3:]
    # seg_train_batch_main.add_train_pairs(pairs_file, image_main_root)
    # gt_image_files = ['/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train/DS_park_image_1/1637826786.780972004.png']
    seg_train_batch_main.add_image_origin(image_origin_files)
    # seg_train_batch_main.add_gt_labelme_json(gt_labelme_json_files)
    # seg_train_batch_main.add_gt_labelme(gt_image_files)
    seg_train_batch_main.add_gt_mask(gt_mask_files)
    # seg_train_batch_main.add_pred_mask(pred_mask_files)

    seg_train_batch_main.remove_dataset_name_diff_in_pairs()

    # remove_folder(save_label_mask_dir)
    # seg_train_batch_main.save_gt_label_mask(save_label_mask_dir)

    image_main_root = ''
    seg_train_batch_main.save_train_pairs(save_pairs_file, image_main_root)

    # seg_train_batch_main.compare_label_mask()