from tqdm import tqdm
import numpy as np
from perception.base.seg_group_files import SegGroupFiles
from perception.base.segmentation_evaluate import single_image_iou, \
    eval_compare, calc_confusion_matrix, single_image_iou_from_confusion_matrix


class SegEvalBatch(SegGroupFiles):
    def __init__(self):
        super(SegEvalBatch, self).__init__()

    def eval_mIou_acc(self, threshold_classes, ignore_index=255):
        num_classes = len(threshold_classes)
        names = self._get_names('file name')
        ious = []
        for name in tqdm(names):
            pairs = self._get_image_pairs(name)
            gt_label_mask, label_mask_file = self._read_gt_label_mask(pairs)
            pred_label_mask = self._read_pred_label_mask(pairs)
            if gt_label_mask is None or pred_label_mask is None:
                continue
            iou = single_image_iou(pred_label_mask,
                                   gt_label_mask,
                                   num_classes,
                                   ignore_index)
            self._set_iou(pairs, iou)
            ious.append(iou)
        eval_compare(ious, threshold_classes)

    def eval_confusion_matrix(self, num_classes, ignore_index=255):
        names = self._get_names('file name')

        mcm = np.zeros((num_classes, num_classes), np.float)
        num_valid_image = 0
        for name in names:
            pairs = self._get_image_pairs(name)
            gt_label_mask, label_mask_file = self._read_gt_label_mask(pairs)
            pred_label_mask = self._read_pred_label_mask(pairs)
            if gt_label_mask is None or pred_label_mask is None:
                continue
            cm = calc_confusion_matrix(pred_label_mask,
                                   gt_label_mask,
                                   num_classes,
                                   ignore_index)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm[np.isnan(cm)] = 0
            mcm += cm
            num_valid_image += 1
        mcm /= num_valid_image
        print(mcm)

    def save_low_precision_to_list(self, list_file, threshold_classes):
        from file.file_list import file_list_main
        names = self._get_names('file name')
        image_files = []
        for name in names:
            pairs = self._get_image_pairs(name)
            iou = self._get_iou(pairs)

            for class_idx in range(len(threshold_classes)):
                if np.isnan(iou[class_idx]):
                    continue
                if iou[class_idx] < threshold_classes[class_idx]:
                    print(class_idx, iou[class_idx], threshold_classes[class_idx])
                    image_file = self._get_image_file(pairs)
                    image_files.append(image_file)
                    break
        file_list_main.write_list(list_file, image_files)


seg_eval_batch_main = SegEvalBatch()

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

    pairs_file = '/data8/duzhe/dataset/opendata/cityscapes/lists/fine/val/pair_relative.txt'
    image_main_root = '/data8/duzhe/dataset/opendata/cityscapes'
    image_origin_dir = '/data9/duzhe/dataset/untouch/segmentation/road/train/images'
    gt_mask_dir = '/data9/duzhe/dataset/untouch/segmentation/road/train/labels'
    save_pairs_file = '/data9/duzhe/dataset/untouch/segmentation/road/train/pairs.txt'
    pred_mask_dir = '/nas2/untouch_data/srcData/auto_dirve/OpenData_test/big/label_mask_prediction'
    pred_mask_files = file_list_main.find_files(pred_mask_dir, ['png', 'jpg'], recursive=True)
    gt_mask_files = file_list_main.find_files(gt_mask_dir, ['png', 'jpg'], recursive=True)
    # gt_image_files = file_list_main.find_files(gt_labelme_dir, ['png', 'jpg'], recursive=True)
    # image_origin_files = file_list_main.find_files(image_origin_dir, ['png', 'jpg'], recursive=True)
    # gt_image_files = file_list_main.keep_key(gt_image_files, 'image_', -2)
    # pred_mask_files = pred_mask_files[3:]
    seg_eval_batch_main.add_train_pairs(pairs_file, image_main_root)
    # gt_image_files = ['/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train/DS_park_image_1/1637826786.780972004.png']
    # seg_eval_batch_main.add_image_origin(image_origin_files)
    # seg_eval_batch_main.add_gt_labelme(gt_image_files)
    # seg_eval_batch_main.add_gt_mask(gt_mask_files)
    seg_eval_batch_main.add_pred_mask(pred_mask_files)

    seg_eval_batch_main.eval_mIou_acc(threshold_classes)
    # seg_eval_batch_main.eval_confusion_matrix(3)

    list_file = 'xx_labelme.list'
    threshold_classes[2] = 0.5
    seg_eval_batch_main.save_low_precision_to_list(list_file, threshold_classes)
