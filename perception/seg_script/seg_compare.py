import numpy as np
from pathlib import Path
import cv2
from perception.base.seg_label_mask_convertor import seg_label_mask_convertor_main
from perception.base.segmentation_evaluate import single_image_iou,\
    single_image_iou_from_confusion_matrix, eval_compare
def pair_image_file(src_image_files, dst_image_files):
    src_name_map = {}
    for file in src_image_files:
        src_name_map[Path(file).parts[-1]] = file

    image_pairs = []
    for file in dst_image_files:
        name = Path(file).parts[-1]
        if name in src_name_map:
            image_pairs.append([src_name_map[name], file])
    return image_pairs


def calc_mIoU_labelme(image_pairs, num_classes, ignore_index):
    IoUs = []
    for src_image_file, dst_image_file in image_pairs:
        src_label_mask = seg_label_mask_convertor_main.label_mask_from_labelme_base(src_image_file)
        dst_label_mask = seg_label_mask_convertor_main.label_mask_from_labelme_base(dst_image_file)
        iou = single_image_iou_from_confusion_matrix(src_label_mask, dst_label_mask, num_classes, ignore_index)
        # iou = single_image_iou(src_label_mask, dst_label_mask, num_classes)
        IoUs.append(iou)
    return IoUs

def calc_IoUs_label_mask_file(image_pairs, num_classes, ignore_index):
    IoUs = []
    for src_image_file, dst_image_file in image_pairs:
        src_label_mask = cv2.imread(src_image_file, 0)
        dst_label_mask = cv2.imread(dst_image_file, 0)
        iou = single_image_iou_from_confusion_matrix(src_label_mask, dst_label_mask, num_classes, ignore_index)
        IoUs.append(iou)
    return IoUs


if __name__ == '__main__':
    from file.file_list import file_list_main

    src_dir = '/nas2/untouch_data/tools/ceshi/1'
    dst_dir = '/nas2/untouch_data/tools/ceshi/2'
    num_classes = 3
    threshold_classes = [0.98, 0.9, 0.99]
    src_image_files = file_list_main.find_files(src_dir, ['png', 'jpg'], recursive=True)
    dst_image_files = file_list_main.find_files(dst_dir, ['png', 'jpg'], recursive=True)

    image_pairs = pair_image_file(src_image_files, dst_image_files)

    print('src image[', len(src_image_files),
          '], dst image[', len(dst_image_files),
          '], compared image[', len(image_pairs), ']')
    if len(image_pairs) == 0:
        print('no match pairs!!!')
        exit()

    IoUs = calc_mIoU_labelme(image_pairs, num_classes, ignore_index=255)
    eval_compare(IoUs, threshold_classes)

    for idx, [src_image_file, dst_image_file] in enumerate(image_pairs):
        print('IoU[', IoUs[idx], ']---', Path(src_image_file).parts[-1])
