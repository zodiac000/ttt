import numpy as np
import cv2

def single_mask_iou(pred_i, label_i):
    intersect_i = np.logical_and(pred_i, label_i)
    pred_area_i = np.sum(pred_i.astype('int32'))
    label_area_i = np.sum(label_i.astype('int32'))
    intersect_area_i = np.sum(intersect_i.astype('int32'))
    union_i = pred_area_i + label_area_i - intersect_area_i
    if union_i == 0:
        iou_i = np.nan
    else:
        iou_i = intersect_area_i / union_i
    return iou_i

def single_image_iou(pred, label, num_classes, ignore_index=255):
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(
                             pred.shape, label.shape))
    mask = label != ignore_index
    #pred里，也有一些因为crop导致部分区域是不预测的，其值也是255
    mask = np.bitwise_and(mask, pred != ignore_index)

    class_iou = []
    for i in range(num_classes):
        pred_i = np.logical_and(pred == i, mask)
        label_i = np.logical_and(label == i, mask)
        iou = single_mask_iou(pred_i, label_i)
        class_iou.append(iou)

    return class_iou

# 设标签宽W，长H
def calc_confusion_matrix(gt_mask, pred_mask, max_label_num, ignore_index=255):  # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为19）
    '''
    两张mask标签图，生成混淆矩阵，混淆矩阵的对角线代表对上的像素个数
    a为（H×W）的mask矩阵，reshape成一维数组
    b同a
    '''
    if not pred_mask.shape == gt_mask.shape:
        raise ValueError('Shape of `pred` and `gt should be equal, '
                         'but there are {} and {}.'.format(
                             pred_mask.shape, gt_mask.shape))
    gt_mask_1dim = gt_mask.flatten()
    pred_mask_1dim = pred_mask.flatten()
    k = (gt_mask_1dim >= 0) & (gt_mask_1dim < max_label_num) & (gt_mask_1dim != ignore_index)
    # k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景）
    hist = np.bincount(max_label_num * gt_mask_1dim[k].astype(int) + pred_mask_1dim[k],
                minlength=max_label_num ** 2)[:max_label_num * max_label_num] \
        .reshape(max_label_num, max_label_num)
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    return hist

def class_iou_from_confusion_matrix(hist):  # 分别为每个类别计算mIoU，hist的形状(n, n)
    '''
    :param hist:
    :return:
    '''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)

def single_image_iou_from_confusion_matrix(pred_mask_img, gt_mask_img, num_classes, ignore_index=255):
    if gt_mask_img.shape != pred_mask_img.shape:
        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        return None
    hist = calc_confusion_matrix(gt_mask_img, pred_mask_img, num_classes, ignore_index)
    IoU = class_iou_from_confusion_matrix(hist)
    return IoU

def calc_eval(IoUs, threshold_classes):
    num_classes = len(threshold_classes)
    mIoU = np.zeros((num_classes), np.float)
    accs = np.zeros((len(threshold_classes)), np.float)
    class_count = np.zeros((num_classes), np.int)
    for iou in IoUs:
        for class_idx in range(len(threshold_classes)):
            if not np.isnan(iou[class_idx]):
                mIoU[class_idx] += iou[class_idx]
                class_count[class_idx] += 1
            if iou[class_idx] > threshold_classes[class_idx]:
                accs[class_idx] += 1.0
    mIoU /= class_count
    return mIoU, accs

def eval_compare(IoUs, threshold_classes):
    calc_eval(IoUs, threshold_classes)
    print('mIoU:', mIoU)
    print('acc num:', accs)
    print('acc(%):', accs / len(IoUs))


if __name__ == '__main__':
    import cv2
    import copy
    image_path1 = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/lane_dataset/makring_data/mark_output/label_mask_prediction/1616006182299.png'
    image_path2 = image_path1

    pred_mask_img = cv2.imread(image_path1, 0)
    gt_mask_img = copy.deepcopy(pred_mask_img)
    gt_mask_img[100:200, 100:400] = 255

    iou = single_image_iou_from_confusion_matrix(pred_mask_img, gt_mask_img, 19)
    print(iou)
