import os
import io
import numpy as np
import cv2
import copy
import PIL
from perception.base.load_cityscapes_config import CityScapesConfig
import matplotlib.pyplot as plt
from image_alg.image_base import resize


class SegView(object):
    def __init__(self):
        self.cityscapes_config = CityScapesConfig()
        self.color_map = self.cityscapes_config.get_color_map()

        self.diff_result = {}

    def calc_color_mask(self, label_mask):
        c1 = cv2.LUT(label_mask, self.color_map[:, 0])
        c2 = cv2.LUT(label_mask, self.color_map[:, 1])
        c3 = cv2.LUT(label_mask, self.color_map[:, 2])
        pseudo_img = np.dstack((c1, c2, c3))
        return pseudo_img

    def calc_image_mask(self, image, label_mask, num_classes, ignore_ids=[], weight=0.5):
        # 默认lable_mask中值为num_classes-1为背景
        pseudo_img = self.calc_color_mask(label_mask)

        pseudo_img_mask = (label_mask >= 0) & (label_mask < num_classes - 1)
        for id in ignore_ids:
            pseudo_img_mask = pseudo_img_mask & (label_mask != id)
        pseudo_img_mask = pseudo_img_mask.astype(np.uint8)
        inv_pseudo_img_mask = 1 - pseudo_img_mask

        im = image.astype(np.float)
        image_mask =  cv2.bitwise_and(im, im, mask=inv_pseudo_img_mask) \
                     + weight * cv2.bitwise_and(im, im, mask=pseudo_img_mask) \
                     + (1.0 - weight) * cv2.bitwise_and(pseudo_img,
                                             pseudo_img,
                                             mask=pseudo_img_mask)
        return image_mask.astype(np.uint8)

    def calc_diff(self,
                  image,
                  pred_label_mask,
                  gt_label_mask,
                  num_pred_classes=3,
                  pred_weight=0.5,
                  num_gt_classes=19,
                  gt_weight=0.0):
        pred_vis_img = None
        if pred_label_mask is not None:
            pred_vis_img = self.calc_image_mask(image, pred_label_mask, num_pred_classes, weight=pred_weight)
            pred_idxs = np.where(pred_label_mask == 255)
            pred_vis_img[pred_idxs[0], pred_idxs[1]] = 255
        gt_vis_img = None
        if gt_label_mask is not None:
            gt_vis_img = self.calc_image_mask(image, gt_label_mask, num_gt_classes, weight=gt_weight)
            gt_idxs = np.where(gt_label_mask == 255)
            gt_vis_img[gt_idxs[0], gt_idxs[1]] = 255

        diff_gt_vis_img = None
        diff_ratio = None
        diff_label_mask = None
        if pred_label_mask is not None and gt_label_mask is not None:
            same_idxs = np.where(pred_label_mask == gt_label_mask)
            diff_idxs = np.where(pred_label_mask != gt_label_mask)
            diff_ratio = 1.0 - len(same_idxs[0])/(pred_label_mask.shape[0] * pred_label_mask.shape[1])
            diff_label_mask = copy.deepcopy(gt_label_mask)
            diff_label_mask[same_idxs[0], same_idxs[1]] = 255
            diff_gt_vis_img = self.calc_image_mask(image, diff_label_mask, num_pred_classes)
            tmp = np.zeros(gt_label_mask.shape, np.uint8)
            tmp[diff_idxs[0], diff_idxs[1]] = pred_label_mask[diff_idxs[0], diff_idxs[1]]
            pred255_idxs = np.where(tmp == 255)
            diff_gt_vis_img[pred255_idxs[0], pred255_idxs[1]] = 255
            tmp[diff_idxs[0], diff_idxs[1]] = gt_label_mask[diff_idxs[0], diff_idxs[1]]
            gt255_idxs = np.where(tmp == 255)
            diff_gt_vis_img[gt255_idxs[0], gt255_idxs[1]] = 255

        self.diff_result = {
            'image':image,
            'diff_gt_vis_img':diff_gt_vis_img,
            'gt_vis_img':gt_vis_img,
            'pred_vis_img':pred_vis_img,
            'diff_ratio':diff_ratio,
            'diff_label_mask':diff_label_mask
        }
        return image, diff_gt_vis_img, gt_vis_img, pred_vis_img

    def _diff_ratio_label_mask(self, gt_label_mask, pred_label_mask):
            num_diff = np.sum(pred_label_mask != gt_label_mask)
            return num_diff/(pred_label_mask.shape[0] * pred_label_mask.shape[1])

    def cv_view(self, image, diff_gt_vis_img, gt_vis_img, pred_vis_img):
        h, w = image.shape[:2]
        image_view = np.zeros((h*2, w*2, 3), np.uint8)
        image_view[:h, :w, :] = image
        if gt_vis_img is not None:
            image_view[h:2*h, :w, :] = gt_vis_img
        if diff_gt_vis_img is not None:
            image_view[:h, w:2*w, :] = diff_gt_vis_img
        if pred_vis_img is not None:
            image_view[h:2*h, w:2*w, :] = pred_vis_img

        return image_view

    def plt_view(self, image, diff_gt_vis_img, gt_vis_img, pred_vis_img):
        fig = plt.figure(figsize=(24, 12))
        ax1 = fig.add_subplot(221)
        ax1.imshow(image[:, :, ::-1])
        ax1.set_title("cvImg")

        ax2 = fig.add_subplot(222)
        if diff_gt_vis_img is not None:
            ax2.imshow(diff_gt_vis_img[:, :, ::-1])
        ax2.set_title("diff_gt_vis_img")

        ax3 = fig.add_subplot(223)
        if gt_vis_img is not None:
            ax3.imshow(gt_vis_img[:, :, ::-1])
        ax3.set_title("gt_label")

        ax4 = fig.add_subplot(224)
        if pred_vis_img is not None:
            ax4.imshow(pred_vis_img[:, :, ::-1])
        ax4.set_title("pred_label")

        buffer_ = io.BytesIO()
        fig.savefig(buffer_, format="png")
        buffer_.seek(0)
        image_view = PIL.Image.open(buffer_)
        # 转换为numpy array
        image_view = np.asarray(image_view)
        return image_view

    def export_diff_image(self, mode='plt', scale=-1):
        if mode == 'plt':
            image_view = self.plt_view(self.diff_result['image'],
                          self.diff_result['diff_gt_vis_img'],
                          self.diff_result['gt_vis_img'],
                          self.diff_result['pred_vis_img'])
        elif mode == 'cv':
            image_view = self.cv_view(self.diff_result['image'],
                          self.diff_result['diff_gt_vis_img'],
                          self.diff_result['gt_vis_img'],
                          self.diff_result['pred_vis_img'])
        else:
            image_view = None
            return
        if scale < 0:
            scale = 900/image_view.shape[0]
        image_view = resize(image_view, scale, scale)

        self.diff_result['image_view'] = image_view
        return image_view

    def show_diff(self, win_name='', mode='plt'):
        if mode == 'plt':
            plt.show()
        elif mode == 'cv':
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, self.diff_result['image_view'])
            cv2.waitKey(0)

    def save_diff(self, diff_file):
        cv2.imwrite(diff_file, self.diff_result['image_view'])


seg_view_main = SegView()

if __name__ == '__main__':
    image_file = '/nas2/untouch_data/srcData/auto_dirve/OpenData/cityscapes/cityscapes/leftImg8bit/train/stuttgart/stuttgart_000021_000019_leftImg8bit.png'
    pred_label_mask_file = '/nas2/untouch_data/srcData/auto_dirve/OpenData_test/big/label_mask_prediction/stuttgart_000021_000019_leftImg8bit.png'
    gt_label_mask_file = '/nas2/untouch_data/srcData/auto_dirve/OpenData/cityscapes/cityscapes/gtFine/train_mask/stuttgart/label_mask_prediction/stuttgart_000021_000019_leftImg8bit.png'
    image = cv2.imread(image_file)
    pred_label_mask = cv2.imread(pred_label_mask_file, 0)
    gt_label_mask = cv2.imread(gt_label_mask_file, 0)
    num_classes = 3

    # diff_image = seg_view.calc_color_mask(pred_label_mask)
    seg_view_main.calc_diff(image, pred_label_mask, gt_label_mask, num_classes)
    seg_view_main.export_diff_image(mode='cv')
    seg_view_main.show_diff(mode='cv')
    # diff_image = seg_view.export_diff_image()
    # cv2.imshow('xx', gt_label_mask)
    # cv2.waitKey(0)
