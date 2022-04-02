import os
import cv2
from pathlib import Path
from perception.base.seg_view import seg_view_main
from file.file_utils import remove_folder, mkdirs
from view.base.img_view import ImgView
from perception.base.seg_group_files import SegGroupFiles

class SegViewBatch(SegGroupFiles):
    def __init__(self):
        super(SegViewBatch, self).__init__()
        self._img_view = ImgView()
        pass

    def save_diff_image(self,
                        save_dir,
                        is_show=True,
                        mode='plt',
                        num_pred_classes=3,
                        pred_weight=0.5,
                        num_gt_classes=19,
                        gt_weight=0.0):
        '''
        :param num_classes:
        :param save_dir: 键盘s，保存diff信息的地址
        :param is_show: 是否需要显示，若不需要显示，则直接保存所有的diff图
        :param mode: cv表示使用opencv来绘制diff图，plt标志使用plt绘制diff图
        :param gt_weight: 用于合成原始图像与gt混合图的权重，0代表不显示原始图像，1代表纯color mask图像
        :return:
        '''
        self._save_image_list(save_dir)
        names = self._get_names('image')
        self._img_view.init()

        idx = 0
        while idx < len(names):
            name = names[idx]
            image, gt_label_mask, pred_label_mask, pairs = self._read_pairs_map(name, num_pred_classes)
            if image is None:
                idx += 1
                print('image is None')
                continue
            image_file = pairs["image_file"]
            print(idx, image_file)

            seg_view_main.calc_diff(image,
                                    pred_label_mask,
                                    gt_label_mask,
                                    num_pred_classes,
                                    pred_weight,
                                    num_gt_classes,
                                    gt_weight)
            image_view = seg_view_main.export_diff_image(mode=mode)
            # seg_view_main.show_diff()

            if is_show:
                self._img_view.set_image(image_view)
                image_file_text = '/'.join(Path(image_file).parts[-3:])
                self._img_view.show_text(point=None, text=image_file_text, color=(0, 0, 255), font_scale=2)
                diff_ratio = seg_view_main.diff_result["diff_ratio"]
                if diff_ratio is not None:
                    self._img_view.show_text(point=None, text=str(diff_ratio), color=(0, 0, 255), font_scale=2)
                idx, key = self._img_view.show(delay=0)

                if key == ord('s'):
                    save_file = os.path.join(save_dir, self._get_filename_from_image_file(image_file))
                    cv2.imwrite(save_file, self._img_view.image)
                    # seg_view_main.save_diff(save_file)
                    self._save_image_list(save_dir, image_file)
            else:
                save_file = os.path.join(save_dir, self._get_filename_from_image_file(image_file))
                seg_view_main.save_diff(save_file)

    def _save_image_list(self, save_dir, image_file = None):
        image_list_file = os.path.join(save_dir, 'image_files_labelme.list')
        if image_file is None:
            mkdirs(save_dir)
            with open(image_list_file, 'w') as fp:
                fp.write('')
        else:
            with open(image_list_file, 'a') as fp:
                fp.write(image_file+'\n')

seg_view_batch_main = SegViewBatch()

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
    gt_weight = Config.get("gt_weight", 0.0)

    pred_mask_dir = '/nas2/untouch_data/srcData/auto_dirve/OpenData_test/train_ljj/predict/once_pairs_0319/once_raw03_3/label_mask_prediction'
    gt_mask_dir = '/data8/ljj/dataset/RoadDataset/train/labels'
    image_origin_dir = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train_image_label/images'
    # gt_labelme_json_dir = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train_label_rule_prev/labels_prev/cityscapes_aachen'
    pred_mask_files = file_list_main.find_files(pred_mask_dir, ['png', 'jpg'], recursive=True)
    gt_mask_files = file_list_main.find_files(gt_mask_dir, ['png', 'jpg'], recursive=True)
    image_origin_files = file_list_main.find_files(image_origin_dir, ['png', 'jpg'], recursive=True)
    # gt_labelme_json_files = file_list_main.find_files(gt_labelme_json_dir, ['json'], recursive=True)
    # gt_image_files = file_list_main.find_files(gt_labelme_dir, ['png', 'jpg'], recursive=True)
    # gt_image_files = file_list_main.keep_key(gt_image_files, 'image_', -2)
    # pred_mask_files = pred_mask_files[3:]
    # seg_view_batch_main.add_train_pairs(pairs_file, image_main_root)
    # seg_view_batch_main.add_gt_labelme(gt_image_files)
    # seg_view_batch_main.add_gt_labelme_json(gt_labelme_json_files)
    seg_view_batch_main.add_image_origin(image_origin_files)
    seg_view_batch_main.add_gt_mask(gt_mask_files)
    seg_view_batch_main.add_pred_mask(pred_mask_files)

    #可以认为注释掉，就可以避免删除了
    # remove_folder(save_diff_dir)

    seg_view_batch_main.save_diff_image(save_dir=save_diff_dir,
                                        is_show=True,
                                        mode='cv',
                                        num_pred_classes=num_pred_classes,
                                        pred_weight=pred_weight,
                                        num_gt_classes=num_gt_classes,
                                        gt_weight=gt_weight)

    # seg_view_batch_main.save_train_pairs(save_pairs_file, image_main_root)
    # seg_view_batch_main.save_gt_label_mask(save_label_mask_dir)
