import os
import cv2
import numpy as np
from perception.base.seg_group_files import SegGroupFiles
from view.base.img_view import ImgView
from pathlib import Path


class SegUnlabelDefect(SegGroupFiles):
    def __init__(self):
        super(SegUnlabelDefect, self).__init__()
        self._img_view = ImgView()
        # # 保存地址
        self._defect_pic_pathlist_str = ""

    def show_defect(self, check_ratio=0.7):
        names = self._get_names()
        self._img_view.init()
        self._img_view.set_scale(0.5)
        idx = 0
        while idx < len(names):
            name = names[idx]
            pairs = self.group_file_map[name]
            image_file = pairs["image_file"]
            img_list = self._read_img(name)
            valid_contours, low_conf_norm_area, hist_ratio = self._get_low_confidence_area(name, check_ratio)
            canvas_img = self._calc_canvas_show(img_list, valid_contours, low_conf_norm_area)
            if canvas_img is None:
                idx += 1
                continue

            self._img_view.set_image(canvas_img)
            image_file_text = '/'.join(Path(image_file).parts[-3:])
            self._img_view.show_text(point=None, text=image_file_text, color=(0, 0, 255), font_scale=2)

            idx, key = self._img_view.show(delay=0)

    def export_defect_list(self, save_dir, save_file_name='badcase', check_ratio=0.7):
        self._find_defect(check_ratio)
        os.makedirs(save_dir, exist_ok=True)
        badcase_file = os.path.join(save_dir, save_file_name + "_labelme.txt")
        with open(badcase_file, "w") as fid:
            fid.writelines(self._defect_pic_pathlist_str)

    def _find_defect(self, check_ratio=0.7):
        names = self._get_names()
        for i, name in enumerate(names):
            valid_contours, low_conf_area, hist_ratio = self._get_low_confidence_area(name, check_ratio)
            if low_conf_area > 800 or hist_ratio >= 0.35:
                value = self.group_file_map[name]
                src_path = value["image_file"]
                self._defect_pic_pathlist_str += src_path + "\n"
                print(src_path)

    def _calc_canvas_show(self, img_list, valid_contours, low_conf_norm_area):
        cv2.drawContours(img_list[1], valid_contours, -1, (0, 0, 255), 2)
        cv2.drawContours(img_list[2], valid_contours, -1, (0, 0, 255), 2)
        cv2.putText(img_list[1], "area: " + str(int(low_conf_norm_area)), (20, 90), 1, 3, (2, 0, 255), 3)

        canvas_img = np.zeros((img_list[1].shape[0] * 2, img_list[1].shape[1] * 2, 3), np.uint8)
        canvas_img[:img_list[1].shape[0], :img_list[1].shape[1], :] = img_list[0]
        canvas_img[img_list[1].shape[0]:, img_list[1].shape[1]:, :] = img_list[1]
        canvas_img[:img_list[1].shape[0], img_list[1].shape[1]:, :] = cv2.applyColorMap(255 - img_list[2],
                                                                                        cv2.COLORMAP_JET)
        return canvas_img

    def add_prob_mask(self, image_files):
        print('image origin files has ', len(image_files))
        self.name_order['prob'] = []
        for image_file in image_files:
            name = self.get_name(image_file)
            pairs = self._get_image_pair_map_extend(name)
            pairs["prob_mask_file"] = image_file
            self.name_order['prob'].append(name)

    def _read_img(self, file_name):
        value = self.group_file_map[file_name]
        src_path = value["image_file"]
        pred_path = value["pred_mask_file"]
        prob_path = value["prob_mask_file"]
        src_img = cv2.imread(src_path, -1)
        pred_img = cv2.imread(pred_path, -1)
        prob_img = cv2.imread(prob_path, -1)
        # print(src_path, pred_img, prob_path)
        img_list = [src_img, pred_img, prob_img]
        return img_list

    def _get_pair_names_list(self):
        # src_img = cv2.imread(src_path, -1)
        # pred_img = cv2.imread(pred_path, -1)
        # prob_img = cv2.imread(prob_path, -1)
        pass

    def _preprocess(self, prob_img):
        norm_img_shape = (1280, 720)
        min_defect_norm_area = 500
        down_mask_out_ration = 0.08
        img_shape = prob_img.shape[:2]
        min_defect_area = min_defect_norm_area / (norm_img_shape[0] * norm_img_shape[1]) * (img_shape[0] * img_shape[1])
        down_mask_out_area = np.array([[0, int((1 - down_mask_out_ration) * img_shape[0])],
                                       [img_shape[1], int((1 - down_mask_out_ration) * img_shape[0])],
                                       [img_shape[1], img_shape[0]],
                                       [0, img_shape[0]]])
        return down_mask_out_area, min_defect_area

    def _find_defect_contours(self, prob_img, min_defect_area, check_ratio=0.7):
        prob_img_show = prob_img.copy()
        img_shape = prob_img.shape[:2]
        norm_img_shape = (1280, 720)
        mask = np.array(prob_img_show < int(check_ratio * 255), np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.erode(mask, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        low_conf_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            # 过滤掉小面积
            if area < min_defect_area:
                continue
            # 过滤掉上部分区域
            is_in_contour = cv2.pointPolygonTest(contour, (10, 10), False)
            if (is_in_contour == 1):
                continue
            valid_contours.append(contour)
            low_conf_area += area
        low_conf_norm_area = low_conf_area / (img_shape[0] * img_shape[1]) * (norm_img_shape[0] * norm_img_shape[1])
        return valid_contours, low_conf_norm_area

    def _calculate_histgrom(self, cv_img, ratio):
        hist, _ = np.histogram(cv_img, bins=256)  # 用numpy包计算直方图
        hist_ratio = np.sum(hist[0:int(ratio * 255)]) / np.sum(hist)
        return hist, hist_ratio

    def _get_low_confidence_area(self, file_name, check_ratio=0.7):
        img_list = self._read_img(file_name)
        prob_img = img_list[2]
        down_mask_out_area, min_defect_area = self._preprocess(prob_img)
        prob_img[prob_img == 0] = 255
        valid_contours, low_conf_norm_area = self._find_defect_contours(prob_img, min_defect_area,
                                                                        check_ratio=0.7)
        _, hist_ratio = self._calculate_histgrom(prob_img, check_ratio)
        return valid_contours, low_conf_norm_area, hist_ratio


if __name__ == '__main__':
    from file.file_list import file_list_main
    from perception.config.seg_select_unlabel_defects_config import untouch_road as SelectConfig

    # 原图
    src_img_dir = SelectConfig.get("src_img_dir")
    # 预测彩图
    pred_dir = SelectConfig.get("pred_dir")
    # 预测置信度图
    prob_dir = SelectConfig.get("prob_dir")
    # 保存地址
    save_dir = SelectConfig.get("save_dir")

    image_origin_files_all = file_list_main.find_files(src_img_dir, ['png', 'jpg'], recursive=True)
    image_pred_files_all = file_list_main.find_files(pred_dir, ['png', 'jpg'], recursive=True)
    image_prob_files_all = file_list_main.find_files(prob_dir, ['png', 'jpg'], recursive=True)

    seg_pic_data = SegUnlabelDefect()
    seg_pic_data.add_image_origin(image_origin_files_all)
    seg_pic_data.add_pred_mask(image_pred_files_all)
    seg_pic_data.add_prob_mask(image_prob_files_all)

    seg_pic_data.export_defect_list(save_dir, "test_demo1", 0.7)

    seg_pic_data.show_defect()
