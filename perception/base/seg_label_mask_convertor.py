import os
import cv2
from image_alg.mask_contours import mask_to_contour
from geometry.shapely_tool import polygon_extend
from labelme_tool.labelme_tool import labelme_tool_main
from perception.base.load_cityscapes_config import CityScapesConfig
import copy
import numpy as np

class SegLabelMaskConvertor(object):
    def __init__(self):
        self.cityscapes_config = CityScapesConfig()

    def label_mask_file_to_labelme(self, label_mask_file, image_file, labelme_file=None, precision='int'):
        label_mask = cv2.imread(label_mask_file, 0)
        return self.label_mask_to_labelme(label_mask, image_file, labelme_file, precision)

    def label_mask_to_labelme(self, label_mask, image_file, labelme_file=None, precision='int'):
        num_classes = np.max(label_mask)
        labelFile = labelme_tool_main.create_labelfile(image_file, labelme_file=labelme_file)
        labelFile.shapes = []
        for train_id in range(num_classes):
            mask = (label_mask == train_id) * 255
            mask = mask.astype(np.uint8)
            if not self.cityscapes_config.is_keep_trainId(train_id):
                continue
            label = self.cityscapes_config.get_name(train_id)
            if label is None:
                continue
            contours = mask_to_contour(mask, min_area=300.0)
            for group_id, contour in enumerate(contours):
                if contour.shape[0] < 5:
                    continue
                contour = contour.reshape((-1, 2)).tolist()
                if precision == 'float':
                    contour = polygon_extend(contour, 0.5)
                labelme_tool_main.add_shape(labelFile,
                                            label,
                                            shape_type='polygon',
                                            points=contour,
                                            group_id=group_id)
        labelme_tool_main.save_label_file(labelFile)

    def label_mask_from_labelme_base(self, image_file=None, labelme_file=None):
        labels_map, img_shape = self.label_from_labelme(image_file, labelme_file)
        label_mask = self.dataset_label_to_label_mask(img_shape, labels_map, "base")

        return label_mask

    def label_from_labelme(self, image_file=None, labelme_file=None):
        if labelme_file is None:
            labelme_file = labelme_tool_main.to_labelme_file(image_file)
        labelFile = labelme_tool_main.load_labelfile(labelme_file)
        width = labelFile.imageWidth
        height = labelFile.imageHeight
        img_shape = [height, width]
        labels_map = {}
        for shape in labelFile.shapes:
            if shape['shape_type'] != 'polygon':
                print('[warning]:shape_type[', shape['shape_type'], '] is not polygon')
                continue
            pts = np.round(np.array(shape["points"])).astype(np.int32)
            label = shape['label']
            label = self.cityscapes_config.replace_name(label)
            if self.cityscapes_config.is_ignore_all_labels(label):
                return None, None

            if not self.cityscapes_config.check_name(label):
                print(label, ' is not in config')
                continue

            if label not in labels_map:
                labels_map[label] = []
            labels_map[label].append(pts)
        return labels_map, img_shape

    def dataset_label_to_label_mask(self, img_shape, labels_map, dataset_name):
        if labels_map is None:
            return None
        width = img_shape[1]
        height = img_shape[0]
        background_id = self.cityscapes_config.get_background_id()
        label_mask = np.ones((height, width), np.uint8) * background_id

        seg_cover_order_list = self.cityscapes_config.get_seg_cover(dataset_name)
        if len(seg_cover_order_list) == 0:
            seg_cover_order_list = self.cityscapes_config.get_seg_cover("base")

        label_mask = self.label_to_label_mask_order(label_mask, labels_map, seg_cover_order_list)
        return label_mask

    def label_to_label_mask_order(self, label_mask, labels_map, seg_cover_order_list):
        background_id = self.cityscapes_config.get_background_id()
        for label in seg_cover_order_list:
            if label not in labels_map:
                continue
            ptss = labels_map[label]
            if self.cityscapes_config.check_name(label):
                    cv2.fillPoly(label_mask, ptss, self.cityscapes_config.get_trainId(label))
        for label, ptss in labels_map.items():
            if label in seg_cover_order_list:
                continue
            trainId = self.cityscapes_config.get_trainId(label)
            if trainId == 255:
                continue
            cv2.fillPoly(label_mask, ptss, background_id)
        for label, ptss in labels_map.items():
            trainId = self.cityscapes_config.get_trainId(label)
            if trainId != 255:
                continue
            cv2.fillPoly(label_mask, ptss, 255)
        return label_mask

    def _to_polyline(self, shape, contour, label, group_id):
        shape['label'] = label
        shape['points'] = contour
        shape['group_id'] = group_id
        shape['shape_type'] = 'polygon'
        return shape

seg_label_mask_convertor_main = SegLabelMaskConvertor()

if __name__ == '__main__':
    from file.file_list import file_list_main
    from pathlib import Path
    from tqdm import tqdm
    from file.file_utils import remove_folder, mkdirs
    seg_label_mask_convertor = SegLabelMaskConvertor()

    test_data_name = 'mark_output'   #'2022-1-12'
    #path = '/data8/ljj/code/drive/lane/lane_proj/contrib/CityscapesSOTA'
    path = '/data8/tjk'
    image_list = os.path.join(path, 'data/lists/', test_data_name+'.txt')
    label_mask_dir = os.path.join(path, 'output/', test_data_name, 'label_mask_prediction')

    image_files = file_list_main.read_list(image_list)
    # image_files = image_files[325:]
    for image_file in tqdm(image_files):
        image_name = Path(image_file).parts[-1]
        label_mask_file = os.path.join(label_mask_dir, os.path.splitext(image_name)[0] + ".png")
        if not os.path.exists(label_mask_file):
            continue
        seg_label_mask_convertor_main.label_mask_file_to_labelme(label_mask_file, image_file)

        # seg_label_mask_convertor_main.from_labelme(image_file)
