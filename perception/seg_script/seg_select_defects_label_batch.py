import os
import cv2
from pathlib import Path
import sys
# sys.path.insert(0, "/data8/wenbin/works/code/drive/road_scripts")
from file.file_utils import remove_folder, mkdirs
from perception.base.seg_group_files import SegGroupFiles
from perception.base.segmentation_evaluate import single_image_iou, calc_eval
from tqdm import tqdm


from pdb import set_trace

class SegDefectsLabelBatch(SegGroupFiles):
    def __init__(self, num_class, iou_threashold, debug=True):
        super(SegDefectsLabelBatch, self).__init__()
        self.num_class = num_class
        self.iou_threashold = iou_threashold
        self.defects_list = []
        self.debug = debug

    def save_defects_list(self, outfile):
        from file.file_utils import write_list
        save_dir = Path(outfile).parent
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        write_list(outfile, self.defects_list)

    def load_defects_list(self, outfile):
        defects_list = []
        with open(outfile, 'r') as file:
            lines = file.readlines()
            for line in lines:
                decode = self._decode_defect_line(line)
                if decode[0] is None:
                    continue
                defects_list.append(decode)
        self.defects_list = defects_list

    def _decode_defect_line(self, line):
        words = line.split(' ')
        if len(words) != 6:
            return [None] * 6
        image_file = words[0].strip()
        labelme_file = words[1].strip()
        pred_color_file = words[2].strip()
        road_iou = float(words[3].strip())
        sidewalk_iou = float(words[4].strip())
        background_iou = float(words[5].strip())
        return [image_file, labelme_file, pred_color_file, road_iou, sidewalk_iou, background_iou]

    def search_defects(self):
        names = self._get_names('image')
        self.defects_list = []
        pbar = tqdm(names)
        for name in pbar:
            pairs = self._get_image_pairs(name)
            image_file = pairs["image_file"]
            gt_mask, pred_mask = self._read_gt_pred_mask(pairs)
            if gt_mask is None or pred_mask is None:
                continue
            ious, diffs = self._cal_ious_diffs(gt_mask, pred_mask)
            if self._check_defect(ious, diffs):
                self._add_defects(image_file, gt_labelme_file, pred_color_file, ious)

    def sorted_by_label(self, label_name):
        sort_index = -1
        if label_name == 'road':
            sort_index = -3
        elif label_name == 'sidewalk':
            sort_index = -2
        self.defects_list = sorted(self.defects_list, key=lambda x: x[sort_index])

    def _read_gt_pred_mask(self, pairs):
        gt_labelme_file = pairs["gt_labelme_file"]
        if gt_labelme_file is None:
            return None
        gt_mask_file = pairs["gt_label_mask_file"]
        pred_mask_file = pairs["pred_mask_file"]
        pred_color_file = pairs["pred_color_file"]
        if pred_color_file is None:
            return None
        gt_mask = cv2.imread(gt_mask_file, cv2.IMREAD_UNCHANGED)
        pred_mask = cv2.imread(pred_mask_file, cv2.IMREAD_UNCHANGED)
        return gt_mask, pred_mask

    def _check_defect(self, ious, diffs):
        if self._check_defect_with_iou(ious):
            return True

        if self._check_defect_with_diff_area(diffs):
            return True
        return False

    def _check_defect_with_iou(self, iou):
        #doesn't include images without sidewalk
        if iou[1] == 0:
            return False

        for i in range(len(iou)):
            if iou[i] < self.iou_threashold[i]:
                return True
        return False

    def _check_defect_with_diff_area(self, diffs):
        #TODO
        return False

    def _add_defects(self, image_file, labelme_file, pred_color_file, ious):
        line = [image_file, labelme_file, pred_color_file]+ious
        self.defects_list.append(line)

    def _cal_ious_diffs(self, gt, pred):
        ious = single_image_iou(gt, pred, self.num_class, ignore_index=255)
        diffs = None #TODO
        return ious, diffs

    def export_images(self, save_dir, top):
        import shutil
        print("Number of defected images: {}".format(len(self.defects_list)))
        if top > len(self.defects_list):
            top = len(self.defects_list)
        pbar = tqdm(self.defects_list[:top])
        for pair in pbar:
            info = pair.split(" ")
            image = info[0]
            labelme = info[1]
            image_name = Path(image).parts[-1]
            labelme_name = Path(labelme).parts[-1]
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            shutil.copyfile(image, os.path.join(save_dir, image_name))
            shutil.copyfile(labelme, os.path.join(save_dir, labelme_name))

if __name__ == '__main__':
    from file.file_list import file_list_main
    from perception.config.seg_select_defects_label_config import untouch_road as Config

    image_dir = Config.get("image_dir", "")
    gt_mask_dir = Config.get("gt_mask_dir", "")
    gt_color_dir = Config.get("gt_color_dir", "")
    gt_labelme_dir = Config.get("gt_labelme_dir", "")
    pred_mask_dir = Config.get("pred_mask_dir", "")
    pred_color_dir = Config.get("pred_color_dir", "")
    save_dir = Config.get("save_dir", "")

    num_classes = Config.get("num_classes", 3)
    iou_thr = Config.get("iou_thr")
    area_thr = Config.get("area_thr")

    label_name = Config.get("label_name")
    top = Config.get("top_n")
    load = Config.get("load")

    seg_defects_label_batch = SegDefectsLabelBatch(num_classes, iou_thr)

    image_origin_files = file_list_main.find_files(image_dir, ['png', 'jpg'], recursive=True)
    gt_mask_files = file_list_main.find_files(gt_mask_dir, ['png', 'jpg'], recursive=True)
    gt_labelme_files = file_list_main.find_files(gt_labelme_dir, ['json'], recursive=True)
    pred_mask_files = file_list_main.find_files(pred_mask_dir, ['png', 'jpg'], recursive=True)
    pred_color_files = file_list_main.find_files(pred_color_dir, ['png', 'jpg'], recursive=True)

    image_origin_files = image_origin_files[:10]

    seg_defects_label_batch.add_image_origin(image_origin_files)
    seg_defects_label_batch.add_gt_mask(gt_mask_files)
    seg_defects_label_batch.add_pred_mask(pred_mask_files)
    seg_defects_label_batch.add_gt_labelme_json(gt_labelme_files)
    seg_defects_label_batch.add_pred_color(pred_color_files)

    seg_defects_label_batch.remove_dataset_name_diff_in_pairs()

##############################################################################

    if not load:
        #calculate ious from masks
        seg_defects_label_batch.search_defects()
        seg_defects_label_batch.sorted_by_label(label_name)
        seg_defects_label_batch.save_defects_list(os.path.join(save_dir, "../defects_labelme.txt"))
    else:
        #load ious from existing file
        seg_defects_label_batch.load_defects_list(os.path.join(save_dir, "../defects_labelme.txt"))



