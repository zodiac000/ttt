import cv2
import json
import glob
import os
import shutil
import numpy as np
from tqdm import tqdm

class SegLabelTool():
    def __init__(self):        
        self.background_trainId = 2
        self.ignore_frame_flag = ["ignore"]
        self.false_label_remap = {
            "ingore":"ignore",
            "road block":"roadblock", 
            "road_block":"roadblock", 
            "roadblockgroup":"roadblock", 
            "road block group":"roadblock", 
            }
        self.label_2_trainID = {
            "freespace": 0,
            "road": 0,
            "sidewalk": 1,
            "roadblock": 2,
            "parking":255,
            "void": 255,
            "ground": 255}     
        # 处理遮挡关系，大ID覆盖小ID
        self.sort_anno_names = ['__background__', 'freespace', 'road', 'sidewalk', 'parking', 'guard rail',
                           'road block', 'roadblock', 'road_block', 'roadblockgroup','road block group',
                           'void', 'ground', 'ignore',
                           'patch', 'road edge', 'road edge low', 'white  dash line',
                           'white solid line', 'yellow dash line', 'yellow solid line']
        self.sortAnno_2_labelId_map = self.gen_anno_labelId_map()  
        
        self.color_map = self.get_color_map_list(256)

    def get_color_map_list(self, num_classes):
        """ Returns the color map for visualizing the segmentation mask,
            which can support arbitrary number of classes.
        Args:
            num_classes: Number of classes
        Returns:
            The color map
        """
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
        color_map = np.array(color_map).astype("uint8")
        
        color_map[0,:] = (128, 64, 128)
        color_map[1,:] = (244, 35, 232)
        color_map[2,:] = (70, 70, 70)

        return color_map



        
    def gen_anno_labelId_map(self):
        self.lane_class_len = len(self.sort_anno_names)  # 种类
        self.lane_class_len_idx = np.arange(self.lane_class_len).tolist()
        cc = zip(self.sort_anno_names, self.lane_class_len_idx)
        anno_2_labelId = dict(cc)
        return anno_2_labelId

    def show_classID(self):
        print("类别 ID: \n", self.label_2_trainID)    

    def check_ignore(self, json_info):    
        for sub_shape in json_info["shapes"]:
            if sub_shape["label"] in self.ignore_frame_flag:
                return True
        return False
    def remap_label(self,json_info):
        for sub_shape in json_info["shapes"]:
            if sub_shape["label"] in self.false_label_remap:
                sub_shape["label"] = self.false_label_remap[sub_shape["label"]]
    
    def sort_anno_bylabelId(self, shape_arr):
        labelIds = [self.sortAnno_2_labelId_map[sub_shape_arr["label"]] for sub_shape_arr in shape_arr]
        sortIdx  = sorted(range(len(labelIds)), key=lambda x: labelIds[x])
        shape_arr = [shape_arr[Idx] for Idx in sortIdx]
        return shape_arr
                    
    def fill_contour_seg(self, cvImg, json_info, color=False):
        seg_label = np.zeros(cvImg.shape[:2], np.uint8)
        label_mask = np.zeros(cvImg.shape[:2], np.uint8)
        sub_shape_arr = []
        for sub_shape in json_info["shapes"]:
            lane_label = sub_shape["label"]
            if lane_label not in self.label_2_trainID.keys():
                continue
            sub_shape_arr.append(sub_shape)
        sub_shape_arr = self.sort_anno_bylabelId(sub_shape_arr)

        for sub_shape in sub_shape_arr:
            lane_label = sub_shape["label"]
            if lane_label not in self.label_2_trainID.keys():
                continue
            clsId = self.label_2_trainID[lane_label]
            points = np.round(np.array(sub_shape["points"])).astype(np.int32)
            cv2.fillPoly(seg_label, [points], clsId)
            cv2.fillPoly(label_mask, [points], 1)
            
        background_mask = (1-label_mask).astype(np.bool)
        seg_label[background_mask] = self.background_trainId
        
        if color:
            # cv2.fillPoly(pesuo_color_img, [points], self.color_map[clsId].tolist())
            c1 = cv2.LUT(seg_label, self.color_map[:, 0])
            c2 = cv2.LUT(seg_label, self.color_map[:, 1])
            c3 = cv2.LUT(seg_label, self.color_map[:, 2])
            pseudo_img = np.dstack((c1, c2, c3))

            color_img = cv2.addWeighted(cvImg, 0.5, pseudo_img, 0.5, 0)
            return seg_label, color_img
        else:
            return seg_label, None

def parse_contour_seg(anno_file):
    with open(anno_file, "r") as fid:
        json_info = json.load(fid)
    return json_info

def gen_pair_list(path, label_posfix="_labelme.json", suffixs=['png', 'jpg', 'bmp']):
    imgfiles = []
    pair_img_files = []
    pair_label_files = []
    for suffix in suffixs:
        path_re = path + '/*.' + suffix
        imgfiles.extend(glob.glob(path_re))
    
    for sub_img_file in imgfiles:
        img_type = sub_img_file[sub_img_file.rfind("."):]
        label_file = sub_img_file.replace(img_type, label_posfix)
        if not os.path.exists(label_file):
            continue
        pair_img_files.append(sub_img_file)
        pair_label_files.append(label_file)
    return zip(pair_img_files, pair_label_files)

def gather_labels(json_info, sum_lables):
    for sub_shape in json_info["shapes"]:
        if sub_shape["label"] not in sum_lables:
            sum_lables[sub_shape["label"]] = 1
        else:
            sum_lables[sub_shape["label"]] += 1

if __name__ == '__main__':
    pass
    img_json_root_path = "/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train"    
    dst_root         = "/data7/zlh/Adatasets/auto_dirve/segDet/datasets/20220228_train"
    dst_img_path     = "%s/images/"%dst_root
    dst_label_path   = "%s/labels/"%dst_root
    label_color_path = "%s/color_labels/"%dst_root

    SegLabelToolObj = SegLabelTool()
    SegLabelToolObj.show_classID()
    
    pair_lists = gen_pair_list(img_json_root_path+"/**", label_posfix="_labelme.json")

    sum_lables = {}
    for i, pair_arr in tqdm(enumerate(pair_lists)):
        img_file  = pair_arr[0]
        json_file = pair_arr[1]

        sub_folder_name = os.path.split(img_file)#[0].split("/")[-1]
        basename = os.path.basename(img_file)
        frame_id = basename[:basename.rfind('.')]

        # if frame_id != "1637745824.015319109":
        #     continue
        cv_img = cv2.imread(img_file)
        json_info = parse_contour_seg(json_file)
        gather_labels(json_info, sum_lables)
        
        SegLabelToolObj.remap_label(json_info)

        if(SegLabelToolObj.check_ignore(json_info)):
            continue

        seg_mask, render_img = SegLabelToolObj.fill_contour_seg(cv_img, json_info, True)
        
        # cv2.imwrite("render_img.png", render_img)

        save_dir_img_path = os.path.join(dst_img_path, sub_folder_name)
        save_dir_label_color_path = os.path.join(label_color_path, sub_folder_name)
        save_dir_label_png_path = os.path.join(dst_label_path, sub_folder_name)

        os.makedirs(save_dir_img_path, exist_ok=True)
        os.makedirs(save_dir_label_color_path, exist_ok=True)
        os.makedirs(save_dir_label_png_path, exist_ok=True)

        dst_img_name = os.path.join(save_dir_img_path, frame_id + ".png")
        dst_label_name = os.path.join(save_dir_label_png_path, frame_id + ".png")
        dst_color_name = os.path.join(save_dir_label_color_path, frame_id + ".jpg")
        # print(img_final_name)

        cv2.imwrite(dst_label_name, seg_mask)
        cv2.imwrite(dst_color_name, render_img)
        cv2.imwrite(dst_img_name, cv_img)
