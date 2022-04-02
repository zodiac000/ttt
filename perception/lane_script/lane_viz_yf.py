import cv2, stat
import numpy as np
import matplotlib.pyplot as plt
import json
import PIL
import PIL.Image
import PIL.ImageDraw
#from tools.condlanenet.common import COLORS
import glob
import os
from tqdm import tqdm


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]


def lane_load_viz_data(src_img_dir, gt_dir, pre_dir, dst_dir, scale,save_file=True,pred_suffixs=".json"):

    os.makedirs(dst_dir, exist_ok=True)
    gt_filelist = gen_list(gt_dir, suffixs=["json"])
    is_vis_gt = True
    is_vis_valid = False

    for i in range(len(gt_filelist)):
        sub_prefile = gt_filelist[i].replace("\\","/")
        frame_id = sub_prefile.split("/")[-1][:-5]
        gt_file = os.path.join(gt_dir,frame_id,".json")
        gt_file = gt_file[:-6] + gt_file[-5:]
        pre_file = os.path.join(pre_dir, frame_id, pred_suffixs)
        pre_file = pre_file[:-11] + pre_file[-4:]

        img_file = os.path.join(src_img_dir, frame_id.replace(".lines","") + ".jpg")
        cvImg = cv2.imread(img_file)
        cvImg_show = cvImg.copy()

        if os.path.exists(gt_file) and is_vis_gt:  # 可视化gt
            if os.path.exists(pre_file):
                if pre_file[-4:] == "json":
                    cvImg_show = vis_one_for_demo(gt_anno, pre_file, lane_width=2, color=(0, 0, 255))
                else:
                    pre_anno = lane_load_labels(pre_file)
                    cvImg_show = vis_one_for_demo(pre_anno, cvImg_show, lane_width=2, color=(0, 0, 255))
            gt_anno = lane_load_labels(gt_file)
            cvImg_show = vis_one_for_demo(gt_anno, cvImg_show, lane_width=2, color=(255, 0, 0))
        # if os.path.exists(pre_file) and is_vis_valid:  #可视化预测图
        #     pre_anno = lane_load_labels(pre_file)
        #     cvImg_show = vis_one_for_demo(pre_anno,cvImg_show,lane_width=2,color=(0,0,255))
        #
        # if os.path.exists(pre_file) and is_vis_valid:  #可视化
        #     pre_anno = lane_load_labels(pre_file)
        #     cvImg_show = vis_one_for_demo(pre_anno,cvImg_show,lane_width=2,color=(0,0,255))
            h, w = cvImg.shape[0], cvImg.shape[1]
            ori_img = cvImg.copy()
            cvImg = cv2.resize(ori_img,(int(w * scale), int(h * scale)))
            cvImg_show = cv2.resize(cvImg_show,(int(w * scale), int(h * scale)))
            pad_img = np.concatenate((cvImg, cvImg_show), axis=1)
            dst_file = os.path.join(dst_dir, frame_id[:-6] + ".jpg")
            if save_file is True:
                cv2.imwrite(dst_file, pad_img)
            cv2.imshow(dst_file,pad_img)
            cv2.waitKey(0)
            os.chmod(dst_file, stat.S_IRWXO + stat.S_IRWXG + stat.S_IRWXU)

def lane_load_labels(label_file):
    with open(label_file, 'r') as fid:
        strs = fid.read()
        data  = json.loads(strs)  #以字典形式加载坐标
    lanes = []
    for line in data.values():
        for i in range(len(line)):  #遍历所有线条
            arr = line[i]
            lanes_0 = []
            for j in range(len(arr)):  #获取当前线条标注点
                #lane_pts = [tuple(float(value)) for value in arr[j].values()]
                lane_pts = []
                for k in arr[j].values():  #获取线上坐标（y,x）
                    lane_pts.append(k)
                lane_pts.reverse()
                lane_pt = tuple(list(map(float,lane_pts))) #转换数据类型
                lanes_0.append(lane_pt)
            lanes_0 = tuple(lanes_0)  #将每条标注线转化为tuple
            # lane_pts = [tuple([float(arr[i]), float(arr[i+1])]) for i in range(0,len(arr),2)]
            lanes.append(lanes_0)  #添加每条标注线到list中
    return lanes


def vis_one_for_demo(results,
                    cvImg,
                    lane_width=11,
                    color=(255,0,0)):
    img_pil = PIL.Image.fromarray(cvImg)
    #for idx, pred_lane in enumerate(results):
    for i in range(len(results)):  #获取每一条标注线
        PIL.ImageDraw.Draw(img_pil).line(
                xy=results[i], fill=color, width=lane_width)  #将标注线条画到对应数据中。
        img = np.array(img_pil, dtype=np.uint8)
    return img 

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

def gen_list(path, suffixs=['txt']):
    filelists = []
    for suffix in suffixs:
        path_re = path + '/*.' + suffix
        filelists.extend(glob.glob(path_re))
    return filelists


if __name__ == '__main__':

    #原数据路径
    src_img_dir = "F:/learn/python/lane_proj/auto_drive/OpenData/Curvelanes/valid/images/"
    #标签路径
    gt_dir      = "F:/learn/python/lane_proj/auto_drive/OpenData/Curvelanes/valid/labels/"
    #预测图路径
    pre_dir     = "F:/learn/python/lane_proj/auto_drive/OpenData/Curvelanes/valid/images/"
    #润色图路径
    dst_dir     = "F:/learn/python/lane_proj/auto_drive/OpenData/Curvelanes/valid/curvelanes_render"

    lane_load_viz_data(src_img_dir, gt_dir, pre_dir, dst_dir,scale=0.28,save_file=True,pred_suffixs=".json")

