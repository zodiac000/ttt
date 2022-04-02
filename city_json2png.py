# import cv2
# import json
# import glob
# import os
# import shutil
# import numpy as np
#
# color_map = [(128, 64, 128), (152, 251, 152), (70, 130, 180),
#              (111, 74, 0), (244, 35, 232), (180, 165, 180),
#              (153, 153, 153), (220, 220, 0), (220, 20, 60), (107, 142, 35), (0, 0, 0)
#              ]
#
#
# def parse_contour_seg(anno_file):
#     with open(anno_file, "r") as fid:
#         json_info = json.load(fid)
#     return json_info
#
#
# def remove_non_seg_label(json_info, laneTypeMapObj):
#     delIdxs = []
#     for i, sub_shape in enumerate(json_info["shapes"]):
#         lane_label = sub_shape["label"]
#         if lane_label in laneTypeMapObj.group_label_names:  # group_label_names = ["white dash line group", "yellow dash line group"]
#             delIdxs.append(i)
#
# def get_dirname(root_path):
#     return os.listdir(root_path)
#
#
# def fill_contour_seg(json_info):
#     seg_mask = np.zeros((1024, 2048), np.uint8)
#     # print(type(seg_mask))
#     sub_shape_arr = []
#     for sub_shape in json_info["objects"]:
#         lane_label = sub_shape["label"]
#         # if lane_label not in laneTypeMapObj.lane_name2Id.keys():
#         #     continue
#         sub_shape_arr.append(sub_shape)
#
#     for sub_shape in sub_shape_arr:
#         lane_label = sub_shape["label"]
#         # print(lane_label)
#
#         # print(clsId)
#         points = np.round(np.array(sub_shape["polygon"])).astype(np.int32)
#         if lane_label == "road":
#             clsId = 0
#         elif lane_label == "sidewalk":
#             clsId = 1
#         # elif lane_label == "unlabeled" or lane_label == "rectification border" \
#         #         or lane_label == "ego vehicle" \
#         #         or lane_label == "out of roi" \
#         #         or lane_label == "ground":
#         elif lane_label == "ego vehicle":
#             clsId = 5
#         else:
#             clsId = 15
#         #     ####
#         #     if color:
#         #         cv2.fillPoly(color_img, [points], color_map[clsId])
#         #     else:
#         #         if clsId == 10:
#         #             cv2.fillPoly(src_img_copy, [points], color_map[clsId])
#         #             clsId = 0
#         cv2.fillPoly(seg_mask, [points], clsId)
#     # if color:
#     #     return color_img, None, False
#     # else:
#     return seg_mask
#
#
# if __name__ == '__main__':
#     # 原始 json所在路径
#     json_root_path = "/nas2/untouch_data/srcData/auto_dirve/OpenData/cityscapes/cityscapes/gtFine/val/munster"
#     # mask图保存的路径
#     label_png_path = "/nas2/untouch_data/srcData/auto_dirve/OpenData/cityscapes/cityscapes/gtFine/val_mask/"
#     img_dir_list = get_dirname(json_root_path)
#     sum_lables = []
#     index = 0
#     for i, json_file_name in enumerate(img_dir_list):
#         if json_file_name.split(".")[-1] != "json":
#             continue
#         # print(json_file_name)
#         index += 1
#         json_file_path = os.path.join(json_root_path, json_file_name)
#         json_info = parse_contour_seg(json_file_path)
#         # print(json_info)
#
#         for sub_shape in json_info["objects"]:
#             sum_lables.append(sub_shape["label"])
#
#         seg_mask = fill_contour_seg(json_info)
#         # print(type(seg_mask))
#         # cv2.imshow("11", seg_mask)
#         # cv2.waitKey(0)
#
#         label_name = json_file_name.replace("_gtFine_polygons.json", "_leftImg8bit.png")
#         # print(label_name)
#
#         city_name = json_root_path.split("/")[-1]
#         # label_mask_prediction
#         os.makedirs(os.path.join(label_png_path, city_name, "label_mask_prediction"), exist_ok=True)
#         cv2.imwrite(os.path.join(label_png_path, city_name, "label_mask_prediction", label_name), seg_mask)
#         print(index, os.path.join(label_png_path, city_name, label_name))
#         # break
import cv2
import json
import glob
import os
import shutil
import numpy as np

color_map = [(128, 64, 128), (152, 251, 152), (70, 130, 180),
             (111, 74, 0), (244, 35, 232), (180, 165, 180),
             (153, 153, 153), (220, 220, 0), (220, 20, 60), (107, 142, 35), (0, 0, 0)
             ]


def parse_contour_seg(anno_file):
    with open(anno_file, "r") as fid:
        json_info = json.load(fid)
    return json_info


def remove_non_seg_label(json_info, laneTypeMapObj):
    delIdxs = []
    for i, sub_shape in enumerate(json_info["shapes"]):
        lane_label = sub_shape["label"]
        if lane_label in laneTypeMapObj.group_label_names:  # group_label_names = ["white dash line group", "yellow dash line group"]
            delIdxs.append(i)


def get_dirname(root_path):
    return os.listdir(root_path)


cityscapes_lable = []


def fill_contour_seg(json_info):
    seg_mask = np.zeros((1024, 2048), np.uint8)
    # print(type(seg_mask))
    sub_shape_arr = []
    for sub_shape in json_info["objects"]:
        lane_label = sub_shape["label"]
        # if lane_label not in laneTypeMapObj.lane_name2Id.keys():
        #     continue
        sub_shape_arr.append(sub_shape)

    for sub_shape in sub_shape_arr:
        lane_label = sub_shape["label"]
        # print(lane_label)

        # print(clsId)
        clsId = 0
        points = np.round(np.array(sub_shape["polygon"])).astype(np.int32)
        if lane_label == "road":
            clsId = 0
        elif lane_label == "sidewalk":
            clsId = 1
        elif lane_label == "parking":
            clsId = 28
        elif lane_label == "ego vehicle" or lane_label == "ground" or lane_label == "rectification border":
            clsId = 29
        else:
            clsId = 30
        #     ####
        #     if color:
        #         cv2.fillPoly(color_img, [points], color_map[clsId])
        #     else:
        #         if clsId == 10:
        #             cv2.fillPoly(src_img_copy, [points], color_map[clsId])
        #             clsId = 0
        cv2.fillPoly(seg_mask, [points], clsId)
    # if color:
    #     return color_img, None, False
    # else:
    return seg_mask


if __name__ == '__main__':
    # 原始 json所在路径
    city_list=['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf',
               'erfurt', 'hamburg', 'hanover', 'jena', 'krefeld', 'monchengladbach',
               'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']
    # city_list = sorted(os.listdir("/nas2/untouch_data/srcData/auto_dirve/OpenData/cityscapes/cityscapes/gtFine/train"))
    # print(city_list)
    json_root_path = "/nas2/untouch_data/srcData/auto_dirve/OpenData/cityscapes/cityscapes/gtFine/val/lindau"
    # mask图保存的路径
    label_png_path = "/nas2/untouch_data/srcData/auto_dirve/OpenData/cityscapes/cityscapes/gtFine/val_mask/"
    # json_root_path = "/nas2/untouch_data/srcData/auto_dirve/OpenData/cityscapes/cityscapes/gtFine/train/darmstadt"
    # mask图保存的路径
    # label_png_path = "/nas2/untouch_data/srcData/auto_dirve/OpenData/cityscapes/cityscapes/gtFine/train_mask/"
    img_dir_list = get_dirname(json_root_path)
    sum_lables = []
    index = 0
    for i, json_file_name in enumerate(img_dir_list):
        if json_file_name.split(".")[-1] != "json":
            continue
        print(json_file_name)
        index += 1
        json_file_path = os.path.join(json_root_path, json_file_name)
        json_info = parse_contour_seg(json_file_path)
        # print(json_info)

        for sub_shape in json_info["objects"]:
            sum_lables.append(sub_shape["label"])
        seg_mask = fill_contour_seg(json_info)
        # print(type(seg_mask))
        # cv2.imshow("11", seg_mask)
        # cv2.waitKey(0)

        label_name = json_file_name.replace("_gtFine_polygons.json", "_leftImg8bit.png")
        # print(label_name)
        city_name = json_root_path.split("/")[-1]
        # label_mask_prediction
        os.makedirs(os.path.join(label_png_path, city_name, "label_mask_prediction"), exist_ok=True)
        cv2.imwrite(os.path.join(label_png_path, city_name, "label_mask_prediction", label_name), seg_mask)
        print(index, os.path.join(label_png_path, city_name, label_name))
    #     # break
