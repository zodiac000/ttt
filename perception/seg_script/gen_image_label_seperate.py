import os
from file.file_list import file_list_main
from file.json_tool import load_json,save_json
from file.file_utils import mkdirs
origin_dir = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train_label_rule_prev/images'

folder_infos = os.listdir(origin_dir)
for folder in folder_infos:
    if folder != 'once_raw03_9':
        continue
    image_path = os.path.join(origin_dir, folder)
    if not os.path.isdir(image_path):
        continue
    files = os.listdir(image_path)

    save_json_path = origin_dir+'/../labels_prev/'+folder
    mkdirs(save_json_path)
    for file in files:
        if '_labelme.json' not in file:
            continue
        json_file = os.path.join(image_path, file)
        json_map = load_json(json_file)
        save_json_file = os.path.join(save_json_path, file)
        save_json(save_json_file, json_map)