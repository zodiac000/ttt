#/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import shutil
import glob
from pathlib import Path
import numpy as np

def remove_folder(dir):
    if not os.path.exists(dir):
        return
    shutil.rmtree(dir)

def mkdirs(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
        return True
    return False

def copyfile(src_file, dst_dir):
    shutil.copy(src_file, dst_dir)

def movefile(src_file, dst_dir):
    shutil.move(src_file, dst_dir)

def copyfolder(src, dst_dir):
    file_names = glob.glob(src)

    for file in file_names:
        copyfile(file, dst_dir)

def create_last_folders(root_dir, full_file, num_last_folder):
    '''
    新建多级文件夹
    以full_file的最后几级目录，加上root_dir进行新建多级文件夹
    root_dir + Path(full_file).parts(-num_last_folder)
    '''
    path = os.path.join(root_dir, '/'.join(Path(full_file).parts[-num_last_folder-1:-1]))
    mkdirs(path)
    return path

def change_file_suffix(filename, suffix):
    '''
    :param filename: 针对_labelme.json, _facepp.json做了处理
    :param suffix: [.json]. Note:has .
    :return:
    '''
    file = filename
    path = Path(file)
    stem = str(path.stem)
    ori_suffix = str(path.suffix)
    if ori_suffix == '.json':
        sub_suffix = stem.split('_')[-1]
        if sub_suffix == 'labelme' or sub_suffix == 'facepp' or sub_suffix == 'standard':
            stem = stem[:-len(sub_suffix)-1]
    parent = str(path.parent)
    if parent == '.':
        return stem + suffix
    return parent + '/' + stem + suffix

def change_file_to_image(filename, not_only_has_file=False):
    img_file = change_file_suffix(filename, '.png', not_only_has_file)
    if not os.path.isfile(img_file):
        img_file = change_file_suffix(filename, '.jpg', not_only_has_file)
    return img_file

def read_pt(file, skip_row=0):
    if not os.path.isfile(file):
        return None
    data = np.loadtxt(file, comments='#', delimiter=' ', unpack=False, skiprows=skip_row)
    return data.tolist()

def write_pt(file, data, fmt='%.18e'):
    np.savetxt(file, np.array(data), fmt=fmt)

def write_list(file_name, list_data, separator= ' ', model='create', is_sub=False):
    line = ''
    for data in list_data:
        if isinstance(data, (list, tuple)):
            line += write_list(file_name, data, separator, model, is_sub=True) + '\n'
        else:
            line += str(data) + separator

    if is_sub is False:
        if model == 'create':
            fp = open(file_name, 'w')
        elif model == 'app':
            fp = open(file_name, 'a')
        fp.writelines(line)
    return line
