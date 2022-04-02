from file.file_list import file_list_main
import os
from pathlib import Path
from file.file_utils import copyfile, mkdirs

if __name__ == '__main__':
    folder = '000523'
    list_file = '/nas2/untouch_data/srcData/auto_dirve/OpenData_test/train_ljj/bad case/raw_cam09_p2/'+folder+'/image_files_labelme.list'
    dst_path = '/nas2/untouch_data/srcData/auto_dirve/OpenData_test/sfnet_res18/once_badcase/cam_09/raw_cam09_p2'
    dst_path = os.path.join(dst_path, folder)

    image_files = file_list_main.read_list(list_file)

    if len(image_files) == 0:
        exit()
    mkdirs(dst_path)
    for image_file in image_files:
        dst_file = os.path.join(dst_path, Path(image_file).name)
        copyfile(image_file, dst_path)
