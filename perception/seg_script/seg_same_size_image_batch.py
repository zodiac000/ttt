import cv2
import os
from pathlib import Path
from file.file_utils import remove_folder, mkdirs
from file.file_list import file_list_main
from image_alg.image_base import resize

def save_list(save_list_file, image_files):
    file_list_main.write_list(save_list_file, image_files)

def save_from_resize_ratio(image_files, output_dir, size=(1920, 1080)):
    resize_image_files = []
    for image_file in image_files:
        image = cv2.imread(image_file)
        (h, w) = image.shape[:2]
        fx = max(size[0]/w, size[1]/h)
        image = resize(image, fx, fx)
        output_file = os.path.join(output_dir, Path(image_file).parts[-1])
        resize_image_files.append(output_file)
        cv2.imwrite(output_file, image)
    return resize_image_files

if __name__ == '__main__':
    image_dir = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/lane_dataset/Multi_Park_data_collection/all/other_park_ad_week_4'
    save_path = '/data8/ljj/code/drive/lane/lane_proj/contrib/CityscapesSOTA/data/'
    save_list_file = os.path.join(save_path, 'lists/2022-1-5.txt')
    output_dir = os.path.join(save_path, 'images/2022-1-5')

    remove_folder(output_dir)
    mkdirs(output_dir)

    image_files = file_list_main.find_files(image_dir, ['png', 'jpg'], recursive=True)
    image_files = file_list_main.sort_filename_files(image_files)
    # image_files = file_list_main.sort_timestamp_files(image_files)

    resize_image_files = save_from_resize_ratio(image_files, output_dir)
    file_list_main.write_list(save_list_file, resize_image_files)

