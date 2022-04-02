from file.file_list import file_list_main
import random
'''
由文件夹得到所有图像文件列表
setting in config
'''

if __name__ == '__main__':
    image_path = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train_label_rule_prev/images/DS_park_image_5'
    image_path = '/nas2/untouch_data/srcData/auto_dirve/OpenData/once/raw_cam03/raw_cam03_p9/data/000567/cam03'
    save_list_file = '/nas2/untouch_data/srcData/auto_dirve/OpenData_test/train_ljj/list/000567_cam03.txt'

    # path = '/nas/untouch_data/SrcData/wuling/zh_export/2020-10-10'
    # save_list_file = os.path.join(path, Path(path).parts[-1] + '_labelme.list')

    image_files = file_list_main.find_files(image_path, ['jpg', 'png'], recursive=True)
    # image_files = file_list_main.shuffle_and_sampling(image_files)
    image_files = file_list_main.ignore_key(image_files, '_mask.')
    image_files = sorted(image_files)
    # image_files = file_list_main.sort_filename_files(image_files)
    # image_files = file_list_main.sort_timestamp_files(image_files)
    file_list_main.write_list(save_list_file, image_files)
