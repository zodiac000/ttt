import os
import cv2
from pathlib import Path
from perception.base.seg_view import seg_view_main
from file.file_utils import remove_folder, mkdirs
from perception.base.seg_group_files import SegGroupFiles

class LaneViewBatch(SegGroupFiles):
    def __init__(self):
        super(LaneViewBatch, self).__init__()

    def _save_image_list(self, save_dir, image_file = None):
        image_list_file = os.path.join(save_dir, 'image_files_labelme.list')
        if image_file is None:
            mkdirs(save_dir)
            with open(image_list_file, 'w') as fp:
                fp.write('')
        else:
            with open(image_list_file, 'a') as fp:
                fp.write(image_file+'\n')

lane_view_batch_main = LaneViewBatch()

if __name__ == '__main__':
    from file.file_list import file_list_main
    from file.save_workspace import workspace_main

    save_diff_dir = '/nas2/untouch_data/srcData/auto_dirve/OpenData_test/train_ljj/diff_view'
    image_origin_dir = '/nas2/auto_drive/OpenData/Curvelanes/tuneresult/train/curvelanes_conditionlane'
