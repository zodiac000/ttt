from file.file_list import file_list_main
from pathlib import Path
import os

class SegModifyFilenameBatch(object):
    def __init__(self):
        pass

    def modify_filename(self, path, suffixs=['png', 'jpg', 'bmp'], recursive=False, sepearate='#'):
        files = file_list_main.find_files(path, suffixs, recursive)
        for file in files:
            file_path = Path(file)
            [dataset_name, filename] = file_path.parts[-2:]
            words = filename.split('#')
            filename = words[-1]
            filename_new = dataset_name+'#'+filename
            parent = str(file_path.parent)
            file_new = os.path.join(parent, filename_new)
            os.rename(file, file_new)

seg_modify_filename_batch_main = SegModifyFilenameBatch()

if __name__ == '__main__':
    path = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train_image_label'
    image_path = os.path.join(path, 'images')
    label_path = os.path.join(path, 'labels-v0.1')
    seg_modify_filename_batch_main.modify_filename(image_path, ['jpg', 'png'],recursive=True)
    seg_modify_filename_batch_main.modify_filename(label_path, ['json'],recursive=True)