#/usr/bin/env python
# -*- coding: UTF-8 -*-
from labelme_tool.label_file import LabelFile
from file.file_utils import *
from PIL import Image
import copy

from labelme_tool.label_file_json import labelFile_shape_head

class LabelmeTool(object):
    def __init__(self):
        self.labelme_suffix = '_labelme.json'

    def load_labelfile(self, label_file, full_path=None):
        if not os.path.isfile(label_file):
            return None
        if os.path.getsize(label_file) == 0:
            os.remove(label_file)
            return None
        labelFile = LabelFile(label_file)
        if full_path is not None:
            labelFile.imagePath = os.path.join(full_path, labelFile.imagePath)
        return labelFile

    def create_labelfile(self, image_path, labelme_file=None):
        im = Image.open(image_path)
        width, height = im.size
        labelFile = LabelFile()
        if labelme_file is None:
            labelFile.filename = change_file_suffix(image_path, self.labelme_suffix)
        else:
            #check if the given labelme suffix is valid
            if self.labelme_suffix not in labelme_file:
                print("suffix is not in the correct format of '_labelme.json'")
                return
            labelFile.filename = labelme_file
        labelFile.imagePath = Path(image_path).parts[-1]
        labelFile.imageData = None
        labelFile.imageHeight = height
        labelFile.imageWidth = width
        labelFile.shapes = []
        return labelFile

    def create_shape(self):
        return copy.deepcopy(labelFile_shape_head[0])

    def add_shape(self, labelFile, label, shape_type, points, group_id=None):
        if not isinstance(points, (list)):
            return
        shape = self.create_shape()
        shape['label'] = label
        shape['shape_type'] = shape_type
        shape['points'] = points

        if group_id is None:
            group_id = 0
            for shape in labelFile.shapes:
                group_id = max(shape['group_id'], group_id)
            group_id += 1
        shape['group_id'] = group_id

        labelFile.shapes.append(shape)

    def save_label_file(self, labelFile):
        labelFile.save(labelFile.filename,
                       labelFile.shapes,
                       labelFile.imagePath,
                       labelFile.imageHeight,
                       labelFile.imageWidth)

    def _read_label_file(self, image_file):
        # 如果good=1，则后面转换不会更新该labelme文件
        labelme_file = change_file_suffix(image_file, self.labelme_suffix)
        if not os.path.isfile(labelme_file):
            labelFile = self.create_labelfile(image_file)
        else:
            labelFile = self.load_labelfile(labelme_file)
        labelFile.imagePath = image_file
        return labelFile

    def to_labelme_file(self, filename, parent_dir=None):
        if parent_dir is None:
            return change_file_suffix(filename, self.labelme_suffix)
        else:
            pathfile = Path(filename)
            labelme_file = os.path.join(parent_dir,
                                        pathfile.stem+self.labelme_suffix)
            return labelme_file

    def to_filename(self, json_file):
        return str(Path(json_file).name)[:-len(self.labelme_suffix)]

    def image_fullpath_to_name_in_labelmefile(self, labelme_file):
        labelFile = self.load_labelfile(labelme_file)
        labelFile.imagePath = Path(labelFile.imagePath).parts[-1]
        self.save_label_file(labelFile)

labelme_tool_main = LabelmeTool()
