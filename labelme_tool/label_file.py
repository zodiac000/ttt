import base64
import io
import json
import os.path as osp

class LabelFileError(Exception):
    pass


class LabelFile(object):

    suffix = '_labelme.json'

    def __init__(self, filename=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        self.imageWidth = None
        self.imageHeight = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    def load(self, filename):
        keys = [
            'version',
            'imageData',
            'imagePath',
            'shapes',  # polygonal annotations
            'flags',   # image level flags
            'imageHeight',
            'imageWidth',
        ]
        shape_keys = [
            'label',
            'points',
            'group_id',
            'shape_type',
            'flags',
            'other_data',
        ]
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            version = data.get('version')
            # if version is None:
            #     print(
            #         'Loading JSON file ({}) of unknown version'
            #         .format(filename)
            #     )
            flags = data.get('flags') or {}
            imageWidth = data.get('imageWidth')
            imageHeight = data.get('imageHeight')
            imagePath = data['imagePath']
            shapes = [
                dict(
                    label=s['label'],
                    points=s['points'],
                    shape_type=s.get('shape_type', 'polygon'),
                    flags=s.get('flags', {}),
                    group_id=s.get('group_id'),
                    other_data={
                        k: v for k, v in s.items() if k not in shape_keys
                    }
                )
                for s in data['shapes']
            ]
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.filename = filename
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.otherData = otherData

    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # os.chmod(filename, 0o777)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix
