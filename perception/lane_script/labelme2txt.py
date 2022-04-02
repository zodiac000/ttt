import os
from perception.base.seg_group_files import SegGroupFiles
from file.json_tool import load_json
from file.file_utils import write_list
from labelme_tool.labelme_tool import labelme_tool_main

class Labelme2txtBatch(SegGroupFiles):
    def __init__(self):
        super(Labelme2txtBatch, self).__init__()
        self.txt_suffix = ".txt"


    def save_curvelane_txt(self, labelme_txt_dir):
        names = self._get_names('gt')
        for name in names:
            pairs = self._get_image_pairs(name)
            gt_labelme_file = pairs['gt_labelme_file']

            labelFile = labelme_tool_main.load_labelfile(gt_labelme_file)
            lines = []
            for shape in labelFile.shapes:
                if shape['label'] == 'line':
                    points = []
                    for point in shape['points']:
                        points.append(str(point[0]))
                        points.append(str(point[1]))
                    lines.append(points)
            parent_dir, outfile_fullstem = self.get_parent_fullstem(labelme_txt_dir, name)
            outfile = outfile_fullstem + self.txt_suffix
            if not os.path.isdir(parent_dir):
                os.makedirs(parent_dir)

            write_list(outfile, lines)
            os.system("chmod -R 777 {}".format(labelme_txt_dir))

labelme2txt_main = Labelme2txtBatch()

if __name__ == "__main__":
    from file.file_list import file_list_main
    labelme_json_path = "/data4/wb/lane_detection/multi_data/json_labelme"
    curvelane_txt_path = "/data4/wb/lane_detection/multi_data/curvelane_txt"

    labelme_json_files = file_list_main.find_files(labelme_json_path, ['json'], recursive=True)

    labelme2txt_main.set_num_last_folder(-2)
    labelme2txt_main.add_gt_labelme_json(labelme_json_files)

    labelme2txt_main.save_curvelane_txt(curvelane_txt_path)






