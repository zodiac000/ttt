import sys
sys.path.insert(0, "../../")
import os
import cv2
from pathlib import Path
from perception.base.seg_view import seg_view_main
from file.file_utils import remove_folder, mkdirs
from view.base.img_view import ImgView, get_color_map
from perception.base.seg_group_files import SegGroupFiles
from labelme_tool.labelme_tool import labelme_tool_main

class LaneViewBatch(SegGroupFiles):
    def __init__(self):
        super(LaneViewBatch, self).__init__()
        self._img_view = ImgView()
        self.color_map = get_color_map(10)
        pass

    def add_pred_labelme(self, pred_labelme_files):
        print('pred labelme file has ', len(pred_labelme_files))
        self.name_order['pred'] = []
        for image_file in pred_labelme_files:
            name = self.get_name(image_file)
            pairs = self._get_image_pair_map_extend(name, add_key='pred_labelme_file')
            pairs["pred_labelme_file"] = labelme_tool_main.to_labelme_file(image_file)
            self.name_order['pred'].append(name)

    def save_diff_image(self,
                        save_dir,
                        is_show=True):
        '''
        :param num_classes:
        :param save_dir: 键盘s，保存diff信息的地址
        :param is_show: 是否需要显示，若不需要显示，则直接保存所有的diff图
        :param mode: cv表示使用opencv来绘制diff图，plt标志使用plt绘制diff图
        :param gt_weight: 用于合成原始图像与gt混合图的权重，0代表不显示原始图像，1代表纯color mask图像
        :return:
        '''
        self._save_image_list(save_dir)
        names = self._get_names('file name')
        names = names[::-1]
        self._img_view.init()
        self._img_view.set_scale(0.35)

        idx = 0
        while idx < len(names):
            name = names[idx]
            pairs = self._get_image_pairs(name)
            image_file = self._get_image_file(pairs)
            gt_labelme_file = pairs['gt_labelme_file']

            image = cv2.imread(image_file)
            if image is None:
                idx += 1
                print('image is None')
                continue
            print(idx, image_file)

            if is_show:
                self._img_view.set_image(image)
                self._draw_lane(gt_labelme_file)
                image_file_text = '/'.join(Path(image_file).parts[-2:])
                self._img_view.show_text(point=None, text=image_file_text, color=(0, 0, 255), font_scale=2)
                idx, key = self._img_view.show(delay=0)

                if key == ord('s'):
                    save_file = os.path.join(save_dir, self._get_filename_from_image_file(image_file))
                    cv2.imwrite(save_file, self._img_view.image)
                    self._save_image_list(save_dir, image_file)
            else:
                save_file = os.path.join(save_dir, self._get_filename_from_image_file(image_file))
                seg_view_main.save_diff(save_file)

    def _draw_lane(self, labelme_json_file):
        if labelme_json_file is None:
            return
        labelFile = labelme_tool_main.load_labelfile(labelme_json_file)
        for idx, shape in enumerate(labelFile.shapes):
            label = shape['label']
            shape_type = shape['shape_type']
            points = shape['points']
            group_id = shape['group_id']
            if group_id is None:
                group_id = idx

            if shape_type != 'linestrip':
                continue
            if label != 'lane':
                continue
            self._img_view.show_line(points, color=self.color_map[group_id], thickness=1)
            pass

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

    workspace_main.load()

    update = True
    # folder = ["6", "12", "18", "24", "30", "36", "42", "48"]
    sub_folders = [str(idx * 6) for idx in list(range(1, 9))]
    sub_folder_index = 7

    save_diff_dir = '/nas2/untouch_data/srcData/wenbin/lane_det/ambiguous/{}'.format(sub_folders[sub_folder_index])
    image_origin_dir = '/nas2/auto_drive/OpenData/Curvelanes/tuneresult/train/curvelanes_split/{}'.format(sub_folders[sub_folder_index])
    # gt_labelme_file_dir = '/data4/tjk/project/lane_detection/auto_mark_data/src'

    if not os.path.isdir(save_diff_dir):
        os.makedirs(save_diff_dir)
    if update or workspace_main.get('image_origin_dir') != image_origin_dir:
        image_origin_files = file_list_main.find_files(image_origin_dir, ['png', 'jpg'], recursive=True)
        # gt_labelme_files = file_list_main.find_files(gt_labelme_file_dir, ['json'], recursive=True)
        lane_view_batch_main.add_image_origin(image_origin_files)
        # lane_view_batch_main.add_gt_labelme_json(gt_labelme_files)
        workspace_main.clear()
        workspace_main.set('image_origin_dir', image_origin_dir)
        workspace_main.set('lane_view_batch_main', lane_view_batch_main)
        workspace_main.save()
    lane_view_batch_main = workspace_main.get('lane_view_batch_main')

    #可以认为注释掉，就可以避免删除了
    # remove_folder(save_diff_dir)
    lane_view_batch_main.save_diff_image(save_dir=save_diff_dir,
                                        is_show=True)