from pathlib import Path
from view.base.img_view import *
# from file.dataset_info_convert import *
import shutil

class ImgsView(object):
    def __init__(self):
        self._img_show = ImgView()
        self._save_path = ''
        self.labels = []
        self.plugin_funcs = []
        self.plugin_names_map = {}
        self._draw_result_func = None # must return labels

        self.add_plugin_func('save', self._save_result)

        self.image_process_func = None
        pass
    def set_setting(self, settings):
        self._img_show.set_setting(settings)

    def set_scale(self, scale, x_offset, y_offset):
        self._img_show.set_scale(scale, x_offset, y_offset)

    def set_save_path(self, save_path):
        self._save_path = save_path

    def get_save_path(self):
        return self._save_path

    def add_plugin_func(self, plugin_name, plugin_func):
        if plugin_name == 'draw':
            self._draw_result_func = plugin_func
        else:
            if plugin_name not in self.plugin_names_map:
                self.plugin_names_map[plugin_name] = len(self.plugin_funcs)
                self.plugin_funcs.append([plugin_name, plugin_func])
            else:
                self.plugin_funcs[self.plugin_names_map[plugin_name]] = [plugin_name, plugin_func]
                print(plugin_name, ' function is over!')

    def set_image_process_func(self, image_process_func):
        self.image_process_func = image_process_func


    def _run_plugin_func(self, img_file, key):
        for plugin_name, plugin_func in self.plugin_funcs:
            plugin_func(img_file, key)

    def _draw_result(self, img_file, key=None):
        if self._draw_result_func is not None:
            labels = self._draw_result_func(img_file, key)
            self._img_show.show_labels(labels)
        pass

    def _save_result(self, image_file, key):
        if key != ord('s'):
            return
        if len(self._save_path) == 0:
            return
        parts = Path(image_file).parts
        root = self._save_path + '/' + '#'.join(parts[-4:-1])

        if not os.path.exists(root):
            os.mkdir(root)
        shutil.copyfile(image_file, os.path.join(root, parts[-1]))
        pass

    def show_image(self, image_files, scale=1.0, delay=1):
        self._img_show.set_scale(scale)
        idx = 0
        self._img_show.init()
        colors = [(0, 0, 255), (0, 255, 0)]
        color = colors[0]
        diff_info = [0, 1] #idx-1, diff count
        key = ''
        while idx < len(image_files):
            img_file = image_files[idx]
            image = cv2.imread(img_file)
            if self.image_process_func is not None:
                image = self.image_process_func(image)
            _, _, _ = self._img_show.set_image(image)
            if image is None:
                print(img_file)
                idx += 1
                continue

            if idx + 1 < len(image_files):
                print('idx-1, idx, diff count: ', diff_info[0], idx, diff_info[1], img_file)
                if idx > 0 and (diff_info[0] < idx
                                and str(Path(image_files[idx - 1]).stem) != str(Path(image_files[idx]).stem)):
                    color = colors[diff_info[1] % 2]
                    diff_info[1] += 1
                elif diff_info[0] >= idx and str(Path(image_files[idx]).stem) != str(Path(image_files[idx + 1]).stem):
                    color = colors[diff_info[1] % 2]
                    diff_info[1] -= 1

            diff_info[0] = idx

            parts = Path(img_file).parts
            img_path = os.path.join(parts[-4], parts[-3], parts[-2], parts[-1])
            self._img_show.show_text(point=None, text=img_path, color=color, font_scale=1)

            self._draw_result(img_file)
            idx, key = self._img_show.show(delay=delay)
            print('idx: ', idx)
            self._run_plugin_func(img_file, key)

            if self._img_show.skip_to_end:
                break

        return key
    def show_folder(self, all_image_files, scale=1.0, delay=1):
        folders_map = self._to_folder_map(all_image_files)
        self._img_show.set_scale(scale)
        for parent, image_files in folders_map.items():
            self._img_show.init()
            self.show_image(image_files, scale, delay)

    def _to_folder_map(self, image_files):
        folders_map = {}
        for image_file in image_files:
            parent = str(Path(image_file).parent)
            if parent not in folders_map:
                folders_map[parent] = []
            folders_map[parent].append(image_file)
        return folders_map
imgs_view_main = ImgsView()
