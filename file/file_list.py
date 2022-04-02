from pathlib import Path
import glob
import os
import shutil
import random

class FileList(object):
    def __init__(self):
        self.file_idx_map = None
        self.statuss = None
        pass

    def find_files(self, path, suffixs=['png', 'jpg', 'bmp'], recursive=False):
        if len(path) == 0:
            print('find_files[path] is null')
            return []
        files = []
        for suffix in suffixs:
            if recursive:
                path_re = path + '/**/*.' + suffix
            else:
                path_re = path + '/*.' + suffix
            files.extend(glob.glob(path_re, recursive=recursive))
        return files

    def get_filename(self, files):
        files = [str(Path(file).name) for file in files]
        return files

    def ignore_key(self, files, key, num_last_folders=0):
        keep_files = []
        for file in files:
            local_path = '/'.join(Path(file).parts[num_last_folders:])
            if key in local_path:
                continue
            keep_files.append(file)
        return keep_files

    def ignore_folders(self, files, folders):
        if len(folders) == 0:
            return files
        lines = []
        for line in files:
            parent = str(Path(line.strip()).parent)
            for ignore in folders:
                if ignore in parent:
                    line = None
                    break
            if line is not None:
                lines.append(line.strip())
        return lines

    def keep_key(self, files, key, num_last_folders):
        '''
        :param files:
        :param key:
        :param num_last_folders: 从路径中的最后几个，查找关键词
        :return:
        '''
        keep_files = []
        for file in files:
            local_path = '/'.join(Path(file).parts[num_last_folders:])
            if key not in local_path:
                continue
            keep_files.append(file)
        return keep_files

    def keep_folders(self, files, folders):
        if len(folders) == 0:
            return files
        lines = []
        for line in files:
            parent = str(Path(line.strip()).parent)
            for ignore in folders:
                if ignore not in parent:
                    continue
                lines.append(line.strip())
        return lines

    def read_list(self, list_file):
        if not os.path.isfile(list_file):
            return []
        with open(list_file, "r") as f:
            readlines = []
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0 or line[0] == '#':
                    continue
                readlines.append(line)
        files = [v.strip() for v in readlines]
        return files

    def read_list_with_status(self, list_file):
        readlines = self.read_list(list_file)
        files, statuss = self._split_files_status(readlines)
        return files, statuss

    def write_list(self, list_file, lists):
        with open(list_file, 'w') as fp:
            fp.writelines('\n'.join(lists))

    def sort_fullname_files(self, files):
        files.sort()
        return files

    def sort_filename_files(self, files):
        files.sort(key=lambda x: str(Path(x).stem))
        return files

    def sort_timestamp_files(self, files):
        files.sort(key=lambda x: float(Path(x).stem))
        return files

    def sort_bundle_files(self, files):
        files = self.sort_filename_files(files)
        filess = []
        name_old = ''
        file_one = []
        for file in files:
            name = str(Path(file).stem)
            if name != name_old:
                file_one.sort()
                filess += file_one
                file_one = []
                name_old = name
            file_one.append(file)
        filess += file_one
        return filess

    def recovery_statuss_begin(self, files, statuss):
        file_idx_map = {}
        for idx, file in enumerate(files):
            file_idx_map[file] = idx
        self.file_idx_map = file_idx_map
        self.statuss = statuss
        return file_idx_map

    def recovery_statuss_end(self, files):
        statuss = []
        for file in files:
            statuss.append(self.statuss[self.file_idx_map.get(file)])
        return statuss

    def search_standard_file(self, files, accept_field='standard'):
        valid_file_map = {}
        for file in files:
            words = file.split('/')
            filename = words[-1]
            filename = filename.split('.')[0]
            timestamp = filename.split('_')[0]
            timestamp = os.path.join(words[-3], words[-2], timestamp)
            if timestamp not in valid_file_map:
                valid_file_map[timestamp] = file
            if accept_field in filename:
                valid_file_map[timestamp] = file
        return list(valid_file_map.values())

    def copy_file(self, image_files, dst_dir):
        for image_file in image_files:
            dst_file = os.path.join(dst_dir, Path(image_file).parts[-1])
            shutil.copyfile(image_file, dst_file)

    def shuffle_and_sampling(self, image_files, ratio=-1, max_len=-1):
        random.shuffle(image_files)
        files = image_files
        if ratio > 0:
            num_image = len(image_files)
            files = image_files[:int(num_image * ratio)]
        if max_len > 0:
            max_len = min(max_len, len(image_files))
            files = image_files[:max_len]
        return files

    def _split_files_status(self, lines):
        files = []
        statuss = []
        for line in lines:
            file_name, status = self._line_status(line)
            files.append(file_name)
            statuss.append(status)
        return files, statuss

    def _line_status(self, file):
        words = file.strip().split(' ')
        if len(words) == 1:
            return words[0], '1'
        else:
            return words[0], words[1]

file_list_main = FileList()