import pickle
import os

class WorkSpace(object):
    def __init__(self, bak_file='data/workspace.bin'):
        path = os.path.dirname(bak_file)
        if not os.path.exists(path):
            os.mkdir(path)
        self.bak_file = bak_file
        self.work_space = {}

        self.history = []
        pass

    def set(self, str_key, x):
        self.history.append(str_key)
        self.work_space[str_key] = x

    def check(self, str_key):
        if str_key in self.work_space:
            return True
        else:
            return False

    def get(self, str_key):
        if str_key not in self.work_space:
            return None
        return self.work_space[str_key]

    def save(self):
        if len(self.history) == 0:
            return

        with open(self.bak_file, 'wb') as fp:
            pickle.dump(self.work_space, fp)

    def load(self):
        if not os.path.isfile(self.bak_file):
            return

        with open(self.bak_file, 'rb') as fp:
            self.work_space = pickle.load(fp)
    def clear(self):
        self.work_space = {}

        self.history = []

workspace_main = WorkSpace()