#/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np

def get_color_map(num_classes):
    num_classes += 1
    color_map = []
    for i in range(0, num_classes):
        j = 0
        lab = i
        color = [0, 0, 0]
        while lab:
            color[0] |= (((lab >> 0) & 1) << (7 - j))
            color[1] |= (((lab >> 1) & 1) << (7 - j))
            color[2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
        color_map.append(tuple(color))
    color_map = color_map[1:]
    return color_map

class ViewSettings(object):
    def __init__(self):
        self.win_auto_size = True

class ImgView(object):
    def __init__(self, scale=1.0, x_offset=0.0, y_offset=0.0):
        self.scale = scale
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.roi = None
        self.size = None
        self.image = None
        # self.img_infos = Stack(max_len=100)
        self.init()

        self.text_origin = (10, 30)
        self.text_step = 30
        self.skip_step = 1
        self.skip_scale = 1
        self.slow_scale = 1

        self.settings = ViewSettings()

    def create_image(self, img_size=(600, 600, 3)):
        self.image = np.zeros(img_size, dtype=np.uint8)
        return self.image

    def init(self):
        self.text_count = 0
        self.img_count = 0
        self.status = -1
        self.skip_to_end = False

    def set_setting(self, settings):
        self.settings = settings

    def set_scale(self, scale=1.0, x_offset=-1, y_offset=-1):
        if scale > 0:
            self.scale = scale
        if x_offset >= 0:
            self.x_offset = x_offset
        if y_offset >= 0:
            self.y_offset = y_offset

    #roi = xyxy
    def set_roi(self, roi, size=()):
        if roi is None:
            return
        x1 = int(roi[0])
        y1 = int(roi[1])
        x2 = int(roi[2])
        y2 = int(roi[3])
        self.roi = [x1, y1, x2, y2]

        scale = -1
        if len(size):
            scale = max(size[0], size[1]) / max(x2 - x1, y2 - y1)
        self.set_scale(scale, x1, y1)

    def _cut_roi(self, image, roi):
        if roi is None or image is None:
            return image
        return image[roi[1]:roi[3], roi[0]:roi[2]]

    def to_point(self, x, y, type='tuple'):
        if np.isnan(x):
            x = np.nan_to_num(x)
        if np.isnan(y):
            y = np.nan_to_num(y)

        x0 = int((x - self.x_offset) * self.scale)
        y0 = int((y - self.y_offset) * self.scale)

        if x0 < -9999 or x0 > 9999:
            x0 = 0
        if y0 < -9999 or y0 > 9999:
            y0 = 0

        if type == 'tuple':
            return (x0, y0)
        else:
            return [x0, y0]

    def set_image(self, img, roi=None):
        self.img_count += self.skip_step * self.skip_scale
        image = img
        if image is None:
            return self.img_count, None, image
        image = self._cut_roi(image, self.roi)
        h, w = image.shape[0:2]
        self.image = cv2.resize(image, (int(w * self.scale), int(h * self.scale)))

        self.text_count = 0
        return self.img_count, None, image

    def get_image(self):
        return self.image

    def set_image_file(self, img_file, roi=None):
        self.img_count += self.skip_step * self.skip_scale
        image = cv2.imread(img_file)
        if image is None:
            return self.img_count, img_file, image
        image = self._cut_roi(image, self.roi)
        (h, w, c) = np.shape(image)
        self.image = cv2.resize(image, (int(w * self.scale), int(h * self.scale)))

        self.text_count = 0
        self.set_img_info(self.img_count, img_file)
        return self.img_count, img_file, image

    def set_img_info(self, idx, img_file):
        # img_info = ImgInfo()
        # img_info.img_file = img_file
        # img_info.idx = idx
        # self.img_infos.push(img_info)
        pass

    def show_point(self, point, color=(255, 255, 0), radius=2):
        xx = (255, 255, 255)
        self.image = cv2.circle(self.image, self.to_point(point[0], point[1]), int(radius * self.scale), color, -1)

    def show_points(self, points, color=(255, 255, 0), radius=2):
        for pt in points:
            self.show_point(pt, color, radius)

    def show_line(self, pts, color=(255, 255, 0), thickness=2, show_point=True):
        if len(pts) < 2:
            return
        self.show_polygon(pts, color, is_closed=False, thickness=thickness, show_point=show_point)

    def show_polygon(self, pts, color=(255, 255, 0), is_closed=True, thickness=20, show_point=True):
        ptt = []
        r = 7
        for pt in pts:
            ptt.append([self.to_point(pt[0], pt[1])])
            if show_point:
                self.show_point(pt, color, radius=r)
        ptt = [np.array(ptt)]
        self.image = cv2.polylines(self.image, ptt, is_closed, color, thickness=thickness, lineType=cv2.LINE_8)

    def show_landmark(self, landmark, color=(0, 0, 255), radius=2):
        if landmark is None or len(landmark) == 0:
            print('no find landmark')
            return
        for idx in range(0, len(landmark)):
            self.show_point(landmark[idx], color, radius)
        self.show_point(landmark[0], color, radius*3)
        self.show_point(landmark[1], color, radius*3)
        return

    def show_bbox(self, bbox, color=(255, 0, 0), mode='xywh', thickness=2):
        if bbox is None:
            return
        if mode == 'xywh':
            p0 = [bbox[0], bbox[1]]
            p1 = [bbox[0] + bbox[2], bbox[1] + bbox[3]]
        elif mode == 'xyxy':
            p0 = [bbox[0], bbox[1]]
            p1 = [bbox[2], bbox[3]]
        elif mode == 'cxywh':
            p0 = [bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2]
            p1 = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        self.image = cv2.rectangle(self.image, self.to_point(p0[0], p0[1]),
                            self.to_point(p1[0], p1[1]),
                            color, thickness)

    def show_bbox3d(self, bbox, color=(0, 255, 0), thickness=1, show_idx=False):
        qs = bbox.astype(np.int32)
        for k in range(0, 4):
            # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4  # 地面上的4个点
            # use LINE_AA for opencv3
            # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
            cv2.line(self.image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            if show_idx:
                cv2.putText(self.image, str(i), (int(qs[i, 0]), int(qs[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(self.image, str(j), (int(qs[j, 0]), int(qs[j, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            i, j = k + 4, (k + 1) % 4 + 4  # 上侧的点
            cv2.line(self.image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            if show_idx:
                cv2.putText(self.image, str(i), (int(qs[i, 0]), int(qs[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(self.image, str(j), (int(qs[j, 0]), int(qs[j, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            i, j = k, k + 4
            cv2.line(self.image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        return self.image

    def show_text(self, text, point=None, color=(0, 0, 255), font_scale=2, thickness=1):
        if point is None:
            pt = (int(self.text_origin[0]), int(self.text_origin[1] + self.text_count * self.text_step))
            self.text_count += 1
        else:
            pt = self.to_point(point[0], point[1])
        self.image = cv2.putText(self.image, text, pt, cv2.FONT_HERSHEY_PLAIN,
                          font_scale, color, thickness, cv2.LINE_AA)
        return

    def show_axis(self, point, x, y, z, axis_len=1.0, thickness=2):
        x0 = [(point[0] + x[0] * axis_len), (point[1] + x[1] * axis_len)]
        y0 = [(point[0] + y[0] * axis_len), (point[1] + y[1] * axis_len)]
        z0 = [(point[0] + z[0] * axis_len), (point[1] + z[1] * axis_len)]

        self.show_line([point, x0], color=(0, 0, 255), thickness=thickness, show_point=False)
        self.show_line([point, y0], color=(0, 255, 0), thickness=thickness, show_point=False)
        self.show_line([point, z0], color=(255, 0, 0), thickness=thickness, show_point=False)

    def show_labels(self, labels):
        '''
        :param labels: 定义了label的绘画参数，比如点，线，及属性
        :return:
        '''

        for name, label in labels:
            if name == 'landmark':
                self.show_text(text=self._get_value(label, 'name', 'landmark'),
                               color=self._get_value(label, 'color', (0, 0, 255)),
                               font_scale=2,
                               thickness=1)
                self.show_landmark(
                    label['pts'],
                    radius=self._get_value(label, 'r', 1),
                    color=self._get_value(label, 'color', (0, 0, 255)))
            elif name == 'text':
                self.show_text(
                    label['text'],
                    point = self._get_value(label, 'pt', None),
                    color=self._get_value(label, 'color', (0, 0, 255)),
                    font_scale=self._get_value(label, 'font', 1),
                    thickness=self._get_value(label, 'thickness', 1))
            elif name == 'bbox':
                pt = label['bbox'][:2]
                pt = [pt[0], pt[1] - 10]
                self.show_text(text=self._get_value(label, 'text', 'bbox'),
                               point = pt,
                               color=self._get_value(label, 'color', (0, 0, 255)),
                               font_scale=2,
                               thickness=2)
                self.show_bbox(
                    label['bbox'],
                    color=self._get_value(label, 'color', (0, 0, 255)),
                    mode=self._get_value(label, 'mode', 'xywh'),
                    thickness=self._get_value(label, 'thickness', 2),
                )
            elif name == 'axis':
                pt = label['pt']
                self.show_axis(pt,
                               x=self._get_value(label, 'x', (1, 0, 0)),
                               y=self._get_value(label, 'y', (0, 1, 0)),
                               z=self._get_value(label, 'z', (0, 0, 1)),
                               axis_len=self._get_value(label, 'axis_len', 1),
                               thickness=self._get_value(label, 'thickness', 2),
                               )
    def _get_value(self, label_map, key, default_value):
        if key in label_map:
            return label_map[key]
        else:
            return default_value

    def show(self, win_name='img', delay=0):
        if self.image is None:
            print('no image show!')
            return
        if self.settings.win_auto_size:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, self.image)
        return self.keyboard(delay)

    def destroy_window(self):
        cv2.destroyAllWindows()

    def keyboard(self, delay):
        if self.status == -1:
            key = cv2.waitKey(delay * self.slow_scale)
        else:
            key = cv2.waitKey(0)
        if key == ord(' '):
            if delay == 0:
                self.status = -1
            else:
                self.status *= -1
        elif key == ord('a'):
            self.img_count -= 1 + self.skip_step * self.skip_scale
            self.status = 1
        elif key == ord('d'):
            self.img_count += 1 - self.skip_step * self.skip_scale
            self.status = 1
        elif key == ord('1'):
            self.img_count += 100
        elif key == ord('0'):
            self.img_count += 1000
        elif key == ord('n'):
            self.skip_to_end = True
        elif key == ord('p'):
            self.skip_to_end = True
        elif key == ord('x'):
            self.skip_scale += 1
        elif key == ord('z'):
            self.skip_scale = max(1, self.skip_scale - 1)
        elif key == ord('o'):
            self.skip_scale = 1
            self.slow_scale = 1
        elif key == ord('c'):
            cv2.waitKey(0)
        elif key == ord(','):
            self.slow_scale = 1
        elif key == ord('q'):
            print("exiting ...")
            self.destroy_window()
            exit()

        if self.img_count < 0:
            self.img_count = 0
            return self.img_count, key
        return self.img_count, key


img_view_main = ImgView()

