import cv2
import numpy as np
from geometry.geometry import nearest_point2d_pair_brute_force

def seperate_mask(mask_img):
    color_map = {}
    for row in range(mask_img.shape[0]):
        for col in range(mask_img.shape[1]):
            color = tuple(mask_img[row, col].tolist())
            if len(color_map) > 33:
                return ValueError('color map is to large[', len(color_map), ']')
            if color not in color_map:
                color_map[color] = np.zeros(mask_img.shape[:2], np.uint8)
            color_map[color][row, col] = 255
    return color_map

def mask_to_contour(mask, epsilion=1, min_area=1.0):
    contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        approxes.append(cv2.approxPolyDP(cnt, epsilion, True))
    return approxes

def remove_redundant_polygon(polygon, area_anchor_point, cross_point=None):
    nearest_point_pair, point_pairs = nearest_point2d_pair_brute_force(polygon, dis_thres=3)
    pairs = []
    for pair in point_pairs:
        if abs(pair[0] - pair[1]) < 5:
            continue
        if nearest_distance2d([area_anchor_point, pair[2]]) < 5:
            continue
        pairs.append(pair)
    if len(pairs) == 1:
        del polygon[pairs[0][0]+1:pairs[0][1]-1]
        return True
    else:
        return False

def draw_contour(image, approxes, color, thickness=1):
    cv2.polylines(image, approxes, True, color, thickness)

if __name__ == '__main__':
    mask_file = '/data8/ljj/code/drive/lane/lane_proj/output/test_result/pseudo_color_prediction/data_1/3.png'
    img = cv2.imread(mask_file)

    color_map = seperate_mask(img)
    for color, mask in color_map.items():
        approxes = mask_to_contour(mask)
    cv2.imshow('xx', img)
    cv2.waitKey(0)