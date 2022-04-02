import cv2

def resize(image, scale_x, scale_y):
    img = cv2.resize(image, None, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    return img