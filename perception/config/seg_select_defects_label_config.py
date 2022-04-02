untouch_road = dict(
image_dir="/data4/tjk/lane_detection/DataSet/RoadData/train_data/images",
gt_mask_dir="/data4/tjk/lane_detection/DataSet/RoadData/train_data/labels",
gt_color_dir="/data4/tjk/lane_detection/DataSet/RoadData/train_data/color_labels",
gt_labelme_dir="/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train_image_label/labels-v1.0",
pred_mask_dir="/data4/tjk/lane_detection/DataSet/RoadData/test_data/new/pseudo_color_prediction",
pred_color_dir="/data4/tjk/lane_detection/DataSet/RoadData/test_data/new/added_prediction",
save_dir = '/data4/wb/auto_diff/0.6',
num_classes=3,
iou_thr=[0.7, 0.6, 0.7],
area_thr=[2000, 2000, 2000],
label_name='sidewalk',
top_n=1000,
load=False,
)

