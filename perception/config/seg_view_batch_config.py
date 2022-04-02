
cityscapes_dataset = dict(
pairs_file="/nas2/untouch_data/srcData/auto_dirve/OpenData/cityscapes/cityscapes/train.list",
image_main_root="/nas2/untouch_data/srcData/auto_dirve/OpenData/cityscapes/cityscapes",
pred_mask_dir="/nas2/untouch_data/srcData/auto_dirve/OpenData_test/cityscapes_ljj/label_mask_prediction",
gt_labelme_dir="/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train",
save_diff_dir="/nas2/untouch_data/srcData/auto_dirve/OpenData_test/cityscapes_ljj/diff_view",
save_label_mask_dir = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/output_test/train_mask',
num_pred_classes=3,
gt_weight=0
)



untouch_train_dataset = dict(
pairs_file='/data8/duzhe/dataset/opendata/cityscapes/lists/fine/train/pair_relative.txt',
image_main_root = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train',
image_origin_dir = '/data4/tjk/lane_detection/DataSet/RoadData/train_data/images',
pred_mask_dir = '/home/public/data/data8/ljj/dataset/RoadDataset/train/labels/DS_park_image_1',
# gt_mask_dir = '/home/public/data/data8/ljj/dataset/RoadDataset/train/lh/labels/DS_park_image_1',
gt_mask_dir = '/data8/ljj/dataset/RoadDataset/train/labels',
gt_labelme_dir = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train',
save_diff_dir = "/nas2/untouch_data/srcData/auto_dirve/OpenData_test/train_ljj/diff_view",
save_label_mask_dir = '/data8/ljj/dataset/RoadDataset/train/labels',
save_pairs_file = '/data8/ljj/dataset/RoadDataset/list/train/pair_relative.txt',
num_pred_classes=3,
gt_weight=0.5,
threshold_classes = [0.98, 0.9, 0.99]
)

untouch_train_dataset_dz = dict(
pairs_file='/data8/duzhe/dataset/opendata/cityscapes/lists/fine/train/pair_relative.txt',
image_main_root = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train/',
pred_mask_dir = '/nas2/untouch_data/srcData/auto_dirve/OpenData_test/big/label_mask_prediction',
gt_labelme_dir = '/nas2/untouch_data/srcData/auto_dirve/dongsheng-car_pic_data/RoadDataset/train',
save_dir = "/data8/duzhe/code/auto-drive/segmentation/paddleseg/bad_case/once/debug",
save_label_mask_dir = '/data8/duzhe/code/auto-drive/segmentation/paddleseg/bad_case/once/debug',
save_pairs_file = '/data8/duzhe/code/auto-drive/segmentation/paddleseg/bad_case/once/debug/pair_relative1.txt',
num_classes=3,
gt_weight=0.5
)
