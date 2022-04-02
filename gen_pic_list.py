import os
pic_root = "/nas2/untouch_data/srcData/auto_dirve/OpenData/cityscapes/cityscapes/gtFine/train_mask/weimar/label_mask_prediction/"
save_file = "/nas2/untouch_data/srcData/auto_dirve/OpenData/cityscapes/cityscapes/gtFine/train_mask/weimar/weimar.txt"
save_lists = ""
# save_lists +=


pic_names = os.listdir(pic_root)
for pic_name in pic_names:
    save_lists += pic_root + pic_name+"\n"
    print(pic_root + pic_name)
with open(save_file, "w") as fid:
    fid.writelines(save_lists)