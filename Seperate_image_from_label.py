import glob
import shutil
import os

src_dir = "/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/Split_full_image/Yolo_class_drop2"
dst_dir = "/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/Split_full_image/image_only"
for pngfile in glob.iglob(os.path.join(src_dir, "*.png")):
    shutil.move(pngfile, dst_dir)
