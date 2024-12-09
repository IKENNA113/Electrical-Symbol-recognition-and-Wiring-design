import cv2
import os
import shutil
import numpy as np

np.random.seed(42)  # Set seed for reproducibility

def compute_pixel_intensity(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return np.mean(image)

def add_low_intensity_images(src_folder, dest_folder1, dest_folder2, selected_folder_path1, selected_folder_path2, percentage=0.1):
    # List all images in the source and destination folders
    src_image_files = [f for f in os.listdir(src_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    # Calculate the number of images to add based on percentage of images in the destination folders
    num_images_to_add1 = int(len(os.listdir(dest_folder1)) * percentage)
    num_images_to_add2 = int(len(os.listdir(dest_folder2)) * percentage)
    
    # Compute intensity for each image in the source folder and sort them based on intensity in ascending order
    src_image_files_sorted = sorted(src_image_files, key=lambda x: compute_pixel_intensity(os.path.join(src_folder, x)))
    
    # Create the selected folders using the provided paths
    os.makedirs(selected_folder_path1, exist_ok=True)
    os.makedirs(selected_folder_path2, exist_ok=True)
    
    # Interleave the selection of images for the two folders
    for i in range(max(num_images_to_add1, num_images_to_add2)):
        if i < num_images_to_add1:
            shutil.move(os.path.join(src_folder, src_image_files_sorted.pop(0)), os.path.join(selected_folder_path1))
        if i < num_images_to_add2:
            shutil.move(os.path.join(src_folder, src_image_files_sorted.pop(0)), os.path.join(selected_folder_path2))

# Paths
no_annotation_folder = '/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/DATASET2610/Patches_for_training/patches610_negative_samples'
with_annotation_folder1 = '/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/DATASET2610/Patches_for_training/Patch_split_Neg/train/images'
with_annotation_folder2 = '/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/DATASET2610/Patches_for_training/Patch_split_Neg/valid/images'
selected_folder_path1 = '/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/DATASET2610/Patches_for_training/Neg_Intensity1'
selected_folder_path2 = '/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/DATASET2610/Patches_for_training/Neg_Intensity2'

# Add 10% of the number of images in with_annotation_folder1 and with_annotation_folder2 from no_annotation_folder to the specified selected folders
add_low_intensity_images(no_annotation_folder, with_annotation_folder1, with_annotation_folder2, selected_folder_path1, selected_folder_path2, percentage=0.1)

