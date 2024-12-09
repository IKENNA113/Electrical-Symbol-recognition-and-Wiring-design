import os
import random
import shutil

def move_images(source_directory, target_directory, file_count):
    images = os.listdir(source_directory)
    random.shuffle(images)
    selected_images = images[:file_count]

    for image in selected_images:
        source_path = os.path.join(source_directory, image)
        target_path = os.path.join(target_directory, image)
        shutil.move(source_path, target_path)

def move_negative_samples(negative_directory, train_directory, val_directory, train_percentage, val_percentage):
    train_images = os.listdir(train_directory)
    val_images = os.listdir(val_directory)
    
    train_count = int(len(train_images) * train_percentage)
    val_count = int(len(val_images) * val_percentage)
    
    move_images(negative_directory, train_directory, train_count)
    move_images(negative_directory, val_directory, val_count)


# Provide the paths and percentages here
negative_directory_path = '/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/DATASET2610/Negative_sampling/patches610_negative_samples'
train_directory_path = '/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/DATASET2610/Negative_sampling/Dataset3/train/images'
val_directory_path = '/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/DATASET2610/Negative_sampling/Dataset3/valid/images'
train_percentage = 0.1
val_percentage = 0.1

move_negative_samples(negative_directory_path, train_directory_path, val_directory_path, train_percentage, val_percentage)
