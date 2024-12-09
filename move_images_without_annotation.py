import os
import shutil

def move_files_with_annotations(directory, destination_directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            continue  # Skip text files (annotations)
        
        file_name_without_extension = os.path.splitext(filename)[0]
        annotation_file = file_name_without_extension + '.txt'
        if not os.path.exists(os.path.join(directory, annotation_file)):
            file_path = os.path.join(directory, filename)
            destination_path = os.path.join(destination_directory, filename)
            shutil.move(file_path, destination_path)
            print(f"Moved file: {filename} to {destination_directory}")

# Usage example
move_files_with_annotations("/home/ikenna/anaconda3/envs/env_new/yolov7/Object Detection2_Aug/patch_data_aug2/patched_data/images", "/home/ikenna/anaconda3/envs/env_new/yolov7/Object Detection2_Aug/patch_data_aug2/patched_data/yolo_patches_negative_samples_aug2")

