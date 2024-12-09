import os
from shutil import copyfile
import cv2

def resize_images(input_dir, output_dir, new_size):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Iterate through each file in the input directory
    for file in os.listdir(input_dir):
        # Check if the file is an image
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            # Read the image
            img = cv2.imread(os.path.join(input_dir, file))
            # Resize the image
            resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            # Write the resized image to the output directory
            cv2.imwrite(os.path.join(output_dir, file), resized_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # Check if the file is a txt file
        if file.endswith(".txt"):
            # Copy the txt file to the output directory
            copyfile(os.path.join(input_dir, file), os.path.join(output_dir, file))





input_dir = "/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/JSON_to_Yolo"
output_dir = "/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/Resize_Image"
new_size = (11520, 8320)

resize_images(input_dir, output_dir, new_size)
