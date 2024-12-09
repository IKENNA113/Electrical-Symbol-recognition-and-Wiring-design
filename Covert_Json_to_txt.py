import os
import json
import cv2
import glob

input_folder = "/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/DATASET610"
output_folder = "/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/JSON_to_Yolo"

# Load class names from classes.txt file
with open('/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/Electrical_class60.txt', 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Create a dictionary to map class names to indices
class_dict = {class_name: i for i, class_name in enumerate(class_names)}
# print(class_dict)

# Get the JSON files in the input folder
json_files = glob.glob(os.path.join(input_folder, '*.json'))
# print(json_files)

for json_file in json_files:
    # print('json file issue: ', json_file)
    # Load the JSON content from the file
    with open(json_file, 'r') as f:
        json_content = json.load(f)

    # Get the image filename from the JSON filename
    image_filename = os.path.basename(json_file).replace('.json', '.png')
    # Get the path of the input and output image files
    input_image_path = os.path.join(input_folder, image_filename)
    output_image_path = os.path.join(output_folder, image_filename)

    # Load the image from the input folder
    image = cv2.imread(input_image_path)

    # Get the image dimensions
    image_height, image_width, _ = image.shape


    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the output file for writing
    output_file_path = os.path.join(output_folder, os.path.basename(json_file).replace('.json', '.txt'))
    with open(output_file_path, 'w') as output_file:
        # Loop over the annotations in the JSON content
        for annotation in json_content[0]['annotations']:
            class_name = annotation['class']
            x = annotation['x']
            y = annotation['y']
            width = annotation['width']
            height = annotation['height']
            
            x_center = (x + width / 2) / image_width
            y_center = (y + height / 2) / image_height
            normalized_width = width / image_width
            normalized_height = height / image_height

            # Get the class index from the dictionary
            class_name = class_name.rstrip() #comment if your class doesn't have space in the end.
            class_index = class_dict[class_name]

            # Write the YOLO format annotation to the output file
            # output_file.write(f"{class_index} {x_center} {y_center} {box_width} {box_height}\n")

            output_file.write(f"{class_index} {x_center} {y_center} {normalized_width} {normalized_height}\n")

    # Copy the image to the output folder
    cv2.imwrite(output_image_path, image)



