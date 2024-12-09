import cv2
import os

def load_class_names(classes_file):
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def draw_bounding_boxes(image_folder, classes_file, output_folder):
    class_names = load_class_names(classes_file)
    
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(image_folder, f"{os.path.splitext(filename)[0]}.txt")
            
            # Load image
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            # width, height, _ = image.shape

            # Read label file
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                # Parse label information
                class_index, x_center, y_center, bbox_width, bbox_height = map(float, line.split())

                # Convert YOLO format to coordinates in image
                x_min = int((x_center - bbox_width / 2) * width)
                y_min = int((y_center - bbox_height / 2) * height)
                x_max = int((x_center + bbox_width / 2) * width)
                y_max = int((y_center + bbox_height / 2) * height)

                # Draw bounding box on image
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Put class label on image
                class_name = class_names[int(class_index)]
                cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Save image with bounding boxes to output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)

# Example usage
image_folder = '/home/ikenna/anaconda3/envs/env_new/yolov7/Object Detection2_Aug/draw data'  # Replace with your image folder path
classes_file = '/home/ikenna/anaconda3/envs/env_new/yolov7/Object Detection2_Aug/custom_30.TXT'  # Replace with your classes file path
output_folder = '/home/ikenna/anaconda3/envs/env_new/yolov7/Object Detection2_Aug/draw data/output_draw'  # Replace with your output folder path

draw_bounding_boxes(image_folder, classes_file, output_folder)
