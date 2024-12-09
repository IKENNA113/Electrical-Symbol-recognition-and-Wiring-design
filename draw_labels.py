import cv2
import os
import numpy as np

def load_class_names(classes_file):
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def get_color_for_class(class_index, num_classes):
    # Use HSV color space to generate diverse colors and then convert to RGB
    hue = 255.0 * class_index / num_classes
    saturation = 170
    value = 255
    color = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
    return (int(color[0]), int(color[1]), int(color[2]))

def draw_bounding_boxes(image_folder, classes_file, output_folder):
    class_names = load_class_names(classes_file)
    num_classes = len(class_names)

    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(image_folder, f"{os.path.splitext(filename)[0]}.txt")

            image = cv2.imread(image_path)
            height, width, _ = image.shape

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                class_index, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
                x_min = int((x_center - bbox_width / 2) * width)
                y_min = int((y_center - bbox_height / 2) * height)
                x_max = int((x_center + bbox_width / 2) * width)
                y_max = int((y_center + bbox_height / 2) * height)

                class_color = get_color_for_class(int(class_index), num_classes)

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), class_color, 2)
                class_name = class_names[int(class_index)]
                cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_color, 2)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)

# Example usage paths kept same as provided
image_folder = '/home/ikenna/anaconda3/envs/env_new/yolov7/FULL_SPLITING_IMG/DATASET4B/test_label/images'
classes_file = '/home/ikenna/anaconda3/envs/env_new/yolov7/FULL_SPLITING_IMG/DATASET4B/Focussymbol.txt'
output_folder = '/home/ikenna/anaconda3/envs/env_new/yolov7/FULL_SPLITING_IMG/DATASET4B/test_label/image_draw_label'

draw_bounding_boxes(image_folder, classes_file, output_folder)

