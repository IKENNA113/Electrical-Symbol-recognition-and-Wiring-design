import os
import shutil

def drop_and_update_labels(input_dir, old_classes_file, new_classes_file, output_dir):
    # Load old and new classes
    with open(old_classes_file, "r") as f:
        old_classes = [line.strip() for line in f.readlines()]
    with open(new_classes_file, "r") as f:
        new_classes = [line.strip() for line in f.readlines()]

    # Create a mapping from old class indexes to new class indexes
    class_index_mapping = {}
    for old_index, old_class in enumerate(old_classes):
        if old_class in new_classes:
            new_index = new_classes.index(old_class)
            class_index_mapping[old_index] = new_index

    # Loop over annotation files
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".txt"):
            continue

        # Read annotation file
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Filter out bboxes belonging to excluded classes and update class indexes
        new_lines = []
        for line in lines:
            values = line.strip().split()
            class_index = int(values[0])
            if class_index in class_index_mapping:
                new_class_index = class_index_mapping[class_index]
                values[0] = str(new_class_index)
                new_line = " ".join(values)
                new_lines.append(new_line)

        # Write updated annotations to file
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "w") as f:
            f.writelines("\n".join(new_lines))

        # Copy image to output directory
        image_name = os.path.splitext(file_name)[0] + ".png"
        input_image_path = os.path.join(input_dir, image_name)
        output_image_path = os.path.join(output_dir, image_name)
        shutil.copy(input_image_path, output_image_path)



input_dir = "/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/Resize_Image"
output_dir = "/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/Yolo_class_drop"
old_classes_file = "/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/Electrical_class60.txt"
new_classes_file = "/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/Focussymbol.txt"

drop_and_update_labels(input_dir, old_classes_file, new_classes_file, output_dir)
