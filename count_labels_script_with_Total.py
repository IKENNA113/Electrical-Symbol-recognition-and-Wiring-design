import os

# Function to count bounding boxes for each class
def count_bounding_boxes(directory_path):
    class_counts = {}

    # Read the classes.txt file
    with open('/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/DATASET2610/Augmentaed_training_patches/Dataset3/Focussymbol.txt', 'r') as classes_file:
        classes = classes_file.read().splitlines()

    # Initialize counts for each class
    for class_name in classes:
        class_counts[class_name] = 0

    # Iterate over the files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory_path, file_name)

            # Read the label file
            with open(file_path, 'r') as label_file:
                labels = label_file.read().splitlines()

            # Count bounding boxes for each class in the label file
            for label in labels:
                class_index = int(label.split()[0])
                class_name = classes[class_index]
                class_counts[class_name] += 1

    return class_counts

# Directory path containing the images and labels
directory_path = '/home/ikenna/anaconda3/envs/env_new/yolov7/Updated_Experiment610/DATASET2610/Augmentaed_training_patches/Dataset3/DATASET3B/train (copy)/labels'

# Call the function to count bounding boxes
bounding_box_counts = count_bounding_boxes(directory_path)

# Print the results
total_boxes = 0
for class_name, count in bounding_box_counts.items():
    print(f"{class_name}: {count} bounding boxes")
    total_boxes += count

print(f"\nTotal bounding boxes across all classes: {total_boxes}")

