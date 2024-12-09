import os
import random
import shutil


def split_data(images_dir: str, labels_dir: str, classes_file: str, output_dir: str, train_ratio: float = 0.7, test_ratio: float = 0.15):
    # Load the list of classes
    with open(classes_file, 'r') as f:
        classes = f.read().splitlines()

    # Create train, test, and validation directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    valid_dir = os.path.join(output_dir, 'valid')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Create "images" and "labels" directories inside train, test, and validation
    train_images_dir = os.path.join(train_dir, 'images')
    train_labels_dir = os.path.join(train_dir, 'labels')
    test_images_dir = os.path.join(test_dir, 'images')
    test_labels_dir = os.path.join(test_dir, 'labels')
    valid_images_dir = os.path.join(valid_dir, 'images')
    valid_labels_dir = os.path.join(valid_dir, 'labels')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)

    # Get the list of image files and their corresponding labels
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    # Get the number of files for each class
    class_counts = count_files_by_class(label_files, classes, labels_dir)

    # Calculate the number of samples for each split
    num_samples = len(image_files)
    num_train = int(num_samples * train_ratio)
    num_test = int(num_samples * test_ratio)
    num_valid = num_samples - num_train - num_test

    # Randomly shuffle the image files along with their corresponding labels
    random.seed(42)
    image_label_files = list(zip(image_files, label_files))
    random.shuffle(image_label_files)
    image_files, label_files = zip(*image_label_files)

    # Split the image files into train, test, and validation sets
    train_files = image_files[:num_train]
    test_files = image_files[num_train:num_train + num_test]
    valid_files = image_files[num_train + num_test:]

    # Move the image and label files to the respective directories
    move_files(train_files, labels_dir, images_dir, train_images_dir, train_labels_dir)
    move_files(test_files, labels_dir, images_dir, test_images_dir, test_labels_dir)
    move_files(valid_files, labels_dir, images_dir, valid_images_dir, valid_labels_dir)

    # Print the class-wise file counts in each split
    print_class_counts(train_files, class_counts, 'Train')
    print_class_counts(test_files, class_counts, 'Test')
    print_class_counts(valid_files, class_counts, 'Validation')


def count_files_by_class(label_files: list, classes: list, labels_dir: str):
    class_counts = {c: 0 for c in classes}
    for label_file in label_files:
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            labels = f.read().splitlines()
        for label in labels:
            class_id = int(label.split()[0])
            class_name = classes[class_id]
            class_counts[class_name] += 1
    return class_counts


def move_files(files: list, labels_dir: str, target_dir: str, target_images_dir: str, target_labels_dir: str):
    for file in files:
        image_file = file
        label_file = file[:-4] + '.txt'
        image_path = os.path.join(target_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)
        shutil.copy(image_path, target_images_dir)
        shutil.copy(label_path, target_labels_dir)


def print_class_counts(files: list, class_counts: dict, split_name: str):
    print(f'{split_name} Class Counts:')
    total_files = len(files)
    for class_name, count in class_counts.items():
        percentage = (count / total_files) * 100
        print(f'{class_name}: {count} files ({percentage:.2f}%)')



def main():
    global images_dir, labels_dir, classes_file, output_dir
    images_dir = '/home/ikenna/anaconda3/envs/env_new/yolov7/FULL_SPLITING_IMG/image_only'
    labels_dir = '/home/ikenna/anaconda3/envs/env_new/yolov7/FULL_SPLITING_IMG/label_only'
    classes_file = '/home/ikenna/anaconda3/envs/env_new/yolov7/FULL_SPLITING_IMG/Focussymbol.txt'
    output_dir = '/home/ikenna/anaconda3/envs/env_new/yolov7/FULL_SPLITING_IMG/testingsplit'

    split_data(images_dir, labels_dir, classes_file, output_dir)


if __name__ == '__main__':
    main()
