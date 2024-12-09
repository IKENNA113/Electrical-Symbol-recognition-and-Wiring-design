import albumentations as A
import os
import glob
import random
from tqdm import tqdm
import cv2

# Set seed for reproducibility
random.seed(42)

# Convert YOLO format to Pascal VOC
def yolo_to_pascal(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = box[0] / dw
    y = box[1] / dh
    w = box[2] / dw
    h = box[3] / dh
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

# Convert Pascal VOC format to YOLO
def pascal_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return [x*dw, y*dh, w*dw, h*dh]

def augment_images_for_class(data_dir, augmented_dir, thresholds, class_counts):
    # Define transformations
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomScale(scale_limit=0.1, p=0.5, interpolation=cv2.INTER_LINEAR),
        A.Resize(640, 640)  # Ensure the output is always 640x640
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    # Progress bar for augmentation
    pbar = tqdm(total=sum(thresholds.values()), desc="Augmenting images", dynamic_ncols=True)

    image_files = glob.glob(os.path.join(data_dir, "*.png"))
    random.shuffle(image_files)

    for random_image_file in image_files:
        label_file = random_image_file.replace('.png', '.txt')

        with open(label_file, 'r') as f:
            lines = f.read().splitlines()
            bboxes = []
            class_labels = []
            for line in lines:
                line_class_id = int(line.split(' ')[0])
                if line_class_id in thresholds.keys() and class_counts[line_class_id] < thresholds[line_class_id]:
                    image = cv2.imread(random_image_file)
                    height, width = image.shape[:2]
                    box = list(map(float, line.split(' ')[1:5]))
                    box = yolo_to_pascal((width, height), box)
                    bboxes.append(box)
                    class_labels.append(line_class_id)

            # Augment only if there's a bounding box of the class of interest in the image
            if bboxes:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

                # Replace any / in the filename for compatibility
                base_name = os.path.basename(random_image_file).replace('.png', f'_aug.png')
                new_image_path = os.path.join(augmented_dir, base_name)
                new_label_path = new_image_path.replace('.png', '.txt')

                cv2.imwrite(new_image_path, augmented['image'])

                with open(new_label_path, 'w') as f:
                    for box, label in zip(augmented['bboxes'], augmented['class_labels']):
                        box = pascal_to_yolo((width, height), box)
                        f.write(f"{label} {' '.join(map(str, box))}\n")

                        class_counts[label] += 1
                        if class_counts[label] >= thresholds[label]:
                            del thresholds[label]

                pbar.update(1)
                if not thresholds:
                    break

    pbar.close()

def main():
    path_to_data = input("Enter the path to your YOLO dataset: ")
    path_to_class_file = input("Enter the path to your class file: ")
    path_to_save = input("Enter the directory where you want to save the augmented data: ")

    with open(path_to_class_file, 'r') as file:
        classes = file.read().splitlines()

    augmented_dir = path_to_save
    os.makedirs(augmented_dir, exist_ok=True)

    # Initialize class counts
    class_counts = {i: 0 for i, _ in enumerate(classes)}

    # Calculate class counts
    for label_file in glob.glob(os.path.join(path_to_data, "*.txt")):
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.split(' ')[0])
                class_counts[class_id] += 1

    # Display initial counts
    for class_id, count in class_counts.items():
        print(f"{classes[class_id]}: {count} bounding boxes")

    # Determine augmentation targets based on class counts
    thresholds = {}
    for class_id, count in class_counts.items():
        if count < 100:
            thresholds[class_id] = count * 4
        elif count < 1000:
            thresholds[class_id] = count * 2
        else:
            thresholds[class_id] = count    

    augment_images_for_class(path_to_data, augmented_dir, thresholds, class_counts)

if __name__ == "__main__":
    main()

