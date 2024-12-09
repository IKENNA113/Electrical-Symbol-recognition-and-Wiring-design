import os
import numpy as np
from skimage import io
from collections import defaultdict

def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    scores = boxes[:,4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def reconstruct_image_and_labels(original_patches_dir, detected_patches_dir, detected_labels_dir, output_dir, original_resolution, patch_size, overlap_factor):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    step_size = (int(patch_size[0] * (1 - overlap_factor)), int(patch_size[1] * (1 - overlap_factor)))
    num_patches_width = (original_resolution[1] - patch_size[1]) // step_size[1] + 1
    num_patches_height = (original_resolution[0] - patch_size[0]) // step_size[0] + 1

    patch_groups = defaultdict(list)
    for patch_file in os.listdir(original_patches_dir):
        orig_name = '_'.join(patch_file.split('_')[:-1])
        patch_groups[orig_name].append(patch_file)

    for orig_name in patch_groups.keys():
        # Check if the image is grayscale or RGB
        sample_patch = io.imread(os.path.join(original_patches_dir, patch_groups[orig_name][0]))
        if len(sample_patch.shape) == 2:  # Grayscale
            reconstructed_image = np.zeros(original_resolution, dtype=np.uint8)
        else:  # RGB
            reconstructed_image = np.zeros((*original_resolution, 3), dtype=np.uint8)

        all_detections = []

        for y in range(num_patches_height):
            for x in range(num_patches_width):
                patch_idx = y*num_patches_width + x
                patch_name = f"{orig_name}_{patch_idx}.png"
                label_name = f"{orig_name}_{patch_idx}.txt"

                if os.path.exists(os.path.join(detected_patches_dir, patch_name)):
                    patch = io.imread(os.path.join(detected_patches_dir, patch_name))
                    label_path = os.path.join(detected_labels_dir, label_name)
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            for line in f.readlines():
                                class_id, x_center, y_center, width, height, confidence = map(float, line.strip().split())
                                x1 = (x_center - width/2) * patch_size[1] + x * step_size[1]
                                y1 = (y_center - height/2) * patch_size[0] + y * step_size[0]
                                x2 = (x_center + width/2) * patch_size[1] + x * step_size[1]
                                y2 = (y_center + height/2) * patch_size[0] + y * step_size[0]
                                all_detections.append([x1, y1, x2, y2, confidence, class_id])

                else:
                    patch = io.imread(os.path.join(original_patches_dir, patch_name))

                reconstructed_image[y*step_size[0]:y*step_size[0]+patch_size[0], x*step_size[1]:x*step_size[1]+patch_size[1]] = patch

        # Apply NMS to remove duplicate detections
        all_detections = np.array(all_detections)
        nms_detections = non_max_suppression(all_detections, 0.5)

        # Save reconstructed image and labels
        io.imsave(os.path.join(output_dir, orig_name + '.png'), reconstructed_image)
        with open(os.path.join(output_dir, orig_name + '.txt'), 'w') as f:
            for det in nms_detections:
                x_center = (det[0] + det[2]) / 2 / original_resolution[1]
                y_center = (det[1] + det[3]) / 2 / original_resolution[0]
                width = (det[2] - det[0]) / original_resolution[1]
                height = (det[3] - det[1]) / original_resolution[0]
                f.write(f"{int(det[5])} {x_center} {y_center} {width} {height}\n")

# Parameters
original_patches_dir = "/content/yolov7/DATASET4B_JOIN_PATCHES2/images2"
detected_patches_dir = "/content/yolov7/DATASET4B_JOIN_PATCHES2/ResultDataset47B"
detected_labels_dir = "/content/yolov7/DATASET4B_JOIN_PATCHES2/labels_detected"
output_dir = "/content/drive/MyDrive/ResultDataset4/label"
original_resolution = (8320, 11520)
patch_size = (640, 640)
overlap_factor = 0.5
# Reconstruct images and labels
reconstruct_image_and_labels(original_patches_dir, detected_patches_dir, detected_labels_dir, output_dir, original_resolution, patch_size, overlap_factor)


