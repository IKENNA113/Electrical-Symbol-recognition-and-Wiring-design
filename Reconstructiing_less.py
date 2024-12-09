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

    # print("idxs :- ", idxs)
    # exit()

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

        #custom logic to remove the boxes which are not required in patched images
        #check if the last box shares the border with the current box, if so remove that
        # print("xx1 :- ", xx1)
        # exit()
        
            # pick.pop()
            # idxs = np.delete(idxs, idxs[-1])
            

    return boxes[pick].astype("int")

def reconstruct_image_and_labels(detected_patches_dir, detected_labels_dir, output_dir, original_resolution, patch_size, overlap_factor):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    step_size = (int(patch_size[0] * (1 - overlap_factor)), int(patch_size[1] * (1 - overlap_factor)))
    num_patches_width = (original_resolution[1] - patch_size[1]) // step_size[1] + 1
    print("Step size:", step_size, "Number of patches width:", num_patches_width)
    # exit()

    patch_groups = defaultdict(list)
    for patch_file in os.listdir(detected_patches_dir):
        # Check if the filename follows the expected pattern 'name_idx.ext'
        if not patch_file.endswith(('.png', '.jpg', '.jpeg')) or '_' not in patch_file:
            print(f"Skipping file {patch_file} as it does not match the pattern.")
            continue
        try:
            orig_name, patch_idx_str = patch_file.rsplit('_', 1)
            patch_idx = int(patch_idx_str.split('.')[0])
            patch_groups[orig_name].append((patch_file, patch_idx))
        except ValueError:
            print(f"Skipping file {patch_file} due to a parsing error.")
            continue

    for orig_name, patches in patch_groups.items():
        sample_patch = io.imread(os.path.join(detected_patches_dir, patches[0][0]))
        if len(sample_patch.shape) == 2:  # Grayscale
            reconstructed_image = np.zeros(original_resolution, dtype=np.uint8)
        else:  # RGB
            reconstructed_image = np.zeros((*original_resolution, 3), dtype=np.uint8)

        all_detections = []

        for patch_file, patch_idx in patches:
            x_idx = patch_idx % num_patches_width
            y_idx = patch_idx // num_patches_width

            patch_path = os.path.join(detected_patches_dir, patch_file)
            patch = io.imread(patch_path)
            label_name = patch_file.replace('.png', '.txt').replace('.jpg', '.txt')
            label_path = os.path.join(detected_labels_dir, label_name)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        class_id, x_center, y_center, width, height, confidence = map(float, line.strip().split())
                        # Adjust coordinates to the full image scale
                        x1 = x_center - (width / 2)
                        y1 = y_center - (height / 2)
                        x2 = x_center + (width / 2)
                        y2 = y_center + (height / 2)

                        # Convert from relative to absolute coordinates
                        x1_abs = x1 * patch_size[1] + x_idx * step_size[1]
                        y1_abs = y1 * patch_size[0] + y_idx * step_size[0]
                        x2_abs = x2 * patch_size[1] + x_idx * step_size[1]
                        y2_abs = y2 * patch_size[0] + y_idx * step_size[0]

                        all_detections.append([x1_abs, y1_abs, x2_abs, y2_abs, confidence, class_id])

            print(f"Reconstructing {orig_name} from patch {patch_file} at position ({x_idx}, {y_idx}), patch size: {patch.shape}")

            reconstructed_image[y_idx * step_size[0]:y_idx * step_size[0] + patch_size[0], x_idx * step_size[1]:x_idx * step_size[1] + patch_size[1]] = patch

        all_detections = np.array(all_detections)
        nms_detections = non_max_suppression(all_detections, 0.1)

        # Save reconstructed image and labels
        io.imsave(os.path.join(output_dir, orig_name + '.png'), reconstructed_image)
        with open(os.path.join(output_dir, orig_name + '.txt'), 'w') as f:
            for det in nms_detections:
                # Convert absolute coordinates back to relative format for saving
                x_center = (det[0] + det[2]) / 2 / original_resolution[1]
                y_center = (det[1] + det[3]) / 2 / original_resolution[0]
                width = (det[2] - det[0]) / original_resolution[1]
                height = (det[3] - det[1]) / original_resolution[0]
                f.write(f"{int(det[5])} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# Parameters
detected_patches_dir = "/home/ubuntu/Downloads/YOLOV8/wiring_yolo/yolov8L_NEG_AUG_PredictALL"
detected_labels_dir = "/home/ubuntu/Downloads/YOLOV8/wiring_yolo/labels"
output_dir = "/home/ubuntu/Downloads/YOLOV8/wiring_yolo/Reconstructed_imageALL"
original_resolution = (8320, 11520)
patch_size = (640, 640)
overlap_factor = 0.5

# Reconstruct images and labels
reconstruct_image_and_labels(detected_patches_dir, detected_labels_dir, output_dir, original_resolution, patch_size, overlap_factor)

