import os
import numpy as np
import cv2
from skimage import io
from collections import defaultdict

# Define a color map for class IDs
color_map = {
    0: (0, 255, 255),    # Yellow for 'DUPLEX RECEPTACLE'
    1: (0, 255, 0),      # Green for 'LIGHT FIXTURE RECESSED'
    2: (255, 0, 0),      # Red for 'LIGHT FIXTURE SURFACE'
    3: (255, 255, 0),    # Cyan for 'TELEVISION OUTLET'
}

for i in range(4, 30):
    color_map[i] = (255, 255, 0)  # Cyan for all other classes

id_to_name = {
    0: 'DUPLEX RECEPTACLE',
    1: 'LIGHT FIXTURE RECESSED',
    2: 'LIGHT FIXTURE SURFACE',
    3: 'TELEVISION OUTLET',
    4: 'SINGLE POLE SWITCH',
    5: 'TELEPHONE OUTLET',
    6: 'THIN LIGHT FIXTURE',
    7: 'JUNCTION BOX',
    8: 'EMERGENCY NIGHT LIGHT',
    9: 'CYLINDER LIGHT FIXTURE',
    10: 'QUAD RECEPTACLE',
    11: 'EXIT SIGN',
    12: 'NON-FUSED DISCONNECT SWITCH',
    13: 'EMERGENCY BATTERY PACK',
    14: 'PANELBOARD',
    15: 'OS',
    16: 'FIRE ALARM SYSTEM STROBE INSTALLED/WALL MOUNTED',
    17: 'led type 1',
    18: 'FIXTURE TYPE 7',
    19: '1X4 STRIP LIGHT FIXTURE',
    20: 'MOTOR',
    21: 'SPECIAL RRECEPTACLE',
    22: 'SQUARE APERTURE DOWNLIGHT',
    23: 'SINGLE POLE KEYED SWITCH',
    24: 'FIRE ALARM SYSTEM- CELING SMOKE DETECTOR',
    25: 'DOUBLE DUPLEX RECEPTACLE',
    26: 'FIRE ALARM SYSTEM- PULL STATON INSTALLED',
    27: 'EXIT SIGN 3',
    28: 'FLUSH IN CEILING MOUNTED SPEAKER WITH BACKBOX',
    29: 'SURFACE SPEAKER',
}

# Non-max suppression function
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

# Clear original annotations function
def clear_original_annotations(image, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)  # Fill with white color
    return image

# Reconstruct image and labels function
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

        # Clear original annotations
        reconstructed_image = clear_original_annotations(reconstructed_image, nms_detections)

        # Draw bounding boxes, class labels, and confidence scores on the reconstructed image
        for det in nms_detections:
            x1, y1, x2, y2, confidence, class_id = map(int, det)
            color = color_map.get(class_id, (255, 255, 0))  # Default to cyan if class ID not in color_map

            # Draw new bounding box
            cv2.rectangle(reconstructed_image, (x1, y1), (x2, y2), color, 2)

            # Draw class label and confidence score
            label_text = f"{id_to_name[class_id]} {confidence:.2f}"
            cv2.putText(reconstructed_image, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save reconstructed image and labels
        io.imsave(os.path.join(output_dir, orig_name + '.png'), reconstructed_image)

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

