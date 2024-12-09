import os
import numpy as np
from skimage import io
from typing import List, Union

def xywh_to_xyxy(lines: List[str], img_height: int, img_width: int) -> List[List[int]]:
    labels = []
    for _, cur_line in enumerate(lines):
        cur_line = cur_line.split(" ")
        cur_line[-1] = cur_line[-1].split("\n")[0]

        x, y, w, h = list(map(float, cur_line[1:]))
        x = int(x * img_width)
        y = int(y * img_height)
        w = int(w * img_width)
        h = int(h * img_height)

        left, top, right, bottom = x - w // 2, y - h // 2, x + w // 2, y + h // 2
        labels.append([int(cur_line[0]), left, top, right, bottom])

    return labels

def xyxy_to_xywh(label: List[int], img_width: int, img_height: int) -> List[float]:
    x1, y1, x2, y2 = list(map(float, label[1:]))
    w = x2 - x1
    h = y2 - y1

    x_cen = round((x1 + w / 2) / img_width, 6)
    y_cen = round((y1 + h / 2) / img_height, 6)
    w = round(w / img_width, 6)
    h = round(h / img_height, 6)

    return [label[0], x_cen, y_cen, w, h]

class Patcher:
    def __init__(self, path_to_save: str, base_path: str, label_path: str) -> None:
        self.path_to_save = path_to_save
        self.create_folders()
        self.base_path = base_path
        self.label_path = label_path

    def create_folders(self) -> None:
        os.makedirs(self.path_to_save, exist_ok=True)
        os.makedirs(os.path.join(self.path_to_save, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.path_to_save, "labels"), exist_ok=True)

    def patch_sampler(self, img: np.ndarray, fname: str, patch_width: int = 640, patch_height: int = 640,) -> None:
        img_height, img_width, _ = img.shape
        if img_height < patch_height or img_width < patch_width:
            return

        overlap_factor = 0.5  # 50% overlap
        step_width = int(patch_width * (1 - overlap_factor))
        step_height = int(patch_height * (1 - overlap_factor))
        
        y_start = 0

        label_path = os.path.join(self.label_path, f"{fname}.txt")
        with open(label_path) as f:
            lines = f.readlines()

        all_labels = xywh_to_xyxy(lines, img_height, img_width)

        idx = 0
        while y_start + patch_height <= img_height:
            x_start = 0
            while x_start + patch_width <= img_width:
                
                x_end = x_start + patch_width
                y_end = y_start + patch_height

                cropped = img[y_start:y_end, x_start:x_end]

                cur_labels = []
                for label in all_labels:
                    cur_label = label.copy()

                    if (
                        label[1] > x_start
                        and label[2] > y_start
                        and label[3] < x_end
                        and label[4] < y_end
                    ):
                        cur_label[1] -= x_start
                        cur_label[2] -= y_start
                        cur_label[3] -= x_start
                        cur_label[4] -= y_start

                        label_yolo = xyxy_to_xywh(cur_label, patch_width, patch_height)
                        cur_labels.append(label_yolo)

                if len(cur_labels):
                    with open(
                        os.path.join(self.path_to_save, "labels", f"{fname}_{idx}.txt"), "a"
                    ) as f:
                        f.write(
                            "\n".join(
                                "{} {} {} {} {}".format(*tup) for tup in cur_labels
                            )
                        )
                        f.write("\n")

                io.imsave(
                    os.path.join(self.path_to_save, "images", f"{fname}_{idx}.png"), cropped
                )

                x_start += step_width
                idx += 1

            y_start += step_height

def main():
    image_path = "/home/ikenna/anaconda3/envs/env_new/yolov7/FULL_SPLITING_IMG/Splited_images/test/images"  
    label_path = "/home/ikenna/anaconda3/envs/env_new/yolov7/FULL_SPLITING_IMG/Splited_images/test/labels"  
    path_to_save = "/home/ikenna/anaconda3/envs/env_new/yolov7/FULL_SPLITING_IMG/Splited_images/test/Test_overlap_patches" 

    patcher = Patcher(path_to_save=path_to_save, base_path=image_path, label_path=label_path)

    for fname in os.listdir(image_path):
        if fname.endswith(".png"):
            img = io.imread(os.path.join(image_path, fname))
            patcher.patch_sampler(img, fname[:-4])  # the [:-4] is to remove the ".png" from filename

if __name__ == "__main__":
    main()

