import os
import cv2
import numpy as np
from skimage import measure

def mask_to_yolo_txt(mask_path, out_path, class_id=0):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape

    # Find contours
    contours = measure.find_contours(mask, 0.5)

    with open(out_path, "w") as f:
        for contour in contours:
            if len(contour) < 3:  # need polygon
                continue
            # Normalize coordinates (x/w, y/h)
            coords = []
            for y, x in contour:  
                coords.append(x / w)
                coords.append(y / h)
            coords = [str(round(c, 6)) for c in coords]
            f.write(f"{class_id} " + " ".join(coords) + "\n")

def process_folder(mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for mask_file in os.listdir(mask_dir):
        if not mask_file.endswith(".png"):
            continue
        mask_path = os.path.join(mask_dir, mask_file)
        out_path = os.path.join(out_dir, mask_file.replace(".png", ".txt"))
        mask_to_yolo_txt(mask_path, out_path)

# Convert train and val
process_folder("data/train/labels", "data/train/labels_yolo")
process_folder("data/val/labels", "data/val/labels_yolo")
print("✅ Masks converted to YOLO format")
