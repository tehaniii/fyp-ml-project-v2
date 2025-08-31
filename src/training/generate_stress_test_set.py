import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# --- CONFIG ---
SRC_DIR = 'data/Validation'
DST_DIR = 'data/StressTest'
IMG_SIZE = (224, 224)  # Resize for model input
NUM_VARIANTS = 3       # Augmented images per original

# --- Create destination directory ---
os.makedirs(DST_DIR, exist_ok=True)

# --- Augmentation Functions ---
def rotate(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def random_crop(image):
    h, w = image.shape[:2]
    top = random.randint(0, h // 10)
    left = random.randint(0, w // 10)
    bottom = random.randint(int(h * 0.9), h)
    right = random.randint(int(w * 0.9), w)
    return image[top:bottom, left:right]

# --- Process Each Class Folder ---
for class_dir in os.listdir(SRC_DIR):
    src_class_path = os.path.join(SRC_DIR, class_dir)
    dst_class_path = os.path.join(DST_DIR, class_dir)
    os.makedirs(dst_class_path, exist_ok=True)

    for img_name in tqdm(os.listdir(src_class_path), desc=f"Processing {class_dir}"):
        src_img_path = os.path.join(src_class_path, img_name)
        image = cv2.imread(src_img_path)

        if image is None:
            print(f"⚠️ Could not read image: {src_img_path}")
            continue

        image = cv2.resize(image, IMG_SIZE)

        for i in range(NUM_VARIANTS):
            variant = image.copy()

            # Random augmentations
            if random.random() < 0.7:
                variant = rotate(variant, angle=random.choice([-45, -30, 30, 45]))
            if random.random() < 0.5:
                variant = adjust_brightness(variant, factor=random.uniform(0.5, 1.5))
            if random.random() < 0.4:
                variant = blur(variant)
            if random.random() < 0.4:
                variant = random_crop(variant)
                variant = cv2.resize(variant, IMG_SIZE)

            # Save augmented image
            base_name = Path(img_name).stem
            out_name = f"{base_name}_aug{i}.jpg"
            dst_img_path = os.path.join(dst_class_path, out_name)
            cv2.imwrite(dst_img_path, variant)
