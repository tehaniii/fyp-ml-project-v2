import os
import cv2
import albumentations as A
from tqdm import tqdm

# Paths
train_dir = 'data/Train/Victoria_Quarter_Farthing'
val_dir = 'data/Validation/Victoria_Quarter_Farthing'
output_size = 224

# Define augmentation pipelines
train_aug = A.Compose([
    A.Rotate(limit=15),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    A.RandomScale(scale_limit=0.1),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.Affine(translate_percent=0.05),
    A.Resize(output_size, output_size)
])

val_aug = A.Compose([
    A.Rotate(limit=10),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
    A.Resize(output_size, output_size)
])

# Augment Function
def augment_images(folder, transform, num_aug):
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in tqdm(images, desc=f'Augmenting in {folder}'):
        img_path = os.path.join(folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        for i in range(1, num_aug + 1):
            augmented = transform(image=image)['image']
            out_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(folder, out_name), augmented)

# Run augmentation
augment_images(train_dir, train_aug, num_aug=20)
augment_images(val_dir, val_aug, num_aug=7)
