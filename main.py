import os
from PIL import Image
from collections import defaultdict
import pandas as pd

dataset_root = "C:/Users/tehan/Music/coin_project/data"
splits = ['Train', 'Validation']
class_stats = defaultdict(list)

for split in splits:
    for class_name in os.listdir(os.path.join(dataset_root, split)):
        class_path = os.path.join(dataset_root, split, class_name)
        if os.path.isdir(class_path):
            image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
            for img in image_files:
                img_path = os.path.join(class_path, img)
                try:
                    with Image.open(img_path) as im:
                        width, height = im.size
                        file_size_kb = os.path.getsize(img_path) / 1024
                        class_stats['Split'].append(split)
                        class_stats['Class'].append(class_name)
                        class_stats['Image'].append(img)
                        class_stats['Width'].append(width)
                        class_stats['Height'].append(height)
                        class_stats['Size_KB'].append(round(file_size_kb, 2))
                except Exception as e:
                    print(f"Error with {img_path}: {e}")

df = pd.DataFrame(class_stats)
summary = df.groupby(['Split', 'Class']).agg(
    Images=('Image', 'count'),
    Avg_Width=('Width', 'mean'),
    Avg_Height=('Height', 'mean'),
    Avg_Size_KB=('Size_KB', 'mean')
).reset_index()

print(summary)
