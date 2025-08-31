import os
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- Config ---
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = 'data/Train'
VAL_DIR = 'data/Validation'
MODEL_OUTPUT_PATH = 'models/coin_classifier_v3.h5'
LABEL_MAP_OUTPUT_PATH = 'class_labels_v3.json'

# --- Reproducibility ---
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Data Augmentation ---
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=SEED
)

# --- Build Model ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # We'll fine-tune later if needed

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# --- Compute Class Weights ---
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# --- Train ---
print("🚀 Training model...")
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weights
)

# --- Save Model ---
os.makedirs('models', exist_ok=True)
model.save(MODEL_OUTPUT_PATH)
print(f"✅ Model saved to: {MODEL_OUTPUT_PATH}")

# --- Save Label Map ---
label_map = train_generator.class_indices
with open(LABEL_MAP_OUTPUT_PATH, "w") as f:
    json.dump({v: k for k, v in label_map.items()}, f)
print(f"✅ Class labels saved to: {LABEL_MAP_OUTPUT_PATH}")
