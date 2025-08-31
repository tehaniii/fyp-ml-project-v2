import os
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIG ---
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15  # Safe now with EarlyStopping
MODEL_INPUT_PATH = "models/coin_classifier_v3.h5"
MODEL_OUTPUT_PATH = "models/coin_classifier_v3_finetuned.h5"
BEST_MODEL_PATH = "models/best_coin_classifier_v3.h5"
TRAIN_DIR = "data/Train"
VAL_DIR = "data/Validation"

# --- REPRODUCIBILITY ---
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Load the Base Model ---
print("📥 Loading base model...")
model = load_model(MODEL_INPUT_PATH)

# --- Unfreeze Last 30 Layers ---
print("🔧 Unfreezing last 30 layers...")
trainable_count = 0
for layer in model.layers[-30:]:
    if hasattr(layer, 'trainable'):
        layer.trainable = True
        trainable_count += 1
print(f"✅ Unfroze {trainable_count} layers.")

# --- Add Dropout Before Output Layer ---
x = model.layers[-2].output  # Dense(128) layer
x = Dropout(0.3)(x)
predictions = model.layers[-1](x)
model = Model(inputs=model.input, outputs=predictions)

# --- Recompile the Model ---
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("🧠 Model recompiled with low learning rate.")

# --- Data Generators ---
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=40,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=25,
    brightness_range=(0.4, 1.6),
    horizontal_flip=True,
    vertical_flip=True,
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

# --- Class Weights ---
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# --- Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
]

# --- Train ---
print("🚀 Fine-tuning the model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# --- Save Fine-Tuned Model ---
os.makedirs("models", exist_ok=True)
model.save(MODEL_OUTPUT_PATH)
print(f"\n✅ Fine-tuned model saved to: {MODEL_OUTPUT_PATH}")
