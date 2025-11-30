# backend/train_brain_best.py

import tensorflow as tf
import os, json
from tensorflow.keras import layers, models

# Dataset directories
BASE = "backend/models/brain_tumor"
TRAIN_DIR = f"{BASE}/Training"
TEST_DIR = f"{BASE}/Testing"

# Output model files
OUT_MODEL = "backend/models/cnn_brain_best.keras"
OUT_CLASSMAP = "backend/models/brain_class_map_best.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="training",
    seed=SEED,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="validation",
    seed=SEED,
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Save class map
class_map = {i: c for i, c in enumerate(train_ds.class_names)}
with open(OUT_CLASSMAP, "w") as f:
    json.dump(class_map, f, indent=2)

# ----------------------------------------------------------
# Add AUGMENTATION to Dataset (NOT inside model)
# ----------------------------------------------------------

augment_layer = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.20),
])

def augment(images, labels):
    return augment_layer(images, training=True), labels

train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ----------------------------------------------------------
# Build Model (Fixed for GRAD-CAM)
# ----------------------------------------------------------

base = tf.keras.applications.MobileNetV3Small(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMG_SIZE, 3)
)   # ❌ REMOVED pooling="avg" so conv layers remain for Grad-CAM

base.trainable = False  # Stage 1 freeze

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
x = base(x, training=False)

# Now output is CONV feature map → apply GAP manually
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(len(class_map), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------------------------------------
# Stage 1 Training
# ----------------------------------------------------------
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8
)

# ----------------------------------------------------------
# Stage 2 Fine-Tuning
# ----------------------------------------------------------
base.trainable = True  # Unfreeze base

fine_tune_at = int(len(base.layers) * 0.6)
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ----------------------------------------------------------
# Save Final Model
# ----------------------------------------------------------
model.save(OUT_MODEL)
print("🔥 FAST + ACCURATE Brain CNN saved at", OUT_MODEL)
