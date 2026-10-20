"""
Train a binary EMPTY/OCCUPIED parking-space image classifier.
Requires a dataset/ folder with:
  dataset/empty/
  dataset/occupied/
Run after collecting labelled images:
  python train_model.py
"""
from pathlib import Path
import tensorflow as tf

DATA_DIR = Path("dataset")
MODEL_PATH = "parking_model.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

for folder in ("empty", "occupied"):
    p = DATA_DIR / folder
    if not p.exists() or not any(p.glob("*")):
        raise SystemExit(
            f"Missing images in {p}. Run capture_dataset.py and collect both classes first."
        )

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["empty", "occupied"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["empty", "occupied"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# MobileNetV2 provides useful visual features without training a huge model from scratch.
backbone = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
backbone.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
x = backbone(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_loss", save_best_only=True
    ),
]

model.fit(train_ds, validation_data=val_ds, epochs=12, callbacks=callbacks)
loss, accuracy = model.evaluate(val_ds, verbose=0)
print(f"Validation accuracy: {accuracy:.1%}")
print(f"Model saved to {MODEL_PATH}")
print("Important: validation accuracy is only meaningful with correctly labelled, varied images.")
