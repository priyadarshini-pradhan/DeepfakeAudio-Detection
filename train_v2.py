import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ============================================
# Paths
# ============================================

X_PATH = "Features/X_v2.npy"
Y_PATH = "Features/y_v2.npy"

MODEL_PATH = "Models/deepfake_cnn_v2.keras"

os.makedirs("Models", exist_ok=True)


# ============================================
# Load Data
# ============================================

print("Loading V2 features...")

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================
# Train / Validation Split
# ============================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# ============================================
# Class Weights
# ============================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, weights)
)

print("\nClass weights:")
print(class_weights)


# ============================================
# Build V2 CNN
# ============================================

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        input_shape=(40, 400, 1)
    ),

    BatchNormalization(),

    MaxPooling2D(
        (2, 2)
    ),

    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    MaxPooling2D(
        (2, 2)
    ),

    Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    MaxPooling2D(
        (2, 2)
    ),

    GlobalAveragePooling2D(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        2,
        activation="softmax"
    )
])


# ============================================
# Compile
# ============================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================
# Model Summary
# ============================================

model.summary()


# ============================================
# Callbacks
# ============================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


# ============================================
# Train
# ============================================

print("\n================================")
print("STARTING V2 TRAINING")
print("================================")

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=30,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ],
    verbose=1
)


# ============================================
# Final Validation
# ============================================

print("\n================================")
print("V2 TRAINING COMPLETED")
print("================================")

loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print(
    f"Validation Loss     : {loss:.4f}"
)

print(
    f"Validation Accuracy : {accuracy:.4f}"
)

print(
    f"\nModel saved to:\n{MODEL_PATH}"
)