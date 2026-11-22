"""
train_model.py
Trains a LeNet CNN to classify faces as "smiling" or "not smiling".

Dataset folder structure (e.g. SMILEsmileD dataset):
    dataset/
        smiling/       -> images of smiling faces
        not_smiling/   -> images of non-smiling faces
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------- Settings ----------------
DATASET_DIR = "dataset"
CLASSES = ["not_smiling", "smiling"]   # label 0, label 1
IMG_SIZE = 28
EPOCHS = 30          # upper limit - early stopping usually stops sooner
BATCH_SIZE = 64
MODEL_PATH = "smile_lenet.h5"


# ---------------- 1. Load and preprocess images ----------------
def load_dataset():
    images, labels = [], []

    for label, class_name in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, class_name)
        for file in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, file))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # grayscale
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))  # 28x28
            images.append(gray)
            labels.append(label)

    X = np.array(images, dtype="float32") / 255.0      # scale to 0-1
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)           # add channel
    y = np.array(labels)
    return X, y


# ---------------- 2. Build LeNet ----------------
def build_lenet():
    model = Sequential([
        Conv2D(20, (5, 5), padding="same", activation="relu",
               input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),

        Conv2D(50, (5, 5), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(500, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------------- 3. Class weights (handle imbalance) ----------------
def get_class_weights(y):
    # weight = total / (num_classes * count_of_class)
    counts = np.bincount(y)
    total = len(y)
    weights = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
    print("Class counts :", dict(enumerate(counts)))
    print("Class weights:", weights)
    return weights


# ---------------- 4. Plot training curves ----------------
def plot_history(history):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()


# ---------------- Main ----------------
if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_dataset()
    print("Total images:", len(X))

    # Split: 80% train / 20% test (test is only used ONCE, at the very end)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Take 10% of the training data as validation (to watch progress / stop early)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

    # Data augmentation: add a mirrored copy of every TRAINING image
    # (a flipped smile is still a smile -> double the data for free)
    X_train = np.concatenate([X_train, np.flip(X_train, axis=2)])
    y_train = np.concatenate([y_train, y_train])
    print("Train:", len(X_train), "| Val:", len(X_val), "| Test:", len(X_test))

    class_weights = get_class_weights(y_train)

    model = build_lenet()
    model.summary()

    callbacks = [
        # stop when val_loss hasn't improved for 4 epochs, keep the best weights
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        # halve the learning rate when val_loss is stuck for 2 epochs
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    history = model.fit(
        X_train, to_categorical(y_train, 2),
        validation_data=(X_val, to_categorical(y_val, 2)),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
    )

    # Evaluate
    preds = model.predict(X_test).argmax(axis=1)
    print(classification_report(y_test, preds, target_names=CLASSES))

    model.save(MODEL_PATH)
    print("Model saved to", MODEL_PATH)

    plot_history(history)
