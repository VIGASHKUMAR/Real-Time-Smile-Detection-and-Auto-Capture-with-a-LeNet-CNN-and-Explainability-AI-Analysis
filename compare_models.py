"""
compare_models.py
Trains several smile classifiers on the SAME train/test split and compares them.

Models compared:
  1. Haar Smile Cascade  - classic OpenCV, no training
  2. HOG + SVM           - classic machine learning
  3. LeNet               - small CNN (the one in the resume)
  4. Mini-Xception       - small modern CNN (separable convs + residuals)
  5. MobileNetV2         - transfer learning from ImageNet

Output:
  - results table printed in terminal  (also saved to results.csv)
  - comparison_chart.png  (accuracy / F1 / speed / size)
  - confusion_matrices.png

Dataset folder structure:
    dataset/
        smiling/
        not_smiling/
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- Settings ----------------
DATASET_DIR = "dataset"
CLASSES = ["not_smiling", "smiling"]
EPOCHS = 15
BATCH_SIZE = 64
USE_MOBILENET = True   # set False to skip it (slow on CPU / needs internet for weights)


# ======================================================
# 1. Load data (once, at 96x96 grayscale - resized per model later)
# ======================================================
def load_dataset(size=96):
    images, labels = [], []
    for label, class_name in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, class_name)
        for file in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(cv2.resize(img, (size, size)))
            labels.append(label)
    return np.array(images), np.array(labels)


def resize_all(images, size):
    return np.array([cv2.resize(img, (size, size)) for img in images])


def class_weights(y):
    counts = np.bincount(y)
    return {i: len(y) / (len(counts) * c) for i, c in enumerate(counts)}


# ======================================================
# 2. The models
# ======================================================

# ---- Model 1: Haar smile cascade (no training) ----
def haar_predict(images):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    preds = []
    for img in images:
        mouth_area = img[img.shape[0] // 2:, :]     # smile is in lower half of face
        smiles = cascade.detectMultiScale(mouth_area, scaleFactor=1.1, minNeighbors=15)
        preds.append(1 if len(smiles) > 0 else 0)
    return np.array(preds)


# ---- Model 2: HOG features + Linear SVM ----
hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

def hog_features(images):
    return np.array([hog.compute(cv2.resize(img, (64, 64))).flatten() for img in images])


# ---- Model 3: LeNet (28x28) ----
def build_lenet(size=28):
    return models.Sequential([
        layers.Input((size, size, 1)),
        layers.Conv2D(20, 5, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(50, 5, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(500, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax"),
    ])


# ---- Model 4: Mini-Xception (48x48) ----
def build_mini_xception(size=48):
    inputs = layers.Input((size, size, 1))
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    # 4 blocks: separable convolutions + a shortcut (residual) connection
    for filters in [16, 32, 64, 128]:
        shortcut = layers.Conv2D(filters, 1, strides=2, padding="same")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.add([x, shortcut])

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return models.Model(inputs, outputs)


# ---- Model 5: MobileNetV2 transfer learning (96x96, 3 channels) ----
def build_mobilenet(size=96):
    try:
        base = MobileNetV2(input_shape=(size, size, 3), include_top=False, weights="imagenet")
        base.trainable = False          # freeze pretrained layers
        print("MobileNetV2: using ImageNet pretrained weights")
    except Exception:
        base = MobileNetV2(input_shape=(size, size, 3), include_top=False, weights=None)
        print("MobileNetV2: pretrained weights unavailable -> training from scratch")

    inputs = layers.Input((size, size, 3))
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)   # MobileNet expects -1..1
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return models.Model(inputs, outputs)


# ======================================================
# 3. Helpers to train/evaluate a Keras model
# ======================================================
def prepare(images, size, channels=1):
    x = resize_all(images, size).astype("float32") / 255.0
    x = x[..., np.newaxis]
    if channels == 3:
        x = np.repeat(x, 3, axis=-1)    # gray -> fake RGB
    return x


def train_keras(model, X_train, y_train, X_test):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start = time.time()
    model.fit(X_train, y_train,
              validation_split=0.1,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              class_weight=class_weights(y_train),
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
              verbose=2)
    train_time = time.time() - start

    start = time.time()
    preds = model.predict(X_test, verbose=0).argmax(axis=1)
    ms_per_img = (time.time() - start) / len(X_test) * 1000

    size_mb = model.count_params() * 4 / 1e6     # float32 = 4 bytes
    return preds, train_time, ms_per_img, size_mb, model.count_params()


# ======================================================
# 4. Run everything
# ======================================================
if __name__ == "__main__":
    print("Loading dataset...")
    images, labels = load_dataset()
    print("Images:", len(images), "| Class counts:", np.bincount(labels))

    img_train, img_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42)

    results = []
    all_preds = {}

    def record(name, preds, train_time, ms, size_mb, params):
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "F1 (smile)": f1_score(y_test, preds),
            "Train time (s)": train_time,
            "ms / image": ms,
            "Size (MB)": size_mb,
            "Parameters": params,
        })
        all_preds[name] = preds
        print(f"--> {name}: accuracy = {results[-1]['Accuracy']:.3f}\n")

    # 1. Haar
    print("\n=== Haar Smile Cascade ===")
    start = time.time()
    preds = haar_predict(img_test)
    record("Haar Cascade", preds, 0, (time.time() - start) / len(img_test) * 1000, 0.0, 0)

    # 2. HOG + SVM
    print("=== HOG + SVM ===")
    Xtr, Xte = hog_features(img_train), hog_features(img_test)
    svm = LinearSVC(C=0.1, class_weight="balanced", max_iter=5000)
    start = time.time()
    svm.fit(Xtr, y_train)
    train_time = time.time() - start
    start = time.time()
    preds = svm.predict(Xte)
    ms = (time.time() - start) / len(Xte) * 1000
    params = svm.coef_.size + 1
    record("HOG + SVM", preds, train_time, ms, params * 4 / 1e6, params)

    # 3-5. CNNs
    cnn_models = [
        ("LeNet",         build_lenet,         28, 1),
        ("Mini-Xception", build_mini_xception, 48, 1),
    ]
    if USE_MOBILENET:
        cnn_models.append(("MobileNetV2", build_mobilenet, 96, 3))
    for name, build_fn, size, channels in cnn_models:
        print(f"=== {name} ===")
        X_train = prepare(img_train, size, channels)
        X_test = prepare(img_test, size, channels)
        record(name, *train_keras(build_fn(size), X_train, y_train, X_test))

    # ---------------- Results table ----------------
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    pd.set_option("display.width", 120)
    print("\n================ RESULTS ================")
    print(df.round(3).to_string(index=False))

    # ---------------- Comparison chart ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    names = df["Model"]

    axes[0].bar(names, df["Accuracy"], color="#4C72B0", label="Accuracy")
    axes[0].bar(names, df["F1 (smile)"], color="#55A868", width=0.4, label="F1 (smile)")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Accuracy and F1 (higher is better)")
    axes[0].legend(loc="lower right")

    axes[1].bar(names, df["ms / image"], color="#C44E52")
    axes[1].set_title("Prediction time per image, ms (lower is better)")

    axes[2].bar(names, df["Size (MB)"], color="#8172B2")
    axes[2].set_title("Model size, MB (lower is better)")

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig("comparison_chart.png", dpi=120)

    # ---------------- Confusion matrices ----------------
    fig, axes = plt.subplots(1, len(all_preds), figsize=(4 * len(all_preds), 4))
    for ax, (name, preds) in zip(axes, all_preds.items()):
        ConfusionMatrixDisplay(confusion_matrix(y_test, preds),
                               display_labels=["no smile", "smile"]).plot(ax=ax, colorbar=False)
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=120)

    print("\nSaved: results.csv, comparison_chart.png, confusion_matrices.png")
