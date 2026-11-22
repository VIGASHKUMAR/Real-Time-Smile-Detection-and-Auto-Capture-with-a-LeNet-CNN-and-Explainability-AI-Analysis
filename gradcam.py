"""
gradcam.py
Explainability: shows WHICH PART of the face made the CNN decide
"smiling" or "not smiling", as a heatmap (red = important, blue = ignored).

Three methods:
  - LayerCAM  (default, used live) - Grad-CAM variant that works better on small CNNs
  - Grad-CAM  (the classic method, kept for comparison)
  - Occlusion (cover parts of the face and measure the drop - slow but direct check)

Used in two ways:
  1. Imported by smile_selfie.py (press 'g' in the live window)
  2. Run on its own to make pictures for your report / README:

     python gradcam.py                          # random faces from dataset/  -> gradcam_examples.png
     python gradcam.py photo1.jpg photo2.png    # your own photos             -> gradcam_photos.png
     python gradcam.py --region-test            # numbers: which face region the model relies on
"""

import os
import sys
import random
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "smile_lenet.h5"
IMG_SIZE = 28


# ======================================================
# Grad-CAM
# ======================================================
class GradCAM:
    """
    Idea in 4 steps:
      1. Run the face through the CNN and keep the output of the LAST conv layer
         (50 feature maps of 14x14 - each map is one learned "pattern detector").
      2. Ask: "if this feature map got a bit stronger, would the decision
         get more confident?"  -> that is the GRADIENT.
      3. Average each map's gradient -> one importance weight per map.
      4. Heatmap = weighted sum of the maps (keep only positive values).
    """

    def __init__(self, model):
        self.model = model
        conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        self.last_conv = conv_layers[-1]          # the 2nd conv layer of LeNet (14x14x50)
        self.output_layer = model.layers[-1]      # Dense(2, softmax)

    def _forward(self, x):
        """Run the model layer by layer, remembering the last conv output.
        The final layer is computed WITHOUT softmax (raw scores = 'logits'),
        because softmax flattens the gradients when the model is very sure."""
        conv_output = None
        for layer in self.model.layers[:-1]:
            x = layer(x)
            if layer is self.last_conv:
                conv_output = x
        logits = tf.matmul(x, self.output_layer.kernel) + self.output_layer.bias
        return conv_output, logits

    def explain(self, face_input, method="layercam"):
        """face_input: array of shape (1, 28, 28, 1), values 0-1.
        method: "layercam" (default, sharper for small CNNs like LeNet) or "gradcam".
        Returns (smile_probability, heatmap 14x14 with values 0-1, predicted_class)."""
        x = tf.convert_to_tensor(face_input, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_output, logits = self._forward(x)
            probs = tf.nn.softmax(logits)[0]
            predicted = int(tf.argmax(probs))
            # score = how much the model prefers its chosen class over the other one
            score = logits[0, predicted] - logits[0, 1 - predicted]

        grads = tape.gradient(score, conv_output)[0]           # (14, 14, 50)
        maps = conv_output[0]                                   # (14, 14, 50)

        if method == "gradcam":
            # Grad-CAM: ONE weight per map (average gradient over all positions)
            weights = tf.reduce_mean(grads, axis=(0, 1))        # (50,)
            heatmap = tf.nn.relu(tf.reduce_sum(maps * weights, axis=-1))
        else:
            # LayerCAM: a weight for EVERY position (positive gradients only).
            # LeNet uses Flatten + Dense, so each position has its own weights -
            # averaging them away (Grad-CAM) blurs the result.
            heatmap = tf.reduce_sum(tf.nn.relu(grads) * maps, axis=-1)

        heatmap = heatmap.numpy()
        heatmap = heatmap / (heatmap.max() + 1e-8)              # scale to 0-1
        return float(probs[1]), heatmap, predicted


def occlusion_map(model, face_input, patch=7, stride=3):
    """Model-independent check: cover a small grey square at every position and
    measure how much the prediction drops. Big drop = important region.
    Slower than Grad-CAM, so it's only used for the stand-alone pictures."""
    x = face_input[0]
    base = model(face_input, training=False).numpy()[0]
    predicted = int(np.argmax(base))

    covered, positions = [], []
    for i in range(0, IMG_SIZE - patch + 1, stride):
        for j in range(0, IMG_SIZE - patch + 1, stride):
            img = x.copy()
            img[i:i + patch, j:j + patch] = 0.5
            covered.append(img)
            positions.append((i, j))
    preds = model.predict(np.array(covered), verbose=0)[:, predicted]

    heat, count = np.zeros((IMG_SIZE, IMG_SIZE)), np.zeros((IMG_SIZE, IMG_SIZE))
    for (i, j), p in zip(positions, preds):
        heat[i:i + patch, j:j + patch] += base[predicted] - p
        count[i:i + patch, j:j + patch] += 1
    heat = np.maximum(heat / np.maximum(count, 1), 0)
    return heat / (heat.max() + 1e-8)


def overlay_heatmap(face_img, heatmap, alpha=0.45):
    """Colour the heatmap (blue -> red) and blend it on top of the face image."""
    if face_img.ndim == 2:                                      # grayscale -> 3 channels
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    h, w = face_img.shape[:2]
    heat = cv2.resize(heatmap, (w, h))
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    return cv2.addWeighted(face_img, 1 - alpha, heat, alpha, 0)


def preprocess(face_gray):
    """Same preprocessing as training: 28x28, 0-1, shape (1, 28, 28, 1)."""
    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    return face.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def remove_duplicate_faces(faces):
    """The Haar cascade sometimes returns 2 boxes for the SAME face (slightly
    shifted / different size). Keep the biggest box and drop any box whose
    centre lies inside an already-kept box."""
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    kept = []
    for (x, y, w, h) in faces:
        cx, cy = x + w / 2, y + h / 2
        if any(kx < cx < kx + kw and ky < cy < ky + kh for (kx, ky, kw, kh) in kept):
            continue
        kept.append((x, y, w, h))
    return kept


def tight_box(x, y, w, h):
    """Cut the Haar box down to match the training framing (eyebrows to chin)."""
    return y + int(0.20 * h), y + h, x + int(0.18 * w), x + w - int(0.18 * w)


# ======================================================
# Stand-alone mode: make Grad-CAM pictures
# ======================================================
def explain_face(cam, face_gray):
    """Run all three explanation methods on one face."""
    face_input = preprocess(face_gray)
    prob, layercam, _ = cam.explain(face_input, method="layercam")
    _, gradcam, _ = cam.explain(face_input, method="gradcam")
    occlusion = occlusion_map(cam.model, face_input)
    return prob, {"Grad-CAM": gradcam, "LayerCAM": layercam, "Occlusion": occlusion}


def show_grid(items, title, out_path):
    """items: list of (face_gray, heatmaps_dict, smile_prob, caption).
    Row 1 = input face, then one row per explanation method."""
    import matplotlib.pyplot as plt

    methods = list(items[0][1].keys())
    rows, n = 1 + len(methods), len(items)
    fig, axes = plt.subplots(rows, n, figsize=(max(2.2 * n, 6), 2.3 * rows + 0.8), squeeze=False)

    for i, (face, heatmaps, prob, caption) in enumerate(items):
        big = cv2.resize(face, (140, 140))
        label = f"Smile {prob*100:.0f}%" if prob >= 0.5 else f"No smile {(1-prob)*100:.0f}%"

        axes[0, i].imshow(big, cmap="gray")
        axes[0, i].set_title(f"{caption}\n{label}", fontsize=9)
        for r, method in enumerate(methods, start=1):
            axes[r, i].imshow(cv2.cvtColor(overlay_heatmap(big, heatmaps[method]), cv2.COLOR_BGR2RGB))
        for r in range(rows):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])

    for r, method in enumerate(methods, start=1):
        axes[r, 0].set_ylabel(method, fontsize=10)

    fig.suptitle(title + "\n(red = region that drove the decision)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print("Saved:", out_path)
    plt.show()


def explain_dataset(cam, per_class=4):
    """Random faces from dataset/smiling and dataset/not_smiling."""
    items = []
    random.seed(1)
    for folder, name in [("smiling", "smiling"), ("not_smiling", "not smiling")]:
        path = os.path.join("dataset", folder)
        for file in random.sample(sorted(os.listdir(path)), per_class):
            face = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            if face is None:
                continue
            prob, heatmaps = explain_face(cam, face)
            items.append((face, heatmaps, prob, f"true: {name}"))
    show_grid(items, "Where does the model look? (dataset faces)", "gradcam_examples.png")


def explain_photos(cam, paths):
    """Find faces in your own photos and explain each one."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    items = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print("Could not read:", path)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
        faces = remove_duplicate_faces(faces)
        if len(faces) == 0:
            print("No face found in:", path)
        for (x, y, w, h) in faces:
            top, bottom, left, right = tight_box(x, y, w, h)
            face = gray[top:bottom, left:right]
            prob, heatmaps = explain_face(cam, face)
            items.append((face, heatmaps, prob, os.path.basename(path)[:18]))
    if items:
        show_grid(items, "Where does the model look? (your photos)", "gradcam_photos.png")


def region_test(cam, n_images=600):
    """Numbers for the README: which face region does the model rely on?
    1) Cover each region of confident smiles and measure the drop in smile probability.
    2) Measure what share of each heatmap falls in each region."""
    random.seed(0)
    files = random.sample(sorted(os.listdir("dataset/smiling")), n_images)
    X = np.array([cv2.resize(cv2.imread(os.path.join("dataset/smiling", f), 0), (IMG_SIZE, IMG_SIZE))
                  for f in files]).astype("float32")[..., None] / 255.0
    p = cam.model.predict(X, verbose=0)[:, 1]
    X, p = X[p >= 0.8], p[p >= 0.8]
    print(f"Confident smiles tested: {len(X)}\n")

    regions = {"forehead": (0, 7), "eyes": (7, 14), "nose": (14, 19), "mouth+chin": (19, 28)}

    print("1) Cover a region -> average smile probability")
    for name, (a, b) in regions.items():
        covered = X.copy()
        covered[:, a:b] = X.mean()
        p_cov = cam.model.predict(covered, verbose=0)[:, 1]
        print(f"   {name:11s} {p.mean():.2f} -> {p_cov.mean():.2f}   (drop {p.mean() - p_cov.mean():.2f})")

    print("\n2) Share of heatmap in each region (first 100 faces)")
    for method in ["gradcam", "layercam", "occlusion"]:
        mass = np.zeros(len(regions))
        for x in X[:100]:
            if method == "occlusion":
                h = occlusion_map(cam.model, x[None])
            else:
                h = cv2.resize(cam.explain(x[None], method=method)[1], (IMG_SIZE, IMG_SIZE))
            for k, (a, b) in enumerate(regions.values()):
                mass[k] += h[a:b].sum()
        shares = "  ".join(f"{n} {s*100:.0f}%" for n, s in zip(regions, mass / mass.sum()))
        print(f"   {method:10s} {shares}")


if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH)
    cam = GradCAM(model)

    if len(sys.argv) > 1 and sys.argv[1] == "--region-test":
        region_test(cam)
    elif len(sys.argv) > 1:
        explain_photos(cam, sys.argv[1:])
    else:
        explain_dataset(cam)
