"""
explain.py
Model-agnostic explanations (LIME and SHAP) for the smile model,
and a faithfulness check that compares ALL explanation methods.

Model-agnostic = these methods never look inside the CNN. They only change
the input image and watch how the output changes, so they work on any model.

  LIME  - cuts the face into superpixels (natural regions), hides random
          combinations of them, and fits a simple linear model:
          "smile score ~ w1*region1 + w2*region2 + ..."  -> the weights explain.
  SHAP  - game theory (Shapley values): the fair average contribution of
          every region, over many combinations. Contributions ADD UP to the
          prediction and can be positive (pro-smile) or negative (anti-smile).

Run:
  python explain.py                  # 8 dataset faces -> explain_lime_shap.png
  python explain.py photo1.jpg ...   # your own photos -> explain_photos.png
  python explain.py --faithfulness   # which method focuses where (all methods)
"""

import os
import sys
import contextlib
import random
import warnings
import cv2
import numpy as np
import tensorflow as tf

from gradcam import (GradCAM, occlusion_map, preprocess, tight_box,
                     remove_duplicate_faces, IMG_SIZE)

warnings.filterwarnings("ignore")

MODEL_PATH = "smile_lenet.h5"
VIEW = 64                 # explanations work on a 64x64 view (superpixels need room)
LIME_SAMPLES = 1000       # how many perturbed copies LIME creates per face
SHAP_EVALS = 500          # how many model calls SHAP may use per face

# face regions on the 28x28 grid (rows) - used for the faithfulness check
REGIONS = {"forehead": (0, 7), "eyes": (7, 14), "nose": (14, 19), "mouth+chin": (19, 28)}


# ======================================================
# A "black box" predict function: what LIME and SHAP talk to
# ======================================================
def make_predict_fn(model):
    """Takes a batch of 64x64 RGB images (values 0-1), returns [P(no smile), P(smile)].
    Inside, it does exactly the training preprocessing: gray, 28x28."""
    def predict(images):
        images = np.asarray(images, dtype="float32")
        gray = images[..., 0] if images.ndim == 4 else images     # all 3 channels are equal
        small = np.array([cv2.resize(g, (IMG_SIZE, IMG_SIZE)) for g in gray])
        return model.predict(small[..., None], verbose=0, batch_size=256)
    return predict


def to_view(face_gray):
    """Gray face (any size, 0-255) -> 64x64 RGB float image in 0-1."""
    face = cv2.resize(face_gray, (VIEW, VIEW)).astype("float32") / 255.0
    return np.stack([face] * 3, axis=-1)


# ======================================================
# LIME
# ======================================================
def lime_explain(predict, view_img, label):
    """Returns (heatmap 64x64 of superpixel weights, segments)."""
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm

    # ~40 superpixels of similar brightness (compactness keeps them roundish)
    segmenter = SegmentationAlgorithm("slic", n_segments=40, compactness=10, sigma=1, start_label=0)
    explainer = lime_image.LimeImageExplainer(random_state=0)
    with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):   # hide LIME's progress bar
        explanation = explainer.explain_instance(
            view_img.astype("double"), predict, labels=(label,), top_labels=None,
            hide_color=0.5, num_samples=LIME_SAMPLES, segmentation_fn=segmenter,
            batch_size=256,
        )
    segments = explanation.segments
    heat = np.zeros(segments.shape, dtype="float32")
    for seg_id, weight in explanation.local_exp[label]:
        heat[segments == seg_id] = weight
    return heat, segments


# ======================================================
# SHAP
# ======================================================
def shap_explain(predict, view_img, label):
    """Returns a 64x64 map of SHAP values for the given class
    (positive = pushes towards that class, negative = pushes away)."""
    import shap

    masker = shap.maskers.Image("blur(16,16)", view_img.shape)   # hidden parts get blurred
    explainer = shap.Explainer(predict, masker, output_names=["no smile", "smile"])
    sv = explainer(view_img[None], max_evals=SHAP_EVALS, batch_size=100,
                   outputs=[label], silent=True)
    return sv.values[0, ..., 0].sum(axis=-1)          # sum the 3 (identical) channels


# ======================================================
# All methods for one face
# ======================================================
def explain_all(model, cam, predict, face_gray):
    """Run every method on one face. Returns (smile_prob, label, dict of maps).
    Maps are float arrays; LIME/SHAP are signed, the others are 0-1."""
    face_input = preprocess(face_gray)
    prob, layercam, label = cam.explain(face_input, method="layercam")
    view_img = to_view(face_gray)
    lime_map, segments = lime_explain(predict, view_img, label)
    return prob, label, {
        "LayerCAM": layercam,
        "Occlusion": occlusion_map(model, face_input),
        "LIME": lime_map,
        "SHAP": shap_explain(predict, view_img, label),
    }, segments


# ======================================================
# Figure
# ======================================================
def show_grid(items, title, out_path):
    """items: list of (face_gray, prob, maps, segments, caption)."""
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    methods = ["LayerCAM", "Occlusion", "LIME", "SHAP"]
    rows, n = 1 + len(methods), len(items)
    fig, axes = plt.subplots(rows, n, figsize=(max(2.2 * n, 7), 2.3 * rows + 0.9), squeeze=False)

    for i, (face, prob, maps, segments, caption) in enumerate(items):
        view = cv2.resize(face, (VIEW, VIEW))
        label = f"Smile {prob*100:.0f}%" if prob >= 0.5 else f"No smile {(1-prob)*100:.0f}%"
        axes[0, i].imshow(view, cmap="gray")
        axes[0, i].set_title(f"{caption}\n{label}", fontsize=9)

        for r, method in enumerate(methods, start=1):
            ax = axes[r, i]
            heat = cv2.resize(maps[method].astype("float32"), (VIEW, VIEW))
            if method in ("LIME", "SHAP"):
                # signed: red = supports the prediction, blue = against it
                if method == "LIME":
                    ax.imshow(mark_boundaries(np.stack([view / 255.0] * 3, -1), segments,
                                              color=(1, 1, 0), mode="thin"))
                else:
                    ax.imshow(view, cmap="gray")
                lim = np.abs(heat).max() + 1e-8
                ax.imshow(heat, cmap="bwr", vmin=-lim, vmax=lim, alpha=0.55)
            else:
                ax.imshow(view, cmap="gray")
                ax.imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)

    for r in range(rows):
        for i in range(n):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
    for r, method in enumerate(methods, start=1):
        axes[r, 0].set_ylabel(method, fontsize=10)

    fig.suptitle(title + "\nLayerCAM/Occlusion: red = important.   "
                 "LIME/SHAP: red = supports the prediction, blue = against it "
                 "(LIME shows superpixel borders)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print("Saved:", out_path)
    plt.show()


def explain_dataset(model, cam, predict, per_class=4):
    items = []
    random.seed(1)
    for folder, name in [("smiling", "smiling"), ("not_smiling", "not smiling")]:
        path = os.path.join("dataset", folder)
        for file in random.sample(sorted(os.listdir(path)), per_class):
            face = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            if face is None:
                continue
            prob, _, maps, segments = explain_all(model, cam, predict, face)
            items.append((face, prob, maps, segments, f"true: {name}"))
    show_grid(items, "Model-agnostic vs gradient explanations (dataset faces)",
              "explain_lime_shap.png")


def explain_photos(model, cam, predict, paths):
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
            prob, _, maps, segments = explain_all(model, cam, predict, face)
            items.append((face, prob, maps, segments, os.path.basename(path)[:18]))
    if items:
        show_grid(items, "Model-agnostic vs gradient explanations (your photos)",
                  "explain_photos.png")


# ======================================================
# Faithfulness check: where does each method put its attention?
# ======================================================
def faithfulness(model, cam, predict, n_faces=40):
    """For confident smiles, measure the share of POSITIVE evidence each
    method puts in each face region. The occlusion region test in gradcam.py
    showed the model relies on the mouth, so a faithful method should
    put most of its evidence there."""
    import pandas as pd

    random.seed(0)
    files = random.sample(sorted(os.listdir("dataset/smiling")), 300)
    faces = []
    for f in files:
        face = cv2.imread(os.path.join("dataset/smiling", f), cv2.IMREAD_GRAYSCALE)
        if face is not None and float(model(preprocess(face), training=False)[0, 1]) >= 0.8:
            faces.append(face)
        if len(faces) == n_faces:
            break
    print(f"Testing {len(faces)} confidently smiling faces (LIME and SHAP take a few seconds each)...")

    methods = ["Grad-CAM", "LayerCAM", "Occlusion", "LIME", "SHAP"]
    mass = {m: np.zeros(len(REGIONS)) for m in methods}
    for k, face in enumerate(faces, 1):
        face_input = preprocess(face)
        view_img = to_view(face)
        maps = {
            "Grad-CAM": cam.explain(face_input, method="gradcam")[1],
            "LayerCAM": cam.explain(face_input, method="layercam")[1],
            "Occlusion": occlusion_map(model, face_input),
            "LIME": lime_explain(predict, view_img, 1)[0],
            "SHAP": shap_explain(predict, view_img, 1),
        }
        for m, heat in maps.items():
            heat = cv2.resize(np.maximum(heat, 0).astype("float32"), (IMG_SIZE, IMG_SIZE),
                              interpolation=cv2.INTER_AREA)       # positive evidence on 28x28
            for j, (a, b) in enumerate(REGIONS.values()):
                mass[m][j] += heat[a:b].sum()
        if k % 10 == 0:
            print(f"  {k}/{len(faces)} done")

    table = pd.DataFrame({m: 100 * mass[m] / mass[m].sum() for m in methods},
                         index=list(REGIONS)).T.round(0)
    area = {r: 100 * (b - a) / IMG_SIZE for r, (a, b) in REGIONS.items()}
    table.loc["(share of image area)"] = [round(v) for v in area.values()]
    print("\nShare of positive evidence per face region (%):")
    print(table.to_string())
    table.to_csv("faithfulness.csv")
    print("\nSaved: faithfulness.csv")
    return table


if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH)
    cam = GradCAM(model)
    predict = make_predict_fn(model)

    if len(sys.argv) > 1 and sys.argv[1] == "--faithfulness":
        faithfulness(model, cam, predict)
    elif len(sys.argv) > 1:
        explain_photos(model, cam, predict, sys.argv[1:])
    else:
        explain_dataset(model, cam, predict)
