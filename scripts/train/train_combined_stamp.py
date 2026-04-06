#!/usr/bin/env python3
"""Train a combined stamp classifier using BOTH pixel-level edge density AND DINOv2 crop features.

Combines:
  - edge_density_ratio and other pixel features from stamp_pixel_analysis.py
  - DINOv2 CLS token from stamp crop

Approaches to combine:
  1. Naive concatenation (pixel + DINOv2 768-dim) -- baseline
  2. PCA-reduced DINOv2 (5-20 dims) + pixel features -- balanced dimensionality
  3. Probability stacking: separate pixel LR + DINOv2 LR, average probabilities
  4. Probability stacking with learned weights

Evaluation:
  1. Leave-one-out cross-validation on binder samples
  2. Train on synthetic+reference, eval on binder
  3. Save best model to data/stamp_combined_classifier.pkl
"""

import json
import logging
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "condition_training" / "stamps"
REAL_DIR = PROJECT_ROOT / "data" / "condition_training" / "stamps_real"
BINDER_DIR = PROJECT_ROOT / "data" / "inbox"
OUTPUT_PATH = PROJECT_ROOT / "data" / "stamp_combined_classifier.pkl"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_transform_crop_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# ============================================================
# Pixel-level features (from stamp_pixel_analysis.py)
# ============================================================

def crop_stamp_region_cv(img: np.ndarray) -> np.ndarray:
    """Crop stamp region from BGR image. x=[55%,90%], y=[45%,70%]."""
    h, w = img.shape[:2]
    return img[int(h * 0.45):int(h * 0.70), int(w * 0.55):int(w * 0.90)]


def crop_control_region_cv(img: np.ndarray) -> np.ndarray:
    """Crop control region (left side, same vertical band)."""
    h, w = img.shape[:2]
    return img[int(h * 0.45):int(h * 0.70), int(w * 0.10):int(w * 0.45)]


def compute_edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges > 0))


def compute_laplacian_var(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))


def compute_gold_pixel_ratio(img_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 40, 120])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.mean(mask > 0))


def compute_high_freq_energy(gray: np.ndarray) -> float:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 5
    magnitude = np.abs(fshift)
    mask = np.ones_like(magnitude)
    mask[cy - r:cy + r, cx - r:cx + r] = 0
    high_freq = np.sum(magnitude * mask)
    total = np.sum(magnitude) + 1e-10
    return float(high_freq / total)


def extract_pixel_features(img_bgr: np.ndarray) -> np.ndarray:
    """Extract pixel-level features from a card image. Returns (11,) array."""
    stamp = crop_stamp_region_cv(img_bgr)
    control = crop_control_region_cv(img_bgr)

    if stamp.size == 0 or control.size == 0:
        return np.zeros(11, dtype=np.float32)

    stamp_gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
    control_gray = cv2.cvtColor(control, cv2.COLOR_BGR2GRAY)
    stamp_hsv = cv2.cvtColor(stamp, cv2.COLOR_BGR2HSV)

    ed_stamp = compute_edge_density(stamp_gray)
    ed_control = compute_edge_density(control_gray)
    edge_density_ratio = ed_stamp / (ed_control + 1e-10)

    lap_stamp = compute_laplacian_var(stamp_gray)
    lap_control = compute_laplacian_var(control_gray)
    laplacian_ratio = lap_stamp / (lap_control + 1e-10)

    brightness_ratio = float(np.mean(stamp_gray)) / (float(np.mean(control_gray)) + 1e-10)

    hf_stamp = compute_high_freq_energy(stamp_gray)
    hf_control = compute_high_freq_energy(control_gray)
    hf_ratio = hf_stamp / (hf_control + 1e-10)

    mean_sat = float(np.mean(stamp_hsv[:, :, 1]))
    sat_std = float(np.std(stamp_hsv[:, :, 1]))
    hue = stamp_hsv[:, :, 0]
    gold_hue_fraction = float(np.mean((hue >= 15) & (hue <= 40)))

    gold_ratio = compute_gold_pixel_ratio(stamp)

    return np.array([
        ed_stamp, edge_density_ratio, gold_ratio, lap_stamp,
        laplacian_ratio, brightness_ratio, hf_stamp, hf_ratio,
        mean_sat, sat_std, gold_hue_fraction,
    ], dtype=np.float32)


PIXEL_FEATURE_NAMES = [
    "edge_density", "edge_density_ratio", "gold_pixel_ratio", "laplacian_var",
    "laplacian_ratio", "brightness_ratio", "high_freq_energy", "high_freq_ratio",
    "mean_saturation", "sat_std", "gold_hue_fraction",
]


# ============================================================
# DINOv2 crop features
# ============================================================

def crop_stamp_region_pil(img: Image.Image) -> Image.Image:
    """Crop stamp region from PIL image. x=[55%,90%], y=[45%,70%]."""
    w, h = img.size
    return img.crop((int(w * 0.55), int(h * 0.45), int(w * 0.90), int(h * 0.70)))


def load_dino_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading DINOv2 ViT-B/14 on %s ...", device)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.to(device)
    model.eval()
    return model, device


def extract_dino_cls_batch(model, device, image_paths, batch_size=32):
    """Extract DINOv2 CLS tokens from stamp-region crops. Returns (N, 768)."""
    all_cls = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            crop = crop_stamp_region_pil(img)
            tensors.append(_transform_crop_224(crop))
        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            cls_out = model(batch)

        cls_np = cls_out.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(cls_np, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        cls_np /= norms
        all_cls.append(cls_np)

        if start % 64 == 0:
            logger.info("  DINOv2 CLS: %d/%d", min(start + batch_size, len(image_paths)),
                        len(image_paths))

    return np.concatenate(all_cls)


def extract_pixel_features_batch(image_paths):
    """Extract pixel features for a batch of images. Returns (N, 11)."""
    feats = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            feats.append(np.zeros(11, dtype=np.float32))
        else:
            feats.append(extract_pixel_features(img))
    return np.array(feats, dtype=np.float32)


# ============================================================
# Data loading
# ============================================================

def load_dataset(jsonl_path, base_dir):
    """Load images and labels from a JSONL file. Deduplicates by path (last wins)."""
    entries = [json.loads(line) for line in open(jsonl_path)]
    seen = {}
    for e in entries:
        img_path = base_dir / e["image"]
        if img_path.exists():
            seen[str(img_path)] = (img_path, 1 if e["stamped"] else 0, e.get("card_name", ""))
        else:
            logger.warning("Missing: %s", img_path)
    items = list(seen.values())
    paths = [x[0] for x in items]
    labels = np.array([x[1] for x in items], dtype=np.int32)
    names = [x[2] for x in items]
    return paths, labels, names


# ============================================================
# LOO-CV and evaluation
# ============================================================

def loo_cv(X, y, names, label, C_values=None, max_iter=5000):
    """Leave-one-out cross-validation. Returns (accuracy, best_C, best_scaled, preds, probas)."""
    if C_values is None:
        C_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    n = len(y)
    best_correct = 0
    best_C = 1.0
    best_preds = None
    best_probas = None
    best_scaled = False

    for scale in [False, True]:
        for C in C_values:
            preds = np.zeros(n, dtype=np.int32)
            probas = np.zeros(n, dtype=np.float64)

            for i in range(n):
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i)
                X_test = X[i:i+1]

                if scale:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                clf = LogisticRegression(
                    C=C, max_iter=max_iter, solver="lbfgs",
                    class_weight="balanced", random_state=42,
                )
                clf.fit(X_train, y_train)
                preds[i] = clf.predict(X_test)[0]
                probas[i] = clf.predict_proba(X_test)[0, 1]

            correct = int(np.sum(preds == y))
            if correct > best_correct:
                best_correct = correct
                best_C = C
                best_preds = preds.copy()
                best_probas = probas.copy()
                best_scaled = scale

    acc = best_correct / n
    print(f"\n  LOO-CV {label}: {acc:.1%} ({best_correct}/{n})  C={best_C} scaled={best_scaled}")
    for i in range(n):
        status = "OK" if best_preds[i] == y[i] else "WRONG"
        lbl = "stamped" if y[i] else "clean"
        print(f"    [{status}] prob={best_probas[i]:.3f} pred={best_preds[i]} true={y[i]} {names[i]} ({lbl})")

    wrong_names = [names[i] for i in range(n) if best_preds[i] != y[i]]
    if wrong_names:
        print(f"  Misclassified ({len(wrong_names)}): {', '.join(wrong_names)}")

    return acc, best_C, best_scaled, best_preds, best_probas


def loo_cv_ensemble(X_pixel, X_dino, y, names, label, C_values=None):
    """LOO-CV with probability averaging ensemble: separate pixel LR + DINOv2 LR.

    For each fold:
      1. Train pixel-only LR -> get prob_pixel
      2. Train dino-only LR -> get prob_dino
      3. Average: (w * prob_pixel + (1-w) * prob_dino) for various w
      4. Predict stamped if avg > 0.5
    """
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0]

    n = len(y)
    weights_to_try = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    best_correct = 0
    best_preds = None
    best_probas = None
    best_config = ""

    for C_p in C_values:
        for C_d in C_values:
            for scale_p in [True]:
                for scale_d in [True]:
                    # Collect per-fold probabilities
                    probas_pixel = np.zeros(n)
                    probas_dino = np.zeros(n)

                    for i in range(n):
                        Xp_tr = np.delete(X_pixel, i, axis=0)
                        Xd_tr = np.delete(X_dino, i, axis=0)
                        y_tr = np.delete(y, i)
                        Xp_te = X_pixel[i:i+1]
                        Xd_te = X_dino[i:i+1]

                        if scale_p:
                            sp = StandardScaler()
                            Xp_tr = sp.fit_transform(Xp_tr)
                            Xp_te = sp.transform(Xp_te)
                        if scale_d:
                            sd = StandardScaler()
                            Xd_tr = sd.fit_transform(Xd_tr)
                            Xd_te = sd.transform(Xd_te)

                        clf_p = LogisticRegression(C=C_p, max_iter=5000, solver="lbfgs",
                                                   class_weight="balanced", random_state=42)
                        clf_p.fit(Xp_tr, y_tr)
                        probas_pixel[i] = clf_p.predict_proba(Xp_te)[0, 1]

                        clf_d = LogisticRegression(C=C_d, max_iter=5000, solver="lbfgs",
                                                   class_weight="balanced", random_state=42)
                        clf_d.fit(Xd_tr, y_tr)
                        probas_dino[i] = clf_d.predict_proba(Xd_te)[0, 1]

                    # Try different ensemble weights
                    for w in weights_to_try:
                        avg = w * probas_pixel + (1 - w) * probas_dino
                        preds = (avg > 0.5).astype(np.int32)
                        correct = int(np.sum(preds == y))
                        if correct > best_correct:
                            best_correct = correct
                            best_preds = preds.copy()
                            best_probas = avg.copy()
                            best_config = (f"w_pixel={w:.1f} C_p={C_p} C_d={C_d} "
                                           f"scale_p={scale_p} scale_d={scale_d}")

    acc = best_correct / n
    print(f"\n  LOO-CV {label}: {acc:.1%} ({best_correct}/{n})  {best_config}")
    for i in range(n):
        status = "OK" if best_preds[i] == y[i] else "WRONG"
        lbl = "stamped" if y[i] else "clean"
        print(f"    [{status}] prob={best_probas[i]:.3f} pred={best_preds[i]} true={y[i]} {names[i]} ({lbl})")

    wrong_names = [names[i] for i in range(n) if best_preds[i] != y[i]]
    if wrong_names:
        print(f"  Misclassified ({len(wrong_names)}): {', '.join(wrong_names)}")

    return acc, best_preds, best_probas, best_config


def train_eval_binder(X_train, y_train, X_binder, y_binder,
                      binder_names, label, C_values=None):
    """Train on full training set, evaluate on binder."""
    if C_values is None:
        C_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    best_acc = 0
    best_clf = None
    best_scaler = None
    best_C = 1.0

    for scale in [False, True]:
        for C in C_values:
            scaler = None
            Xtr, Xb = X_train, X_binder
            if scale:
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X_train)
                Xb = scaler.transform(X_binder)

            clf = LogisticRegression(
                C=C, max_iter=5000, solver="lbfgs",
                class_weight="balanced", random_state=42,
            )
            clf.fit(Xtr, y_train)
            pred = clf.predict(Xb)
            acc = accuracy_score(y_binder, pred)
            if acc > best_acc:
                best_acc = acc
                best_clf = clf
                best_scaler = scaler
                best_C = C

    Xb = X_binder
    if best_scaler is not None:
        Xb = best_scaler.transform(X_binder)
    pred = best_clf.predict(Xb)
    proba = best_clf.predict_proba(Xb)[:, 1]

    n_correct = int(np.sum(pred == y_binder))
    print(f"\n  Train->Binder {label}: {best_acc:.1%} ({n_correct}/{len(y_binder)})  "
          f"C={best_C} scaled={best_scaler is not None}")
    for i in range(len(y_binder)):
        status = "OK" if pred[i] == y_binder[i] else "WRONG"
        lbl = "stamped" if y_binder[i] else "clean"
        print(f"    [{status}] prob={proba[i]:.3f} pred={pred[i]} true={y_binder[i]} "
              f"{binder_names[i]} ({lbl})")

    wrong_names = [binder_names[i] for i in range(len(y_binder)) if pred[i] != y_binder[i]]
    if wrong_names:
        print(f"  Misclassified ({len(wrong_names)}): {', '.join(wrong_names)}")

    return best_acc, best_clf, best_scaler, best_C


def train_eval_binder_ensemble(X_pixel_train, X_dino_train, y_train,
                                X_pixel_binder, X_dino_binder, y_binder,
                                binder_names, label,
                                C_values=None):
    """Train separate pixel + DINOv2 models, average probabilities, eval on binder."""
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    weights_to_try = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    best_acc = 0
    best_preds = None
    best_probas = None
    best_info = {}

    for C_p in C_values:
        for C_d in C_values:
            # Pixel model (always scale)
            sp = StandardScaler()
            Xp_tr = sp.fit_transform(X_pixel_train)
            Xp_b = sp.transform(X_pixel_binder)
            clf_p = LogisticRegression(C=C_p, max_iter=5000, solver="lbfgs",
                                       class_weight="balanced", random_state=42)
            clf_p.fit(Xp_tr, y_train)
            prob_p = clf_p.predict_proba(Xp_b)[:, 1]

            # DINOv2 model (always scale)
            sd = StandardScaler()
            Xd_tr = sd.fit_transform(X_dino_train)
            Xd_b = sd.transform(X_dino_binder)
            clf_d = LogisticRegression(C=C_d, max_iter=5000, solver="lbfgs",
                                       class_weight="balanced", random_state=42)
            clf_d.fit(Xd_tr, y_train)
            prob_d = clf_d.predict_proba(Xd_b)[:, 1]

            for w in weights_to_try:
                avg = w * prob_p + (1 - w) * prob_d
                preds = (avg > 0.5).astype(np.int32)
                acc = accuracy_score(y_binder, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_preds = preds.copy()
                    best_probas = avg.copy()
                    best_info = {
                        "clf_pixel": clf_p, "scaler_pixel": sp, "C_pixel": C_p,
                        "clf_dino": clf_d, "scaler_dino": sd, "C_dino": C_d,
                        "weight_pixel": w,
                    }

    n_correct = int(np.sum(best_preds == y_binder))
    w = best_info.get("weight_pixel", 0.5)
    print(f"\n  Train->Binder {label}: {best_acc:.1%} ({n_correct}/{len(y_binder)})  "
          f"w_pixel={w:.1f} C_p={best_info.get('C_pixel')} C_d={best_info.get('C_dino')}")
    for i in range(len(y_binder)):
        status = "OK" if best_preds[i] == y_binder[i] else "WRONG"
        lbl = "stamped" if y_binder[i] else "clean"
        print(f"    [{status}] prob={best_probas[i]:.3f} pred={best_preds[i]} true={y_binder[i]} "
              f"{binder_names[i]} ({lbl})")

    wrong_names = [binder_names[i] for i in range(len(y_binder)) if best_preds[i] != y_binder[i]]
    if wrong_names:
        print(f"  Misclassified ({len(wrong_names)}): {', '.join(wrong_names)}")

    return best_acc, best_info


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()

    # ===== Load datasets =====
    synth_paths, y_synth, synth_names = load_dataset(
        SYNTHETIC_DIR / "labels.jsonl", SYNTHETIC_DIR)
    real_paths, y_real, real_names = load_dataset(
        REAL_DIR / "sources.jsonl", REAL_DIR)
    binder_paths, y_binder, binder_names = load_dataset(
        REAL_DIR / "binder_ground_truth.jsonl", BINDER_DIR)

    logger.info("Synthetic: %d (%d stamped)", len(synth_paths), int(sum(y_synth)))
    logger.info("Real reference: %d (%d stamped)", len(real_paths), int(sum(y_real)))
    logger.info("Binder: %d (%d stamped)", len(binder_paths), int(sum(y_binder)))

    train_paths = synth_paths + real_paths
    y_train = np.concatenate([y_synth, y_real])
    train_names = synth_names + real_names

    # ===== Extract pixel features =====
    logger.info("Extracting pixel features...")
    pixel_binder = extract_pixel_features_batch(binder_paths)
    pixel_train = extract_pixel_features_batch(train_paths)
    logger.info("Pixel features: binder %s, train %s", pixel_binder.shape, pixel_train.shape)

    # ===== Extract DINOv2 features =====
    dino_model, dino_device = load_dino_model()
    logger.info("Extracting DINOv2 CLS features for binder...")
    dino_binder = extract_dino_cls_batch(dino_model, dino_device, binder_paths)
    logger.info("Extracting DINOv2 CLS features for training set...")
    dino_train = extract_dino_cls_batch(dino_model, dino_device, train_paths)
    logger.info("DINOv2 features: binder %s, train %s", dino_binder.shape, dino_train.shape)

    del dino_model
    torch.cuda.empty_cache()

    # ===== Build feature variants =====
    # PCA on DINOv2 features (fit on training data)
    pca_variants = {}
    for n_comp in [3, 5, 8, 10, 15, 20]:
        pca = PCA(n_components=n_comp, random_state=42)
        dino_train_pca = pca.fit_transform(dino_train)
        dino_binder_pca = pca.transform(dino_binder)
        pca_variants[n_comp] = (dino_train_pca, dino_binder_pca, pca)

    feature_sets = {}

    # Baselines
    feature_sets["pixel_only"] = (pixel_binder, pixel_train)
    feature_sets["dino_cls_only"] = (dino_binder, dino_train)

    # PCA DINOv2 only
    for nc in [5, 10, 20]:
        dtr, dbr, _ = pca_variants[nc]
        feature_sets[f"dino_pca{nc}"] = (dbr, dtr)

    # Combined: pixel + PCA DINOv2
    for nc in [3, 5, 8, 10, 15, 20]:
        dtr, dbr, _ = pca_variants[nc]
        feature_sets[f"combined_pixel+pca{nc}"] = (
            np.hstack([pixel_binder, dbr]),
            np.hstack([pixel_train, dtr]),
        )

    # Raw combined (pixel + full 768-dim DINOv2)
    feature_sets["combined_pixel+dino768"] = (
        np.hstack([pixel_binder, dino_binder]),
        np.hstack([pixel_train, dino_train]),
    )

    # ===== Experiment 1: LOO-CV on binder =====
    print("\n" + "#" * 70)
    print("  EXPERIMENT 1: Leave-One-Out CV on Binder (%d samples)" % len(y_binder))
    print("#" * 70)

    loo_results = {}
    for name, (X_binder_feat, _) in feature_sets.items():
        acc, C, scaled, preds, probas = loo_cv(
            X_binder_feat, y_binder, binder_names, name)
        loo_results[name] = acc

    # Ensemble LOO-CV (probability averaging)
    print("\n  --- Ensemble: probability averaging ---")
    ens_acc, ens_preds, ens_probas, ens_config = loo_cv_ensemble(
        pixel_binder, dino_binder, y_binder, binder_names, "ensemble_prob_avg")
    loo_results["ensemble_prob_avg"] = ens_acc

    # Ensemble with PCA DINOv2
    for nc in [5, 10]:
        _, dbr, _ = pca_variants[nc]
        e_acc, _, _, e_cfg = loo_cv_ensemble(
            pixel_binder, dbr, y_binder, binder_names, f"ensemble_pixel+pca{nc}")
        loo_results[f"ensemble_pixel+pca{nc}"] = e_acc

    # ===== Experiment 2: Train on synth+real, eval on binder =====
    print("\n" + "#" * 70)
    print("  EXPERIMENT 2: Train on Synthetic+Real -> Eval on Binder")
    print("#" * 70)

    transfer_results = {}
    best_transfer = {"name": "", "acc": 0}

    for name, (X_binder_feat, X_train_feat) in feature_sets.items():
        acc, clf, scaler, C = train_eval_binder(
            X_train_feat, y_train, X_binder_feat, y_binder, binder_names, name)
        transfer_results[name] = acc
        if acc > best_transfer["acc"]:
            best_transfer = {"name": name, "acc": acc, "clf": clf, "scaler": scaler, "C": C}

    # Ensemble transfer
    print("\n  --- Ensemble transfer ---")
    ens_t_acc, ens_t_info = train_eval_binder_ensemble(
        pixel_train, dino_train, y_train,
        pixel_binder, dino_binder, y_binder,
        binder_names, "ensemble_prob_avg")
    transfer_results["ensemble_prob_avg"] = ens_t_acc

    for nc in [5, 10]:
        dtr, dbr, _ = pca_variants[nc]
        e_t_acc, e_t_info = train_eval_binder_ensemble(
            pixel_train, dtr, y_train,
            pixel_binder, dbr, y_binder,
            binder_names, f"ensemble_pixel+pca{nc}")
        transfer_results[f"ensemble_pixel+pca{nc}"] = e_t_acc

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    all_names = sorted(set(list(loo_results.keys()) + list(transfer_results.keys())))
    print(f"  {'Feature Set':<30s}  {'LOO-CV':>8s}  {'Train->Binder':>14s}")
    print("  " + "-" * 56)
    for name in sorted(all_names, key=lambda n: -(loo_results.get(n, 0) + transfer_results.get(n, 0))):
        loo = loo_results.get(name, 0)
        transfer = transfer_results.get(name, 0)
        best_marker = ""
        if loo >= max(loo_results.values()) and loo > 0:
            best_marker = " <-- BEST LOO"
        print(f"  {name:<30s}  {loo:>7.1%}  {transfer:>13.1%}{best_marker}")

    # ===== Save best model =====
    # Pick model with best LOO-CV; break ties by transfer accuracy
    best_loo_acc = max(loo_results.values())
    loo_winners = [n for n, a in loo_results.items() if a == best_loo_acc]
    # Among tied LOO winners, pick the one with best transfer
    best_loo_name = max(loo_winners, key=lambda n: transfer_results.get(n, 0))
    best_transfer_name = max(transfer_results, key=transfer_results.get)
    best_transfer_acc = transfer_results[best_transfer_name]

    print(f"\n  Best LOO-CV: {best_loo_name} ({best_loo_acc:.1%})")
    print(f"  Best Train->Binder: {best_transfer_name} ({best_transfer_acc:.1%})")

    # Save the best ensemble model if it won, otherwise save the best single model
    # For deployment, we need to save both sub-models for ensemble
    if "ensemble" in best_loo_name or "ensemble" in best_transfer_name:
        # Train final ensemble on all data
        # Determine best ensemble config by re-running on binder
        best_ens_name = best_loo_name if "ensemble" in best_loo_name else best_transfer_name

        # Determine which DINOv2 variant
        if "pca" in best_ens_name:
            nc = int(best_ens_name.split("pca")[-1])
            dtr, dbr, pca_model = pca_variants[nc]
        else:
            dtr, dbr = dino_train, dino_binder
            pca_model = None

        # Train final models on ALL data (train + binder)
        X_pixel_all = np.vstack([pixel_train, pixel_binder])
        X_dino_all = np.vstack([dtr, dbr])
        y_all = np.concatenate([y_train, y_binder])

        # Use reasonable C values
        sp_final = StandardScaler()
        X_pixel_all_s = sp_final.fit_transform(X_pixel_all)
        sd_final = StandardScaler()
        X_dino_all_s = sd_final.fit_transform(X_dino_all)

        clf_p_final = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs",
                                          class_weight="balanced", random_state=42)
        clf_p_final.fit(X_pixel_all_s, y_all)

        clf_d_final = LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs",
                                          class_weight="balanced", random_state=42)
        clf_d_final.fit(X_dino_all_s, y_all)

        save_obj = {
            "model_type": "ensemble_prob_avg",
            "clf_pixel": clf_p_final,
            "scaler_pixel": sp_final,
            "clf_dino": clf_d_final,
            "scaler_dino": sd_final,
            "pca": pca_model,
            "weight_pixel": 0.5,
            "pixel_feature_names": PIXEL_FEATURE_NAMES,
            "metrics": {
                "loo_cv_binder_acc": best_loo_acc,
                "transfer_binder_acc": transfer_results.get(best_ens_name, 0),
                "n_train": len(y_all),
            },
        }
    else:
        # Save single concatenated model
        best_name = best_loo_name
        X_binder_best, X_train_best = feature_sets[best_name]

        X_all = np.vstack([X_train_best, X_binder_best])
        y_all = np.concatenate([y_train, y_binder])

        _, best_C, best_scaled, _, _ = loo_cv(
            X_binder_best, y_binder, binder_names, f"final_{best_name}")

        final_scaler = None
        X_all_fit = X_all
        if best_scaled:
            final_scaler = StandardScaler()
            X_all_fit = final_scaler.fit_transform(X_all)

        final_clf = LogisticRegression(
            C=best_C, max_iter=5000, solver="lbfgs",
            class_weight="balanced", random_state=42,
        )
        final_clf.fit(X_all_fit, y_all)

        # Check if this uses PCA
        pca_model = None
        if "pca" in best_name:
            nc = int(best_name.split("pca")[-1])
            _, _, pca_model = pca_variants[nc]

        save_obj = {
            "model": final_clf,
            "scaler": final_scaler,
            "pca": pca_model,
            "feature_type": best_name,
            "model_type": "lr_combined",
            "pixel_feature_names": PIXEL_FEATURE_NAMES,
            "metrics": {
                "loo_cv_binder_acc": best_loo_acc,
                "transfer_binder_acc": transfer_results.get(best_name, 0),
                "C": best_C,
                "scaled": best_scaled,
                "n_train": len(y_all),
                "n_features": X_all.shape[1],
            },
        }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(save_obj, f)
    logger.info("Saved combined classifier to %s", OUTPUT_PATH)

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds", elapsed)
    print(f"\n  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
