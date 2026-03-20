#!/usr/bin/env python3
"""LBP texture classifier for stamp/holo detection on Pokemon cards.

Extracts Local Binary Pattern histograms from multiple card regions
and compares LBP, edge density, and DINOv2 features for separating
stamped reverse holo cards from normal/holofoil cards.

Usage:
    python scripts/lbp_stamp_classifier.py
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INBOX_DIR = PROJECT_ROOT / "data" / "inbox"
GT_PATH = PROJECT_ROOT / "data" / "condition_training" / "stamps_real" / "binder_ground_truth.jsonl"

# LBP parameters
LBP_RADIUS = 2
LBP_NPOINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"
# uniform LBP with P=16, R=2 produces P+2=18 bins
LBP_NBINS = LBP_NPOINTS + 2


def load_ground_truth():
    """Load binder ground truth, resolving image paths."""
    entries = []
    with open(GT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            img_path = INBOX_DIR / entry["image"]
            if not img_path.exists():
                logger.warning("Image not found: %s", img_path)
                continue
            entry["image_path"] = str(img_path)
            entries.append(entry)
    return entries


def extract_lbp_histogram(gray_region: np.ndarray) -> np.ndarray:
    """Compute normalized LBP histogram for a grayscale region."""
    lbp = local_binary_pattern(gray_region, LBP_NPOINTS, LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_NBINS, range=(0, LBP_NBINS), density=True)
    return hist.astype(np.float32)


def extract_regions(img_path: str) -> dict:
    """Extract card regions as grayscale arrays.

    Returns dict with keys: stamp_region, border_region, artwork_center, full_card.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Stamp region: bottom-right quadrant of the artwork area
    # Cards have ~10% border on each side, artwork is roughly middle 80%
    # Stamp is in bottom-right of the artwork, roughly bottom 40%, right 40%
    stamp_region = gray[int(h * 0.55):int(h * 0.90), int(w * 0.55):int(w * 0.90)]

    # Border region: card frame (top 15% and bottom 10%, left/right 12%)
    border_top = gray[0:int(h * 0.12), int(w * 0.1):int(w * 0.9)]
    border_bottom = gray[int(h * 0.88):, int(w * 0.1):int(w * 0.9)]
    border_left = gray[int(h * 0.12):int(h * 0.88), 0:int(w * 0.1)]
    border_right = gray[int(h * 0.12):int(h * 0.88), int(w * 0.9):]
    # Combine border strips into one region by stacking
    max_w = max(border_top.shape[1], border_bottom.shape[1])
    # Just use top and bottom borders (most consistent)
    border_region = np.vstack([border_top, border_bottom])

    # Artwork center: middle of the card
    artwork_center = gray[int(h * 0.20):int(h * 0.55), int(w * 0.15):int(w * 0.85)]

    return {
        "stamp_region": stamp_region,
        "border_region": border_region,
        "artwork_center": artwork_center,
        "full_card": gray,
    }


def extract_lbp_features(img_path: str) -> np.ndarray:
    """Extract multi-region LBP histogram features from a card image.

    Returns a feature vector combining LBP histograms from multiple regions
    plus statistical texture descriptors.
    """
    regions = extract_regions(img_path)
    features = []

    for region_name in ["stamp_region", "border_region", "artwork_center"]:
        region = regions[region_name]
        # LBP histogram
        lbp_hist = extract_lbp_histogram(region)
        features.append(lbp_hist)

        # Additional texture stats from LBP image
        lbp_img = local_binary_pattern(region, LBP_NPOINTS, LBP_RADIUS, method=LBP_METHOD)
        features.append(np.array([
            np.mean(lbp_img),
            np.std(lbp_img),
            np.median(lbp_img),
        ], dtype=np.float32))

    # Cross-region contrast: stamp vs border, stamp vs artwork
    stamp_hist = features[0]  # first region's histogram
    border_hist = features[2]  # second region's histogram
    artwork_hist = features[4]  # third region's histogram

    # Chi-squared distance between histograms
    def chi_sq(a, b):
        denom = a + b + 1e-10
        return np.sum((a - b) ** 2 / denom)

    features.append(np.array([
        chi_sq(stamp_hist, border_hist),
        chi_sq(stamp_hist, artwork_hist),
        chi_sq(border_hist, artwork_hist),
    ], dtype=np.float32))

    return np.concatenate(features)


def extract_edge_features(img_path: str) -> np.ndarray:
    """Extract edge density features from card regions."""
    regions = extract_regions(img_path)
    features = []

    for region_name in ["stamp_region", "border_region", "artwork_center"]:
        region = regions[region_name]
        # Canny edge detection
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.mean(edges > 0)

        # Laplacian variance (focus/texture measure)
        lap = cv2.Laplacian(region, cv2.CV_64F)
        lap_var = np.var(lap)
        lap_mean = np.mean(np.abs(lap))

        # Sobel gradients
        sobelx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

        features.append(np.array([
            edge_density,
            lap_var,
            lap_mean,
            np.mean(grad_mag),
            np.std(grad_mag),
        ], dtype=np.float32))

    return np.concatenate(features)


def extract_glcm_features(img_path: str) -> np.ndarray:
    """Extract GLCM (Gray-Level Co-Occurrence Matrix) texture features."""
    from skimage.feature import graycomatrix, graycoprops

    regions = extract_regions(img_path)
    features = []

    for region_name in ["stamp_region", "border_region", "artwork_center"]:
        region = regions[region_name]
        # Quantize to 64 levels for GLCM
        quantized = (region / 4).astype(np.uint8)
        glcm = graycomatrix(quantized, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2],
                            levels=64, symmetric=True, normed=True)

        props = []
        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
            vals = graycoprops(glcm, prop)
            props.append(vals.mean())
            props.append(vals.std())

        features.append(np.array(props, dtype=np.float32))

    return np.concatenate(features)


def extract_combined_features(img_path: str) -> np.ndarray:
    """LBP + edge + GLCM combined."""
    lbp = extract_lbp_features(img_path)
    edge = extract_edge_features(img_path)
    glcm = extract_glcm_features(img_path)
    return np.concatenate([lbp, edge, glcm])


def extract_dino_features(img_path: str) -> np.ndarray:
    """Extract DINOv2 features for comparison (uses existing stamp classifier infra)."""
    from cardprice.ml.stamp_classifier import _extract_features, _patch_stats, _get_region_patches

    cls_token, patch_tokens = _extract_features(img_path)
    br = _get_region_patches(patch_tokens, 'br')
    bc = _get_region_patches(patch_tokens, 'bottom_center')
    bl = _get_region_patches(patch_tokens, 'bl')

    # Simplified: CLS + bottom-right stats
    br_stats = _patch_stats(br)
    bc_stats = _patch_stats(bc)
    br_mean = np.mean(br, axis=0)
    bl_mean = np.mean(bl, axis=0)

    return np.concatenate([cls_token, br_stats, bc_stats, br_mean - bl_mean])


def leave_one_out_cv(X, y, labels, model_class, model_kwargs=None, scale=True):
    """Leave-one-out cross-validation. Returns predictions and probabilities."""
    n = len(y)
    preds = np.zeros(n, dtype=int)
    probs = np.zeros(n, dtype=float)

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i + 1]

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        clf = model_class(**(model_kwargs or {}))
        clf.fit(X_train, y_train)
        preds[i] = clf.predict(X_test)[0]
        if hasattr(clf, "predict_proba"):
            probs[i] = clf.predict_proba(X_test)[0, 1]
        elif hasattr(clf, "decision_function"):
            probs[i] = clf.decision_function(X_test)[0]

    return preds, probs


def print_results(name, y_true, y_pred, probs, labels, entries):
    """Print detailed classification results."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy: {acc:.1%} ({int(np.sum(y_true == y_pred))}/{len(y_true)})")
    print()

    # Per-sample results
    for i, entry in enumerate(entries):
        correct = "OK" if y_true[i] == y_pred[i] else "MISS"
        stamp_str = "STAMP" if y_true[i] == 1 else "clean"
        pred_str = "STAMP" if y_pred[i] == 1 else "clean"
        prob_str = f"{probs[i]:.3f}" if probs[i] > 0 else "n/a"
        print(f"  [{correct}] {entry['card_name']:25s} true={stamp_str:5s} pred={pred_str:5s} prob={prob_str}  ({entry.get('variant', '')})")

    print()
    return acc


def main():
    entries = load_ground_truth()
    logger.info("Loaded %d ground truth entries", len(entries))

    # Handle duplicate entry for Dragonite (lines 12 and 17 -- same image, different labels)
    # Keep line 17 (stamped=true, prerelease) as it's the correction
    seen = {}
    deduped = []
    for e in entries:
        key = e["image_path"]
        if key in seen:
            logger.info("Duplicate image %s -- keeping later entry (stamped=%s)", key, e["stamped"])
            # Replace the earlier entry
            for j, d in enumerate(deduped):
                if d["image_path"] == key:
                    deduped[j] = e
                    break
        else:
            seen[key] = True
            deduped.append(e)
    entries = deduped
    logger.info("After dedup: %d entries", len(entries))

    # Labels: 1=stamped, 0=not stamped
    y = np.array([1 if e["stamped"] else 0 for e in entries])
    labels = [e["card_name"] for e in entries]

    logger.info("Class distribution: %d stamped, %d clean", np.sum(y == 1), np.sum(y == 0))
    print()

    # ---- Extract all features ----
    print("Extracting features...")

    # 1) LBP features
    print("  LBP histograms...", flush=True)
    X_lbp = np.array([extract_lbp_features(e["image_path"]) for e in entries])
    print(f"    Shape: {X_lbp.shape}")

    # 2) Edge density features
    print("  Edge density...", flush=True)
    X_edge = np.array([extract_edge_features(e["image_path"]) for e in entries])
    print(f"    Shape: {X_edge.shape}")

    # 3) GLCM features
    print("  GLCM texture...", flush=True)
    X_glcm = np.array([extract_glcm_features(e["image_path"]) for e in entries])
    print(f"    Shape: {X_glcm.shape}")

    # 4) Combined (LBP + edge + GLCM)
    X_combined = np.concatenate([X_lbp, X_edge, X_glcm], axis=1)
    print(f"  Combined shape: {X_combined.shape}")

    # 5) DINOv2 features
    print("  DINOv2 features...", flush=True)
    X_dino = np.array([extract_dino_features(e["image_path"]) for e in entries])
    print(f"    Shape: {X_dino.shape}")

    # 6) LBP + DINOv2 combined
    X_lbp_dino = np.concatenate([X_lbp, X_dino], axis=1)
    print(f"  LBP+DINOv2 shape: {X_lbp_dino.shape}")

    # ---- Leave-one-out CV for each feature set ----
    feature_sets = {
        "LBP only": X_lbp,
        "Edge density only": X_edge,
        "GLCM only": X_glcm,
        "LBP + Edge + GLCM": X_combined,
        "DINOv2 (baseline)": X_dino,
        "LBP + DINOv2": X_lbp_dino,
    }

    models = {
        "LogisticRegression": (LogisticRegression, {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}),
        "SVM-RBF": (SVC, {"C": 1.0, "kernel": "rbf", "probability": True, "class_weight": "balanced"}),
    }

    results = {}
    for feat_name, X_feat in feature_sets.items():
        for model_name, (model_cls, model_kw) in models.items():
            full_name = f"{feat_name} + {model_name}"
            preds, probs = leave_one_out_cv(X_feat, y, labels, model_cls, model_kw)
            acc = print_results(full_name, y, preds, probs, labels, entries)
            results[full_name] = acc

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  SUMMARY: Leave-One-Out CV Accuracy")
    print("=" * 60)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "#" * int(acc * 40)
        print(f"  {acc:5.1%}  {bar:40s}  {name}")

    # ---- Feature importance analysis for best LBP model ----
    print("\n" + "=" * 60)
    print("  LBP Feature Analysis")
    print("=" * 60)

    # Look at LBP histogram differences between stamped and clean
    stamped_idx = np.where(y == 1)[0]
    clean_idx = np.where(y == 0)[0]

    for region_idx, region_name in enumerate(["stamp_region", "border_region", "artwork_center"]):
        # Each region contributes LBP_NBINS + 3 features
        feat_size = LBP_NBINS + 3
        start = region_idx * feat_size
        end = start + LBP_NBINS  # just the histogram part

        stamped_hist = X_lbp[stamped_idx, start:end].mean(axis=0)
        clean_hist = X_lbp[clean_idx, start:end].mean(axis=0)
        diff = np.abs(stamped_hist - clean_hist)

        print(f"\n  {region_name}:")
        print(f"    Mean histogram diff (stamped vs clean): {diff.mean():.4f}")
        print(f"    Max histogram diff: {diff.max():.4f} (bin {diff.argmax()})")
        print(f"    Stamped LBP mean: {X_lbp[stamped_idx, end:end+3].mean(axis=0)}")
        print(f"    Clean LBP mean:   {X_lbp[clean_idx, end:end+3].mean(axis=0)}")

    # Edge density comparison
    print(f"\n  Edge density comparison:")
    for region_idx, region_name in enumerate(["stamp_region", "border_region", "artwork_center"]):
        start = region_idx * 5
        stamped_edge = X_edge[stamped_idx, start].mean()
        clean_edge = X_edge[clean_idx, start].mean()
        print(f"    {region_name}: stamped={stamped_edge:.4f} clean={clean_edge:.4f} diff={abs(stamped_edge-clean_edge):.4f}")


if __name__ == "__main__":
    main()
