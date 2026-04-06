#!/usr/bin/env python3
"""
Ensemble stamp detector: combines pixel-level features + DINOv2 features.

Extracts diverse signal types from each binder card:
  - edge_density_ratio (stamp vs control region)
  - DINOv2 CLS token (224x224 full card) — reduced via PCA
  - DINOv2 patch stats from bottom-right 4x4 patches — reduced via PCA
  - Mean brightness difference (stamp vs control)
  - Saturation variance in stamp region
  - High-frequency energy ratio (FFT)
  - Gradient magnitude in stamp region
  - Color channel ratios (R/G, R/B)

Trains multiple classifiers with leave-one-out CV on binder samples.
Uses PCA to reduce DINOv2 dimensions and avoid curse of dimensionality.
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

warnings.filterwarnings("ignore")

BASE = Path("/home/godli/cardprice")
INBOX = BASE / "data/inbox"
GT_PATH = BASE / "data/condition_training/stamps_real/binder_ground_truth.jsonl"
OUT_MODEL = BASE / "data/stamp_ensemble_classifier.pkl"


# ============================================================================
# Pixel-level feature extraction
# ============================================================================

def crop_stamp_region(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(h * 0.45):int(h * 0.70), int(w * 0.55):int(w * 0.90)]


def crop_control_region(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(h * 0.45):int(h * 0.70), int(w * 0.10):int(w * 0.45)]


def compute_edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges > 0))


def compute_high_freq_energy(gray: np.ndarray) -> float:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 5
    magnitude = np.abs(fshift)
    mask = np.ones_like(magnitude)
    mask[cy - r:cy + r, cx - r:cx + r] = 0
    return float(np.sum(magnitude * mask) / (np.sum(magnitude) + 1e-10))


def compute_gradient_magnitude(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))


def extract_pixel_features(img: np.ndarray) -> dict:
    """Extract all pixel-level features from a card image."""
    stamp = crop_stamp_region(img)
    control = crop_control_region(img)

    stamp_gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
    control_gray = cv2.cvtColor(control, cv2.COLOR_BGR2GRAY)
    stamp_hsv = cv2.cvtColor(stamp, cv2.COLOR_BGR2HSV)
    stamp_rgb = cv2.cvtColor(stamp, cv2.COLOR_BGR2RGB).astype(np.float32)

    stamp_edges = compute_edge_density(stamp_gray)
    control_edges = compute_edge_density(control_gray)

    stamp_brightness = float(np.mean(stamp_gray))
    control_brightness = float(np.mean(control_gray))

    sat = stamp_hsv[:, :, 1].astype(np.float32)

    stamp_hf = compute_high_freq_energy(stamp_gray)
    control_hf = compute_high_freq_energy(control_gray)

    stamp_grad = compute_gradient_magnitude(stamp_gray)
    control_grad = compute_gradient_magnitude(control_gray)

    mean_r = float(np.mean(stamp_rgb[:, :, 0]))
    mean_g = float(np.mean(stamp_rgb[:, :, 1]))
    mean_b = float(np.mean(stamp_rgb[:, :, 2]))

    lap_stamp = cv2.Laplacian(stamp_gray, cv2.CV_64F)
    lap_control = cv2.Laplacian(control_gray, cv2.CV_64F)
    laplacian_var = float(np.var(lap_stamp))
    control_lap_var = float(np.var(lap_control))

    hue = stamp_hsv[:, :, 0]
    lower = np.array([15, 40, 120])
    upper = np.array([40, 255, 255])
    gold_mask = cv2.inRange(stamp_hsv, lower, upper)

    return {
        "edge_density": stamp_edges,
        "edge_density_ratio": stamp_edges / (control_edges + 1e-10),
        "brightness_diff": stamp_brightness - control_brightness,
        "brightness_ratio": stamp_brightness / (control_brightness + 1e-10),
        "sat_variance": float(np.var(sat)),
        "sat_mean": float(np.mean(sat)),
        "hf_ratio": stamp_hf / (control_hf + 1e-10),
        "grad_ratio": stamp_grad / (control_grad + 1e-10),
        "rg_ratio": mean_r / (mean_g + 1e-10),
        "rb_ratio": mean_r / (mean_b + 1e-10),
        "laplacian_var": laplacian_var,
        "laplacian_ratio": laplacian_var / (control_lap_var + 1e-10),
        "gold_hue_fraction": float(np.mean((hue >= 15) & (hue <= 40))),
        "gold_pixel_ratio": float(np.mean(gold_mask > 0)),
        "hue_std": float(np.std(hue.astype(np.float32))),
        "stamp_grad_mag": stamp_grad,
    }


# ============================================================================
# DINOv2 feature extraction
# ============================================================================

def extract_dino_raw(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw DINOv2 CLS token and bottom-right patch stats.

    Returns:
        cls_token: (768,)
        br_stats: (768*3,) = mean, std, max of bottom-right 4x4 patches
    """
    import torch
    from PIL import Image as PILImage
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from cardprice.ml.dino_matcher import _load_model
    model, device = _load_model()

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_out = model(tensor)
        patch_out = model.get_intermediate_layers(tensor, n=1)
        patch_tokens = patch_out[0].squeeze(0)

    cls_np = cls_out.cpu().numpy().astype(np.float32).squeeze()
    norm = np.linalg.norm(cls_np)
    if norm > 0:
        cls_np /= norm

    patches_np = patch_tokens.cpu().numpy().astype(np.float32)
    pnorms = np.linalg.norm(patches_np, axis=1, keepdims=True)
    pnorms[pnorms == 0] = 1.0
    patches_np /= pnorms

    grid = patches_np.reshape(16, 16, 768)
    br_patches = grid[12:, 12:, :].reshape(-1, 768)

    br_mean = np.mean(br_patches, axis=0)
    br_std = np.std(br_patches, axis=0)
    br_max = np.max(br_patches, axis=0)
    br_stats = np.concatenate([br_mean, br_std, br_max])

    return cls_np, br_stats


# ============================================================================
# Data loading
# ============================================================================

def load_binder_data() -> list[dict]:
    samples = []
    with open(GT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            img_path = INBOX / entry["image"]
            key = str(img_path)
            # Deduplicate: last entry wins
            samples = [s for s in samples if s["path"] != key]
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  WARN: Could not load {img_path}")
                continue
            samples.append({
                "img": img,
                "label": int(entry["stamped"]),
                "name": entry.get("card_name", "unknown"),
                "path": key,
                "entry": entry,
            })
    return samples


# ============================================================================
# Feature matrix building with PCA for DINOv2
# ============================================================================

PIXEL_FEATURE_NAMES = [
    "edge_density", "edge_density_ratio",
    "brightness_diff", "brightness_ratio",
    "sat_variance", "sat_mean",
    "hf_ratio", "grad_ratio",
    "rg_ratio", "rb_ratio",
    "laplacian_var", "laplacian_ratio",
    "gold_hue_fraction", "gold_pixel_ratio",
    "hue_std", "stamp_grad_mag",
]


def extract_all_features(samples: list[dict]) -> tuple:
    """Extract pixel features + raw DINOv2 vectors for all samples.

    Returns:
        pixel_X: (n, 16) pixel features
        dino_cls: (n, 768) CLS tokens
        dino_br: (n, 2304) BR patch stats
        y: (n,) labels
    """
    pixel_feats = []
    cls_tokens = []
    br_stats_list = []
    labels = []

    print(f"\nExtracting features from {len(samples)} samples...")
    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {s['name']} ({'STAMP' if s['label'] else 'clean'})")

        pf = extract_pixel_features(s["img"])
        pixel_feats.append([pf[k] for k in PIXEL_FEATURE_NAMES])

        cls_tok, br_stats = extract_dino_raw(s["img"])
        cls_tokens.append(cls_tok)
        br_stats_list.append(br_stats)

        labels.append(s["label"])

    pixel_X = np.array(pixel_feats, dtype=np.float32)
    dino_cls = np.array(cls_tokens, dtype=np.float32)
    dino_br = np.array(br_stats_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    return pixel_X, dino_cls, dino_br, y


def evaluate_loo(clf, X, y, clf_name: str, sample_names: list[str],
                 verbose: bool = True) -> tuple[float, list]:
    """Leave-one-out CV. Returns (accuracy, list of errors)."""
    loo = LeaveOneOut()
    correct = 0
    errors = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf_fold = clone(clf)
        clf_fold.fit(X_train, y_train)
        pred = clf_fold.predict(X_test)[0]
        if pred == y_test[0]:
            correct += 1
        else:
            errors.append((sample_names[test_idx[0]], y_test[0], pred))

    acc = correct / len(y)
    if verbose:
        print(f"\n  {clf_name}: LOO = {correct}/{len(y)} = {acc:.1%}")
        for name, true, pred in errors:
            t_str = "stamped" if true else "clean"
            p_str = "stamped" if pred else "clean"
            print(f"    MISS: {name} — true={t_str}, pred={p_str}")
    return acc, errors


def evaluate_loo_with_pca(X_pixel, X_dino, y, n_components, clf, clf_name,
                          sample_names, verbose=True):
    """LOO where PCA is fitted inside each fold (no data leakage)."""
    loo = LeaveOneOut()
    correct = 0
    errors = []

    for train_idx, test_idx in loo.split(X_pixel):
        # Pixel features
        Xp_train = X_pixel[train_idx]
        Xp_test = X_pixel[test_idx]

        # DINOv2 features with PCA fitted on train only
        Xd_train_raw = X_dino[train_idx]
        Xd_test_raw = X_dino[test_idx]

        if n_components > 0 and X_dino.shape[1] > 0:
            nc = min(n_components, Xd_train_raw.shape[0], Xd_train_raw.shape[1])
            pca = PCA(n_components=nc, random_state=42)
            Xd_train = pca.fit_transform(Xd_train_raw)
            Xd_test = pca.transform(Xd_test_raw)
            X_train = np.hstack([Xp_train, Xd_train])
            X_test = np.hstack([Xp_test, Xd_test])
        else:
            X_train = Xp_train
            X_test = Xp_test

        y_train, y_test = y[train_idx], y[test_idx]
        clf_fold = clone(clf)
        clf_fold.fit(X_train, y_train)
        pred = clf_fold.predict(X_test)[0]
        if pred == y_test[0]:
            correct += 1
        else:
            errors.append((sample_names[test_idx[0]], y_test[0], pred))

    acc = correct / len(y)
    if verbose:
        print(f"\n  {clf_name}: LOO = {correct}/{len(y)} = {acc:.1%}")
        for name, true, pred in errors:
            t_str = "stamped" if true else "clean"
            p_str = "stamped" if pred else "clean"
            print(f"    MISS: {name} — true={t_str}, pred={p_str}")
    return acc, errors


def get_feature_importances(clf, feature_names, X, y):
    clf_fitted = clone(clf)
    clf_fitted.fit(X, y)
    if hasattr(clf_fitted, 'feature_importances_'):
        importances = clf_fitted.feature_importances_
    elif hasattr(clf_fitted, 'coef_'):
        importances = np.abs(clf_fitted.coef_[0])
    elif hasattr(clf_fitted, 'named_steps'):
        final = list(clf_fitted.named_steps.values())[-1]
        if hasattr(final, 'feature_importances_'):
            importances = final.feature_importances_
        elif hasattr(final, 'coef_'):
            importances = np.abs(final.coef_[0])
        else:
            return []
    else:
        return []
    pairs = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    return pairs


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  ENSEMBLE STAMP DETECTOR TRAINING")
    print("=" * 70)

    samples = load_binder_data()
    print(f"\nLoaded {len(samples)} binder samples")
    n_stamped = sum(s["label"] for s in samples)
    print(f"  Stamped: {n_stamped}, Clean: {len(samples) - n_stamped}")
    for s in samples:
        tag = "STAMP" if s["label"] else "clean"
        print(f"    [{tag:5s}] {s['name']}")

    sample_names = [s["name"] for s in samples]

    # Extract features
    pixel_X, dino_cls, dino_br, y = extract_all_features(samples)
    dino_all = np.hstack([dino_cls, dino_br])
    print(f"\n  Pixel features: {pixel_X.shape}")
    print(f"  DINOv2 CLS: {dino_cls.shape}")
    print(f"  DINOv2 BR stats: {dino_br.shape}")
    print(f"  DINOv2 all: {dino_all.shape}")

    # ====================================================================
    # SECTION 1: Pixel-only classifiers
    # ====================================================================
    print("\n" + "=" * 70)
    print("  SECTION 1: PIXEL-ONLY FEATURES (16 dims)")
    print("=" * 70)

    classifiers_pixel = {
        "LR(C=0.1)": Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=2000, C=0.1))]),
        "LR(C=1.0)": Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=2000, C=1.0))]),
        "LR(C=10)": Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=2000, C=10.0))]),
        "RF(d=2)": RandomForestClassifier(n_estimators=200, max_depth=2, min_samples_leaf=1, random_state=42),
        "RF(d=3)": RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=1, random_state=42),
        "RF(d=None)": RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2, random_state=42),
        "GB(d=1,lr=0.1)": GradientBoostingClassifier(n_estimators=50, max_depth=1, learning_rate=0.1, random_state=42),
        "GB(d=2,lr=0.05)": GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42),
        "SVM(rbf)": Pipeline([('s', StandardScaler()), ('c', SVC(kernel='rbf', C=1.0, probability=True))]),
        "SVM(rbf,C=10)": Pipeline([('s', StandardScaler()), ('c', SVC(kernel='rbf', C=10.0, probability=True))]),
        "KNN(k=3)": Pipeline([('s', StandardScaler()), ('c', KNeighborsClassifier(n_neighbors=3))]),
        "KNN(k=5)": Pipeline([('s', StandardScaler()), ('c', KNeighborsClassifier(n_neighbors=5))]),
        "AdaBoost(d=1)": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42),
    }

    results_pixel = {}
    for name, clf in classifiers_pixel.items():
        acc, errs = evaluate_loo(clf, pixel_X, y, f"Pixel {name}", sample_names)
        results_pixel[name] = (acc, errs)

    # ====================================================================
    # SECTION 2: Pixel + PCA(DINOv2) with various n_components
    # ====================================================================
    print("\n" + "=" * 70)
    print("  SECTION 2: PIXEL + PCA(DINOv2) COMBINATIONS")
    print("=" * 70)

    results_combined = {}
    dino_sources = {
        "cls": dino_cls,
        "br": dino_br,
        "all": dino_all,
    }

    best_classifiers = [
        ("LR(C=1)", Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=2000, C=1.0))])),
        ("RF(d=2)", RandomForestClassifier(n_estimators=200, max_depth=2, min_samples_leaf=1, random_state=42)),
        ("SVM(rbf)", Pipeline([('s', StandardScaler()), ('c', SVC(kernel='rbf', C=1.0, probability=True))])),
        ("KNN(3)", Pipeline([('s', StandardScaler()), ('c', KNeighborsClassifier(n_neighbors=3))])),
    ]

    for dino_name, dino_X in dino_sources.items():
        for n_comp in [2, 3, 5, 8, 10]:
            for clf_name, clf in best_classifiers:
                full_name = f"pixel+PCA{n_comp}({dino_name})_{clf_name}"
                acc, errs = evaluate_loo_with_pca(
                    pixel_X, dino_X, y, n_comp, clf, full_name,
                    sample_names, verbose=False)
                results_combined[full_name] = (acc, errs)
                if acc >= 0.8:
                    print(f"  ** {full_name}: {acc:.1%}")

    # Print top results
    print("\n  Top 15 combined configurations:")
    sorted_combined = sorted(results_combined.items(), key=lambda x: -x[1][0])
    for i, (name, (acc, errs)) in enumerate(sorted_combined[:15]):
        err_names = [e[0] for e in errs]
        print(f"    {i+1:2d}. {acc:.1%}  {name}")
        if errs:
            print(f"        misses: {', '.join(err_names)}")

    # ====================================================================
    # SECTION 3: DINOv2-only with PCA
    # ====================================================================
    print("\n" + "=" * 70)
    print("  SECTION 3: DINOv2-ONLY (PCA reduced)")
    print("=" * 70)

    results_dino_only = {}
    dummy_pixel = np.zeros((len(y), 0), dtype=np.float32)  # empty pixel features
    for dino_name, dino_X in dino_sources.items():
        for n_comp in [2, 3, 5, 8, 10]:
            for clf_name, clf in best_classifiers:
                # Use pixel_X as zeros so only dino matters
                full_name = f"PCA{n_comp}({dino_name})_{clf_name}"
                # Just apply PCA on the dino features alone
                acc, errs = evaluate_loo_with_pca(
                    np.zeros((len(y), 0), dtype=np.float32),
                    dino_X, y, n_comp, clf, full_name,
                    sample_names, verbose=False)
                results_dino_only[full_name] = (acc, errs)

    print("\n  Top 10 DINOv2-only configurations:")
    sorted_dino = sorted(results_dino_only.items(), key=lambda x: -x[1][0])
    for i, (name, (acc, errs)) in enumerate(sorted_dino[:10]):
        err_names = [e[0] for e in errs]
        print(f"    {i+1:2d}. {acc:.1%}  {name}")
        if errs:
            print(f"        misses: {', '.join(err_names)}")

    # ====================================================================
    # SECTION 4: Voting ensembles of the best combinations
    # ====================================================================
    print("\n" + "=" * 70)
    print("  SECTION 4: VOTING ENSEMBLES")
    print("=" * 70)

    # Gather best pixel-only configs
    best_pixel = sorted(results_pixel.items(), key=lambda x: -x[1][0])[:3]
    print(f"\n  Best pixel-only:")
    for name, (acc, _) in best_pixel:
        print(f"    {acc:.1%} {name}")

    # Gather best combined configs
    print(f"\n  Best combined:")
    for name, (acc, _) in sorted_combined[:3]:
        print(f"    {acc:.1%} {name}")

    # Build voting ensembles from pixel-only classifiers
    # (They all operate on the same feature space, so voting is straightforward)
    vote_pixel_estimators = []
    for name, (acc, _) in best_pixel:
        vote_pixel_estimators.append((name, classifiers_pixel[name]))

    if len(vote_pixel_estimators) >= 2:
        vote_pixel = VotingClassifier(estimators=vote_pixel_estimators, voting='soft')
        acc, errs = evaluate_loo(vote_pixel, pixel_X, y,
                                 "Voting(top3 pixel)", sample_names)
        results_pixel["Voting(top3)"] = (acc, errs)

    # Broader ensemble: all pixel classifiers with accuracy >= 0.6
    good_pixel = [(n, classifiers_pixel[n]) for n, (a, _) in results_pixel.items()
                  if a >= 0.6 and n in classifiers_pixel]
    if len(good_pixel) >= 3:
        vote_all = VotingClassifier(estimators=good_pixel, voting='soft')
        acc, errs = evaluate_loo(vote_all, pixel_X, y,
                                 "Voting(all good pixel)", sample_names)
        results_pixel["Voting(all_good)"] = (acc, errs)

    # ====================================================================
    # SECTION 5: Feature importance (pixel features)
    # ====================================================================
    print("\n" + "=" * 70)
    print("  SECTION 5: FEATURE IMPORTANCE")
    print("=" * 70)

    # RF importances
    rf_imp = RandomForestClassifier(n_estimators=500, max_depth=3, random_state=42)
    imp = get_feature_importances(rf_imp, PIXEL_FEATURE_NAMES, pixel_X, y)
    print("\n  Random Forest feature importances:")
    for i, (fname, v) in enumerate(imp):
        bar = "#" * int(v * 100)
        print(f"    {i+1:2d}. {fname:<25s} {v:.4f} {bar}")

    # Logistic regression coefficients
    lr_imp = Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=2000, C=1.0))])
    imp_lr = get_feature_importances(lr_imp, PIXEL_FEATURE_NAMES, pixel_X, y)
    print("\n  Logistic Regression |coefficients|:")
    for i, (fname, v) in enumerate(imp_lr):
        bar = "#" * int(v * 20)
        print(f"    {i+1:2d}. {fname:<25s} {v:.4f} {bar}")

    # Per-feature accuracy (single feature)
    print("\n  Single-feature LOO accuracy (best threshold):")
    for j, fname in enumerate(PIXEL_FEATURE_NAMES):
        X_single = pixel_X[:, j:j+1]
        # Try LR
        lr_single = Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=2000, C=1.0))])
        acc_s, _ = evaluate_loo(lr_single, X_single, y, fname, sample_names, verbose=False)
        print(f"    {fname:<25s} {acc_s:.1%}")

    # ====================================================================
    # GRAND SUMMARY
    # ====================================================================
    print("\n" + "=" * 70)
    print("  GRAND SUMMARY")
    print("=" * 70)

    all_results = {}
    for name, (acc, errs) in results_pixel.items():
        all_results[f"pixel/{name}"] = (acc, errs)
    for name, (acc, errs) in results_combined.items():
        all_results[f"combined/{name}"] = (acc, errs)
    for name, (acc, errs) in results_dino_only.items():
        all_results[f"dino/{name}"] = (acc, errs)

    sorted_all = sorted(all_results.items(), key=lambda x: -x[1][0])
    best_name, (best_acc, best_errs) = sorted_all[0]

    print(f"\n  {'Rank':<5s} {'Accuracy':>8s}  {'Configuration'}")
    print(f"  {'-'*5} {'-'*8}  {'-'*50}")
    for i, (name, (acc, errs)) in enumerate(sorted_all[:25]):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1:4d}  {acc:>7.1%}  {name}{marker}")

    print(f"\n  Best: {best_name} = {best_acc:.1%}")
    if best_errs:
        print(f"  Errors ({len(best_errs)}):")
        for name, true, pred in best_errs:
            t = "stamped" if true else "clean"
            p = "stamped" if pred else "clean"
            print(f"    {name}: true={t}, pred={p}")

    # ====================================================================
    # Save best model
    # ====================================================================
    print(f"\n  Saving best model...")

    # Determine the best configuration and rebuild it
    # We need to identify feature set and classifier
    if best_name.startswith("pixel/"):
        # Pixel-only
        clf_key = best_name.split("/", 1)[1]
        if clf_key in classifiers_pixel:
            best_clf = clone(classifiers_pixel[clf_key])
        else:
            # Voting ensemble — rebuild
            best_clf = VotingClassifier(
                estimators=vote_pixel_estimators, voting='soft')
        best_clf.fit(pixel_X, y)
        save_data = {
            "model": best_clf,
            "feature_names": PIXEL_FEATURE_NAMES,
            "feature_type": "pixel_only",
            "uses_dino": False,
            "metrics": {"loo_acc": best_acc, "n_samples": len(y),
                        "n_stamped": int(np.sum(y)), "n_clean": int(np.sum(y == 0))},
        }
    elif best_name.startswith("combined/"):
        # Pixel + PCA(DINOv2)
        # Parse: pixel+PCA{n}({source})_{clf}
        config = best_name.split("/", 1)[1]
        # e.g. pixel+PCA5(cls)_LR(C=1)
        import re
        m = re.match(r'pixel\+PCA(\d+)\((\w+)\)_(.+)', config)
        if m:
            n_comp = int(m.group(1))
            dino_source = m.group(2)
            clf_tag = m.group(3)
        else:
            n_comp, dino_source, clf_tag = 5, "cls", "LR(C=1)"

        dino_X = dino_sources[dino_source]
        # Find the classifier
        clf_map = {n: c for n, c in best_classifiers}
        base_clf = clone(clf_map.get(clf_tag, best_classifiers[0][1]))

        # Fit PCA on all data for the saved model
        nc = min(n_comp, dino_X.shape[0], dino_X.shape[1])
        pca = PCA(n_components=nc, random_state=42)
        dino_reduced = pca.fit_transform(dino_X)
        X_combined = np.hstack([pixel_X, dino_reduced])
        base_clf.fit(X_combined, y)

        dino_feature_names = [f"pca_{dino_source}_{i}" for i in range(nc)]
        save_data = {
            "model": base_clf,
            "pca": pca,
            "dino_source": dino_source,
            "n_pca_components": nc,
            "feature_names": PIXEL_FEATURE_NAMES + dino_feature_names,
            "pixel_feature_names": PIXEL_FEATURE_NAMES,
            "feature_type": f"pixel+pca{nc}_{dino_source}",
            "uses_dino": True,
            "metrics": {"loo_acc": best_acc, "n_samples": len(y),
                        "n_stamped": int(np.sum(y)), "n_clean": int(np.sum(y == 0))},
        }
    else:
        # DINOv2-only
        config = best_name.split("/", 1)[1]
        import re
        m = re.match(r'PCA(\d+)\((\w+)\)_(.+)', config)
        if m:
            n_comp = int(m.group(1))
            dino_source = m.group(2)
            clf_tag = m.group(3)
        else:
            n_comp, dino_source, clf_tag = 5, "cls", "LR(C=1)"

        dino_X = dino_sources[dino_source]
        clf_map = {n: c for n, c in best_classifiers}
        base_clf = clone(clf_map.get(clf_tag, best_classifiers[0][1]))

        nc = min(n_comp, dino_X.shape[0], dino_X.shape[1])
        pca = PCA(n_components=nc, random_state=42)
        dino_reduced = pca.fit_transform(dino_X)
        base_clf.fit(dino_reduced, y)

        dino_feature_names = [f"pca_{dino_source}_{i}" for i in range(nc)]
        save_data = {
            "model": base_clf,
            "pca": pca,
            "dino_source": dino_source,
            "n_pca_components": nc,
            "feature_names": dino_feature_names,
            "feature_type": f"pca{nc}_{dino_source}",
            "uses_dino": True,
            "uses_pixel": False,
            "metrics": {"loo_acc": best_acc, "n_samples": len(y),
                        "n_stamped": int(np.sum(y)), "n_clean": int(np.sum(y == 0))},
        }

    with open(OUT_MODEL, "wb") as f:
        pickle.dump(save_data, f)

    print(f"  Saved model to: {OUT_MODEL}")
    print(f"  Config: {best_name}")
    print(f"  LOO accuracy: {best_acc:.1%}")
    print(f"  Features: {len(save_data['feature_names'])} "
          f"(uses_dino={save_data.get('uses_dino', False)})")
    print("\nDone.")


if __name__ == "__main__":
    main()
