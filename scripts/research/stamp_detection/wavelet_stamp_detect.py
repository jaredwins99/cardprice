#!/usr/bin/env python3
"""Wavelet-based stamp detection for Pokemon cards.

Uses 2D discrete wavelet transform to detect diagonal stamp text overlays.
Stamps (e.g., "TEAM ROCKET", "CRYSTAL GUARDIANS") are diagonal text that
produces distinctive high-energy diagonal detail coefficients.

Approach:
  1. Crop the stamp region (bottom-right quadrant of card)
  2. Apply 2-3 level 2D DWT using Haar or Daubechies wavelets
  3. Compute energy of each subband (cH, cV, cD at each level)
  4. Use diagonal/horizontal energy ratios as features
  5. Train simple classifier (threshold or logistic regression)
  6. Evaluate on 17 binder ground truth cards

Usage:
    python scripts/research/stamp_detection/wavelet_stamp_detect.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pywt
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INBOX_DIR = PROJECT_ROOT / "data" / "inbox"
BINDER_GT_PATH = (
    PROJECT_ROOT / "data" / "condition_training" / "stamps_real" / "binder_ground_truth.jsonl"
)


def load_binder_ground_truth() -> list[dict]:
    """Load the 17 binder ground truth cards."""
    entries = []
    with open(BINDER_GT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            img_path = INBOX_DIR / entry["image"]
            if not img_path.exists():
                print(f"  WARNING: {img_path} not found, skipping")
                continue
            entry["img_path"] = str(img_path)
            entries.append(entry)
    print(f"Loaded {len(entries)} binder ground truth cards")
    return entries


def crop_stamp_region(img: Image.Image, region: str = "bottom_right") -> np.ndarray:
    """Crop the stamp region from a card image and convert to grayscale.

    Stamps appear in the bottom-right area of EX-era cards, typically
    diagonal text across the card art area.

    Returns grayscale numpy array.
    """
    w, h = img.size

    if region == "bottom_right":
        # Bottom-right quadrant — where EX stamps and prerelease stamps appear
        box = (w // 2, h // 2, w, h)
    elif region == "bottom_half":
        box = (0, h // 2, w, h)
    elif region == "center":
        # Center of card — stamp text crosses here
        box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    elif region == "full":
        box = (0, 0, w, h)
    else:
        raise ValueError(f"Unknown region: {region}")

    crop = img.crop(box).convert("L")
    # Resize to consistent size for comparable wavelet features
    crop = crop.resize((256, 256), Image.LANCZOS)
    return np.array(crop, dtype=np.float64)


def compute_wavelet_features(
    gray: np.ndarray, wavelet: str = "db2", max_level: int = 3
) -> dict:
    """Compute wavelet energy features from a grayscale image.

    Performs 2D DWT decomposition and extracts energy from each subband.

    Returns dict of feature values.
    """
    # Normalize to [0, 1]
    gray = gray / 255.0

    # Multi-level 2D DWT
    coeffs = pywt.wavedec2(gray, wavelet, level=max_level)

    features = {}

    # Approximation energy at coarsest level
    cA = coeffs[0]
    features["cA_energy"] = float(np.sum(cA ** 2))
    features["cA_mean"] = float(np.mean(np.abs(cA)))

    total_detail_energy = 0.0

    for level_idx in range(1, len(coeffs)):
        level = len(coeffs) - level_idx  # level 1 = finest, level max_level = coarsest
        cH, cV, cD = coeffs[level_idx]

        # Energy = sum of squared coefficients
        eH = float(np.sum(cH ** 2))
        eV = float(np.sum(cV ** 2))
        eD = float(np.sum(cD ** 2))
        e_total = eH + eV + eD

        total_detail_energy += e_total

        # Store raw energies
        features[f"L{level}_eH"] = eH
        features[f"L{level}_eV"] = eV
        features[f"L{level}_eD"] = eD
        features[f"L{level}_total"] = e_total

        # Ratios — stamps produce high diagonal energy
        features[f"L{level}_eD_ratio"] = eD / (eH + eV + 1e-10)
        features[f"L{level}_eD_frac"] = eD / (e_total + 1e-10)
        features[f"L{level}_eH_frac"] = eH / (e_total + 1e-10)
        features[f"L{level}_eV_frac"] = eV / (e_total + 1e-10)

        # Mean absolute coefficient values
        features[f"L{level}_cH_mean"] = float(np.mean(np.abs(cH)))
        features[f"L{level}_cV_mean"] = float(np.mean(np.abs(cV)))
        features[f"L{level}_cD_mean"] = float(np.mean(np.abs(cD)))

        # Std of coefficients (texture complexity)
        features[f"L{level}_cH_std"] = float(np.std(cH))
        features[f"L{level}_cV_std"] = float(np.std(cV))
        features[f"L{level}_cD_std"] = float(np.std(cD))

    # Cross-level features
    if max_level >= 2:
        # Ratio of fine-to-coarse diagonal energy
        fine_eD = features.get("L1_eD", 0)
        coarse_eD = features.get(f"L{max_level}_eD", 1e-10)
        features["fine_coarse_eD_ratio"] = fine_eD / (coarse_eD + 1e-10)

    features["total_detail_energy"] = total_detail_energy

    return features


def compute_edge_enhanced_features(gray: np.ndarray, wavelet: str = "db2", max_level: int = 3) -> dict:
    """Apply edge enhancement before wavelet decomposition.

    This amplifies stamp text edges before decomposition.
    """
    from PIL import Image as PILImage

    # Convert to PIL for filtering
    img_pil = PILImage.fromarray(gray.astype(np.uint8))

    # Unsharp mask to enhance edges
    enhanced = img_pil.filter(ImageFilter.EDGE_ENHANCE_MORE)
    enhanced_arr = np.array(enhanced, dtype=np.float64)

    feats = compute_wavelet_features(enhanced_arr, wavelet, max_level)
    return {f"edge_{k}": v for k, v in feats.items()}


def extract_all_features(img_path: str, wavelet: str = "db2", max_level: int = 3) -> dict:
    """Extract full wavelet feature set from a card image."""
    img = Image.open(img_path).convert("RGB")

    all_features = {}

    # Multiple regions
    for region in ["bottom_right", "center", "full"]:
        gray = crop_stamp_region(img, region)

        # Raw wavelet features
        feats = compute_wavelet_features(gray, wavelet, max_level)
        all_features.update({f"{region}_{k}": v for k, v in feats.items()})

        # Edge-enhanced wavelet features (only for bottom_right to keep feature count manageable)
        if region == "bottom_right":
            edge_feats = compute_edge_enhanced_features(gray, wavelet, max_level)
            all_features.update({f"{region}_{k}": v for k, v in edge_feats.items()})

    # Cross-region features: stamp region vs full card
    for level in range(1, max_level + 1):
        br_eD = all_features.get(f"bottom_right_L{level}_eD", 0)
        full_eD = all_features.get(f"full_L{level}_eD", 1e-10)
        all_features[f"cross_L{level}_br_full_eD_ratio"] = br_eD / (full_eD + 1e-10)

        br_eD_frac = all_features.get(f"bottom_right_L{level}_eD_frac", 0)
        full_eD_frac = all_features.get(f"full_L{level}_eD_frac", 0)
        all_features[f"cross_L{level}_eD_frac_diff"] = br_eD_frac - full_eD_frac

    return all_features


def analyze_feature_separability(entries: list[dict], features_list: list[dict]) -> list[tuple]:
    """Find which features best separate stamped from clean cards."""
    labels = [1 if e["stamped"] else 0 for e in entries]

    stamped_idx = [i for i, l in enumerate(labels) if l == 1]
    clean_idx = [i for i, l in enumerate(labels) if l == 0]

    if not stamped_idx or not clean_idx:
        print("ERROR: Need both stamped and clean examples")
        return []

    feature_names = sorted(features_list[0].keys())
    separability = []

    for fname in feature_names:
        vals = [features_list[i][fname] for i in range(len(features_list))]
        stamped_vals = [vals[i] for i in stamped_idx]
        clean_vals = [vals[i] for i in clean_idx]

        s_mean = np.mean(stamped_vals)
        c_mean = np.mean(clean_vals)
        s_std = np.std(stamped_vals) + 1e-10
        c_std = np.std(clean_vals) + 1e-10

        # Fisher's discriminant ratio
        pooled_std = np.sqrt((s_std ** 2 + c_std ** 2) / 2)
        fisher = abs(s_mean - c_mean) / (pooled_std + 1e-10)

        separability.append((fname, fisher, s_mean, c_mean, s_std, c_std))

    separability.sort(key=lambda x: -x[1])
    return separability


def build_classifier(entries: list[dict], features_list: list[dict], top_k: int = 10):
    """Build a simple classifier using top-k wavelet features.

    Uses leave-one-out cross-validation on the 17 samples.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    labels = np.array([1 if e["stamped"] else 0 for e in entries])

    # Get top features by separability
    sep = analyze_feature_separability(entries, features_list)
    top_features = [s[0] for s in sep[:top_k]]

    # Build feature matrix
    X = np.array([[f[fname] for fname in top_features] for f in features_list])

    # Leave-one-out cross-validation
    n = len(entries)
    correct = 0
    predictions = []

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(labels, i)
        X_test = X[i:i + 1]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train_s, y_train)
        pred = clf.predict(X_test_s)[0]
        prob = clf.predict_proba(X_test_s)[0]

        predictions.append({
            "entry": entries[i],
            "pred": int(pred),
            "gt": int(labels[i]),
            "prob_stamped": float(prob[1]),
            "correct": int(pred) == int(labels[i]),
        })
        if int(pred) == int(labels[i]):
            correct += 1

    return correct, n, predictions, top_features


def build_threshold_classifier(entries: list[dict], features_list: list[dict]):
    """Try simple single-feature threshold classifiers."""
    labels = np.array([1 if e["stamped"] else 0 for e in entries])
    sep = analyze_feature_separability(entries, features_list)

    best_acc = 0
    best_result = None

    for fname, fisher, s_mean, c_mean, _, _ in sep[:20]:
        vals = np.array([f[fname] for f in features_list])

        # Try multiple thresholds
        sorted_vals = np.sort(np.unique(vals))
        for j in range(len(sorted_vals) - 1):
            thresh = (sorted_vals[j] + sorted_vals[j + 1]) / 2

            # stamped > threshold or stamped < threshold?
            if s_mean > c_mean:
                preds = (vals > thresh).astype(int)
            else:
                preds = (vals < thresh).astype(int)

            acc = np.mean(preds == labels)
            if acc > best_acc:
                best_acc = acc
                direction = ">" if s_mean > c_mean else "<"
                best_result = (fname, thresh, direction, acc, fisher)

    return best_result


def main():
    print("=" * 70)
    print("WAVELET STAMP DETECTION")
    print("=" * 70)

    # Load ground truth
    entries = load_binder_ground_truth()
    if not entries:
        print("ERROR: No ground truth entries found")
        sys.exit(1)

    stamped_count = sum(1 for e in entries if e["stamped"])
    clean_count = len(entries) - stamped_count
    print(f"  Stamped: {stamped_count}, Clean: {clean_count}")

    # Try multiple wavelets
    for wavelet in ["haar", "db2", "db4"]:
        print(f"\n{'=' * 70}")
        print(f"WAVELET: {wavelet}")
        print(f"{'=' * 70}")

        # Extract features
        print(f"\nExtracting wavelet features ({wavelet})...")
        features_list = []
        for entry in entries:
            feats = extract_all_features(entry["img_path"], wavelet=wavelet, max_level=3)
            features_list.append(feats)

        print(f"  {len(features_list[0])} features per card")

        # Feature separability analysis
        print(f"\nTop 15 features by Fisher discriminant ratio:")
        print(f"  {'Feature':<55s}  {'Fisher':>7s}  {'Stamped':>8s}  {'Clean':>8s}")
        print(f"  {'-' * 55}  {'-' * 7}  {'-' * 8}  {'-' * 8}")

        sep = analyze_feature_separability(entries, features_list)
        for fname, fisher, s_mean, c_mean, s_std, c_std in sep[:15]:
            print(f"  {fname:<55s}  {fisher:7.3f}  {s_mean:8.4f}  {c_mean:8.4f}")

        # Threshold classifier
        print(f"\nBest single-feature threshold classifier:")
        thresh_result = build_threshold_classifier(entries, features_list)
        if thresh_result:
            fname, thresh, direction, acc, fisher = thresh_result
            print(f"  Feature: {fname}")
            print(f"  Threshold: {direction} {thresh:.6f}")
            print(f"  Accuracy: {acc:.1%} ({int(acc * len(entries))}/{len(entries)})")
            print(f"  Fisher ratio: {fisher:.3f}")

        # LOO logistic regression classifier
        for top_k in [3, 5, 10, 15]:
            print(f"\nLogistic Regression (top {top_k} features, LOO CV):")
            correct, n, predictions, top_feats = build_classifier(entries, features_list, top_k=top_k)
            print(f"  Accuracy: {correct}/{n} ({correct / n:.1%})")

            # Show errors
            errors = [p for p in predictions if not p["correct"]]
            if errors:
                print(f"  Errors:")
                for err in errors:
                    e = err["entry"]
                    gt = "stamped" if err["gt"] else "clean"
                    pred = "stamped" if err["pred"] else "clean"
                    print(f"    {e['card_name']:20s} gt={gt:8s} pred={pred:8s} "
                          f"prob_stamped={err['prob_stamped']:.3f} ({e.get('note', '')})")
            else:
                print(f"  PERFECT - no errors!")

        # Show per-card detail for best wavelet
        if wavelet == "db2":
            print(f"\nPer-card diagonal energy analysis (db2, bottom_right, Level 1):")
            print(f"  {'Card':<25s}  {'Label':>7s}  {'eD':>10s}  {'eH':>10s}  "
                  f"{'eV':>10s}  {'D/(H+V)':>8s}  {'D_frac':>7s}")
            print(f"  {'-' * 25}  {'-' * 7}  {'-' * 10}  {'-' * 10}  "
                  f"{'-' * 10}  {'-' * 8}  {'-' * 7}")
            for entry, feats in zip(entries, features_list):
                label = "STAMP" if entry["stamped"] else "clean"
                eD = feats["bottom_right_L1_eD"]
                eH = feats["bottom_right_L1_eH"]
                eV = feats["bottom_right_L1_eV"]
                ratio = eD / (eH + eV + 1e-10)
                frac = feats["bottom_right_L1_eD_frac"]
                print(f"  {entry['card_name']:<25s}  {label:>7s}  {eD:10.4f}  {eH:10.4f}  "
                      f"{eV:10.4f}  {ratio:8.4f}  {frac:7.4f}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
