#!/usr/bin/env python3
"""
Gabor filter bank analysis for stamp texture detection.

EX-era stamp text is rotated ~30-45 degrees. Gabor filters tuned to that
orientation should produce higher energy responses on stamped cards than
on clean cards.

Pipeline:
  1. Build Gabor filter bank: 4 orientations x 3 frequencies = 12 filters
  2. Apply to stamp region of each binder card (17 cards from binder_ground_truth.jsonl)
  3. Compute per-filter energy features
  4. Compute orientation histograms (stamps should peak at ~45 deg)
  5. Train logistic regression classifier on Gabor features
  6. Leave-one-out cross-validation
  7. Report accuracy
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = Path("/home/godli/cardprice")
STAMPS_REAL = BASE / "data" / "condition_training" / "stamps_real"
INBOX = BASE / "data" / "inbox"


# ---------------------------------------------------------------------------
# Stamp region cropping (same as stamp_pixel_analysis.py)
# ---------------------------------------------------------------------------

def crop_stamp_region(img: np.ndarray) -> np.ndarray:
    """Crop bottom-right of artwork area where EX stamps appear.
    x: 55-90%, y: 45-70% of card.
    """
    h, w = img.shape[:2]
    x1, x2 = int(w * 0.55), int(w * 0.90)
    y1, y2 = int(h * 0.45), int(h * 0.70)
    return img[y1:y2, x1:x2]


def crop_control_region(img: np.ndarray) -> np.ndarray:
    """Control region: same vertical band but left side (no stamp)."""
    h, w = img.shape[:2]
    x1, x2 = int(w * 0.10), int(w * 0.45)
    y1, y2 = int(h * 0.45), int(h * 0.70)
    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Gabor filter bank
# ---------------------------------------------------------------------------

def build_gabor_bank(
    orientations: list[float],
    frequencies: list[float],
    ksize: int = 31,
    sigma: float = 4.0,
    gamma: float = 0.5,
) -> list[dict]:
    """Build a bank of Gabor filters.

    Parameters
    ----------
    orientations : list of float
        Orientations in degrees (0, 45, 90, 135).
    frequencies : list of float
        Spatial frequencies (wavelengths in pixels).
    ksize : int
        Kernel size.
    sigma : float
        Gaussian envelope sigma.
    gamma : float
        Spatial aspect ratio.

    Returns
    -------
    List of dicts with keys: kernel, orientation_deg, frequency, label.
    """
    bank = []
    for theta_deg in orientations:
        theta_rad = np.deg2rad(theta_deg)
        for freq in frequencies:
            lambd = 1.0 / freq  # wavelength = 1/frequency
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma,
                theta_rad,
                lambd,
                gamma,
                psi=0,
                ktype=cv2.CV_64F,
            )
            # Normalize kernel so energy comparisons are meaningful
            kernel /= np.abs(kernel).sum() + 1e-10
            bank.append({
                "kernel": kernel,
                "orientation_deg": theta_deg,
                "frequency": freq,
                "label": f"{theta_deg:.0f}deg_f{freq:.3f}",
            })
    return bank


def apply_gabor_bank(gray: np.ndarray, bank: list[dict]) -> dict:
    """Apply Gabor filter bank and compute energy features.

    Returns dict mapping filter label -> energy (mean squared response).
    """
    results = {}
    gray_f = gray.astype(np.float64)
    for filt in bank:
        response = cv2.filter2D(gray_f, cv2.CV_64F, filt["kernel"])
        energy = float(np.mean(response ** 2))
        results[filt["label"]] = energy
    return results


def compute_orientation_histogram(
    gray: np.ndarray,
    bank: list[dict],
) -> dict[float, float]:
    """Compute orientation histogram: sum energy across frequencies per orientation.

    Returns dict mapping orientation_deg -> total energy.
    """
    gray_f = gray.astype(np.float64)
    hist = {}
    for filt in bank:
        response = cv2.filter2D(gray_f, cv2.CV_64F, filt["kernel"])
        energy = float(np.mean(response ** 2))
        theta = filt["orientation_deg"]
        hist[theta] = hist.get(theta, 0.0) + energy
    return hist


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_binder_dataset() -> list[dict]:
    """Load binder ground truth with images."""
    gt_path = STAMPS_REAL / "binder_ground_truth.jsonl"
    entries = []
    with open(gt_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            img_path = INBOX / entry["image"]
            img = cv2.imread(str(img_path))
            if img is not None:
                entry["img"] = img
                entries.append(entry)
            else:
                print(f"  [WARN] Could not load: {img_path}")
    return entries


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_gabor_features(
    img: np.ndarray,
    bank: list[dict],
) -> dict:
    """Extract Gabor features from stamp and control regions.

    Returns a dict with:
      - Per-filter stamp energy
      - Per-filter control energy
      - Per-filter energy ratio (stamp/control)
      - Per-orientation total energy (stamp)
      - Per-orientation energy ratio
      - Orientation dominance features
    """
    stamp = crop_stamp_region(img)
    control = crop_control_region(img)
    stamp_gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
    control_gray = cv2.cvtColor(control, cv2.COLOR_BGR2GRAY)

    # Per-filter energies
    stamp_energies = apply_gabor_bank(stamp_gray, bank)
    control_energies = apply_gabor_bank(control_gray, bank)

    features = {}

    # Raw stamp energies
    for label, energy in stamp_energies.items():
        features[f"stamp_{label}"] = energy

    # Energy ratios (stamp / control)
    for label in stamp_energies:
        ctrl_e = control_energies.get(label, 1e-10)
        features[f"ratio_{label}"] = stamp_energies[label] / (ctrl_e + 1e-10)

    # Orientation histograms
    stamp_orient = compute_orientation_histogram(stamp_gray, bank)
    control_orient = compute_orientation_histogram(control_gray, bank)

    total_stamp_energy = sum(stamp_orient.values()) + 1e-10
    total_control_energy = sum(control_orient.values()) + 1e-10

    for theta, energy in stamp_orient.items():
        features[f"orient_{theta:.0f}_stamp"] = energy
        features[f"orient_{theta:.0f}_normalized"] = energy / total_stamp_energy
        ctrl_e = control_orient.get(theta, 1e-10)
        features[f"orient_{theta:.0f}_ratio"] = energy / (ctrl_e + 1e-10)

    # Orientation dominance: is 45deg the dominant orientation?
    orient_45 = stamp_orient.get(45.0, 0.0)
    orient_0 = stamp_orient.get(0.0, 0.0)
    orient_90 = stamp_orient.get(90.0, 0.0)
    orient_135 = stamp_orient.get(135.0, 0.0)

    features["orient_45_dominance"] = orient_45 / total_stamp_energy
    features["orient_45_vs_0"] = orient_45 / (orient_0 + 1e-10)
    features["orient_45_vs_90"] = orient_45 / (orient_90 + 1e-10)
    features["orient_diagonal_sum"] = (orient_45 + orient_135) / total_stamp_energy
    features["orient_cardinal_sum"] = (orient_0 + orient_90) / total_stamp_energy
    features["diagonal_vs_cardinal"] = (
        (orient_45 + orient_135) / (orient_0 + orient_90 + 1e-10)
    )

    return features


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def leave_one_out_cv(X: np.ndarray, y: np.ndarray) -> tuple[float, list]:
    """Leave-one-out cross-validation with logistic regression.

    Returns (accuracy, list of (true_label, predicted_label, probability)).
    """
    n = len(y)
    results = []
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]
        y_test = y[i]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train_s, y_train)

        pred = clf.predict(X_test_s)[0]
        prob = clf.predict_proba(X_test_s)[0]
        results.append((int(y_test), int(pred), float(prob[1])))

    correct = sum(1 for t, p, _ in results if t == p)
    accuracy = correct / n
    return accuracy, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Gabor Filter Bank — Stamp Texture Analysis")
    print("=" * 70)

    # Build filter bank
    orientations = [0.0, 45.0, 90.0, 135.0]
    frequencies = [0.05, 0.10, 0.20]  # cycles per pixel
    bank = build_gabor_bank(orientations, frequencies, ksize=31, sigma=4.0, gamma=0.5)
    print(f"\nFilter bank: {len(orientations)} orientations x {len(frequencies)} frequencies = {len(bank)} filters")
    for filt in bank:
        print(f"  {filt['label']}: theta={filt['orientation_deg']:.0f} freq={filt['frequency']:.3f}")

    # Load data
    print("\nLoading binder ground truth...")
    entries = load_binder_dataset()
    print(f"  Loaded {len(entries)} cards")

    n_stamped = sum(1 for e in entries if e["stamped"])
    n_clean = len(entries) - n_stamped
    print(f"  Stamped: {n_stamped}, Clean: {n_clean}")

    # Extract features
    print("\nExtracting Gabor features...")
    all_features = []
    labels = []
    names = []
    for entry in entries:
        feats = extract_gabor_features(entry["img"], bank)
        all_features.append(feats)
        labels.append(1 if entry["stamped"] else 0)
        names.append(entry.get("card_name", "unknown"))

    y = np.array(labels)

    # -------------------------------------------------------------------
    # Orientation histogram analysis
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ORIENTATION HISTOGRAM ANALYSIS")
    print("=" * 70)

    print(f"\n  {'Card':<25} {'Stamped':>7}  {'0deg':>10} {'45deg':>10} {'90deg':>10} {'135deg':>10}  {'Peak':>6}")
    print(f"  {'-'*25} {'-'*7}  {'-'*10} {'-'*10} {'-'*10} {'-'*10}  {'-'*6}")

    for i, entry in enumerate(entries):
        feats = all_features[i]
        name = names[i]
        stamped = entry["stamped"]

        orient_vals = {}
        for theta in orientations:
            orient_vals[theta] = feats.get(f"orient_{theta:.0f}_normalized", 0.0)

        peak_orient = max(orient_vals, key=orient_vals.get)

        print(f"  {name:<25} {'YES' if stamped else 'no':>7}  "
              f"{orient_vals[0.0]:>10.4f} {orient_vals[45.0]:>10.4f} "
              f"{orient_vals[90.0]:>10.4f} {orient_vals[135.0]:>10.4f}  "
              f"{peak_orient:>5.0f}d")

    # Aggregate orientation stats by class
    print("\n  --- Aggregate orientation energy (normalized) ---")
    for cls, cls_name in [(1, "STAMPED"), (0, "CLEAN")]:
        indices = [i for i in range(len(labels)) if labels[i] == cls]
        print(f"\n  {cls_name} (n={len(indices)}):")
        for theta in orientations:
            vals = [all_features[i][f"orient_{theta:.0f}_normalized"] for i in indices]
            print(f"    {theta:>5.0f} deg: mean={np.mean(vals):.4f} std={np.std(vals):.4f} "
                  f"min={np.min(vals):.4f} max={np.max(vals):.4f}")

    # Diagonal vs cardinal comparison
    print("\n  --- Diagonal (45+135) vs Cardinal (0+90) energy ratio ---")
    for i, entry in enumerate(entries):
        feats = all_features[i]
        diag_ratio = feats["diagonal_vs_cardinal"]
        print(f"  {names[i]:<25} {'STAMP' if entry['stamped'] else 'clean':>5}  "
              f"diag/card={diag_ratio:.4f}")

    # -------------------------------------------------------------------
    # Per-filter energy comparison
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PER-FILTER ENERGY SEPARATION (stamped vs clean)")
    print("=" * 70)

    feature_keys = sorted(all_features[0].keys())
    separations = {}

    for key in feature_keys:
        s_vals = [all_features[i][key] for i in range(len(labels)) if labels[i] == 1]
        c_vals = [all_features[i][key] for i in range(len(labels)) if labels[i] == 0]
        s_mean, s_std = np.mean(s_vals), np.std(s_vals)
        c_mean, c_std = np.mean(c_vals), np.std(c_vals)
        sep = abs(s_mean - c_mean) / (s_std + c_std + 1e-10)
        separations[key] = sep

    # Top 20 features by separation
    ranked = sorted(separations.items(), key=lambda x: -x[1])
    print(f"\n  {'Feature':<40} {'Stamped':>12} {'Clean':>12} {'Sep':>8}")
    print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*8}")
    for key, sep in ranked[:20]:
        s_vals = [all_features[i][key] for i in range(len(labels)) if labels[i] == 1]
        c_vals = [all_features[i][key] for i in range(len(labels)) if labels[i] == 0]
        print(f"  {key:<40} {np.mean(s_vals):>8.5f}+{np.std(s_vals):<4.3f} "
              f"{np.mean(c_vals):>8.5f}+{np.std(c_vals):<4.3f} {sep:>7.3f}")

    # -------------------------------------------------------------------
    # Classifier: all Gabor features
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CLASSIFIER: LOO-CV with all Gabor features")
    print("=" * 70)

    # Build feature matrix
    X_all = np.array([[all_features[i][k] for k in feature_keys] for i in range(len(labels))])
    print(f"  Feature matrix: {X_all.shape}")

    acc_all, results_all = leave_one_out_cv(X_all, y)
    print(f"\n  LOO-CV Accuracy (all features): {acc_all:.1%} ({sum(1 for t,p,_ in results_all if t==p)}/{len(results_all)})")

    print(f"\n  {'Card':<25} {'True':>5} {'Pred':>5} {'Prob':>6} {'OK':>3}")
    print(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*6} {'-'*3}")
    for i, (true, pred, prob) in enumerate(results_all):
        ok = "OK" if true == pred else "XX"
        print(f"  {names[i]:<25} {true:>5} {pred:>5} {prob:>6.3f} {ok:>3}")

    # -------------------------------------------------------------------
    # Classifier: orientation features only
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CLASSIFIER: LOO-CV with orientation features only")
    print("=" * 70)

    orient_keys = [k for k in feature_keys if k.startswith("orient_")]
    X_orient = np.array([[all_features[i][k] for k in orient_keys] for i in range(len(labels))])
    print(f"  Feature matrix: {X_orient.shape}")

    acc_orient, results_orient = leave_one_out_cv(X_orient, y)
    print(f"\n  LOO-CV Accuracy (orientation features): {acc_orient:.1%} ({sum(1 for t,p,_ in results_orient if t==p)}/{len(results_orient)})")

    for i, (true, pred, prob) in enumerate(results_orient):
        ok = "OK" if true == pred else "XX"
        if true != pred:
            print(f"  MISS: {names[i]:<25} true={true} pred={pred} prob={prob:.3f}")

    # -------------------------------------------------------------------
    # Classifier: ratio features only (stamp/control)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CLASSIFIER: LOO-CV with ratio features only")
    print("=" * 70)

    ratio_keys = [k for k in feature_keys if k.startswith("ratio_")]
    X_ratio = np.array([[all_features[i][k] for k in ratio_keys] for i in range(len(labels))])
    print(f"  Feature matrix: {X_ratio.shape}")

    acc_ratio, results_ratio = leave_one_out_cv(X_ratio, y)
    print(f"\n  LOO-CV Accuracy (ratio features): {acc_ratio:.1%} ({sum(1 for t,p,_ in results_ratio if t==p)}/{len(results_ratio)})")

    for i, (true, pred, prob) in enumerate(results_ratio):
        ok = "OK" if true == pred else "XX"
        if true != pred:
            print(f"  MISS: {names[i]:<25} true={true} pred={pred} prob={prob:.3f}")

    # -------------------------------------------------------------------
    # Classifier: top-N features by separation
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CLASSIFIER: LOO-CV with top-N features by separation")
    print("=" * 70)

    for top_n in [5, 10, 15]:
        top_keys = [k for k, _ in ranked[:top_n]]
        X_top = np.array([[all_features[i][k] for k in top_keys] for i in range(len(labels))])
        acc_top, results_top = leave_one_out_cv(X_top, y)
        n_correct = sum(1 for t, p, _ in results_top if t == p)
        print(f"  Top-{top_n}: {acc_top:.1%} ({n_correct}/{len(results_top)})")
        for i, (true, pred, prob) in enumerate(results_top):
            if true != pred:
                print(f"    MISS: {names[i]:<25} true={true} pred={pred} prob={prob:.3f}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  All features ({X_all.shape[1]}):       {acc_all:.1%}")
    print(f"  Orientation features ({X_orient.shape[1]}):  {acc_orient:.1%}")
    print(f"  Ratio features ({X_ratio.shape[1]}):     {acc_ratio:.1%}")
    for top_n in [5, 10, 15]:
        top_keys = [k for k, _ in ranked[:top_n]]
        X_top = np.array([[all_features[i][k] for k in top_keys] for i in range(len(labels))])
        acc_top, _ = leave_one_out_cv(X_top, y)
        print(f"  Top-{top_n} features:          {acc_top:.1%}")

    print(f"\n  Dataset: {len(entries)} cards ({n_stamped} stamped, {n_clean} clean)")
    print(f"  Filter bank: {len(bank)} filters ({len(orientations)} orient x {len(frequencies)} freq)")


if __name__ == "__main__":
    main()
