#!/usr/bin/env python3
"""
Opacity-based stamp detection: exploit the semi-transparent nature of stamps.

Theory: Stamps are semi-transparent metallic overlays. The stamp region has
TWO layers blended: card art + stamp text. This alpha-blending creates
specific statistical signatures:
  - Higher local variance (two signals mixed)
  - Reduced color saturation (metallic overlay desaturates)
  - Bimodal intensity histogram (art peaks + stamp peaks)
  - Channel decorrelation (metallic has different spectral properties)
  - Higher entropy (stamp adds information)

This script:
  1. Loads 17 binder ground truth cards
  2. Crops stamp region and control region for each
  3. Computes alpha-blending signatures
  4. Compares metrics between stamped and clean cards
  5. Builds threshold classifier on best-separating metrics
  6. Reports accuracy
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import stats as sp_stats

BASE = Path("/home/godli/cardprice")
INBOX = BASE / "data/inbox"
GT_PATH = BASE / "data/condition_training/stamps_real/binder_ground_truth.jsonl"


# ---------------------------------------------------------------------------
# Region cropping
# ---------------------------------------------------------------------------

def crop_stamp_region(img: np.ndarray) -> np.ndarray:
    """Bottom-right of artwork where EX-era / prerelease stamps appear.
    x: 55-90%, y: 45-70% of card."""
    h, w = img.shape[:2]
    return img[int(h * 0.45):int(h * 0.70), int(w * 0.55):int(w * 0.90)]


def crop_control_region(img: np.ndarray) -> np.ndarray:
    """Same vertical band but LEFT side (no stamp). x: 10-45%, y: 45-70%."""
    h, w = img.shape[:2]
    return img[int(h * 0.45):int(h * 0.70), int(w * 0.10):int(w * 0.45)]


# ---------------------------------------------------------------------------
# Alpha-blending signature features
# ---------------------------------------------------------------------------

def local_variance_map(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Compute variance in ksize x ksize windows.
    Var(X) = E[X^2] - E[X]^2."""
    gray_f = gray.astype(np.float64)
    mean = cv2.blur(gray_f, (ksize, ksize))
    mean_sq = cv2.blur(gray_f ** 2, (ksize, ksize))
    var_map = mean_sq - mean ** 2
    # Clamp negatives from float precision
    var_map = np.maximum(var_map, 0.0)
    return var_map


def compute_kurtosis(gray: np.ndarray) -> float:
    """Kurtosis of pixel intensities. Stamp overlay should reduce kurtosis
    (more uniform / platykurtic distribution)."""
    vals = gray.ravel().astype(np.float64)
    k = sp_stats.kurtosis(vals, fisher=True)
    return float(k)


def compute_bimodality_coefficient(gray: np.ndarray) -> float:
    """Bimodality coefficient: BC = (skewness^2 + 1) / kurtosis_excess.
    BC > 0.555 suggests bimodal distribution.
    For stamp regions: two overlapping distributions (art + stamp)."""
    vals = gray.ravel().astype(np.float64)
    n = len(vals)
    if n < 4:
        return 0.0
    skew = sp_stats.skew(vals)
    kurt = sp_stats.kurtosis(vals, fisher=False)  # excess=False -> Pearson
    # BC = (skew^2 + 1) / kurt
    # Kurt must be > 0 to avoid division issues
    if kurt < 1e-10:
        return 0.0
    bc = (skew ** 2 + 1) / kurt
    return float(bc)


def compute_channel_correlation(img_bgr: np.ndarray) -> dict:
    """Correlation between RGB channels. Natural images have high inter-channel
    correlation. Metallic stamp overlay should decorrelate them."""
    b, g, r = [img_bgr[:, :, c].ravel().astype(np.float64) for c in range(3)]
    # Pearson correlations
    rg = np.corrcoef(r, g)[0, 1] if np.std(r) > 0 and np.std(g) > 0 else 1.0
    rb = np.corrcoef(r, b)[0, 1] if np.std(r) > 0 and np.std(b) > 0 else 1.0
    gb = np.corrcoef(g, b)[0, 1] if np.std(g) > 0 and np.std(b) > 0 else 1.0
    mean_corr = (rg + rb + gb) / 3.0
    return {
        "rg": float(rg),
        "rb": float(rb),
        "gb": float(gb),
        "mean": float(mean_corr),
    }


def compute_entropy(gray: np.ndarray) -> float:
    """Shannon entropy of intensity histogram. Stamp adds information -> higher entropy."""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    return float(-np.sum(probs * np.log2(probs)))


def compute_saturation_stats(img_bgr: np.ndarray) -> dict:
    """HSV saturation statistics. Metallic overlay desaturates."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float64)
    return {
        "mean": float(np.mean(sat)),
        "std": float(np.std(sat)),
        "median": float(np.median(sat)),
    }


def compute_gradient_coherence(gray: np.ndarray) -> float:
    """Gradient orientation coherence. Stamp text introduces random edges
    on top of art gradients -> lower coherence."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    # Weight orientations by magnitude
    total_mag = np.sum(mag)
    if total_mag < 1e-10:
        return 1.0
    # Coherence = |sum of unit gradient vectors weighted by magnitude| / sum(mag)
    ux = gx / (mag + 1e-10)
    uy = gy / (mag + 1e-10)
    wx = np.sum(ux * mag)
    wy = np.sum(uy * mag)
    coherence = np.sqrt(wx ** 2 + wy ** 2) / total_mag
    return float(coherence)


# ---------------------------------------------------------------------------
# Feature extraction for one card
# ---------------------------------------------------------------------------

@dataclass
class OpacityFeatures:
    """All opacity-based features for a single card."""
    name: str = ""
    stamped: bool = False

    # Local variance (5x5 window)
    stamp_mean_var: float = 0.0
    control_mean_var: float = 0.0
    var_ratio: float = 0.0  # stamp / control
    var_diff: float = 0.0   # stamp - control

    # Kurtosis
    stamp_kurtosis: float = 0.0
    control_kurtosis: float = 0.0
    kurtosis_diff: float = 0.0  # stamp - control

    # Bimodality coefficient
    stamp_bimodality: float = 0.0
    control_bimodality: float = 0.0
    bimodality_diff: float = 0.0

    # Channel correlation
    stamp_chan_corr: float = 0.0
    control_chan_corr: float = 0.0
    chan_corr_diff: float = 0.0  # stamp - control (expect negative for stamped)

    # Entropy
    stamp_entropy: float = 0.0
    control_entropy: float = 0.0
    entropy_diff: float = 0.0
    entropy_ratio: float = 0.0

    # Saturation
    stamp_sat_mean: float = 0.0
    control_sat_mean: float = 0.0
    sat_diff: float = 0.0  # stamp - control (expect negative for stamped)
    sat_ratio: float = 0.0

    # Gradient coherence
    stamp_coherence: float = 0.0
    control_coherence: float = 0.0
    coherence_diff: float = 0.0

    # Combined / derived
    opacity_score: float = 0.0  # weighted combo of best features


def extract_opacity_features(img: np.ndarray, name: str, stamped: bool) -> OpacityFeatures:
    """Extract all opacity-based features from a card image."""
    feat = OpacityFeatures(name=name, stamped=stamped)

    stamp = crop_stamp_region(img)
    control = crop_control_region(img)

    stamp_gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
    control_gray = cv2.cvtColor(control, cv2.COLOR_BGR2GRAY)

    # 1. Local variance map
    stamp_var_map = local_variance_map(stamp_gray, ksize=5)
    control_var_map = local_variance_map(control_gray, ksize=5)
    feat.stamp_mean_var = float(np.mean(stamp_var_map))
    feat.control_mean_var = float(np.mean(control_var_map))
    feat.var_ratio = feat.stamp_mean_var / (feat.control_mean_var + 1e-10)
    feat.var_diff = feat.stamp_mean_var - feat.control_mean_var

    # 2. Kurtosis
    feat.stamp_kurtosis = compute_kurtosis(stamp_gray)
    feat.control_kurtosis = compute_kurtosis(control_gray)
    feat.kurtosis_diff = feat.stamp_kurtosis - feat.control_kurtosis

    # 3. Bimodality coefficient
    feat.stamp_bimodality = compute_bimodality_coefficient(stamp_gray)
    feat.control_bimodality = compute_bimodality_coefficient(control_gray)
    feat.bimodality_diff = feat.stamp_bimodality - feat.control_bimodality

    # 4. Channel correlation
    stamp_corr = compute_channel_correlation(stamp)
    control_corr = compute_channel_correlation(control)
    feat.stamp_chan_corr = stamp_corr["mean"]
    feat.control_chan_corr = control_corr["mean"]
    feat.chan_corr_diff = feat.stamp_chan_corr - feat.control_chan_corr

    # 5. Entropy
    feat.stamp_entropy = compute_entropy(stamp_gray)
    feat.control_entropy = compute_entropy(control_gray)
    feat.entropy_diff = feat.stamp_entropy - feat.control_entropy
    feat.entropy_ratio = feat.stamp_entropy / (feat.control_entropy + 1e-10)

    # 6. Saturation
    stamp_sat = compute_saturation_stats(stamp)
    control_sat = compute_saturation_stats(control)
    feat.stamp_sat_mean = stamp_sat["mean"]
    feat.control_sat_mean = control_sat["mean"]
    feat.sat_diff = feat.stamp_sat_mean - feat.control_sat_mean
    feat.sat_ratio = feat.stamp_sat_mean / (feat.control_sat_mean + 1e-10)

    # 7. Gradient coherence
    feat.stamp_coherence = compute_gradient_coherence(stamp_gray)
    feat.control_coherence = compute_gradient_coherence(control_gray)
    feat.coherence_diff = feat.stamp_coherence - feat.control_coherence

    return feat


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_binder_cards() -> list[tuple[np.ndarray, bool, str]]:
    """Load binder ground truth cards. Returns (image, stamped, name)."""
    results = []
    with open(GT_PATH) as f:
        for line in f:
            entry = json.loads(line.strip())
            img_path = INBOX / entry["image"]
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  WARN: Could not load {img_path}")
                continue
            results.append((img, entry["stamped"], entry["card_name"]))
    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_features(features: list[OpacityFeatures]):
    """Compute separation metrics and find best features."""
    stamped = [f for f in features if f.stamped]
    clean = [f for f in features if not f.stamped]

    print(f"\n{'='*80}")
    print(f"  OPACITY-BASED STAMP DETECTION ANALYSIS")
    print(f"  Stamped: {len(stamped)}, Clean: {len(clean)}, Total: {len(features)}")
    print(f"{'='*80}")

    # All numeric features to evaluate
    feature_names = [
        # Local variance
        "stamp_mean_var", "var_ratio", "var_diff",
        # Kurtosis
        "stamp_kurtosis", "kurtosis_diff",
        # Bimodality
        "stamp_bimodality", "bimodality_diff",
        # Channel correlation
        "stamp_chan_corr", "chan_corr_diff",
        # Entropy
        "stamp_entropy", "entropy_diff", "entropy_ratio",
        # Saturation
        "stamp_sat_mean", "sat_diff", "sat_ratio",
        # Gradient coherence
        "stamp_coherence", "coherence_diff",
    ]

    print(f"\n  {'Feature':<22} {'Stamped':>14} {'Clean':>14} {'Sep':>8}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*8}")

    separations = {}
    for fname in feature_names:
        s_vals = np.array([getattr(f, fname) for f in stamped])
        c_vals = np.array([getattr(f, fname) for f in clean])
        s_mean, s_std = np.mean(s_vals), np.std(s_vals)
        c_mean, c_std = np.mean(c_vals), np.std(c_vals)
        sep = abs(s_mean - c_mean) / (s_std + c_std + 1e-10)
        separations[fname] = sep
        print(f"  {fname:<22} {s_mean:>8.3f}+{s_std:<5.2f} "
              f"{c_mean:>8.3f}+{c_std:<5.2f} {sep:>7.3f}")

    ranked = sorted(separations.items(), key=lambda x: -x[1])
    print(f"\n  Top features by separation (|mean_diff| / (std_s + std_c)):")
    for i, (fname, sep) in enumerate(ranked[:8]):
        print(f"    {i+1}. {fname}: {sep:.3f}")

    return separations


def build_classifier(features: list[OpacityFeatures]):
    """Build threshold classifier on each feature and combos."""
    stamped = [f for f in features if f.stamped]
    clean = [f for f in features if not f.stamped]

    feature_names = [
        "stamp_mean_var", "var_ratio", "var_diff",
        "stamp_kurtosis", "kurtosis_diff",
        "stamp_bimodality", "bimodality_diff",
        "stamp_chan_corr", "chan_corr_diff",
        "stamp_entropy", "entropy_diff", "entropy_ratio",
        "stamp_sat_mean", "sat_diff", "sat_ratio",
        "stamp_coherence", "coherence_diff",
    ]

    print(f"\n{'='*80}")
    print(f"  THRESHOLD CLASSIFIER (per-feature)")
    print(f"{'='*80}")

    results = []
    for fname in feature_names:
        s_vals = [getattr(f, fname) for f in stamped]
        c_vals = [getattr(f, fname) for f in clean]
        all_vals = sorted(set(s_vals + c_vals))

        best_acc = 0.0
        best_thresh = 0.0
        best_dir = ">"
        best_tp = best_tn = best_fp = best_fn = 0

        for thresh in all_vals:
            # stamped > thresh
            tp = sum(1 for v in s_vals if v > thresh)
            fn = len(s_vals) - tp
            tn = sum(1 for v in c_vals if v <= thresh)
            fp = len(c_vals) - tn
            acc_gt = (tp + tn) / len(features)
            if acc_gt > best_acc:
                best_acc = acc_gt
                best_thresh = thresh
                best_dir = ">"
                best_tp, best_tn, best_fp, best_fn = tp, tn, fp, fn

            # stamped < thresh
            tp2 = sum(1 for v in s_vals if v < thresh)
            fn2 = len(s_vals) - tp2
            tn2 = sum(1 for v in c_vals if v >= thresh)
            fp2 = len(c_vals) - tn2
            acc_lt = (tp2 + tn2) / len(features)
            if acc_lt > best_acc:
                best_acc = acc_lt
                best_thresh = thresh
                best_dir = "<"
                best_tp, best_tn, best_fp, best_fn = tp2, tn2, fp2, fn2

        results.append((best_acc, fname, best_thresh, best_dir,
                         best_tp, best_tn, best_fp, best_fn))
        print(f"  {fname:<22} acc={best_acc:.1%}  "
              f"thresh={best_thresh:>10.4f}  dir={best_dir}  "
              f"TP={best_tp} TN={best_tn} FP={best_fp} FN={best_fn}")

    results.sort(key=lambda x: -x[0])
    print(f"\n  BEST single feature: {results[0][1]} "
          f"{results[0][3]} {results[0][2]:.4f} -> {results[0][0]:.1%}")

    # Print per-card predictions for best feature
    best_fname = results[0][1]
    best_thresh = results[0][2]
    best_dir = results[0][3]
    print(f"\n  Per-card detail for {best_fname} {best_dir} {best_thresh:.4f}:")
    for feat in features:
        val = getattr(feat, best_fname)
        if best_dir == ">":
            pred_stamped = val > best_thresh
        else:
            pred_stamped = val < best_thresh
        correct = pred_stamped == feat.stamped
        marker = "OK" if correct else "WRONG"
        print(f"    [{marker:>5}] {feat.name:<20} val={val:>10.4f}  "
              f"pred={'STAMP' if pred_stamped else 'clean':>5}  "
              f"true={'STAMP' if feat.stamped else 'clean':>5}")

    # 2-feature AND/OR combos
    print(f"\n{'='*80}")
    print(f"  2-FEATURE COMBOS (top 10)")
    print(f"{'='*80}")

    combo_results = []
    for i, (_, f1, t1, d1, _, _, _, _) in enumerate(results):
        for j, (_, f2, t2, d2, _, _, _, _) in enumerate(results):
            if j <= i:
                continue
            # AND combo
            tp_and = sum(
                1 for f in stamped
                if (_check(getattr(f, f1), t1, d1) and
                    _check(getattr(f, f2), t2, d2))
            )
            tn_and = sum(
                1 for f in clean
                if not (_check(getattr(f, f1), t1, d1) and
                        _check(getattr(f, f2), t2, d2))
            )
            acc_and = (tp_and + tn_and) / len(features)
            combo_results.append((acc_and, "AND", f1, t1, d1, f2, t2, d2,
                                   tp_and, tn_and))

            # OR combo
            tp_or = sum(
                1 for f in stamped
                if (_check(getattr(f, f1), t1, d1) or
                    _check(getattr(f, f2), t2, d2))
            )
            tn_or = sum(
                1 for f in clean
                if not (_check(getattr(f, f1), t1, d1) or
                        _check(getattr(f, f2), t2, d2))
            )
            acc_or = (tp_or + tn_or) / len(features)
            combo_results.append((acc_or, "OR", f1, t1, d1, f2, t2, d2,
                                   tp_or, tn_or))

    combo_results.sort(key=lambda x: -x[0])
    for acc, op, f1, t1, d1, f2, t2, d2, tp, tn in combo_results[:10]:
        d1s = ">" if d1 == ">" else "<"
        d2s = ">" if d2 == ">" else "<"
        fn = len(stamped) - tp
        fp = len(clean) - tn
        print(f"    {acc:.1%}  {f1}{d1s}{t1:.3f} {op} {f2}{d2s}{t2:.3f}  "
              f"TP={tp} TN={tn} FP={fp} FN={fn}")

    # Print misclassified for best combo
    if combo_results:
        best_combo = combo_results[0]
        acc, op, f1, t1, d1, f2, t2, d2, _, _ = best_combo
        print(f"\n  Misclassified by best combo ({op}):")
        for feat in features:
            v1 = getattr(feat, f1)
            v2 = getattr(feat, f2)
            c1 = _check(v1, t1, d1)
            c2 = _check(v2, t2, d2)
            if op == "AND":
                pred = c1 and c2
            else:
                pred = c1 or c2
            if pred != feat.stamped:
                print(f"    {feat.name:<20} {f1}={v1:.4f} {f2}={v2:.4f} "
                      f"pred={'STAMP' if pred else 'clean'} "
                      f"true={'STAMP' if feat.stamped else 'clean'}")

    return results, combo_results


def _check(val: float, thresh: float, direction: str) -> bool:
    """Check if value passes threshold in given direction."""
    if direction == ">":
        return val > thresh
    else:
        return val < thresh


# ---------------------------------------------------------------------------
# Detailed per-card dump
# ---------------------------------------------------------------------------

def dump_per_card(features: list[OpacityFeatures]):
    """Print every feature for every card for manual inspection."""
    print(f"\n{'='*80}")
    print(f"  PER-CARD RAW VALUES")
    print(f"{'='*80}")

    for f in features:
        label = "STAMP" if f.stamped else "clean"
        print(f"\n  [{label:>5}] {f.name}")
        print(f"    Local var:     stamp={f.stamp_mean_var:.2f}  "
              f"control={f.control_mean_var:.2f}  "
              f"ratio={f.var_ratio:.3f}  diff={f.var_diff:.2f}")
        print(f"    Kurtosis:      stamp={f.stamp_kurtosis:.3f}  "
              f"control={f.control_kurtosis:.3f}  "
              f"diff={f.kurtosis_diff:.3f}")
        print(f"    Bimodality:    stamp={f.stamp_bimodality:.4f}  "
              f"control={f.control_bimodality:.4f}  "
              f"diff={f.bimodality_diff:.4f}")
        print(f"    Chan corr:     stamp={f.stamp_chan_corr:.4f}  "
              f"control={f.control_chan_corr:.4f}  "
              f"diff={f.chan_corr_diff:.4f}")
        print(f"    Entropy:       stamp={f.stamp_entropy:.3f}  "
              f"control={f.control_entropy:.3f}  "
              f"diff={f.entropy_diff:.3f}  ratio={f.entropy_ratio:.3f}")
        print(f"    Saturation:    stamp={f.stamp_sat_mean:.2f}  "
              f"control={f.control_sat_mean:.2f}  "
              f"diff={f.sat_diff:.2f}  ratio={f.sat_ratio:.3f}")
        print(f"    Coherence:     stamp={f.stamp_coherence:.4f}  "
              f"control={f.control_coherence:.4f}  "
              f"diff={f.coherence_diff:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Opacity-Based Stamp Detection")
    print("=" * 80)
    print("Theory: stamps are semi-transparent metallic overlays that create")
    print("alpha-blending signatures in the underlying image statistics.\n")

    # Load data
    cards = load_binder_cards()
    print(f"Loaded {len(cards)} binder ground truth cards")
    n_stamped = sum(1 for _, s, _ in cards if s)
    n_clean = sum(1 for _, s, _ in cards if not s)
    print(f"  Stamped: {n_stamped}, Clean: {n_clean}")

    # Extract features
    features = []
    for img, stamped, name in cards:
        feat = extract_opacity_features(img, name, stamped)
        features.append(feat)

    # Detailed per-card dump
    dump_per_card(features)

    # Analyze separations
    separations = analyze_features(features)

    # Build classifier
    single_results, combo_results = build_classifier(features)

    # Final summary
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Dataset: {len(features)} binder cards "
          f"({n_stamped} stamped, {n_clean} clean)")

    print(f"\n  Theory validation:")
    theory_checks = [
        ("var_ratio", "Higher local variance in stamp region",
         "stamped should have higher var_ratio"),
        ("sat_diff", "Reduced saturation in stamp region",
         "stamped should have negative sat_diff"),
        ("stamp_bimodality", "Bimodal intensity in stamp region",
         "stamped should have higher bimodality"),
        ("chan_corr_diff", "Channel decorrelation from metallic overlay",
         "stamped should have negative chan_corr_diff"),
        ("entropy_diff", "Higher entropy from added stamp info",
         "stamped should have positive entropy_diff"),
    ]

    stamped_feats = [f for f in features if f.stamped]
    clean_feats = [f for f in features if not f.stamped]

    for fname, theory, expectation in theory_checks:
        s_mean = np.mean([getattr(f, fname) for f in stamped_feats])
        c_mean = np.mean([getattr(f, fname) for f in clean_feats])
        diff = s_mean - c_mean
        direction = "higher" if diff > 0 else "lower"
        print(f"\n    {theory}:")
        print(f"      {fname}: stamped={s_mean:.4f}, clean={c_mean:.4f}, "
              f"stamped is {direction} by {abs(diff):.4f}")
        print(f"      Expected: {expectation}")
        # Check if direction matches theory
        if fname == "var_ratio" and diff > 0:
            print(f"      CONFIRMED")
        elif fname == "sat_diff" and diff < 0:
            print(f"      CONFIRMED")
        elif fname == "stamp_bimodality" and diff > 0:
            print(f"      CONFIRMED")
        elif fname == "chan_corr_diff" and diff < 0:
            print(f"      CONFIRMED")
        elif fname == "entropy_diff" and diff > 0:
            print(f"      CONFIRMED")
        else:
            print(f"      NOT CONFIRMED (opposite direction)")

    if single_results:
        best = single_results[0]
        print(f"\n  Best single feature: {best[1]} -> {best[0]:.1%} accuracy")
    if combo_results:
        best_combo = combo_results[0]
        print(f"  Best 2-feature combo: {best_combo[0]:.1%} accuracy")


if __name__ == "__main__":
    main()
