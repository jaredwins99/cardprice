#!/usr/bin/env python3
"""Multi-scale edge analysis for stamp detection.

Hypothesis: stamps introduce fine-scale text edges that differ from the
card's natural artwork edges. By comparing edge density across multiple
Gaussian blur scales (fine, medium, coarse), we can detect a scale-space
signature unique to stamped cards.

Key features:
  1. Canny edge density at 3 scales (sigma=0.5, 1.0, 2.0)
  2. Fine/coarse edge ratio — stamps should have relatively MORE fine edges
  3. Difference of Gaussians (DoG) — stamps create specific scale-space blobs
  4. Stamp region vs control region comparison at each scale

Runs on 17 binder ground truth cards.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

BASE = Path("/home/godli/cardprice")
INBOX = BASE / "data/inbox"
GT_PATH = BASE / "data/condition_training/stamps_real/binder_ground_truth.jsonl"


# ── Region cropping ──────────────────────────────────────────────────────

def crop_stamp_region(img: np.ndarray) -> np.ndarray:
    """Bottom-right artwork area where EX-era stamps appear."""
    h, w = img.shape[:2]
    return img[int(h * 0.45):int(h * 0.70), int(w * 0.55):int(w * 0.90)]


def crop_control_region(img: np.ndarray) -> np.ndarray:
    """Same vertical band, left side (no stamp expected)."""
    h, w = img.shape[:2]
    return img[int(h * 0.45):int(h * 0.70), int(w * 0.10):int(w * 0.45)]


# ── Multi-scale edge features ───────────────────────────────────────────

SCALES = {
    "fine":   0.5,
    "medium": 1.0,
    "coarse": 2.0,
}


def canny_edge_density(gray: np.ndarray, sigma: float) -> float:
    """Apply Gaussian blur at given sigma, then Canny, return edge pixel fraction."""
    if sigma > 0:
        blurred = gaussian_filter(gray.astype(np.float64), sigma=sigma)
        blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    else:
        blurred = gray
    edges = cv2.Canny(blurred, 50, 150)
    return float(np.mean(edges > 0))


def compute_multiscale_edges(gray: np.ndarray) -> dict:
    """Compute edge density at fine/medium/coarse scales."""
    result = {}
    for name, sigma in SCALES.items():
        result[name] = canny_edge_density(gray, sigma)
    return result


def compute_dog(gray: np.ndarray, sigma1: float, sigma2: float) -> np.ndarray:
    """Difference of Gaussians: G(sigma2) - G(sigma1), with sigma2 > sigma1."""
    g1 = gaussian_filter(gray.astype(np.float64), sigma=sigma1)
    g2 = gaussian_filter(gray.astype(np.float64), sigma=sigma2)
    return g2 - g1


def compute_dog_features(gray: np.ndarray) -> dict:
    """Compute DoG-based features at multiple scale pairs."""
    features = {}

    # DoG: fine detail (0.5 vs 1.0)
    dog_fine = compute_dog(gray, 0.5, 1.0)
    features["dog_fine_mean"] = float(np.mean(np.abs(dog_fine)))
    features["dog_fine_std"] = float(np.std(dog_fine))
    features["dog_fine_energy"] = float(np.mean(dog_fine ** 2))

    # DoG: medium detail (1.0 vs 2.0)
    dog_med = compute_dog(gray, 1.0, 2.0)
    features["dog_med_mean"] = float(np.mean(np.abs(dog_med)))
    features["dog_med_std"] = float(np.std(dog_med))
    features["dog_med_energy"] = float(np.mean(dog_med ** 2))

    # DoG: coarse detail (2.0 vs 4.0)
    dog_coarse = compute_dog(gray, 2.0, 4.0)
    features["dog_coarse_mean"] = float(np.mean(np.abs(dog_coarse)))
    features["dog_coarse_std"] = float(np.std(dog_coarse))
    features["dog_coarse_energy"] = float(np.mean(dog_coarse ** 2))

    # Ratio features: fine vs coarse DoG
    features["dog_fine_coarse_ratio"] = (
        features["dog_fine_energy"] / (features["dog_coarse_energy"] + 1e-10)
    )
    features["dog_fine_med_ratio"] = (
        features["dog_fine_energy"] / (features["dog_med_energy"] + 1e-10)
    )

    return features


def compute_laplacian_of_gaussian(gray: np.ndarray, sigma: float) -> float:
    """LoG response magnitude at given scale."""
    blurred = gaussian_filter(gray.astype(np.float64), sigma=sigma)
    lap = cv2.Laplacian(blurred.astype(np.uint8), cv2.CV_64F)
    return float(np.mean(np.abs(lap)))


# ── Feature extraction per card ──────────────────────────────────────────

def extract_card_features(img: np.ndarray) -> dict:
    """Extract all multi-scale edge features from a card image."""
    stamp = crop_stamp_region(img)
    control = crop_control_region(img)
    stamp_gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
    control_gray = cv2.cvtColor(control, cv2.COLOR_BGR2GRAY)

    feat = {}

    # 1. Multi-scale Canny edge density
    stamp_edges = compute_multiscale_edges(stamp_gray)
    ctrl_edges = compute_multiscale_edges(control_gray)

    for scale_name in SCALES:
        feat[f"stamp_{scale_name}_edges"] = stamp_edges[scale_name]
        feat[f"ctrl_{scale_name}_edges"] = ctrl_edges[scale_name]
        feat[f"ratio_{scale_name}_edges"] = (
            stamp_edges[scale_name] / (ctrl_edges[scale_name] + 1e-10)
        )

    # 2. Cross-scale ratios (the key feature!)
    feat["stamp_fine_coarse_ratio"] = (
        stamp_edges["fine"] / (stamp_edges["coarse"] + 1e-10)
    )
    feat["ctrl_fine_coarse_ratio"] = (
        ctrl_edges["fine"] / (ctrl_edges["coarse"] + 1e-10)
    )
    feat["stamp_fine_medium_ratio"] = (
        stamp_edges["fine"] / (stamp_edges["medium"] + 1e-10)
    )
    feat["ctrl_fine_medium_ratio"] = (
        ctrl_edges["fine"] / (ctrl_edges["medium"] + 1e-10)
    )

    # Relative cross-scale: how much MORE fine-detail the stamp has vs control
    feat["relative_fine_coarse"] = (
        feat["stamp_fine_coarse_ratio"] / (feat["ctrl_fine_coarse_ratio"] + 1e-10)
    )
    feat["relative_fine_medium"] = (
        feat["stamp_fine_medium_ratio"] / (feat["ctrl_fine_medium_ratio"] + 1e-10)
    )

    # 3. DoG features for stamp and control
    stamp_dog = compute_dog_features(stamp_gray)
    ctrl_dog = compute_dog_features(control_gray)
    for k, v in stamp_dog.items():
        feat[f"stamp_{k}"] = v
    for k, v in ctrl_dog.items():
        feat[f"ctrl_{k}"] = v

    # DoG stamp-to-control ratios
    for k in stamp_dog:
        feat[f"ratio_{k}"] = stamp_dog[k] / (ctrl_dog[k] + 1e-10)

    # 4. LoG at multiple scales
    for sigma in [0.5, 1.0, 2.0, 4.0]:
        sname = str(sigma).replace(".", "p")
        feat[f"stamp_log_{sname}"] = compute_laplacian_of_gaussian(stamp_gray, sigma)
        feat[f"ctrl_log_{sname}"] = compute_laplacian_of_gaussian(control_gray, sigma)
        feat[f"ratio_log_{sname}"] = (
            feat[f"stamp_log_{sname}"] / (feat[f"ctrl_log_{sname}"] + 1e-10)
        )

    # 5. Edge density difference (stamp - control) at each scale
    for scale_name in SCALES:
        feat[f"diff_{scale_name}_edges"] = (
            stamp_edges[scale_name] - ctrl_edges[scale_name]
        )

    return feat


# ── Data loading ─────────────────────────────────────────────────────────

def load_binder_ground_truth() -> list[dict]:
    """Load binder ground truth, dedup by image path (last entry wins)."""
    entries = {}
    with open(GT_PATH) as f:
        for line in f:
            entry = json.loads(line.strip())
            entries[entry["image"]] = entry  # last wins for dupes
    return list(entries.values())


# ── Threshold classifier ─────────────────────────────────────────────────

def find_best_threshold(values_stamped: list[float],
                        values_clean: list[float]) -> tuple[float, float, str]:
    """Find optimal threshold and direction. Returns (threshold, accuracy, direction)."""
    all_vals = sorted(set(values_stamped + values_clean))
    best_acc = 0.0
    best_thresh = 0.0
    best_dir = ">"

    for thresh in all_vals:
        # stamped > threshold
        tp = sum(1 for v in values_stamped if v > thresh)
        tn = sum(1 for v in values_clean if v <= thresh)
        acc_gt = (tp + tn) / (len(values_stamped) + len(values_clean))
        if acc_gt > best_acc:
            best_acc, best_thresh, best_dir = acc_gt, thresh, ">"

        # stamped < threshold
        tp2 = sum(1 for v in values_stamped if v < thresh)
        tn2 = sum(1 for v in values_clean if v >= thresh)
        acc_lt = (tp2 + tn2) / (len(values_stamped) + len(values_clean))
        if acc_lt > best_acc:
            best_acc, best_thresh, best_dir = acc_lt, thresh, "<"

    return best_thresh, best_acc, best_dir


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Multi-Scale Edge Analysis for Stamp Detection")
    print("=" * 70)

    # Load data
    entries = load_binder_ground_truth()
    print(f"\nLoaded {len(entries)} ground truth entries")

    cards = []
    for entry in entries:
        img_path = INBOX / entry["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARN: cannot load {img_path}")
            continue
        feat = extract_card_features(img)
        feat["_name"] = entry["card_name"]
        feat["_stamped"] = entry["stamped"]
        feat["_image"] = entry["image"]
        cards.append(feat)

    stamped = [c for c in cards if c["_stamped"]]
    clean = [c for c in cards if not c["_stamped"]]
    print(f"  Stamped: {len(stamped)}, Clean: {len(clean)}, Total: {len(cards)}")

    # ── Per-card raw values ──────────────────────────────────────────────
    print(f"\n{'─' * 90}")
    print("  Per-card edge densities (Canny at 3 scales)")
    print(f"{'─' * 90}")
    print(f"  {'Card':<22} {'Label':<8}  "
          f"{'Fine':>7} {'Med':>7} {'Coarse':>7}  "
          f"{'F/C':>6} {'F/M':>6}  "
          f"{'cFine':>7} {'cMed':>7} {'cCoarse':>7}")

    for c in cards:
        label = "STAMP" if c["_stamped"] else "clean"
        print(f"  {c['_name']:<22} {label:<8}  "
              f"{c['stamp_fine_edges']:7.4f} "
              f"{c['stamp_medium_edges']:7.4f} "
              f"{c['stamp_coarse_edges']:7.4f}  "
              f"{c['stamp_fine_coarse_ratio']:6.2f} "
              f"{c['stamp_fine_medium_ratio']:6.2f}  "
              f"{c['ctrl_fine_edges']:7.4f} "
              f"{c['ctrl_medium_edges']:7.4f} "
              f"{c['ctrl_coarse_edges']:7.4f}")

    # ── Feature-level statistics ─────────────────────────────────────────
    # Collect all feature names (excluding metadata)
    feature_names = sorted(k for k in cards[0] if not k.startswith("_"))

    print(f"\n{'=' * 90}")
    print("  Feature Separation (|mean_s - mean_c| / (std_s + std_c + eps))")
    print(f"{'=' * 90}")
    print(f"  {'Feature':<40} {'Stamped':>14} {'Clean':>14} {'Sep':>7}")
    print(f"  {'─' * 40} {'─' * 14} {'─' * 14} {'─' * 7}")

    separations = {}
    for fname in feature_names:
        s_vals = np.array([c[fname] for c in stamped])
        c_vals = np.array([c[fname] for c in clean])
        s_mean, s_std = np.mean(s_vals), np.std(s_vals)
        c_mean, c_std = np.mean(c_vals), np.std(c_vals)
        sep = abs(s_mean - c_mean) / (s_std + c_std + 1e-10)
        separations[fname] = sep
        print(f"  {fname:<40} {s_mean:>8.4f}+{s_std:<5.3f} "
              f"{c_mean:>8.4f}+{c_std:<5.3f} {sep:>6.3f}")

    # Top features
    ranked = sorted(separations.items(), key=lambda x: -x[1])
    print(f"\n  Top 15 features by separation:")
    for i, (fname, sep) in enumerate(ranked[:15]):
        print(f"    {i + 1:2d}. {fname:<40} sep={sep:.3f}")

    # ── Threshold classifier for each feature ────────────────────────────
    print(f"\n{'=' * 90}")
    print("  Threshold Classifier (best single-feature accuracy)")
    print(f"{'=' * 90}")

    results = []
    for fname in feature_names:
        s_vals = [c[fname] for c in stamped]
        c_vals = [c[fname] for c in clean]
        thresh, acc, direction = find_best_threshold(s_vals, c_vals)
        results.append((acc, fname, thresh, direction))

    results.sort(key=lambda x: -x[0])
    print(f"\n  {'Feature':<40} {'Acc':>6} {'Thresh':>10} {'Dir':>4}")
    print(f"  {'─' * 40} {'─' * 6} {'─' * 10} {'─' * 4}")
    for acc, fname, thresh, direction in results[:20]:
        print(f"  {fname:<40} {acc:5.1%} {thresh:>10.4f}   {direction}")

    # ── Best single-feature classifier detail ────────────────────────────
    best_acc, best_fname, best_thresh, best_dir = results[0]
    print(f"\n{'=' * 90}")
    print(f"  Best classifier: {best_fname} {best_dir} {best_thresh:.4f}")
    print(f"  Accuracy: {best_acc:.1%} ({int(best_acc * len(cards))}/{len(cards)})")
    print(f"{'=' * 90}")

    print(f"\n  Per-card predictions:")
    n_correct = 0
    for c in cards:
        val = c[best_fname]
        if best_dir == ">":
            pred = val > best_thresh
        else:
            pred = val < best_thresh
        correct = pred == c["_stamped"]
        n_correct += correct
        status = "OK" if correct else "WRONG"
        print(f"    [{status:>5}] {c['_name']:<22} "
              f"{'STAMP' if c['_stamped'] else 'clean':<6} "
              f"pred={'STAMP' if pred else 'clean':<6} "
              f"{best_fname}={val:.4f}")

    # ── Multi-feature: try 2-feature combos ──────────────────────────────
    print(f"\n{'=' * 90}")
    print("  2-Feature Combo Search (AND logic)")
    print(f"{'=' * 90}")

    top_feats = [fname for _, fname, _, _ in results[:15]]
    combo_results = []

    for i, f1 in enumerate(top_feats):
        for f2 in top_feats[i + 1:]:
            s1 = [c[f1] for c in stamped]
            c1 = [c[f1] for c in clean]
            s2 = [c[f2] for c in stamped]
            c2 = [c[f2] for c in clean]

            # Try AND / OR with various thresholds
            for t1 in [np.median(s1 + c1), (np.mean(s1) + np.mean(c1)) / 2]:
                for t2 in [np.median(s2 + c2), (np.mean(s2) + np.mean(c2)) / 2]:
                    for d1 in [1, -1]:
                        for d2 in [1, -1]:
                            # AND: both conditions
                            tp_and = sum(
                                1 for j in range(len(stamped))
                                if (d1 * s1[j] > d1 * t1 and d2 * s2[j] > d2 * t2)
                            )
                            tn_and = sum(
                                1 for j in range(len(clean))
                                if not (d1 * c1[j] > d1 * t1 and d2 * c2[j] > d2 * t2)
                            )
                            acc_and = (tp_and + tn_and) / len(cards)

                            # OR: either condition
                            tp_or = sum(
                                1 for j in range(len(stamped))
                                if (d1 * s1[j] > d1 * t1 or d2 * s2[j] > d2 * t2)
                            )
                            tn_or = sum(
                                1 for j in range(len(clean))
                                if not (d1 * c1[j] > d1 * t1 or d2 * c2[j] > d2 * t2)
                            )
                            acc_or = (tp_or + tn_or) / len(cards)

                            d1s = ">" if d1 > 0 else "<"
                            d2s = ">" if d2 > 0 else "<"

                            combo_results.append(
                                (acc_and, f"AND: {f1}{d1s}{t1:.4f} & {f2}{d2s}{t2:.4f}")
                            )
                            combo_results.append(
                                (acc_or, f"OR:  {f1}{d1s}{t1:.4f} | {f2}{d2s}{t2:.4f}")
                            )

    combo_results.sort(key=lambda x: -x[0])
    seen = set()
    print(f"\n  Top 10 combos:")
    count = 0
    for acc, desc in combo_results:
        if desc not in seen:
            seen.add(desc)
            print(f"    {acc:5.1%}  {desc}")
            count += 1
            if count >= 10:
                break

    # ── DoG analysis detail ──────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  DoG Analysis Detail")
    print(f"{'=' * 90}")
    dog_feats = [k for k in feature_names if "dog" in k]
    print(f"\n  {'Card':<22} {'Label':<6}", end="")
    # Print just the key DoG features
    key_dog = ["stamp_dog_fine_energy", "ctrl_dog_fine_energy",
               "ratio_dog_fine_energy", "stamp_dog_fine_coarse_ratio",
               "ctrl_dog_fine_coarse_ratio", "ratio_dog_fine_coarse_ratio"]
    for k in key_dog:
        short = k.replace("stamp_dog_", "s_").replace("ctrl_dog_", "c_").replace("ratio_dog_", "r_")
        print(f" {short:>12}", end="")
    print()

    for c in cards:
        label = "STAMP" if c["_stamped"] else "clean"
        print(f"  {c['_name']:<22} {label:<6}", end="")
        for k in key_dog:
            print(f" {c[k]:>12.4f}", end="")
        print()

    # ── LoG analysis detail ──────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  LoG Analysis (Laplacian of Gaussian at multiple scales)")
    print(f"{'=' * 90}")
    print(f"\n  {'Card':<22} {'Label':<6}  "
          f"{'LoG0.5':>7} {'LoG1.0':>7} {'LoG2.0':>7} {'LoG4.0':>7}  "
          f"{'R0.5':>6} {'R1.0':>6} {'R2.0':>6} {'R4.0':>6}")

    for c in cards:
        label = "STAMP" if c["_stamped"] else "clean"
        print(f"  {c['_name']:<22} {label:<6}  "
              f"{c['stamp_log_0p5']:7.2f} "
              f"{c['stamp_log_1p0']:7.2f} "
              f"{c['stamp_log_2p0']:7.2f} "
              f"{c['stamp_log_4p0']:7.2f}  "
              f"{c['ratio_log_0p5']:6.3f} "
              f"{c['ratio_log_1p0']:6.3f} "
              f"{c['ratio_log_2p0']:6.3f} "
              f"{c['ratio_log_4p0']:6.3f}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Total cards: {len(cards)} ({len(stamped)} stamped, {len(clean)} clean)")
    print(f"  Best single feature: {best_fname}")
    print(f"    Direction: stamped {best_dir} {best_thresh:.4f}")
    print(f"    Accuracy: {best_acc:.1%}")

    if combo_results:
        top_combo_acc, top_combo_desc = combo_results[0]
        print(f"  Best 2-feature combo: {top_combo_acc:.1%}")
        print(f"    {top_combo_desc}")

    # Key insight
    print(f"\n  Key insight: fine/coarse edge ratio")
    print(f"    Hypothesis: stamps add fine text edges -> higher fine/coarse ratio")
    s_fc = [c["stamp_fine_coarse_ratio"] for c in stamped]
    c_fc = [c["stamp_fine_coarse_ratio"] for c in clean]
    print(f"    Stamped fine/coarse: {np.mean(s_fc):.3f} +/- {np.std(s_fc):.3f}")
    print(f"    Clean   fine/coarse: {np.mean(c_fc):.3f} +/- {np.std(c_fc):.3f}")
    sep = abs(np.mean(s_fc) - np.mean(c_fc)) / (np.std(s_fc) + np.std(c_fc) + 1e-10)
    print(f"    Separation: {sep:.3f}")

    print(f"\n  Relative fine/coarse (stamp vs control region):")
    s_rfc = [c["relative_fine_coarse"] for c in stamped]
    c_rfc = [c["relative_fine_coarse"] for c in clean]
    print(f"    Stamped relative F/C: {np.mean(s_rfc):.3f} +/- {np.std(s_rfc):.3f}")
    print(f"    Clean   relative F/C: {np.mean(c_rfc):.3f} +/- {np.std(c_rfc):.3f}")
    sep2 = abs(np.mean(s_rfc) - np.mean(c_rfc)) / (np.std(s_rfc) + np.std(c_rfc) + 1e-10)
    print(f"    Separation: {sep2:.3f}")


if __name__ == "__main__":
    main()
