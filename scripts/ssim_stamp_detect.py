#!/usr/bin/env python3
"""
SSIM-based stamp detection: compare stamp region vs mirrored control region.

Theory: For non-stamped cards, the bottom-right and bottom-left of artwork
have similar texture (both are just card art). For stamped cards, the
bottom-right (stamp region) has LOWER structural similarity to the
bottom-left (control region) because the stamp adds unique structure.

Metrics computed:
  1. SSIM (structural similarity index)
  2. Pixel MSE (mean squared error)
  3. Normalized cross-correlation
  4. Histogram intersection (grayscale + per-channel)
  5. Top-right vs bottom-right comparison (stamp adds structure to bottom only)
"""

import json
import sys
import os
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

BASE = Path("/home/godli/cardprice")
INBOX = BASE / "data/inbox"
GT_PATH = BASE / "data/condition_training/stamps_real/binder_ground_truth.jsonl"


# ---------------------------------------------------------------------------
# Region cropping
# ---------------------------------------------------------------------------

def crop_stamp_region(img: np.ndarray) -> np.ndarray:
    """Bottom-right of artwork where stamp sits: x=[0.55,0.92], y=[0.40,0.68]."""
    h, w = img.shape[:2]
    return img[int(h * 0.40):int(h * 0.68), int(w * 0.55):int(w * 0.92)]


def crop_mirror_control(img: np.ndarray) -> np.ndarray:
    """Bottom-LEFT mirror of stamp region: x=[0.08,0.45], y=[0.40,0.68]."""
    h, w = img.shape[:2]
    return img[int(h * 0.40):int(h * 0.68), int(w * 0.08):int(w * 0.45)]


def crop_top_right(img: np.ndarray) -> np.ndarray:
    """Top-right of artwork: x=[0.55,0.92], y=[0.12,0.40]."""
    h, w = img.shape[:2]
    return img[int(h * 0.12):int(h * 0.40), int(w * 0.55):int(w * 0.92)]


def crop_top_left(img: np.ndarray) -> np.ndarray:
    """Top-left of artwork: x=[0.08,0.45], y=[0.12,0.40]."""
    h, w = img.shape[:2]
    return img[int(h * 0.12):int(h * 0.40), int(w * 0.08):int(w * 0.45)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def resize_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize both crops to the same dimensions (min of each axis)."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    # Ensure minimum size for SSIM window
    h = max(h, 8)
    w = max(w, 8)
    a_r = cv2.resize(a, (w, h))
    b_r = cv2.resize(b, (w, h))
    return a_r, b_r


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM between two BGR images (converted to grayscale)."""
    a_g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    win_size = min(7, min(a_g.shape[0], a_g.shape[1]))
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3
    return ssim(a_g, b_g, win_size=win_size)


def compute_ssim_color(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM on each color channel, averaged."""
    vals = []
    win_size = min(7, min(a.shape[0], a.shape[1]))
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3
    for c in range(3):
        vals.append(ssim(a[:, :, c], b[:, :, c], win_size=win_size))
    return float(np.mean(vals))


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two images (grayscale)."""
    a_g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b_g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return float(np.mean((a_g - b_g) ** 2))


def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation between two grayscale images."""
    a_g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b_g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    a_norm = a_g - np.mean(a_g)
    b_norm = b_g - np.mean(b_g)
    denom = np.sqrt(np.sum(a_norm ** 2) * np.sum(b_norm ** 2))
    if denom < 1e-10:
        return 0.0
    return float(np.sum(a_norm * b_norm) / denom)


def compute_hist_intersection(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram intersection (grayscale, normalized)."""
    a_g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    h_a = cv2.calcHist([a_g], [0], None, [256], [0, 256]).flatten()
    h_b = cv2.calcHist([b_g], [0], None, [256], [0, 256]).flatten()
    h_a = h_a / (h_a.sum() + 1e-10)
    h_b = h_b / (h_b.sum() + 1e-10)
    return float(np.sum(np.minimum(h_a, h_b)))


def compute_hist_intersection_color(a: np.ndarray, b: np.ndarray) -> float:
    """Color histogram intersection (per-channel, averaged)."""
    vals = []
    for c in range(3):
        h_a = cv2.calcHist([a], [c], None, [256], [0, 256]).flatten()
        h_b = cv2.calcHist([b], [c], None, [256], [0, 256]).flatten()
        h_a = h_a / (h_a.sum() + 1e-10)
        h_b = h_b / (h_b.sum() + 1e-10)
        vals.append(np.sum(np.minimum(h_a, h_b)))
    return float(np.mean(vals))


def compute_edge_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM on Canny edge maps (captures structural differences)."""
    a_g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    a_e = cv2.Canny(a_g, 50, 150).astype(np.float64)
    b_e = cv2.Canny(b_g, 50, 150).astype(np.float64)
    win_size = min(7, min(a_e.shape[0], a_e.shape[1]))
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3
    return ssim(a_e, b_e, win_size=win_size, data_range=255.0)


# ---------------------------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------------------------

def load_gt() -> list[dict]:
    """Load binder ground truth, deduplicating by image path (last wins)."""
    entries = {}
    with open(GT_PATH) as f:
        for line in f:
            entry = json.loads(line.strip())
            entries[entry["image"]] = entry
    return list(entries.values())


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_card(entry: dict) -> dict | None:
    img_path = INBOX / entry["image"]
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  WARN: cannot load {img_path}")
        return None

    stamp = crop_stamp_region(img)
    mirror = crop_mirror_control(img)
    top_r = crop_top_right(img)
    top_l = crop_top_left(img)

    # --- Comparison 1: stamp region (bottom-right) vs mirror control (bottom-left) ---
    s, m = resize_pair(stamp, mirror)
    lr_ssim = compute_ssim(s, m)
    lr_ssim_color = compute_ssim_color(s, m)
    lr_mse = compute_mse(s, m)
    lr_ncc = compute_ncc(s, m)
    lr_hist = compute_hist_intersection(s, m)
    lr_hist_color = compute_hist_intersection_color(s, m)
    lr_edge_ssim = compute_edge_ssim(s, m)

    # --- Comparison 2: top-right vs bottom-right (vertical asymmetry) ---
    tr, sr = resize_pair(top_r, stamp)
    tb_ssim = compute_ssim(tr, sr)
    tb_ssim_color = compute_ssim_color(tr, sr)
    tb_mse = compute_mse(tr, sr)
    tb_edge_ssim = compute_edge_ssim(tr, sr)

    # --- Comparison 3: top-left vs top-right (baseline symmetry, no stamp in either) ---
    tl, tr2 = resize_pair(top_l, top_r)
    tt_ssim = compute_ssim(tl, tr2)

    # --- Comparison 4: bottom-left vs top-left (vertical baseline) ---
    ml, tl2 = resize_pair(mirror, top_l)
    bl_tl_ssim = compute_ssim(ml, tl2)

    # Derived: asymmetry score = (top L-R SSIM) - (bottom L-R SSIM)
    # For stamped cards, bottom L-R SSIM should be lower, so asymmetry > 0
    asymmetry = tt_ssim - lr_ssim

    # Derived: vertical asymmetry = (left top-bottom SSIM) - (right top-bottom SSIM)
    # Stamp makes right top-bottom SSIM lower
    vert_asymmetry = bl_tl_ssim - tb_ssim

    return {
        "image": entry["image"],
        "card_name": entry.get("card_name", "?"),
        "stamped": entry["stamped"],
        "variant": entry.get("variant", "?"),
        # Left-right comparison (stamp vs mirror)
        "lr_ssim": lr_ssim,
        "lr_ssim_color": lr_ssim_color,
        "lr_mse": lr_mse,
        "lr_ncc": lr_ncc,
        "lr_hist": lr_hist,
        "lr_hist_color": lr_hist_color,
        "lr_edge_ssim": lr_edge_ssim,
        # Top-bottom comparison (top-right vs bottom-right)
        "tb_ssim": tb_ssim,
        "tb_ssim_color": tb_ssim_color,
        "tb_mse": tb_mse,
        "tb_edge_ssim": tb_edge_ssim,
        # Baselines
        "tt_ssim": tt_ssim,  # top L vs top R (no stamp in either)
        "bl_tl_ssim": bl_tl_ssim,  # bottom-left vs top-left
        # Derived
        "asymmetry": asymmetry,  # top LR SSIM - bottom LR SSIM
        "vert_asymmetry": vert_asymmetry,  # left TB SSIM - right TB SSIM
    }


def compute_separability(results: list[dict], metric: str,
                         lower_is_stamped: bool = True) -> dict:
    """Compute separation score and optimal threshold for a metric."""
    stamped = [r[metric] for r in results if r["stamped"]]
    clean = [r[metric] for r in results if not r["stamped"]]

    s_mean, s_std = np.mean(stamped), np.std(stamped)
    c_mean, c_std = np.mean(clean), np.std(clean)
    separation = abs(s_mean - c_mean) / (s_std + c_std + 1e-10)

    # Find optimal threshold
    all_vals = sorted(set(stamped + clean))
    best_acc = 0.0
    best_thresh = 0.0

    for thresh in all_vals:
        if lower_is_stamped:
            tp = sum(1 for v in stamped if v < thresh)
            tn = sum(1 for v in clean if v >= thresh)
        else:
            tp = sum(1 for v in stamped if v > thresh)
            tn = sum(1 for v in clean if v <= thresh)
        acc = (tp + tn) / (len(stamped) + len(clean))
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    # Also try the other direction
    for thresh in all_vals:
        if lower_is_stamped:
            tp = sum(1 for v in stamped if v > thresh)
            tn = sum(1 for v in clean if v <= thresh)
        else:
            tp = sum(1 for v in stamped if v < thresh)
            tn = sum(1 for v in clean if v >= thresh)
        acc = (tp + tn) / (len(stamped) + len(clean))
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            lower_is_stamped = not lower_is_stamped

    return {
        "metric": metric,
        "stamped_mean": s_mean,
        "stamped_std": s_std,
        "clean_mean": c_mean,
        "clean_std": c_std,
        "separation": separation,
        "best_acc": best_acc,
        "best_thresh": best_thresh,
        "stamped_values": stamped,
        "clean_values": clean,
    }


def main():
    print("SSIM Stamp Detection: Left-Right Symmetry Analysis")
    print("=" * 75)

    gt = load_gt()
    print(f"Loaded {len(gt)} ground truth entries")
    n_stamped = sum(1 for e in gt if e["stamped"])
    n_clean = sum(1 for e in gt if not e["stamped"])
    print(f"  Stamped: {n_stamped}, Clean: {n_clean}")

    # Analyze each card
    results = []
    for entry in gt:
        r = analyze_card(entry)
        if r is not None:
            results.append(r)

    print(f"\nAnalyzed {len(results)} cards")

    # --- Per-card results ---
    print(f"\n{'='*100}")
    print(f"{'Card':<25s} {'Stamp':>5s} {'LR_SSIM':>8s} {'LR_MSE':>8s} "
          f"{'LR_NCC':>8s} {'LR_Hist':>8s} {'EdgeSSIM':>8s} "
          f"{'TB_SSIM':>8s} {'Asym':>8s} {'VAsym':>8s}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: x["stamped"], reverse=True):
        tag = "YES" if r["stamped"] else "no"
        print(f"{r['card_name']:<25s} {tag:>5s} "
              f"{r['lr_ssim']:>8.4f} {r['lr_mse']:>8.1f} "
              f"{r['lr_ncc']:>8.4f} {r['lr_hist']:>8.4f} {r['lr_edge_ssim']:>8.4f} "
              f"{r['tb_ssim']:>8.4f} {r['asymmetry']:>8.4f} {r['vert_asymmetry']:>8.4f}")

    # --- Separability analysis ---
    metrics_to_test = [
        ("lr_ssim", True, "Left-Right SSIM (gray)"),
        ("lr_ssim_color", True, "Left-Right SSIM (color)"),
        ("lr_mse", False, "Left-Right MSE"),
        ("lr_ncc", True, "Left-Right NCC"),
        ("lr_hist", True, "Left-Right Hist Intersection"),
        ("lr_hist_color", True, "Left-Right Hist Color"),
        ("lr_edge_ssim", True, "Left-Right Edge SSIM"),
        ("tb_ssim", True, "Top-Bottom SSIM"),
        ("tb_ssim_color", True, "Top-Bottom SSIM (color)"),
        ("tb_mse", False, "Top-Bottom MSE"),
        ("tb_edge_ssim", True, "Top-Bottom Edge SSIM"),
        ("tt_ssim", True, "Top L-R SSIM (baseline)"),
        ("asymmetry", False, "Asymmetry (top LR - bottom LR)"),
        ("vert_asymmetry", False, "Vert Asymmetry (left TB - right TB)"),
    ]

    print(f"\n{'='*90}")
    print("SEPARABILITY ANALYSIS")
    print(f"{'='*90}")
    print(f"{'Metric':<35s} {'Stamped':>14s} {'Clean':>14s} {'Sep':>6s} {'Acc':>6s} {'Thresh':>8s}")
    print("-" * 90)

    sep_results = []
    for metric, lower_is_stamped, label in metrics_to_test:
        s = compute_separability(results, metric, lower_is_stamped)
        sep_results.append((label, s))
        print(f"{label:<35s} "
              f"{s['stamped_mean']:>6.4f}+{s['stamped_std']:<6.3f} "
              f"{s['clean_mean']:>6.4f}+{s['clean_std']:<6.3f} "
              f"{s['separation']:>6.3f} "
              f"{s['best_acc']:>5.1%} "
              f"{s['best_thresh']:>8.4f}")

    # --- Best metric detailed breakdown ---
    best_label, best_sep = max(sep_results, key=lambda x: x[1]["best_acc"])
    print(f"\n{'='*75}")
    print(f"BEST METRIC: {best_label}")
    print(f"  Accuracy: {best_sep['best_acc']:.1%}")
    print(f"  Threshold: {best_sep['best_thresh']:.4f}")
    print(f"  Separation: {best_sep['separation']:.3f}")
    print(f"{'='*75}")

    # Show misclassified
    metric_name = best_sep["metric"]
    thresh = best_sep["best_thresh"]
    s_mean = best_sep["stamped_mean"]
    c_mean = best_sep["clean_mean"]

    # Determine direction: if stamped mean < clean mean, stamped is below threshold
    stamped_below = s_mean < c_mean

    print(f"\nPer-card classification with {best_label} @ threshold={thresh:.4f}:")
    errors = 0
    for r in results:
        val = r[metric_name]
        if stamped_below:
            pred_stamped = val < thresh
        else:
            pred_stamped = val > thresh
        correct = pred_stamped == r["stamped"]
        status = "OK" if correct else "WRONG"
        if not correct:
            errors += 1
        tag = "stamped" if r["stamped"] else "clean"
        pred_tag = "stamped" if pred_stamped else "clean"
        print(f"  [{status:5s}] {r['card_name']:<25s} "
              f"gt={tag:<8s} pred={pred_tag:<8s} {metric_name}={val:.4f}")

    print(f"\nTotal errors: {errors}/{len(results)}")

    # --- Value distributions for the best few metrics ---
    print(f"\n{'='*75}")
    print("VALUE DISTRIBUTIONS (top 3 metrics by accuracy)")
    print(f"{'='*75}")

    top3 = sorted(sep_results, key=lambda x: -x[1]["best_acc"])[:3]
    for label, s in top3:
        print(f"\n  {label}:")
        print(f"    Stamped:  {sorted(s['stamped_values'])}")
        print(f"    Clean:    {sorted(s['clean_values'])}")
        # Show overlap
        s_min, s_max = min(s["stamped_values"]), max(s["stamped_values"])
        c_min, c_max = min(s["clean_values"]), max(s["clean_values"])
        overlap_lo = max(s_min, c_min)
        overlap_hi = min(s_max, c_max)
        if overlap_lo < overlap_hi:
            print(f"    OVERLAP:  [{overlap_lo:.4f}, {overlap_hi:.4f}]")
            n_overlap_s = sum(1 for v in s["stamped_values"]
                              if overlap_lo <= v <= overlap_hi)
            n_overlap_c = sum(1 for v in s["clean_values"]
                              if overlap_lo <= v <= overlap_hi)
            print(f"    In overlap: {n_overlap_s} stamped, {n_overlap_c} clean")
        else:
            print(f"    NO OVERLAP - perfect separation!")


if __name__ == "__main__":
    main()
