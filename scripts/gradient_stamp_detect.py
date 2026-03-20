#!/usr/bin/env python3
"""Gradient Orientation Histogram (HOG-like) stamp detection.

Theory: EX-era stamps are rotated text (~30-45 degrees). This creates strong
gradients at that specific angle. A magnitude-weighted histogram of gradient
orientations in the stamp region should show a peak at the stamp's rotation
angle for stamped cards, but not for clean cards.

Steps:
  1. Load 17 binder ground truth cards
  2. Crop stamp region (x=[0.55,0.92], y=[0.40,0.68])
  3. Compute Sobel gradients -> orientation + magnitude
  4. Build magnitude-weighted orientation histogram (18 bins, 0-180 degrees)
  5. Compute anisotropy score: max_bin / mean_bin
  6. Report separability and accuracy
  7. Save orientation histograms as plots
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/home/godli/cardprice")
INBOX = BASE / "data" / "inbox"
GT_PATH = BASE / "data" / "condition_training" / "stamps_real" / "binder_ground_truth.jsonl"
OUT_DIR = BASE / "data" / "condition_training" / "stamps_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_BINS = 18  # 0-180 degrees, 10 degrees per bin


def load_ground_truth():
    """Load binder ground truth, handling duplicate entries (last one wins)."""
    entries = {}
    with open(GT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Use image path as key so duplicates override
            entries[rec["image"]] = rec
    return list(entries.values())


def crop_stamp_region(img: np.ndarray) -> np.ndarray:
    """Crop stamp region: x=[0.55, 0.92], y=[0.40, 0.68]."""
    h, w = img.shape[:2]
    x1, x2 = int(w * 0.55), int(w * 0.92)
    y1, y2 = int(h * 0.40), int(h * 0.68)
    return img[y1:y2, x1:x2]


def compute_gradient_histogram(img_region: np.ndarray):
    """Compute magnitude-weighted gradient orientation histogram.

    Returns:
        hist: (NUM_BINS,) magnitude-weighted orientation histogram
        orientations: 2D array of gradient orientations in degrees [0, 180)
        magnitudes: 2D array of gradient magnitudes
    """
    # Convert to grayscale
    if len(img_region.shape) == 3:
        gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_region.copy()

    # Apply slight Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    # atan2 returns [-pi, pi], convert to [0, 180) for unsigned orientation
    orientation = np.degrees(np.arctan2(gy, gx)) % 180.0

    # Build magnitude-weighted histogram
    bin_edges = np.linspace(0, 180, NUM_BINS + 1)
    hist = np.zeros(NUM_BINS)
    for b in range(NUM_BINS):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == NUM_BINS - 1:
            mask = (orientation >= lo) & (orientation <= hi)
        else:
            mask = (orientation >= lo) & (orientation < hi)
        hist[b] = magnitude[mask].sum()

    # Normalize histogram so total = 1
    total = hist.sum()
    if total > 0:
        hist = hist / total

    return hist, orientation, magnitude


def compute_anisotropy(hist: np.ndarray) -> float:
    """Ratio of max bin to mean bin. Higher = more directional."""
    mean_val = hist.mean()
    if mean_val == 0:
        return 0.0
    return hist.max() / mean_val


def compute_stamp_angle_energy(hist: np.ndarray) -> float:
    """Energy in the 30-50 degree range (stamp text angle).

    Bins are 10 degrees each: bin 3 = [30,40), bin 4 = [40,50).
    """
    # Bins covering 30-50 degrees
    return hist[3] + hist[4]


def compute_peak_angle(hist: np.ndarray) -> float:
    """Angle of the peak bin in degrees."""
    bin_centers = np.linspace(5, 175, NUM_BINS)
    return bin_centers[np.argmax(hist)]


def main():
    gt = load_ground_truth()
    print(f"Loaded {len(gt)} ground truth entries")

    results = []
    stamped_hists = []
    clean_hists = []

    for rec in gt:
        img_path = INBOX / rec["image"]
        if not img_path.exists():
            print(f"  SKIP (not found): {rec['image']}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  SKIP (unreadable): {rec['image']}")
            continue

        stamp_crop = crop_stamp_region(img)
        hist, orientations, magnitudes = compute_gradient_histogram(stamp_crop)
        anisotropy = compute_anisotropy(hist)
        stamp_energy = compute_stamp_angle_energy(hist)
        peak_angle = compute_peak_angle(hist)
        is_stamped = rec["stamped"]

        results.append({
            "image": rec["image"],
            "name": rec["card_name"],
            "stamped": is_stamped,
            "anisotropy": anisotropy,
            "stamp_energy": stamp_energy,
            "peak_angle": peak_angle,
            "hist": hist,
        })

        if is_stamped:
            stamped_hists.append(hist)
        else:
            clean_hists.append(hist)

    # Print results table
    print(f"\n{'Card':<30} {'Stamped':>7} {'Aniso':>7} {'30-50E':>7} {'Peak':>6}")
    print("-" * 65)
    for r in results:
        label = "YES" if r["stamped"] else "no"
        print(f"{r['name']:<30} {label:>7} {r['anisotropy']:>7.3f} "
              f"{r['stamp_energy']:>7.4f} {r['peak_angle']:>5.0f}d")

    # Separability analysis
    stamped_aniso = [r["anisotropy"] for r in results if r["stamped"]]
    clean_aniso = [r["anisotropy"] for r in results if not r["stamped"]]
    stamped_energy = [r["stamp_energy"] for r in results if r["stamped"]]
    clean_energy = [r["stamp_energy"] for r in results if not r["stamped"]]

    print(f"\n--- Anisotropy (max/mean ratio) ---")
    print(f"  Stamped: mean={np.mean(stamped_aniso):.3f}, "
          f"min={np.min(stamped_aniso):.3f}, max={np.max(stamped_aniso):.3f}")
    print(f"  Clean:   mean={np.mean(clean_aniso):.3f}, "
          f"min={np.min(clean_aniso):.3f}, max={np.max(clean_aniso):.3f}")

    print(f"\n--- 30-50 degree energy ---")
    print(f"  Stamped: mean={np.mean(stamped_energy):.4f}, "
          f"min={np.min(stamped_energy):.4f}, max={np.max(stamped_energy):.4f}")
    print(f"  Clean:   mean={np.mean(clean_energy):.4f}, "
          f"min={np.min(clean_energy):.4f}, max={np.max(clean_energy):.4f}")

    # Try thresholds for anisotropy
    print(f"\n--- Threshold sweep (anisotropy) ---")
    best_acc = 0
    best_thresh = 0
    for thresh in np.arange(1.0, 3.0, 0.05):
        correct = sum(1 for r in results
                      if (r["anisotropy"] >= thresh) == r["stamped"])
        acc = correct / len(results)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    print(f"  Best threshold: {best_thresh:.2f}, accuracy: {best_acc:.1%} "
          f"({int(best_acc * len(results))}/{len(results)})")

    # Try thresholds for stamp energy
    print(f"\n--- Threshold sweep (30-50 degree energy) ---")
    best_acc_e = 0
    best_thresh_e = 0
    for thresh in np.arange(0.05, 0.30, 0.005):
        correct = sum(1 for r in results
                      if (r["stamp_energy"] >= thresh) == r["stamped"])
        acc = correct / len(results)
        if acc > best_acc_e:
            best_acc_e = acc
            best_thresh_e = thresh
    print(f"  Best threshold: {best_thresh_e:.3f}, accuracy: {best_acc_e:.1%} "
          f"({int(best_acc_e * len(results))}/{len(results)})")

    # Combined score: stamp_energy * anisotropy
    print(f"\n--- Threshold sweep (combined: energy * anisotropy) ---")
    combined = [r["stamp_energy"] * r["anisotropy"] for r in results]
    stamped_comb = [c for c, r in zip(combined, results) if r["stamped"]]
    clean_comb = [c for c, r in zip(combined, results) if not r["stamped"]]
    print(f"  Stamped: mean={np.mean(stamped_comb):.4f}, "
          f"min={np.min(stamped_comb):.4f}, max={np.max(stamped_comb):.4f}")
    print(f"  Clean:   mean={np.mean(clean_comb):.4f}, "
          f"min={np.min(clean_comb):.4f}, max={np.max(clean_comb):.4f}")
    best_acc_c = 0
    best_thresh_c = 0
    for thresh in np.arange(0.05, 0.60, 0.005):
        correct = sum(1 for r, c in zip(results, combined)
                      if (c >= thresh) == r["stamped"])
        acc = correct / len(results)
        if acc > best_acc_c:
            best_acc_c = acc
            best_thresh_c = thresh
    print(f"  Best threshold: {best_thresh_c:.3f}, accuracy: {best_acc_c:.1%} "
          f"({int(best_acc_c * len(results))}/{len(results)})")

    # --- Plots ---

    # 1. Individual histograms per card
    fig, axes = plt.subplots(4, 5, figsize=(20, 14))
    bin_centers = np.linspace(5, 175, NUM_BINS)
    axes = axes.flatten()
    for i, r in enumerate(results):
        if i >= len(axes):
            break
        ax = axes[i]
        color = "red" if r["stamped"] else "steelblue"
        ax.bar(bin_centers, r["hist"], width=9, color=color, alpha=0.7)
        ax.set_title(f"{r['name']}\n{'STAMPED' if r['stamped'] else 'clean'} "
                     f"(aniso={r['anisotropy']:.2f})", fontsize=9)
        ax.set_xlabel("Orientation (deg)", fontsize=7)
        ax.set_ylabel("Weight", fontsize=7)
        ax.tick_params(labelsize=7)
        # Highlight 30-50 degree region
        ax.axvspan(30, 50, alpha=0.15, color="gold")
    # Hide unused axes
    for j in range(len(results), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Gradient Orientation Histograms — Stamp Region\n"
                 "(red=stamped, blue=clean, gold band=30-50 deg)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path1 = OUT_DIR / "gradient_orientation_histograms.png"
    fig.savefig(plot_path1, dpi=150)
    print(f"\nSaved: {plot_path1}")

    # 2. Average histograms: stamped vs clean
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    if stamped_hists:
        avg_stamped = np.mean(stamped_hists, axis=0)
        ax2.bar(bin_centers - 2, avg_stamped, width=4, color="red",
                alpha=0.7, label=f"Stamped (n={len(stamped_hists)})")
    if clean_hists:
        avg_clean = np.mean(clean_hists, axis=0)
        ax2.bar(bin_centers + 2, avg_clean, width=4, color="steelblue",
                alpha=0.7, label=f"Clean (n={len(clean_hists)})")
    ax2.axvspan(30, 50, alpha=0.15, color="gold", label="30-50 deg (stamp angle)")
    ax2.set_xlabel("Gradient Orientation (degrees)")
    ax2.set_ylabel("Normalized Weight")
    ax2.set_title("Average Gradient Orientation: Stamped vs Clean")
    ax2.legend()
    plot_path2 = OUT_DIR / "gradient_avg_stamped_vs_clean.png"
    fig2.savefig(plot_path2, dpi=150)
    print(f"Saved: {plot_path2}")

    # 3. Scatter plot: anisotropy vs stamp energy
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for r in results:
        color = "red" if r["stamped"] else "steelblue"
        marker = "^" if r["stamped"] else "o"
        ax3.scatter(r["anisotropy"], r["stamp_energy"],
                    c=color, marker=marker, s=80, edgecolors="black", linewidth=0.5)
        ax3.annotate(r["name"], (r["anisotropy"], r["stamp_energy"]),
                     fontsize=6, alpha=0.7)
    ax3.set_xlabel("Anisotropy (max/mean)")
    ax3.set_ylabel("30-50 deg Energy")
    ax3.set_title("Stamp Detection: Anisotropy vs Stamp-Angle Energy")
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="red",
               markersize=10, label="Stamped"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=10, label="Clean"),
    ]
    ax3.legend(handles=legend_elements)
    plot_path3 = OUT_DIR / "gradient_scatter.png"
    fig3.savefig(plot_path3, dpi=150)
    print(f"Saved: {plot_path3}")

    plt.close("all")


if __name__ == "__main__":
    main()
