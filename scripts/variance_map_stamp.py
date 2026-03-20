#!/usr/bin/env python3
"""Variance map stamp detector: detect stamps via local pixel variance.

Stamps create high-variance patches (text edges against artwork background).
This script:
1. Loads 17 binder ground truth cards
2. Computes local variance maps using sliding windows
3. Compares stamp region variance vs control region variance
4. Builds a threshold classifier on variance ratio
5. Visualizes variance maps and reports accuracy

Output: data/condition_training/stamps_analysis/variance/
"""

import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INBOX_DIR = PROJECT_ROOT / "data" / "inbox"
GT_PATH = PROJECT_ROOT / "data" / "condition_training" / "stamps_real" / "binder_ground_truth.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "condition_training" / "stamps_analysis" / "variance"

# Stamp region: right side of artwork where EX stamps appear (normalized coords)
# From stamp_classifier.py: [0.55, 0.40, 0.92, 0.68]
STAMP_REGION = (0.55, 0.40, 0.92, 0.68)  # x0, y0, x1, y1

# Control region: left side of artwork, same vertical band
CONTROL_REGION = (0.10, 0.45, 0.45, 0.70)  # x0, y0, x1, y1

# Sliding window size for local variance
WINDOW_SIZE = 15


def load_ground_truth():
    """Load binder ground truth, deduplicating by image path (last wins)."""
    entries = {}
    with open(GT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            img_path = str(INBOX_DIR / obj["image"])
            entries[img_path] = {
                "path": img_path,
                "card_name": obj["card_name"],
                "stamped": obj["stamped"],
                "set_id": obj.get("set_id", ""),
                "variant": obj.get("variant", ""),
                "note": obj.get("note", ""),
            }
    return list(entries.values())


def compute_local_variance(gray, window_size=WINDOW_SIZE):
    """Compute local variance using sliding window via cv2 boxFilter.

    For each pixel, variance = E[X^2] - (E[X])^2 over a window_size x window_size neighborhood.
    """
    gray_f = gray.astype(np.float64)
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(window_size, window_size))
    mean_sq = cv2.boxFilter(gray_f ** 2, ddepth=-1, ksize=(window_size, window_size))
    variance = mean_sq - mean ** 2
    # Clamp numerical noise
    variance = np.maximum(variance, 0.0)
    return variance


def crop_region(img, region):
    """Crop a normalized region (x0, y0, x1, y1) from an image."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = region
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def analyze_card(entry, window_size=WINDOW_SIZE):
    """Analyze a single card image for variance-based stamp detection.

    Returns dict with metrics.
    """
    img = cv2.imread(entry["path"])
    if img is None:
        print(f"  WARNING: Could not read {entry['path']}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance_map = compute_local_variance(gray, window_size)

    # Crop stamp and control regions from variance map
    stamp_var = crop_region(variance_map, STAMP_REGION)
    control_var = crop_region(variance_map, CONTROL_REGION)

    # Metrics
    stamp_mean = float(np.mean(stamp_var))
    control_mean = float(np.mean(control_var))
    stamp_max = float(np.max(stamp_var))
    stamp_median = float(np.median(stamp_var))
    control_median = float(np.median(control_var))

    # Variance ratio (stamp / control), avoiding division by zero
    variance_ratio = stamp_mean / max(control_mean, 1e-6)

    # Hotspots: count pixels in stamp region with variance > threshold
    # Use percentile of the overall card variance as threshold
    overall_p90 = np.percentile(variance_map, 90)
    hotspot_count = int(np.sum(stamp_var > overall_p90))
    hotspot_fraction = hotspot_count / max(stamp_var.size, 1)

    # High-variance edge density: Canny on stamp region vs control
    stamp_gray = crop_region(gray, STAMP_REGION)
    control_gray = crop_region(gray, CONTROL_REGION)
    stamp_edges = cv2.Canny(stamp_gray, 50, 150)
    control_edges = cv2.Canny(control_gray, 50, 150)
    stamp_edge_density = np.mean(stamp_edges > 0)
    control_edge_density = np.mean(control_edges > 0)
    edge_ratio = stamp_edge_density / max(control_edge_density, 1e-6)

    return {
        "path": entry["path"],
        "card_name": entry["card_name"],
        "stamped": entry["stamped"],
        "set_id": entry["set_id"],
        "variant": entry["variant"],
        "stamp_mean_var": stamp_mean,
        "control_mean_var": control_mean,
        "variance_ratio": variance_ratio,
        "stamp_max_var": stamp_max,
        "stamp_median_var": stamp_median,
        "control_median_var": control_median,
        "hotspot_count": hotspot_count,
        "hotspot_fraction": hotspot_fraction,
        "stamp_edge_density": stamp_edge_density,
        "control_edge_density": control_edge_density,
        "edge_ratio": edge_ratio,
        "variance_map": variance_map,
        "gray": gray,
        "img": img,
    }


def find_best_threshold(results, metric_key="variance_ratio"):
    """Find threshold that maximizes accuracy on the given metric."""
    values = [(r[metric_key], r["stamped"]) for r in results]
    values.sort(key=lambda x: x[0])

    best_acc = 0
    best_thresh = 0
    best_direction = ">"  # stamped has higher metric

    # Try both directions
    for direction in [">", "<"]:
        for i in range(len(values)):
            thresh = values[i][0]
            if direction == ">":
                preds = [v >= thresh for v, _ in values]
            else:
                preds = [v < thresh for v, _ in values]
            correct = sum(1 for p, (_, gt) in zip(preds, values) if p == gt)
            acc = correct / len(values)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
                best_direction = direction

    return best_thresh, best_direction, best_acc


def visualize_variance_maps(results, output_dir):
    """Create side-by-side variance map visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    stamped = [r for r in results if r["stamped"]]
    clean = [r for r in results if not r["stamped"]]

    # --- Individual card variance maps ---
    for r in results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        label = "STAMPED" if r["stamped"] else "CLEAN"
        fig.suptitle(f"{r['card_name']} ({r['set_id']}) - {label}\n"
                     f"VarRatio={r['variance_ratio']:.2f}, "
                     f"EdgeRatio={r['edge_ratio']:.2f}, "
                     f"Hotspots={r['hotspot_fraction']:.3f}",
                     fontsize=12)

        # Original image with regions overlaid
        img_rgb = cv2.cvtColor(r["img"], cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        overlay = img_rgb.copy()
        # Draw stamp region (red)
        sx0, sy0, sx1, sy1 = [int(v * d) for v, d in
                               zip(STAMP_REGION, [w, h, w, h])]
        cv2.rectangle(overlay, (sx0, sy0), (sx1, sy1), (255, 0, 0), 2)
        # Draw control region (blue)
        cx0, cy0, cx1, cy1 = [int(v * d) for v, d in
                               zip(CONTROL_REGION, [w, h, w, h])]
        cv2.rectangle(overlay, (cx0, cy0), (cx1, cy1), (0, 0, 255), 2)
        axes[0].imshow(overlay)
        axes[0].set_title("Card (red=stamp, blue=control)")
        axes[0].axis("off")

        # Full variance map
        vmap = r["variance_map"]
        axes[1].imshow(vmap, cmap="hot", vmin=0, vmax=np.percentile(vmap, 99))
        axes[1].set_title(f"Variance Map (win={WINDOW_SIZE})")
        axes[1].axis("off")

        # Stamp vs control region variance maps side by side
        stamp_crop = crop_region(vmap, STAMP_REGION)
        control_crop = crop_region(vmap, CONTROL_REGION)
        vmax = max(np.percentile(stamp_crop, 99), np.percentile(control_crop, 99))
        # Combine them horizontally with a gap
        gap = np.zeros((stamp_crop.shape[0], 10))
        # Resize control to match stamp height
        ctrl_resized = cv2.resize(control_crop,
                                  (control_crop.shape[1],
                                   stamp_crop.shape[0]))
        combined = np.hstack([stamp_crop, gap, ctrl_resized])
        axes[2].imshow(combined, cmap="hot", vmin=0, vmax=vmax)
        axes[2].set_title(f"Stamp region (L) vs Control (R)\n"
                          f"mean={r['stamp_mean_var']:.0f} vs {r['control_mean_var']:.0f}")
        axes[2].axis("off")

        plt.tight_layout()
        safe_name = r["card_name"].replace(" ", "_").replace("'", "")
        fname = f"{'stamped' if r['stamped'] else 'clean'}_{safe_name}_{r['set_id']}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=120, bbox_inches="tight")
        plt.close()

    # --- Summary comparison plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Variance Map Stamp Detection - Summary", fontsize=14)

    # Plot 1: Variance ratio distribution
    stamped_ratios = [r["variance_ratio"] for r in stamped]
    clean_ratios = [r["variance_ratio"] for r in clean]
    ax = axes[0, 0]
    ax.barh(range(len(stamped_ratios)),
            stamped_ratios,
            color="red", alpha=0.7, label="Stamped")
    ax.barh(range(len(stamped_ratios), len(stamped_ratios) + len(clean_ratios)),
            clean_ratios,
            color="blue", alpha=0.7, label="Clean")
    names_s = [r["card_name"][:15] for r in stamped]
    names_c = [r["card_name"][:15] for r in clean]
    ax.set_yticks(range(len(names_s) + len(names_c)))
    ax.set_yticklabels(names_s + names_c, fontsize=8)
    ax.set_xlabel("Variance Ratio (stamp/control)")
    ax.set_title("Variance Ratio by Card")
    ax.legend()
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)

    # Plot 2: Edge ratio distribution
    stamped_edge = [r["edge_ratio"] for r in stamped]
    clean_edge = [r["edge_ratio"] for r in clean]
    ax = axes[0, 1]
    ax.barh(range(len(stamped_edge)),
            stamped_edge,
            color="red", alpha=0.7, label="Stamped")
    ax.barh(range(len(stamped_edge), len(stamped_edge) + len(clean_edge)),
            clean_edge,
            color="blue", alpha=0.7, label="Clean")
    ax.set_yticks(range(len(names_s) + len(names_c)))
    ax.set_yticklabels(names_s + names_c, fontsize=8)
    ax.set_xlabel("Edge Density Ratio (stamp/control)")
    ax.set_title("Edge Ratio by Card")
    ax.legend()
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)

    # Plot 3: Hotspot fraction
    stamped_hot = [r["hotspot_fraction"] for r in stamped]
    clean_hot = [r["hotspot_fraction"] for r in clean]
    ax = axes[1, 0]
    ax.barh(range(len(stamped_hot)),
            stamped_hot,
            color="red", alpha=0.7, label="Stamped")
    ax.barh(range(len(stamped_hot), len(stamped_hot) + len(clean_hot)),
            clean_hot,
            color="blue", alpha=0.7, label="Clean")
    ax.set_yticks(range(len(names_s) + len(names_c)))
    ax.set_yticklabels(names_s + names_c, fontsize=8)
    ax.set_xlabel("Hotspot Fraction (stamp region)")
    ax.set_title("Variance Hotspots by Card")
    ax.legend()

    # Plot 4: Scatter plot - variance ratio vs edge ratio
    ax = axes[1, 1]
    ax.scatter([r["variance_ratio"] for r in stamped],
               [r["edge_ratio"] for r in stamped],
               c="red", s=80, alpha=0.7, label="Stamped", marker="o")
    ax.scatter([r["variance_ratio"] for r in clean],
               [r["edge_ratio"] for r in clean],
               c="blue", s=80, alpha=0.7, label="Clean", marker="x")
    for r in results:
        ax.annotate(r["card_name"][:10], (r["variance_ratio"], r["edge_ratio"]),
                    fontsize=6, alpha=0.7)
    ax.set_xlabel("Variance Ratio")
    ax.set_ylabel("Edge Ratio")
    ax.set_title("Variance Ratio vs Edge Ratio")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved summary_comparison.png")

    # --- Stamped vs Clean side-by-side variance maps ---
    n_stamped = len(stamped)
    n_clean = len(clean)
    n_rows = max(n_stamped, n_clean)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    fig.suptitle("Variance Maps: Stamped (left) vs Clean (right)", fontsize=14)

    for i in range(n_rows):
        for col, group, title in [(0, stamped, "Stamped"), (1, clean, "Clean")]:
            ax = axes[i, col] if n_rows > 1 else axes[col]
            if i < len(group):
                r = group[i]
                vmap = r["variance_map"]
                ax.imshow(vmap, cmap="hot", vmin=0,
                          vmax=np.percentile(vmap, 99))
                ax.set_title(f"{title}: {r['card_name']}\n"
                             f"VR={r['variance_ratio']:.2f} ER={r['edge_ratio']:.2f}",
                             fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sidebyside_variance_maps.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved sidebyside_variance_maps.png")


def main():
    print("=" * 70)
    print("VARIANCE MAP STAMP DETECTOR")
    print("=" * 70)

    # Load ground truth
    gt_entries = load_ground_truth()
    print(f"\nLoaded {len(gt_entries)} ground truth cards")
    n_stamped = sum(1 for e in gt_entries if e["stamped"])
    n_clean = len(gt_entries) - n_stamped
    print(f"  Stamped: {n_stamped}, Clean: {n_clean}")

    # Analyze each card
    print(f"\nAnalyzing cards (window_size={WINDOW_SIZE})...")
    results = []
    for entry in gt_entries:
        r = analyze_card(entry)
        if r is not None:
            results.append(r)
            label = "STAMPED" if r["stamped"] else "clean  "
            print(f"  {label} {r['card_name']:20s} ({r['set_id']:5s}) | "
                  f"VarRatio={r['variance_ratio']:6.2f}  "
                  f"EdgeRatio={r['edge_ratio']:6.2f}  "
                  f"Hotspots={r['hotspot_fraction']:.3f}  "
                  f"StampMean={r['stamp_mean_var']:8.1f}  "
                  f"CtrlMean={r['control_mean_var']:8.1f}")

    # Build threshold classifiers
    print(f"\n{'=' * 70}")
    print("THRESHOLD CLASSIFIER RESULTS")
    print(f"{'=' * 70}")

    metrics_to_try = [
        ("variance_ratio", "Variance Ratio (stamp_mean / control_mean)"),
        ("edge_ratio", "Edge Density Ratio (stamp / control)"),
        ("hotspot_fraction", "Hotspot Fraction in stamp region"),
        ("stamp_mean_var", "Raw stamp region mean variance"),
        ("stamp_edge_density", "Raw stamp edge density"),
    ]

    best_overall_acc = 0
    best_overall_metric = ""

    for metric_key, description in metrics_to_try:
        thresh, direction, acc = find_best_threshold(results, metric_key)
        n_correct = int(acc * len(results))

        # Show predictions at this threshold
        if direction == ">":
            preds = [r[metric_key] >= thresh for r in results]
        else:
            preds = [r[metric_key] < thresh for r in results]

        tp = sum(1 for p, r in zip(preds, results) if p and r["stamped"])
        fp = sum(1 for p, r in zip(preds, results) if p and not r["stamped"])
        fn = sum(1 for p, r in zip(preds, results) if not p and r["stamped"])
        tn = sum(1 for p, r in zip(preds, results) if not p and not r["stamped"])

        print(f"\n  {description}")
        print(f"    Best threshold: {thresh:.4f} (predict stamped if {direction}= threshold)")
        print(f"    Accuracy: {n_correct}/{len(results)} ({acc:.1%})")
        print(f"    TP={tp} FP={fp} FN={fn} TN={tn}")

        if acc > best_overall_acc:
            best_overall_acc = acc
            best_overall_metric = metric_key

        # Show errors
        errors = []
        for p, r in zip(preds, results):
            if p != r["stamped"]:
                errors.append(r)
        if errors:
            print(f"    Errors:")
            for e in errors:
                label = "STAMPED" if e["stamped"] else "CLEAN"
                print(f"      {e['card_name']:20s} gt={label:7s} "
                      f"{metric_key}={e[metric_key]:.4f}")

    # Combined metric: variance_ratio + edge_ratio
    print(f"\n  Combined: variance_ratio + edge_ratio (weighted sum)")
    best_combined_acc = 0
    best_w = 0
    best_combined_thresh = 0
    for w_var in np.arange(0.0, 1.05, 0.05):
        w_edge = 1.0 - w_var
        combined = [w_var * r["variance_ratio"] + w_edge * r["edge_ratio"]
                    for r in results]
        # Try thresholds
        for thresh in sorted(set(combined)):
            preds = [c >= thresh for c in combined]
            correct = sum(1 for p, r in zip(preds, results) if p == r["stamped"])
            acc = correct / len(results)
            if acc > best_combined_acc:
                best_combined_acc = acc
                best_w = w_var
                best_combined_thresh = thresh

    combined_vals = [best_w * r["variance_ratio"] + (1 - best_w) * r["edge_ratio"]
                     for r in results]
    preds = [c >= best_combined_thresh for c in combined_vals]
    tp = sum(1 for p, r in zip(preds, results) if p and r["stamped"])
    fp = sum(1 for p, r in zip(preds, results) if p and not r["stamped"])
    fn = sum(1 for p, r in zip(preds, results) if not p and r["stamped"])
    tn = sum(1 for p, r in zip(preds, results) if not p and not r["stamped"])
    n_correct = int(best_combined_acc * len(results))

    print(f"    Best weights: w_var={best_w:.2f}, w_edge={1-best_w:.2f}")
    print(f"    Best threshold: {best_combined_thresh:.4f}")
    print(f"    Accuracy: {n_correct}/{len(results)} ({best_combined_acc:.1%})")
    print(f"    TP={tp} FP={fp} FN={fn} TN={tn}")

    errors = []
    for p, r, cv in zip(preds, results, combined_vals):
        if p != r["stamped"]:
            errors.append((r, cv))
    if errors:
        print(f"    Errors:")
        for e, cv in errors:
            label = "STAMPED" if e["stamped"] else "CLEAN"
            print(f"      {e['card_name']:20s} gt={label:7s} combined={cv:.4f}")

    # Multi-window analysis
    print(f"\n{'=' * 70}")
    print("MULTI-WINDOW SIZE ANALYSIS")
    print(f"{'=' * 70}")

    for ws in [7, 11, 15, 21, 31]:
        ws_results = []
        for entry in gt_entries:
            r = analyze_card(entry, window_size=ws)
            if r is not None:
                ws_results.append(r)

        thresh, direction, acc = find_best_threshold(ws_results, "variance_ratio")
        _, _, edge_acc = find_best_threshold(ws_results, "edge_ratio")
        n_correct = int(acc * len(ws_results))
        n_edge_correct = int(edge_acc * len(ws_results))
        print(f"  Window={ws:2d}: VarRatio {n_correct}/{len(ws_results)} ({acc:.1%})  "
              f"EdgeRatio {n_edge_correct}/{len(ws_results)} ({edge_acc:.1%})")

    # Visualizations
    print(f"\nGenerating visualizations...")
    visualize_variance_maps(results, str(OUTPUT_DIR))
    print(f"  Output: {OUTPUT_DIR}")

    # Print raw data table
    print(f"\n{'=' * 70}")
    print("RAW DATA TABLE")
    print(f"{'=' * 70}")
    print(f"{'Card':<22s} {'Label':>7s} {'VarRatio':>9s} {'EdgeRatio':>9s} "
          f"{'Hotspot%':>8s} {'StampVar':>9s} {'CtrlVar':>9s} {'StampMax':>9s}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: x["variance_ratio"], reverse=True):
        label = "STAMPED" if r["stamped"] else "clean"
        print(f"{r['card_name']:<22s} {label:>7s} {r['variance_ratio']:9.3f} "
              f"{r['edge_ratio']:9.3f} {r['hotspot_fraction']:8.4f} "
              f"{r['stamp_mean_var']:9.1f} {r['control_mean_var']:9.1f} "
              f"{r['stamp_max_var']:9.1f}")

    print(f"\n{'=' * 70}")
    print(f"BEST SINGLE METRIC: {best_overall_metric} at {best_overall_acc:.1%}")
    print(f"BEST COMBINED: {best_combined_acc:.1%}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
