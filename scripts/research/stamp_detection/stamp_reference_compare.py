#!/usr/bin/env python3
"""Compare binder scans against clean reference images to detect stamps.

Theory: after identifying a card (e.g., ex15-44 Chikorita delta), we have
a clean reference at data/card_images/ex15/ex15-44_normal.png. If the scanned
card is stamped, the stamp region differs from the reference. If not stamped,
the stamp region should match closely (modulo lighting/sleeve effects).

This uses the existing surface_detector.py DINOv2 patch comparison, plus
pixel-level SSIM in the stamp region, to test whether reference comparison
can separate stamped from non-stamped binder scans.

Usage:
    python scripts/research/stamp_detection/stamp_reference_compare.py
    python scripts/research/stamp_detection/stamp_reference_compare.py --verbose
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Card ID mapping for each binder ground truth entry
# ---------------------------------------------------------------------------
# Manually mapped from binder_ground_truth.jsonl card names to card_names.json IDs.
# The ground truth only has names + set_id; we need the actual card_id for reference lookup.

CARD_ID_MAP = {
    # ex15 Delta Species page
    "page_20260305_094228_cards/card_00.png": "ex15-44",   # Chikorita delta
    "page_20260305_094228_cards/card_01.png": "ex15-26",   # Bayleef delta
    "page_20260305_094228_cards/card_02.png": "ex15-4",    # Meganium delta
    "page_20260305_094228_cards/card_03.png": "ex15-67",   # Totodile delta
    "page_20260305_094228_cards/card_04.png": "ex15-27",   # Croconaw delta
    "page_20260305_094228_cards/card_05.png": "ex15-2",    # Feraligatr delta
    "page_20260305_094228_cards/card_06.png": "ex15-45",   # Cyndaquil delta
    "page_20260305_094228_cards/card_07.png": "ex15-36",   # Quilava delta
    "page_20260305_094228_cards/card_08.png": "ex15-12",   # Typhlosion delta
    # Gym Heroes page - Misty's Seadra (PRERELEASE stamp)
    "page_20260307_014406_cards/card_02.png": "gym1-9",    # Misty's Seadra
    # Team Rocket page - Dark Dragonite (stamped differently)
    "page_20260307_015320_cards/card_02.png": "base5-5",   # Dark Dragonite (holo)
    # Fossil page - Dragonite (no stamp)
    "page_20260307_015320_cards/card_05.png": "base3-4",   # Dragonite (Fossil)
    # Neo Revelation - Entei (promo, no stamp per se but firework holo)
    "page_20260307_020047_cards/card_05.png": "neo3-6",    # Entei
    # Fossil - Aerodactyl (PRERELEASE stamp)
    "page_20260307_020047_cards/card_08.png": "base3-1",   # Aerodactyl
}

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INBOX_DIR = PROJECT_ROOT / "data" / "inbox"
CARD_IMAGES_DIR = PROJECT_ROOT / "data" / "card_images"
GT_PATH = PROJECT_ROOT / "data" / "condition_training" / "stamps_real" / "binder_ground_truth.jsonl"

# Stamp region on 16x16 DINOv2 patch grid (from eval_stamp_surface.py)
# The EX-era stamp sits at approximately x: 55-88%, y: 35-55% of card
# On a 16x16 grid: cols 9-14, rows 5-8
STAMP_ROWS = (5, 9)   # rows 5-8 inclusive
STAMP_COLS = (8, 15)   # cols 8-14 inclusive

# For PRERELEASE stamps (WotC era): bottom-right of artwork
# These are larger and positioned differently
PRERELEASE_ROWS = (4, 10)
PRERELEASE_COLS = (6, 16)

# Pixel-level stamp region (fractional coords on the card image)
# EX-era stamp: bottom-right of artwork
PIXEL_STAMP_X0, PIXEL_STAMP_Y0 = 0.50, 0.30
PIXEL_STAMP_X1, PIXEL_STAMP_Y1 = 0.90, 0.58


def find_ref_image(card_id: str) -> str | None:
    """Locate the reference image for a card_id."""
    parts = card_id.split("-", 1)
    if len(parts) != 2:
        return None
    set_id = parts[0]
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = CARD_IMAGES_DIR / set_id / f"{card_id}_normal{ext}"
        if candidate.is_file():
            return str(candidate)
    return None


def load_ground_truth() -> list[dict]:
    """Load binder_ground_truth.jsonl entries that have known card_ids."""
    entries = []
    with open(GT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            image_rel = obj["image"]
            if image_rel not in CARD_ID_MAP:
                continue

            card_id = CARD_ID_MAP[image_rel]
            ref_path = find_ref_image(card_id)
            if ref_path is None:
                print(f"  WARNING: No reference image for {card_id} ({obj['card_name']})")
                continue

            scan_path = str(INBOX_DIR / image_rel)
            if not os.path.exists(scan_path):
                print(f"  WARNING: Scan not found: {scan_path}")
                continue

            entries.append({
                "image": image_rel,
                "scan_path": scan_path,
                "ref_path": ref_path,
                "card_id": card_id,
                "card_name": obj["card_name"],
                "stamped": obj["stamped"],
                "variant": obj.get("variant", ""),
                "set_id": obj.get("set_id", ""),
            })
    return entries


def extract_stamp_region_pixels(img: np.ndarray) -> np.ndarray:
    """Crop the stamp region from a card image (BGR)."""
    h, w = img.shape[:2]
    y0 = int(PIXEL_STAMP_Y0 * h)
    y1 = int(PIXEL_STAMP_Y1 * h)
    x0 = int(PIXEL_STAMP_X0 * w)
    x1 = int(PIXEL_STAMP_X1 * w)
    return img[y0:y1, x0:x1]


def compute_pixel_ssim(scan_crop: np.ndarray, ref_crop: np.ndarray) -> dict:
    """Compute SSIM between two stamp-region crops.

    Both are resized to a common size before comparison.
    Returns SSIM score and per-channel info.
    """
    target_size = (128, 96)  # width x height for stamp region
    scan_resized = cv2.resize(scan_crop, target_size)
    ref_resized = cv2.resize(ref_crop, target_size)

    # Convert to grayscale for SSIM
    scan_gray = cv2.cvtColor(scan_resized, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)

    score, diff_map = ssim(ref_gray, scan_gray, full=True)

    # Also compute per-channel SSIM
    channel_scores = []
    for c in range(3):
        cs = ssim(ref_resized[:, :, c], scan_resized[:, :, c])
        channel_scores.append(cs)

    # Pixel-level absolute difference
    abs_diff = cv2.absdiff(scan_gray.astype(np.float32),
                           ref_gray.astype(np.float32))
    mean_abs_diff = float(np.mean(abs_diff))
    max_abs_diff = float(np.max(abs_diff))

    return {
        "ssim": float(score),
        "ssim_per_channel": channel_scores,
        "mean_pixel_diff": mean_abs_diff,
        "max_pixel_diff": max_abs_diff,
        "diff_map": diff_map,
    }


def compute_dino_patch_stats(anomaly_map: np.ndarray,
                             stamp_rows: tuple = STAMP_ROWS,
                             stamp_cols: tuple = STAMP_COLS) -> dict:
    """Analyze DINOv2 patch similarity in stamp region vs rest."""
    stamp_patch = anomaly_map[stamp_rows[0]:stamp_rows[1],
                              stamp_cols[0]:stamp_cols[1]]

    mask = np.ones_like(anomaly_map, dtype=bool)
    mask[stamp_rows[0]:stamp_rows[1], stamp_cols[0]:stamp_cols[1]] = False
    rest_patches = anomaly_map[mask]

    stamp_mean = float(np.mean(stamp_patch))
    stamp_min = float(np.min(stamp_patch))
    rest_mean = float(np.mean(rest_patches))
    rest_min = float(np.min(rest_patches))
    delta = rest_mean - stamp_mean

    # Count anomalous patches in stamp region
    threshold = 0.85
    stamp_defects = int(np.sum(stamp_patch < threshold))
    stamp_total = stamp_patch.size
    stamp_defect_ratio = stamp_defects / stamp_total if stamp_total > 0 else 0.0

    return {
        "stamp_mean": stamp_mean,
        "stamp_min": stamp_min,
        "rest_mean": rest_mean,
        "rest_min": rest_min,
        "delta": delta,
        "stamp_defects": stamp_defects,
        "stamp_total": stamp_total,
        "stamp_defect_ratio": stamp_defect_ratio,
    }


def run_analysis(entries: list[dict], verbose: bool = False) -> list[dict]:
    """Run reference comparison on all entries."""
    from cardprice.ml.surface_detector import (
        extract_patch_tokens,
        compare_patches,
    )

    results = []

    for entry in entries:
        t0 = time.time()

        scan_path = entry["scan_path"]
        ref_path = entry["ref_path"]
        gt_stamped = entry["stamped"]

        # --- DINOv2 patch comparison ---
        scan_patches = extract_patch_tokens(scan_path)
        ref_patches = extract_patch_tokens(ref_path)
        anomaly_map = compare_patches(scan_patches, ref_patches)

        # Standard stamp region (EX-era)
        dino_stats = compute_dino_patch_stats(anomaly_map)
        # Also check wider region for PRERELEASE stamps
        dino_stats_wide = compute_dino_patch_stats(
            anomaly_map, PRERELEASE_ROWS, PRERELEASE_COLS)

        # --- Pixel-level comparison ---
        scan_img = cv2.imread(scan_path)
        ref_img = cv2.imread(ref_path)

        scan_stamp = extract_stamp_region_pixels(scan_img)
        ref_stamp = extract_stamp_region_pixels(ref_img)
        pixel_stats = compute_pixel_ssim(scan_stamp, ref_stamp)

        # --- Full card stats ---
        full_mean_sim = float(np.mean(anomaly_map))
        full_min_sim = float(np.min(anomaly_map))

        elapsed_ms = (time.time() - t0) * 1000

        result = {
            "image": entry["image"],
            "card_id": entry["card_id"],
            "card_name": entry["card_name"],
            "gt_stamped": gt_stamped,
            "variant": entry["variant"],
            "set_id": entry["set_id"],
            # DINOv2 patch stats (standard region)
            "dino_stamp_mean": dino_stats["stamp_mean"],
            "dino_stamp_min": dino_stats["stamp_min"],
            "dino_rest_mean": dino_stats["rest_mean"],
            "dino_delta": dino_stats["delta"],
            "dino_stamp_defect_ratio": dino_stats["stamp_defect_ratio"],
            # DINOv2 patch stats (wide region for PRERELEASE)
            "dino_wide_stamp_mean": dino_stats_wide["stamp_mean"],
            "dino_wide_delta": dino_stats_wide["delta"],
            "dino_wide_defect_ratio": dino_stats_wide["stamp_defect_ratio"],
            # Pixel-level SSIM
            "pixel_ssim": pixel_stats["ssim"],
            "pixel_mean_diff": pixel_stats["mean_pixel_diff"],
            "pixel_max_diff": pixel_stats["max_pixel_diff"],
            # Full card
            "full_mean_sim": full_mean_sim,
            "full_min_sim": full_min_sim,
            # Timing
            "time_ms": elapsed_ms,
            # Keep anomaly map for visualization
            "_anomaly_map": anomaly_map,
        }
        results.append(result)

        if verbose:
            label = "STAMPED" if gt_stamped else "CLEAN  "
            print(f"  [{label}] {entry['card_name']:20s} ({entry['card_id']:10s})  "
                  f"dino_delta={dino_stats['delta']:+.4f}  "
                  f"stamp_sim={dino_stats['stamp_mean']:.3f}  "
                  f"rest_sim={dino_stats['rest_mean']:.3f}  "
                  f"SSIM={pixel_stats['ssim']:.3f}  "
                  f"pix_diff={pixel_stats['mean_pixel_diff']:.1f}  "
                  f"{elapsed_ms:.0f}ms")

    return results


def print_report(results: list[dict]) -> None:
    """Print analysis report with separability analysis."""
    print("\n" + "=" * 80)
    print("STAMP REFERENCE COMPARISON: Binder Scan vs Clean Reference")
    print("=" * 80)
    print(f"\nTotal cards analyzed: {len(results)}")

    stamped = [r for r in results if r["gt_stamped"]]
    clean = [r for r in results if not r["gt_stamped"]]
    print(f"  Stamped: {len(stamped)}")
    print(f"  Clean:   {len(clean)}")

    # --- DINOv2 Delta Analysis ---
    print("\n" + "-" * 80)
    print("DINOv2 Patch Comparison (stamp region vs rest)")
    print("-" * 80)
    print(f"\n{'Card':<25s} {'Stamped':>7s}  {'Delta':>8s}  {'StampSim':>8s}  "
          f"{'RestSim':>8s}  {'DefectR':>8s}  {'SSIM':>6s}  {'PixDiff':>7s}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: x["dino_delta"], reverse=True):
        label = "YES" if r["gt_stamped"] else "no"
        variant = f" ({r['variant']})" if r["variant"] not in ("normal", "") else ""
        name = f"{r['card_name']}{variant}"
        print(f"{name:<25s} {label:>7s}  {r['dino_delta']:+8.4f}  "
              f"{r['dino_stamp_mean']:8.3f}  {r['dino_rest_mean']:8.3f}  "
              f"{r['dino_stamp_defect_ratio']:8.2f}  "
              f"{r['pixel_ssim']:6.3f}  {r['pixel_mean_diff']:7.1f}")

    # --- Separability Analysis ---
    print("\n" + "-" * 80)
    print("Separability Analysis")
    print("-" * 80)

    for metric_name, get_val, higher_for_stamped in [
        ("DINOv2 Delta (rest-stamp)", lambda r: r["dino_delta"], True),
        ("DINOv2 Wide Delta", lambda r: r["dino_wide_delta"], True),
        ("DINOv2 Stamp Defect Ratio", lambda r: r["dino_stamp_defect_ratio"], True),
        ("DINOv2 Wide Defect Ratio", lambda r: r["dino_wide_defect_ratio"], True),
        ("Pixel SSIM (lower=more diff)", lambda r: r["pixel_ssim"], False),
        ("Pixel Mean Diff", lambda r: r["pixel_mean_diff"], True),
        ("Full Card Mean Similarity", lambda r: r["full_mean_sim"], False),
    ]:
        s_vals = [get_val(r) for r in stamped]
        c_vals = [get_val(r) for r in clean]

        if not s_vals or not c_vals:
            continue

        s_mean = np.mean(s_vals)
        c_mean = np.mean(c_vals)
        s_min, s_max = np.min(s_vals), np.max(s_vals)
        c_min, c_max = np.min(c_vals), np.max(c_vals)

        if higher_for_stamped:
            gap = s_min - c_max
            separable = gap > 0
        else:
            gap = c_min - s_max
            separable = gap > 0

        sep_str = f"SEPARABLE (gap={gap:.4f})" if separable else f"OVERLAP ({gap:.4f})"

        print(f"\n  {metric_name}:")
        print(f"    Stamped: mean={s_mean:.4f}  min={s_min:.4f}  max={s_max:.4f}")
        print(f"    Clean:   mean={c_mean:.4f}  min={c_min:.4f}  max={c_max:.4f}")
        print(f"    => {sep_str}")

    # --- Optimal thresholds ---
    print("\n" + "-" * 80)
    print("Optimal Threshold Search")
    print("-" * 80)

    for metric_name, get_val, higher_means_stamped in [
        ("DINOv2 Delta", lambda r: r["dino_delta"], True),
        ("DINOv2 Wide Delta", lambda r: r["dino_wide_delta"], True),
        ("Pixel SSIM", lambda r: r["pixel_ssim"], False),
        ("Pixel Mean Diff", lambda r: r["pixel_mean_diff"], True),
        ("DINOv2 Stamp Mean Sim", lambda r: r["dino_stamp_mean"], False),
    ]:
        all_vals = [(get_val(r), r["gt_stamped"]) for r in results]
        all_vals.sort(key=lambda x: x[0])

        best_acc = 0
        best_thresh = 0

        # Try each midpoint as threshold
        vals_only = sorted(set(v for v, _ in all_vals))
        for i in range(len(vals_only) - 1):
            thresh = (vals_only[i] + vals_only[i + 1]) / 2
            if higher_means_stamped:
                correct = sum(1 for v, gt in all_vals
                              if (v >= thresh) == gt)
            else:
                correct = sum(1 for v, gt in all_vals
                              if (v < thresh) == gt)
            acc = correct / len(all_vals)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        print(f"  {metric_name}: best_threshold={best_thresh:.4f}  "
              f"accuracy={best_acc:.1%} ({int(best_acc * len(results))}/{len(results)})")

    # --- Heatmap visualization for each card ---
    print("\n" + "-" * 80)
    print("Anomaly Map Summary (16x16 DINOv2 grid)")
    print("-" * 80)
    print("  Stamp region: rows 5-8, cols 8-14")

    for r in results:
        amap = r["_anomaly_map"]
        label = "STAMPED" if r["gt_stamped"] else "CLEAN"
        print(f"\n  {r['card_name']} ({r['card_id']}) [{label}]")
        # Print stamp region values
        stamp_patch = amap[STAMP_ROWS[0]:STAMP_ROWS[1],
                           STAMP_COLS[0]:STAMP_COLS[1]]
        print(f"    Stamp region patch similarities:")
        for row_idx in range(stamp_patch.shape[0]):
            vals = " ".join(f"{v:.2f}" for v in stamp_patch[row_idx])
            print(f"      row {STAMP_ROWS[0]+row_idx}: {vals}")

    # --- Timing ---
    mean_time = np.mean([r["time_ms"] for r in results])
    print(f"\nMean time per card: {mean_time:.0f}ms")
    print("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare binder scans against clean references for stamp detection")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("Loading ground truth...")
    entries = load_ground_truth()
    print(f"Found {len(entries)} cards with known IDs and reference images")

    if not entries:
        print("ERROR: No entries found. Check paths.")
        sys.exit(1)

    for e in entries:
        label = "STAMPED" if e["stamped"] else "clean"
        print(f"  {e['card_name']:20s} ({e['card_id']:10s}) [{label:7s}] "
              f"ref={os.path.basename(e['ref_path'])}")

    print("\nRunning analysis...")
    results = run_analysis(entries, verbose=args.verbose)

    print_report(results)


if __name__ == "__main__":
    main()
