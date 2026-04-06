#!/usr/bin/env python3
"""
Stamp detection via pixel-level analysis of the stamp region.

EX-era stamps are gold/silver metallic text overlaid on card artwork in the
bottom-right area of the art box. This script analyzes pixel-level features
that distinguish stamped from non-stamped cards without ML models.

Features analyzed:
  1. Mean brightness in stamp region vs surrounding artwork
  2. Color channel ratios (R/G, R/B) for gold tint detection
  3. Edge density via Canny — stamps have text edges
  4. HSV hue/saturation distribution
  5. Local contrast (Laplacian variance) — texture energy
  6. High-frequency energy via DCT
  7. Gold pixel ratio — pixels matching gold color profile

Runs on three datasets:
  - Binder scans (the real target)
  - Reference photos from the web
  - Synthetic stamps
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

BASE = Path("/home/godli/cardprice")
STAMPS_REAL = BASE / "data/condition_training/stamps_real"
STAMPS_SYNTH = BASE / "data/condition_training/stamps"
INBOX = BASE / "data/inbox"
OUT_DIR = BASE / "data/condition_training/stamps_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Stamp region cropping ---

def crop_stamp_region(img: np.ndarray) -> np.ndarray:
    """Crop the stamp region: bottom-right of artwork area.

    EX-era stamps appear in roughly:
      x: 55-90% of card width
      y: 45-70% of card height
    This targets the artwork area where the set logo stamp sits.
    """
    h, w = img.shape[:2]
    x1, x2 = int(w * 0.55), int(w * 0.90)
    y1, y2 = int(h * 0.45), int(h * 0.70)
    return img[y1:y2, x1:x2]


def crop_artwork_region(img: np.ndarray) -> np.ndarray:
    """Crop the full artwork area (for comparison baseline).

    Artwork is roughly the top 70% of card, excluding name bar at top.
    """
    h, w = img.shape[:2]
    x1, x2 = int(w * 0.05), int(w * 0.95)
    y1, y2 = int(h * 0.10), int(h * 0.70)
    return img[y1:y2, x1:x2]


def crop_control_region(img: np.ndarray) -> np.ndarray:
    """Crop a control region: same vertical band but left side (no stamp)."""
    h, w = img.shape[:2]
    x1, x2 = int(w * 0.10), int(w * 0.45)
    y1, y2 = int(h * 0.45), int(h * 0.70)
    return img[y1:y2, x1:x2]


# --- Feature extraction ---

@dataclass
class StampFeatures:
    """Pixel-level features extracted from stamp region."""
    # Brightness
    mean_brightness: float = 0.0
    brightness_ratio: float = 0.0  # stamp / control region

    # Color channels (BGR -> RGB for analysis)
    mean_r: float = 0.0
    mean_g: float = 0.0
    mean_b: float = 0.0
    rg_ratio: float = 0.0  # R/G — gold has R/G close to 1.0-1.2
    rb_ratio: float = 0.0  # R/B — gold has high R/B (>1.5)

    # Edge density
    edge_density: float = 0.0  # fraction of Canny edge pixels
    edge_density_ratio: float = 0.0  # stamp edges / control edges

    # HSV
    mean_hue: float = 0.0
    mean_saturation: float = 0.0
    mean_value: float = 0.0
    hue_std: float = 0.0
    sat_std: float = 0.0
    # Gold hue range (roughly 15-40 in OpenCV hue scale 0-180)
    gold_hue_fraction: float = 0.0

    # Texture / contrast
    laplacian_var: float = 0.0  # variance of Laplacian — texture energy
    laplacian_ratio: float = 0.0  # stamp / control

    # High-frequency energy
    high_freq_energy: float = 0.0
    high_freq_ratio: float = 0.0

    # Gold pixel detection
    gold_pixel_ratio: float = 0.0  # fraction of pixels that look "gold"

    # Metadata
    label: bool = False
    name: str = ""
    dataset: str = ""


def compute_edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels that are Canny edges."""
    edges = cv2.Canny(gray, 50, 150)
    return np.mean(edges > 0)


def compute_laplacian_var(gray: np.ndarray) -> float:
    """Variance of Laplacian — measures texture/focus energy."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))


def compute_high_freq_energy(gray: np.ndarray) -> float:
    """High-frequency energy using Fourier transform."""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    # Mask out low frequencies (center 20%)
    r = min(h, w) // 5
    magnitude = np.abs(fshift)
    # Zero out center
    mask = np.ones_like(magnitude)
    mask[cy - r:cy + r, cx - r:cx + r] = 0
    high_freq = np.sum(magnitude * mask)
    total = np.sum(magnitude) + 1e-10
    return float(high_freq / total)


def compute_gold_pixel_ratio(img_bgr: np.ndarray) -> float:
    """Fraction of pixels that match a 'gold' color profile.

    Gold characteristics in HSV:
      Hue: 15-40 (yellow-orange range, OpenCV 0-180 scale)
      Saturation: 40-255 (reasonably saturated)
      Value: 120-255 (bright)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 40, 120])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.mean(mask > 0))


def extract_features(img: np.ndarray, label: bool, name: str,
                     dataset: str) -> Optional[StampFeatures]:
    """Extract all stamp features from a card image."""
    if img is None or img.size == 0:
        return None

    stamp = crop_stamp_region(img)
    control = crop_control_region(img)

    if stamp.size == 0 or control.size == 0:
        return None

    # Convert to different color spaces
    stamp_gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
    control_gray = cv2.cvtColor(control, cv2.COLOR_BGR2GRAY)
    stamp_hsv = cv2.cvtColor(stamp, cv2.COLOR_BGR2HSV)

    # RGB channels (OpenCV loads as BGR)
    stamp_rgb = cv2.cvtColor(stamp, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean_r = float(np.mean(stamp_rgb[:, :, 0]))
    mean_g = float(np.mean(stamp_rgb[:, :, 1]))
    mean_b = float(np.mean(stamp_rgb[:, :, 2]))

    feat = StampFeatures()
    feat.label = label
    feat.name = name
    feat.dataset = dataset

    # Brightness
    feat.mean_brightness = float(np.mean(stamp_gray))
    control_brightness = float(np.mean(control_gray))
    feat.brightness_ratio = feat.mean_brightness / (control_brightness + 1e-10)

    # Color channels
    feat.mean_r = mean_r
    feat.mean_g = mean_g
    feat.mean_b = mean_b
    feat.rg_ratio = mean_r / (mean_g + 1e-10)
    feat.rb_ratio = mean_r / (mean_b + 1e-10)

    # Edge density
    feat.edge_density = compute_edge_density(stamp_gray)
    control_edges = compute_edge_density(control_gray)
    feat.edge_density_ratio = feat.edge_density / (control_edges + 1e-10)

    # HSV
    feat.mean_hue = float(np.mean(stamp_hsv[:, :, 0]))
    feat.mean_saturation = float(np.mean(stamp_hsv[:, :, 1]))
    feat.mean_value = float(np.mean(stamp_hsv[:, :, 2]))
    feat.hue_std = float(np.std(stamp_hsv[:, :, 0]))
    feat.sat_std = float(np.std(stamp_hsv[:, :, 1]))
    # Gold hue fraction (15-40 on OpenCV's 0-180 scale)
    hue = stamp_hsv[:, :, 0]
    feat.gold_hue_fraction = float(np.mean((hue >= 15) & (hue <= 40)))

    # Texture
    feat.laplacian_var = compute_laplacian_var(stamp_gray)
    control_lap = compute_laplacian_var(control_gray)
    feat.laplacian_ratio = feat.laplacian_var / (control_lap + 1e-10)

    # High-frequency energy
    feat.high_freq_energy = compute_high_freq_energy(stamp_gray)
    feat.high_freq_ratio = feat.high_freq_energy / (
        compute_high_freq_energy(control_gray) + 1e-10)

    # Gold pixels
    feat.gold_pixel_ratio = compute_gold_pixel_ratio(stamp)

    return feat


# --- Data loading ---

def load_binder_dataset() -> list[tuple[np.ndarray, bool, str]]:
    """Load binder scan cards with ground truth labels."""
    gt_path = STAMPS_REAL / "binder_ground_truth.jsonl"
    results = []
    with open(gt_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            img_path = INBOX / entry["image"]
            img = cv2.imread(str(img_path))
            if img is not None:
                name = entry.get("card_name", Path(entry["image"]).stem)
                results.append((img, entry["stamped"], name))
            else:
                print(f"  [WARN] Could not load binder image: {img_path}")
    return results


def load_real_photos_dataset() -> list[tuple[np.ndarray, bool, str]]:
    """Load reference photos (web-sourced stamped/clean cards)."""
    src_path = STAMPS_REAL / "sources.jsonl"
    results = []
    with open(src_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            img_path = STAMPS_REAL / entry["image"]
            img = cv2.imread(str(img_path))
            if img is not None:
                name = entry.get("card_name", Path(entry["image"]).stem)
                results.append((img, entry["stamped"], name))
            else:
                print(f"  [WARN] Could not load real photo: {img_path}")
    return results


def load_synthetic_dataset(max_per_class: int = 50) -> list[
        tuple[np.ndarray, bool, str]]:
    """Load synthetic stamps (rendered stamp overlays on clean cards)."""
    labels_path = STAMPS_SYNTH / "labels.jsonl"
    results = []
    stamped_count = 0
    clean_count = 0
    with open(labels_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            is_stamped = entry["stamped"]
            if is_stamped and stamped_count >= max_per_class:
                continue
            if not is_stamped and clean_count >= max_per_class:
                continue
            img_path = STAMPS_SYNTH / entry["image"]
            img = cv2.imread(str(img_path))
            if img is not None:
                name = entry.get("card_id", Path(entry["image"]).stem)
                results.append((img, is_stamped, name))
                if is_stamped:
                    stamped_count += 1
                else:
                    clean_count += 1
            if stamped_count >= max_per_class and clean_count >= max_per_class:
                break
    return results


# --- Analysis and reporting ---

def analyze_feature_separation(features: list[StampFeatures],
                               dataset_name: str) -> dict[str, float]:
    """Compute separation metrics for each feature between stamped/clean.

    Returns a dict of feature_name -> separation score (higher = better).
    Separation = |mean_stamped - mean_clean| / (std_stamped + std_clean + eps).
    """
    stamped = [f for f in features if f.label]
    clean = [f for f in features if not f.label]

    if not stamped or not clean:
        print(f"  [{dataset_name}] Need both stamped and clean samples!")
        return {}

    feature_names = [
        "mean_brightness", "brightness_ratio", "rg_ratio", "rb_ratio",
        "edge_density", "edge_density_ratio", "mean_hue", "mean_saturation",
        "mean_value", "hue_std", "sat_std", "gold_hue_fraction",
        "laplacian_var", "laplacian_ratio", "high_freq_energy",
        "high_freq_ratio", "gold_pixel_ratio",
    ]

    separations = {}
    print(f"\n{'='*70}")
    print(f"  Feature Separation: {dataset_name}")
    print(f"  Stamped: {len(stamped)}, Clean: {len(clean)}")
    print(f"{'='*70}")
    print(f"  {'Feature':<22} {'Stamped':>12} {'Clean':>12} {'Sep':>8}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*8}")

    for fname in feature_names:
        s_vals = np.array([getattr(f, fname) for f in stamped])
        c_vals = np.array([getattr(f, fname) for f in clean])
        s_mean, s_std = np.mean(s_vals), np.std(s_vals)
        c_mean, c_std = np.mean(c_vals), np.std(c_vals)
        sep = abs(s_mean - c_mean) / (s_std + c_std + 1e-10)
        separations[fname] = sep
        print(f"  {fname:<22} {s_mean:>8.3f}+{s_std:<4.2f} "
              f"{c_mean:>8.3f}+{c_std:<4.2f} {sep:>7.3f}")

    # Rank by separation
    ranked = sorted(separations.items(), key=lambda x: -x[1])
    print(f"\n  Top features by separation:")
    for i, (fname, sep) in enumerate(ranked[:5]):
        print(f"    {i+1}. {fname}: {sep:.3f}")

    return separations


def build_threshold_detector(
    features: list[StampFeatures], dataset_name: str
) -> dict:
    """Find optimal thresholds for each feature and combined detector."""
    stamped = [f for f in features if f.label]
    clean = [f for f in features if not f.label]
    if not stamped or not clean:
        return {}

    feature_names = [
        "edge_density", "edge_density_ratio", "gold_pixel_ratio",
        "laplacian_var", "laplacian_ratio", "high_freq_energy",
        "high_freq_ratio", "brightness_ratio", "gold_hue_fraction",
        "mean_saturation", "sat_std",
    ]

    print(f"\n{'='*70}")
    print(f"  Threshold Detector: {dataset_name}")
    print(f"{'='*70}")

    best_feature = None
    best_acc = 0.0
    best_threshold = 0.0
    best_direction = ">"

    for fname in feature_names:
        s_vals = [getattr(f, fname) for f in stamped]
        c_vals = [getattr(f, fname) for f in clean]
        all_vals = sorted(set(s_vals + c_vals))

        best_feat_acc = 0.0
        best_feat_thresh = 0.0
        best_feat_dir = ">"

        for thresh in all_vals:
            # Try stamped > thresh
            tp = sum(1 for v in s_vals if v > thresh)
            tn = sum(1 for v in c_vals if v <= thresh)
            acc_gt = (tp + tn) / (len(s_vals) + len(c_vals))

            # Try stamped < thresh
            tp2 = sum(1 for v in s_vals if v < thresh)
            tn2 = sum(1 for v in c_vals if v >= thresh)
            acc_lt = (tp2 + tn2) / (len(s_vals) + len(c_vals))

            if acc_gt > best_feat_acc:
                best_feat_acc = acc_gt
                best_feat_thresh = thresh
                best_feat_dir = ">"
            if acc_lt > best_feat_acc:
                best_feat_acc = acc_lt
                best_feat_thresh = thresh
                best_feat_dir = "<"

        print(f"  {fname:<22} best_acc={best_feat_acc:.1%}  "
              f"threshold={best_feat_thresh:.4f}  dir={best_feat_dir}")

        if best_feat_acc > best_acc:
            best_acc = best_feat_acc
            best_feature = fname
            best_threshold = best_feat_thresh
            best_direction = best_feat_dir

    print(f"\n  BEST single feature: {best_feature} "
          f"{best_direction} {best_threshold:.4f} -> {best_acc:.1%}")

    # Try a simple 2-feature combo: top 2 by individual accuracy
    # Exhaustive 2-feature search
    print(f"\n  --- 2-Feature Combos (top 5) ---")
    combo_results = []
    for i, f1 in enumerate(feature_names):
        s1 = [getattr(f, f1) for f in stamped]
        c1 = [getattr(f, f1) for f in clean]
        for f2 in feature_names[i + 1:]:
            s2 = [getattr(f, f2) for f in stamped]
            c2 = [getattr(f, f2) for f in clean]
            # Try various threshold combos using median as threshold
            for t1 in [np.median(s1 + c1),
                       (np.mean(s1) + np.mean(c1)) / 2]:
                for t2 in [np.median(s2 + c2),
                           (np.mean(s2) + np.mean(c2)) / 2]:
                    for d1 in [1, -1]:
                        for d2 in [1, -1]:
                            tp = sum(
                                1 for j in range(len(stamped))
                                if (d1 * s1[j] > d1 * t1
                                    or d2 * s2[j] > d2 * t2)
                            )
                            tn = sum(
                                1 for j in range(len(clean))
                                if not (d1 * c1[j] > d1 * t1
                                        or d2 * c2[j] > d2 * t2)
                            )
                            acc = (tp + tn) / len(features)
                            combo_results.append(
                                (acc, f1, t1, d1, f2, t2, d2))

    combo_results.sort(key=lambda x: -x[0])
    for acc, f1, t1, d1, f2, t2, d2 in combo_results[:5]:
        d1s = ">" if d1 > 0 else "<"
        d2s = ">" if d2 > 0 else "<"
        print(f"    {acc:.1%}  {f1}{d1s}{t1:.4f} OR {f2}{d2s}{t2:.4f}")

    return {
        "best_feature": best_feature,
        "best_threshold": best_threshold,
        "best_direction": best_direction,
        "best_accuracy": best_acc,
    }


# --- Visualization ---

def save_stamp_crops_comparison(
    features: list[StampFeatures],
    images: list[np.ndarray],
    dataset_name: str,
):
    """Save side-by-side stamp region crops: stamped vs clean."""
    stamped_imgs = [(img, f) for img, f in zip(images, features) if f.label]
    clean_imgs = [(img, f) for img, f in zip(images, features) if not f.label]

    n_show = min(8, len(stamped_imgs), len(clean_imgs))
    if n_show == 0:
        print(f"  [{dataset_name}] Not enough samples for visualization")
        return

    # Normalize stamp crops to same size
    crop_h, crop_w = 120, 180

    def make_row(img_list, label_text):
        crops = []
        for img, feat in img_list[:n_show]:
            crop = crop_stamp_region(img)
            crop = cv2.resize(crop, (crop_w, crop_h))
            # Add label
            cv2.putText(crop, f"E:{feat.edge_density:.3f}",
                        (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 255, 0), 1)
            cv2.putText(crop, f"G:{feat.gold_pixel_ratio:.3f}",
                        (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 255, 0), 1)
            cv2.putText(crop, f"L:{feat.laplacian_var:.0f}",
                        (2, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 255, 0), 1)
            crops.append(crop)
        row = np.hstack(crops)
        # Add header
        header = np.zeros((25, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, label_text, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return np.vstack([header, row])

    stamped_row = make_row(stamped_imgs, "STAMPED")
    clean_row = make_row(clean_imgs, "CLEAN")

    # Ensure same width
    max_w = max(stamped_row.shape[1], clean_row.shape[1])
    if stamped_row.shape[1] < max_w:
        pad = np.zeros((stamped_row.shape[0],
                        max_w - stamped_row.shape[1], 3), dtype=np.uint8)
        stamped_row = np.hstack([stamped_row, pad])
    if clean_row.shape[1] < max_w:
        pad = np.zeros((clean_row.shape[0],
                        max_w - clean_row.shape[1], 3), dtype=np.uint8)
        clean_row = np.hstack([clean_row, pad])

    comparison = np.vstack([stamped_row, clean_row])
    out_path = OUT_DIR / f"stamp_crops_{dataset_name}.png"
    cv2.imwrite(str(out_path), comparison)
    print(f"  Saved crop comparison: {out_path}")


def save_feature_distributions(features: list[StampFeatures],
                               dataset_name: str):
    """Save a text-based distribution visualization as an image."""
    stamped = [f for f in features if f.label]
    clean = [f for f in features if not f.label]
    if not stamped or not clean:
        return

    key_features = [
        ("edge_density", "Edge Density"),
        ("edge_density_ratio", "Edge Ratio (stamp/ctrl)"),
        ("gold_pixel_ratio", "Gold Pixel Ratio"),
        ("laplacian_var", "Laplacian Variance"),
        ("laplacian_ratio", "Laplacian Ratio"),
        ("brightness_ratio", "Brightness Ratio"),
        ("gold_hue_fraction", "Gold Hue Fraction"),
        ("mean_saturation", "Mean Saturation"),
        ("high_freq_energy", "HF Energy"),
        ("high_freq_ratio", "HF Ratio"),
    ]

    # Create visualization image
    row_h = 60
    img_w = 800
    img_h = len(key_features) * row_h + 40
    viz = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    cv2.putText(viz, f"Feature Distributions: {dataset_name}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    for idx, (fname, display_name) in enumerate(key_features):
        y_base = 40 + idx * row_h
        s_vals = np.array([getattr(f, fname) for f in stamped])
        c_vals = np.array([getattr(f, fname) for f in clean])

        # Normalize to pixel range for visualization
        all_vals = np.concatenate([s_vals, c_vals])
        vmin, vmax = np.min(all_vals), np.max(all_vals)
        if vmax - vmin < 1e-10:
            continue

        def val_to_x(v):
            return int(200 + (v - vmin) / (vmax - vmin) * 550)

        # Label
        cv2.putText(viz, display_name, (5, y_base + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Plot stamped as red dots
        for v in s_vals:
            x = val_to_x(v)
            cv2.circle(viz, (x, y_base + 15), 3, (0, 0, 255), -1)

        # Plot clean as green dots
        for v in c_vals:
            x = val_to_x(v)
            cv2.circle(viz, (x, y_base + 35), 3, (0, 255, 0), -1)

        # Mean markers
        cv2.drawMarker(viz, (val_to_x(np.mean(s_vals)), y_base + 15),
                       (0, 0, 255), cv2.MARKER_DIAMOND, 8, 2)
        cv2.drawMarker(viz, (val_to_x(np.mean(c_vals)), y_base + 35),
                       (0, 255, 0), cv2.MARKER_DIAMOND, 8, 2)

        # Labels for axis
        cv2.putText(viz, f"{vmin:.3f}", (200, y_base + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        cv2.putText(viz, f"{vmax:.3f}", (720, y_base + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    # Legend
    cv2.putText(viz, "Red=Stamped  Green=Clean  Diamond=Mean",
                (200, img_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (180, 180, 180), 1)

    out_path = OUT_DIR / f"distributions_{dataset_name}.png"
    cv2.imwrite(str(out_path), viz)
    print(f"  Saved distributions: {out_path}")


def save_per_card_detail(features: list[StampFeatures],
                         images: list[np.ndarray],
                         dataset_name: str, max_cards: int = 14):
    """Save detailed per-card analysis showing stamp region + features."""
    card_h, card_w = 200, 300
    cols = min(7, len(features))
    rows = min(2, (min(max_cards, len(features)) + cols - 1) // cols)
    n = rows * cols

    img_w = cols * card_w
    img_h = rows * card_h + 30
    detail = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    cv2.putText(detail, f"Per-Card Detail: {dataset_name}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for i in range(min(n, len(features))):
        feat = features[i]
        img = images[i]
        row, col = i // cols, i % cols
        x_off = col * card_w
        y_off = 30 + row * card_h

        # Stamp crop
        stamp_crop = crop_stamp_region(img)
        stamp_crop = cv2.resize(stamp_crop, (card_w - 10, card_h - 50))
        detail[y_off:y_off + card_h - 50,
               x_off + 5:x_off + card_w - 5] = stamp_crop

        # Border color: red=stamped, green=clean
        color = (0, 0, 255) if feat.label else (0, 255, 0)
        cv2.rectangle(detail, (x_off + 2, y_off - 2),
                      (x_off + card_w - 2, y_off + card_h - 2), color, 2)

        # Text info
        ty = y_off + card_h - 45
        info_lines = [
            f"{feat.name[:20]}",
            f"E:{feat.edge_density:.3f} G:{feat.gold_pixel_ratio:.3f}",
            f"L:{feat.laplacian_var:.0f} B:{feat.brightness_ratio:.2f}",
        ]
        for j, line in enumerate(info_lines):
            cv2.putText(detail, line, (x_off + 5, ty + j * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)

    out_path = OUT_DIR / f"detail_{dataset_name}.png"
    cv2.imwrite(str(out_path), detail)
    print(f"  Saved per-card detail: {out_path}")


# --- Main ---

def run_analysis(dataset_name: str,
                 data: list[tuple[np.ndarray, bool, str]]) -> list[
                     StampFeatures]:
    """Run full analysis on a dataset."""
    print(f"\n{'#'*70}")
    print(f"  Dataset: {dataset_name} ({len(data)} images)")
    print(f"{'#'*70}")

    features = []
    images = []
    for img, label, name in data:
        feat = extract_features(img, label, name, dataset_name)
        if feat is not None:
            features.append(feat)
            images.append(img)

    n_stamped = sum(1 for f in features if f.label)
    n_clean = sum(1 for f in features if not f.label)
    print(f"  Extracted features: {len(features)} "
          f"({n_stamped} stamped, {n_clean} clean)")

    if not features:
        return []

    # Analysis
    separations = analyze_feature_separation(features, dataset_name)
    detector = build_threshold_detector(features, dataset_name)

    # Visualizations
    save_stamp_crops_comparison(features, images, dataset_name)
    save_feature_distributions(features, dataset_name)
    save_per_card_detail(features, images, dataset_name)

    # Print misclassified cards using best single feature
    if detector:
        bf = detector["best_feature"]
        bt = detector["best_threshold"]
        bd = detector["best_direction"]
        print(f"\n  Misclassified by {bf} {bd} {bt:.4f}:")
        for feat in features:
            val = getattr(feat, bf)
            if bd == ">":
                pred = val > bt
            else:
                pred = val < bt
            if pred != feat.label:
                print(f"    {feat.name}: {bf}={val:.4f} "
                      f"predicted={'stamped' if pred else 'clean'} "
                      f"actual={'stamped' if feat.label else 'clean'}")

    return features


def main():
    print("Stamp Pixel Analysis")
    print("=" * 70)

    all_features = {}

    # 1. Binder scans (primary target)
    print("\nLoading binder scans...")
    binder_data = load_binder_dataset()
    if binder_data:
        all_features["binder"] = run_analysis("binder", binder_data)

    # 2. Real reference photos
    print("\nLoading real reference photos...")
    real_data = load_real_photos_dataset()
    if real_data:
        all_features["real_photos"] = run_analysis("real_photos", real_data)

    # 3. Synthetic stamps
    print("\nLoading synthetic stamps...")
    synth_data = load_synthetic_dataset(max_per_class=50)
    if synth_data:
        all_features["synthetic"] = run_analysis("synthetic", synth_data)

    # Cross-dataset summary
    print(f"\n\n{'#'*70}")
    print(f"  CROSS-DATASET SUMMARY")
    print(f"{'#'*70}")

    for ds_name, feats in all_features.items():
        stamped = [f for f in feats if f.label]
        clean = [f for f in feats if not f.label]
        if not stamped or not clean:
            continue
        print(f"\n  {ds_name}:")
        for fname in ["edge_density", "gold_pixel_ratio", "laplacian_var",
                       "brightness_ratio", "edge_density_ratio",
                       "laplacian_ratio"]:
            s_vals = [getattr(f, fname) for f in stamped]
            c_vals = [getattr(f, fname) for f in clean]
            print(f"    {fname:<22} stamped={np.mean(s_vals):.4f}"
                  f"+{np.std(s_vals):.3f}  "
                  f"clean={np.mean(c_vals):.4f}+{np.std(c_vals):.3f}")

    # Save raw features as JSON for further analysis
    out_json = OUT_DIR / "features.json"
    json_data = {}
    for ds_name, feats in all_features.items():
        json_data[ds_name] = []
        for f in feats:
            d = {k: v for k, v in f.__dict__.items()
                 if not k.startswith("_")}
            json_data[ds_name].append(d)
    with open(out_json, "w") as fh:
        json.dump(json_data, fh, indent=2)
    print(f"\n  Saved raw features: {out_json}")

    print(f"\n  All visualizations saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
