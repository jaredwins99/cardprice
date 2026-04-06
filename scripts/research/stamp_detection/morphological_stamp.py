#!/usr/bin/env python3
"""Stamp detection via morphological operations, MSER, and stroke width analysis.

Theory: EX-era stamps are metallic text overlaid on card artwork at an angle
(~30-45 degrees). This text creates connected components (letters) that differ
from natural artwork textures. By using morphological operations tuned to the
stamp's angle and text characteristics, we can isolate stamp text.

Methods:
  1. Morphological closing with angled kernels to detect angled text blobs
  2. MSER (Maximally Stable Extremal Regions) for text-like region detection
  3. Stroke Width Transform concepts (edge gradient consistency)
  4. Connected component analysis with text-like filtering

Runs on the 17 binder ground truth cards from binder_ground_truth.jsonl.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path("/home/godli/cardprice")
INBOX = BASE / "data/inbox"
GT_PATH = BASE / "data/condition_training/stamps_real/binder_ground_truth.jsonl"


# ---------------------------------------------------------------------------
# Region cropping
# ---------------------------------------------------------------------------

def crop_stamp_region(img: np.ndarray) -> np.ndarray:
    """Crop the stamp region: x=[0.55, 0.92], y=[0.40, 0.68]."""
    h, w = img.shape[:2]
    return img[int(h * 0.40):int(h * 0.68), int(w * 0.55):int(w * 0.92)]


def crop_control_region(img: np.ndarray) -> np.ndarray:
    """Control region: same y band, left side (no stamp expected)."""
    h, w = img.shape[:2]
    return img[int(h * 0.40):int(h * 0.68), int(w * 0.08):int(w * 0.45)]


# ---------------------------------------------------------------------------
# Method 1: Morphological closing with angled kernels
# ---------------------------------------------------------------------------

def make_angled_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Create a line-shaped structuring element at given angle."""
    # Build a rotated line kernel
    k = np.zeros((length, length), dtype=np.uint8)
    center = length // 2
    rad = np.radians(angle_deg)
    for i in range(length):
        offset = i - center
        x = center + int(offset * np.cos(rad))
        y = center - int(offset * np.sin(rad))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1
    return k


def morphological_text_score(region: np.ndarray) -> dict:
    """Detect text-like structures using morphological operations.

    Pipeline:
      1. Grayscale + adaptive threshold (binary)
      2. Morphological closing with angled kernel (bridges stamp letters)
      3. Connected component analysis
      4. Filter components by aspect ratio, size, and angle
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    results = {}

    # Try multiple thresholding approaches
    # Adaptive threshold - handles varying artwork backgrounds
    binary_adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 5
    )
    # Inverted for bright-on-dark stamps
    binary_adapt_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )

    # Also try OTSU
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    binaries = {
        "adapt": binary_adapt,
        "adapt_inv": binary_adapt_inv,
        "otsu": binary_otsu,
        "otsu_inv": binary_otsu_inv,
    }

    best_score = 0
    best_method = ""

    for bname, binary in binaries.items():
        for angle in [30, 35, 40, 45]:
            for klen in [7, 11, 15]:
                kernel = make_angled_kernel(klen, angle)
                # Morphological closing: dilate then erode
                # Bridges nearby text components along the stamp angle
                closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                # Find connected components
                n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    closed, connectivity=8
                )

                # Filter components that look like text
                text_components = 0
                min_area = h * w * 0.001  # at least 0.1% of region
                max_area = h * w * 0.3    # no more than 30%

                for i in range(1, n_labels):  # skip background
                    area = stats[i, cv2.CC_STAT_AREA]
                    cw = stats[i, cv2.CC_STAT_WIDTH]
                    ch = stats[i, cv2.CC_STAT_HEIGHT]

                    if area < min_area or area > max_area:
                        continue

                    # Text-like: wider than tall (or rotated equivalent)
                    aspect = max(cw, ch) / (min(cw, ch) + 1e-5)
                    if aspect > 1.5 and aspect < 15:
                        # Elongated = text-like
                        text_components += 1

                score = text_components
                key = f"{bname}_a{angle}_k{klen}"
                results[key] = score

                if score > best_score:
                    best_score = score
                    best_method = key

    # Also compute a simpler metric: total text-like component count
    # across best angle/kernel combos
    total_text = sum(results.values())

    return {
        "best_morph_score": best_score,
        "best_morph_method": best_method,
        "total_morph_score": total_text,
        "morph_details": results,
    }


# ---------------------------------------------------------------------------
# Method 2: MSER (Maximally Stable Extremal Regions)
# ---------------------------------------------------------------------------

def mser_text_score(region: np.ndarray) -> dict:
    """Detect text regions using MSER.

    MSER finds regions that are stable across intensity thresholds --
    text letters are typically stable regions against their background.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Create MSER detector with text-tuned parameters
    mser = cv2.MSER_create(
        delta=5,
        min_area=30,
        max_area=int(h * w * 0.15),
        max_variation=0.25,
        min_diversity=0.2,
        max_evolution=200,
        area_threshold=1.01,
        min_margin=0.003,
        edge_blur_size=5,
    )

    # Detect on both original and inverted
    regions_dark, _ = mser.detectRegions(gray)
    regions_bright, _ = mser.detectRegions(255 - gray)

    def filter_text_regions(regions):
        """Filter MSER regions that look like text characters."""
        text_count = 0
        text_area = 0
        for pts in regions:
            if len(pts) < 10:
                continue
            # Fit bounding box
            x, y, rw, rh = cv2.boundingRect(pts)
            area = rw * rh
            aspect = max(rw, rh) / (min(rw, rh) + 1e-5)

            # Text character constraints:
            # - Not too thin (aspect < 8)
            # - Not too square-ish for large blobs (that's artwork)
            # - Reasonable size
            if area < 30:
                continue
            if area > h * w * 0.1:
                continue
            if aspect > 8:
                continue

            # Solidity check (text has moderate solidity)
            hull = cv2.convexHull(pts)
            hull_area = cv2.contourArea(hull)
            if hull_area < 1:
                continue
            solidity = len(pts) / hull_area
            if solidity < 0.2 or solidity > 0.95:
                continue

            text_count += 1
            text_area += area

        return text_count, text_area

    dark_count, dark_area = filter_text_regions(regions_dark)
    bright_count, bright_area = filter_text_regions(regions_bright)

    total_count = dark_count + bright_count
    area_ratio = (dark_area + bright_area) / (h * w + 1e-5)

    return {
        "mser_total_count": total_count,
        "mser_dark_count": dark_count,
        "mser_bright_count": bright_count,
        "mser_area_ratio": area_ratio,
    }


# ---------------------------------------------------------------------------
# Method 3: Stroke Width Transform concepts
# ---------------------------------------------------------------------------

def swt_text_score(region: np.ndarray) -> dict:
    """Approximate Stroke Width Transform for text detection.

    Full SWT is complex; we approximate by:
    1. Compute Canny edges
    2. Compute gradient direction at edges
    3. For text, gradients on opposite sides of a stroke should be ~180 degrees apart
    4. Consistent stroke widths indicate text vs random artwork edges
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Compute edges
    edges = cv2.Canny(gray, 50, 150)

    # Compute gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_dir = np.arctan2(gy, gx)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # For each edge pixel, walk along gradient direction until hitting
    # another edge pixel. Record the distance (stroke width).
    edge_points = np.argwhere(edges > 0)  # (row, col) pairs
    stroke_widths = []

    max_walk = 30  # max stroke width to search
    sample_step = max(1, len(edge_points) // 500)  # sample for speed

    for idx in range(0, len(edge_points), sample_step):
        r, c = edge_points[idx]
        theta = grad_dir[r, c]
        dx = np.cos(theta)
        dy = np.sin(theta)

        # Walk along gradient direction
        for step in range(2, max_walk):
            nr = int(r + step * dy)
            nc = int(c + step * dx)
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                break
            if edges[nr, nc] > 0:
                # Check opposite gradient direction (should be ~180 deg apart)
                other_theta = grad_dir[nr, nc]
                angle_diff = abs(theta - other_theta)
                # Normalize to [0, pi]
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                if abs(angle_diff - np.pi) < np.pi / 4:  # within 45 deg of opposite
                    stroke_widths.append(step)
                break

    if not stroke_widths:
        return {
            "swt_mean_width": 0.0,
            "swt_std_width": 0.0,
            "swt_count": 0,
            "swt_consistency": 0.0,
        }

    sw = np.array(stroke_widths, dtype=np.float32)
    mean_sw = float(np.mean(sw))
    std_sw = float(np.std(sw))
    # Consistency: text has very consistent stroke widths (low CoV)
    consistency = mean_sw / (std_sw + 1e-5) if std_sw > 0 else mean_sw

    return {
        "swt_mean_width": mean_sw,
        "swt_std_width": std_sw,
        "swt_count": len(stroke_widths),
        "swt_consistency": consistency,
    }


# ---------------------------------------------------------------------------
# Method 4: Edge density + gradient orientation histogram
# ---------------------------------------------------------------------------

def edge_orientation_score(region: np.ndarray) -> dict:
    """Analyze edge orientations in stamp region.

    Stamp text at 30-45 degrees should create a peak in the gradient
    orientation histogram at that angle range.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Compute gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.degrees(np.arctan2(gy, gx)) % 180  # 0-180

    # Only consider strong edges
    threshold = np.percentile(mag, 75)
    strong = mag > threshold

    if np.sum(strong) == 0:
        return {
            "edge_aniso": 0.0,
            "stamp_angle_ratio": 0.0,
            "edge_density": 0.0,
        }

    # Histogram of edge orientations (weighted by magnitude)
    hist, bin_edges = np.histogram(
        angle[strong], bins=18, range=(0, 180), weights=mag[strong]
    )
    hist = hist / (hist.sum() + 1e-10)

    # Stamp text angles: 30-45 degrees and 120-135 degrees (perpendicular edges)
    # Bins are 10 degrees each: bin 3 = 30-40, bin 4 = 40-50
    # Also 120-130 (bin 12), 130-140 (bin 13)
    stamp_bins = hist[3] + hist[4] + hist[12] + hist[13]

    # Anisotropy: how peaked is the histogram? Text creates peaks.
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    max_entropy = np.log(18)  # uniform distribution
    anisotropy = 1.0 - entropy / max_entropy

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    return {
        "edge_aniso": float(anisotropy),
        "stamp_angle_ratio": float(stamp_bins),
        "edge_density": edge_density,
    }


# ---------------------------------------------------------------------------
# Method 5: Local Binary Patterns for texture analysis
# ---------------------------------------------------------------------------

def texture_regularity_score(region: np.ndarray) -> dict:
    """Detect regular texture patterns (stamp text) vs irregular (artwork).

    Uses local variance analysis: stamp text creates patches of high local
    contrast against the artwork, with a regular spatial pattern.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # Local variance using box filter
    local_mean = cv2.blur(gray, (11, 11))
    local_sq_mean = cv2.blur(gray**2, (11, 11))
    local_var = local_sq_mean - local_mean**2
    local_var = np.maximum(local_var, 0)

    # High local variance indicates text edges
    high_var_mask = local_var > np.percentile(local_var, 80)
    high_var_fraction = float(np.mean(high_var_mask))

    # Spatial regularity of high-variance regions
    # Text creates bands; artwork creates random blobs
    # Check column-wise and row-wise variance of the high_var_mask
    col_sum = np.mean(high_var_mask, axis=0)
    row_sum = np.mean(high_var_mask, axis=1)
    col_var = float(np.var(col_sum))
    row_var = float(np.var(row_sum))

    # For angled text, check along diagonal
    # Rotate the mask by -35 degrees and check horizontal regularity
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, -35, 1.0)
    rotated_mask = cv2.warpAffine(
        high_var_mask.astype(np.float32), rot_mat, (w, h)
    )
    rot_row_sum = np.mean(rotated_mask, axis=0)
    rot_row_var = float(np.var(rot_row_sum))

    return {
        "high_var_fraction": high_var_fraction,
        "spatial_col_var": col_var,
        "spatial_row_var": row_var,
        "spatial_diag_var": rot_row_var,
        "mean_local_var": float(np.mean(local_var)),
    }


# ---------------------------------------------------------------------------
# Combined analysis
# ---------------------------------------------------------------------------

def analyze_card(img: np.ndarray) -> dict:
    """Run all methods on stamp region and control region."""
    stamp = crop_stamp_region(img)
    control = crop_control_region(img)

    if stamp.size == 0 or control.size == 0:
        return {}

    # Run all methods on stamp region
    stamp_morph = morphological_text_score(stamp)
    stamp_mser = mser_text_score(stamp)
    stamp_swt = swt_text_score(stamp)
    stamp_edge = edge_orientation_score(stamp)
    stamp_tex = texture_regularity_score(stamp)

    # Run all methods on control region (for comparison)
    ctrl_morph = morphological_text_score(control)
    ctrl_mser = mser_text_score(control)
    ctrl_swt = swt_text_score(control)
    ctrl_edge = edge_orientation_score(control)
    ctrl_tex = texture_regularity_score(control)

    return {
        # Morphological
        "stamp_morph_best": stamp_morph["best_morph_score"],
        "ctrl_morph_best": ctrl_morph["best_morph_score"],
        "morph_ratio": stamp_morph["best_morph_score"] / (ctrl_morph["best_morph_score"] + 1e-5),
        "stamp_morph_total": stamp_morph["total_morph_score"],
        "ctrl_morph_total": ctrl_morph["total_morph_score"],
        "morph_total_ratio": stamp_morph["total_morph_score"] / (ctrl_morph["total_morph_score"] + 1e-5),

        # MSER
        "stamp_mser_count": stamp_mser["mser_total_count"],
        "ctrl_mser_count": ctrl_mser["mser_total_count"],
        "mser_count_ratio": stamp_mser["mser_total_count"] / (ctrl_mser["mser_total_count"] + 1e-5),
        "stamp_mser_area": stamp_mser["mser_area_ratio"],
        "ctrl_mser_area": ctrl_mser["mser_area_ratio"],

        # SWT
        "stamp_swt_count": stamp_swt["swt_count"],
        "ctrl_swt_count": ctrl_swt["swt_count"],
        "stamp_swt_consistency": stamp_swt["swt_consistency"],
        "ctrl_swt_consistency": ctrl_swt["swt_consistency"],
        "swt_count_ratio": stamp_swt["swt_count"] / (ctrl_swt["swt_count"] + 1e-5),

        # Edge orientation
        "stamp_edge_aniso": stamp_edge["edge_aniso"],
        "ctrl_edge_aniso": ctrl_edge["edge_aniso"],
        "stamp_angle_ratio": stamp_edge["stamp_angle_ratio"],
        "ctrl_angle_ratio": ctrl_edge["stamp_angle_ratio"],
        "stamp_edge_density": stamp_edge["edge_density"],
        "ctrl_edge_density": ctrl_edge["edge_density"],
        "edge_density_ratio": stamp_edge["edge_density"] / (ctrl_edge["edge_density"] + 1e-5),

        # Texture regularity
        "stamp_high_var": stamp_tex["high_var_fraction"],
        "ctrl_high_var": ctrl_tex["high_var_fraction"],
        "stamp_diag_var": stamp_tex["spatial_diag_var"],
        "ctrl_diag_var": ctrl_tex["spatial_diag_var"],
        "diag_var_ratio": stamp_tex["spatial_diag_var"] / (ctrl_tex["spatial_diag_var"] + 1e-5),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth() -> list[dict]:
    """Load the 17 binder ground truth cards.

    Handles duplicate image paths by taking the last entry.
    """
    entries = []
    seen = {}
    with open(GT_PATH) as f:
        for line in f:
            entry = json.loads(line.strip())
            img_path = INBOX / entry["image"]
            key = str(img_path)
            seen[key] = entry  # last wins for duplicates

    for entry in seen.values():
        img_path = INBOX / entry["image"]
        img = cv2.imread(str(img_path))
        if img is not None:
            entries.append({
                "img": img,
                "stamped": entry["stamped"],
                "name": entry.get("card_name", Path(entry["image"]).stem),
                "path": str(img_path),
            })
        else:
            print(f"  WARN: Could not load {img_path}")

    return entries


# ---------------------------------------------------------------------------
# Threshold sweep + accuracy reporting
# ---------------------------------------------------------------------------

def find_best_threshold(values: list[float], labels: list[bool],
                        feature_name: str) -> tuple[float, float, str]:
    """Find optimal threshold for a feature. Returns (accuracy, threshold, direction)."""
    best_acc = 0.0
    best_thresh = 0.0
    best_dir = ">"

    unique_vals = sorted(set(values))
    # Add midpoints
    thresholds = []
    for i in range(len(unique_vals)):
        thresholds.append(unique_vals[i])
        if i + 1 < len(unique_vals):
            thresholds.append((unique_vals[i] + unique_vals[i + 1]) / 2)

    for t in thresholds:
        # stamped > t
        correct_gt = sum(
            1 for v, l in zip(values, labels)
            if (v > t) == l
        )
        acc_gt = correct_gt / len(values)

        # stamped < t
        correct_lt = sum(
            1 for v, l in zip(values, labels)
            if (v < t) == l
        )
        acc_lt = correct_lt / len(values)

        if acc_gt > best_acc:
            best_acc = acc_gt
            best_thresh = t
            best_dir = ">"
        if acc_lt > best_acc:
            best_acc = acc_lt
            best_thresh = t
            best_dir = "<"

    return best_acc, best_thresh, best_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MORPHOLOGICAL STAMP DETECTION")
    print("=" * 70)

    # Load data
    print("\nLoading binder ground truth cards...")
    cards = load_ground_truth()
    n_stamped = sum(1 for c in cards if c["stamped"])
    n_clean = sum(1 for c in cards if not c["stamped"])
    print(f"Loaded {len(cards)} cards ({n_stamped} stamped, {n_clean} clean)")

    # Analyze each card
    print("\nAnalyzing cards...")
    all_features = {}
    labels = []
    names = []

    for card in cards:
        name = card["name"]
        label = card["stamped"]
        print(f"  {'[S]' if label else '[C]'} {name}...", end="", flush=True)
        feats = analyze_card(card["img"])
        if not feats:
            print(" SKIP (empty region)")
            continue
        print(f" morph={feats['stamp_morph_best']}, "
              f"mser={feats['stamp_mser_count']}, "
              f"swt={feats['stamp_swt_count']}, "
              f"edge_d={feats['stamp_edge_density']:.3f}")

        for key, val in feats.items():
            if key not in all_features:
                all_features[key] = []
            all_features[key].append(val)
        labels.append(label)
        names.append(name)

    # Report per-card details
    print("\n" + "=" * 70)
    print("PER-CARD FEATURE TABLE")
    print("=" * 70)

    key_features = [
        "stamp_morph_best", "morph_ratio", "stamp_mser_count",
        "mser_count_ratio", "stamp_swt_count", "swt_count_ratio",
        "stamp_edge_density", "edge_density_ratio",
        "stamp_angle_ratio", "stamp_high_var", "diag_var_ratio",
    ]

    header = f"{'Card':<20s} {'Label':>5s}"
    for kf in key_features:
        short = kf.replace("stamp_", "s_").replace("ctrl_", "c_")[:12]
        header += f" {short:>12s}"
    print(header)
    print("-" * len(header))

    for i, (name, label) in enumerate(zip(names, labels)):
        row = f"{name[:20]:<20s} {'S' if label else 'C':>5s}"
        for kf in key_features:
            val = all_features[kf][i]
            row += f" {val:>12.3f}"
        print(row)

    # Threshold sweep for each feature
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP - SINGLE FEATURES")
    print("=" * 70)
    print(f"{'Feature':<28s} {'Best Acc':>8s} {'Threshold':>12s} {'Dir':>4s}  "
          f"{'Stamped Mean':>12s} {'Clean Mean':>12s}")
    print("-" * 82)

    feature_accuracies = []

    for fname in sorted(all_features.keys()):
        vals = all_features[fname]
        acc, thresh, direction = find_best_threshold(vals, labels, fname)

        stamped_vals = [v for v, l in zip(vals, labels) if l]
        clean_vals = [v for v, l in zip(vals, labels) if not l]
        s_mean = np.mean(stamped_vals) if stamped_vals else 0
        c_mean = np.mean(clean_vals) if clean_vals else 0

        feature_accuracies.append((acc, fname, thresh, direction, s_mean, c_mean))
        print(f"{fname:<28s} {acc:>7.1%} {thresh:>12.4f} {direction:>4s}  "
              f"{s_mean:>12.4f} {c_mean:>12.4f}")

    # Sort by accuracy
    feature_accuracies.sort(key=lambda x: -x[0])

    print("\n" + "=" * 70)
    print("TOP 10 FEATURES BY ACCURACY")
    print("=" * 70)
    for i, (acc, fname, thresh, direction, s_mean, c_mean) in enumerate(feature_accuracies[:10]):
        print(f"  {i + 1:2d}. {fname:<28s} {acc:>7.1%}  "
              f"threshold={thresh:.4f} ({direction})  "
              f"stamped={s_mean:.4f} clean={c_mean:.4f}")

    # Show misclassifications for top feature
    if feature_accuracies:
        best_acc, best_fname, best_thresh, best_dir, _, _ = feature_accuracies[0]
        vals = all_features[best_fname]

        print(f"\nBest feature: {best_fname} {best_dir} {best_thresh:.4f} "
              f"-> {best_acc:.1%}")

        print("\nMisclassifications:")
        any_wrong = False
        for i, (name, label) in enumerate(zip(names, labels)):
            val = vals[i]
            if best_dir == ">":
                pred = val > best_thresh
            else:
                pred = val < best_thresh
            if pred != label:
                any_wrong = True
                print(f"  {name}: {best_fname}={val:.4f}, "
                      f"predicted={'stamped' if pred else 'clean'}, "
                      f"actual={'stamped' if label else 'clean'}")
        if not any_wrong:
            print("  None!")

    # Method-level summary
    print("\n" + "=" * 70)
    print("METHOD SUMMARY")
    print("=" * 70)

    method_groups = {
        "Morphological": [f for f in all_features if "morph" in f],
        "MSER": [f for f in all_features if "mser" in f],
        "SWT": [f for f in all_features if "swt" in f],
        "Edge Orientation": [f for f in all_features if "edge" in f or "angle" in f],
        "Texture": [f for f in all_features if "var" in f and "swt" not in f],
    }

    for method_name, method_features in method_groups.items():
        best_method_acc = 0
        best_method_feat = ""
        for fname in method_features:
            vals = all_features[fname]
            acc, _, _ = find_best_threshold(vals, labels, fname)
            if acc > best_method_acc:
                best_method_acc = acc
                best_method_feat = fname
        print(f"  {method_name:<20s}: best={best_method_feat:<28s} acc={best_method_acc:.1%}")

    # Try combining top features
    print("\n" + "=" * 70)
    print("COMBINED FEATURES (OR / AND)")
    print("=" * 70)

    # Get top 5 features
    top5 = feature_accuracies[:5]
    # Try all pairs with AND/OR
    best_combo_acc = 0
    best_combo_desc = ""

    for i, (_, f1, t1, d1, _, _) in enumerate(top5):
        for j, (_, f2, t2, d2, _, _) in enumerate(top5):
            if i >= j:
                continue
            v1 = all_features[f1]
            v2 = all_features[f2]

            for combiner in ["OR", "AND"]:
                correct = 0
                for k in range(len(labels)):
                    if d1 == ">":
                        p1 = v1[k] > t1
                    else:
                        p1 = v1[k] < t1
                    if d2 == ">":
                        p2 = v2[k] > t2
                    else:
                        p2 = v2[k] < t2

                    if combiner == "OR":
                        pred = p1 or p2
                    else:
                        pred = p1 and p2

                    if pred == labels[k]:
                        correct += 1

                acc = correct / len(labels)
                desc = f"{f1} {d1} {t1:.4f} {combiner} {f2} {d2} {t2:.4f}"
                if acc > best_combo_acc:
                    best_combo_acc = acc
                    best_combo_desc = desc
                if acc >= 0.9:
                    print(f"  {acc:.1%}: {desc}")

    print(f"\n  Best combo: {best_combo_acc:.1%}: {best_combo_desc}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
