#!/usr/bin/env python3
"""Diagnostic: detect and auto-trim binder sleeve edges from card segments.

For each segment, analyzes the four edges to detect sleeve material:
  1. Dark strip analysis (sleeve = low brightness, low saturation)
  2. Sobel edge detection (card border = strong horizontal/vertical gradient)
  3. Row/column brightness profile gradient (find the transition point)

Then implements auto_trim_sleeve() that crops to the card content.

RESEARCH ONLY — not integrated into production.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

SEGMENT_DIRS = [
    "data/inbox/page_20260228_174819_cards_v4",
    "data/inbox/page_20260228_195512_cards",
    "data/inbox/page_20260228_202134_cards",
]

# How far into the image to search for a card boundary (fraction of dimension)
SEARCH_DEPTH = 0.15

# Minimum Sobel edge strength to consider a row/col as a "card edge"
SOBEL_THRESHOLD = 25.0

# Minimum brightness gradient (per-row/col mean) to indicate a transition
GRAD_THRESHOLD = 2.5

# For dark-strip detection: mean brightness below this = likely sleeve
DARK_THRESH = 60

# Output directory for trimmed results
OUTPUT_DIR = "data/inbox/sleeve_trim_diagnostic"


# ── Edge analysis ────────────────────────────────────────────────────────────

def analyze_edge_strip(gray, hsv, side, depth_frac=0.05):
    """Analyze a 5% edge strip for sleeve characteristics.

    Returns dict with brightness/saturation stats for the strip.
    """
    h, w = gray.shape
    if side == "top":
        d = int(h * depth_frac)
        strip_g = gray[:d, :]
        strip_h = hsv[:d, :, :]
    elif side == "bottom":
        d = int(h * depth_frac)
        strip_g = gray[-d:, :]
        strip_h = hsv[-d:, :, :]
    elif side == "left":
        d = int(w * depth_frac)
        strip_g = gray[:, :d]
        strip_h = hsv[:, :d, :]
    elif side == "right":
        d = int(w * depth_frac)
        strip_g = gray[:, -d:]
        strip_h = hsv[:, -d:, :]
    else:
        raise ValueError(f"Unknown side: {side}")

    return {
        "side": side,
        "depth_px": d,
        "bright_mean": float(np.mean(strip_g)),
        "bright_std": float(np.std(strip_g)),
        "sat_mean": float(np.mean(strip_h[:, :, 1])),
        "val_mean": float(np.mean(strip_h[:, :, 2])),
        "is_dark": float(np.mean(strip_g)) < DARK_THRESH,
    }


def _find_first_sobel_peak(strength, threshold):
    """Find the FIRST strong Sobel peak from the outer edge inward.

    A sleeve-to-card boundary is a single sharp transition: the Sobel
    signal spikes for 3-8 rows/cols then drops. Internal card features
    (text boxes, art borders) also produce edges but tend to keep the
    Sobel elevated for long stretches.

    Strategy:
      1. Scan from outer edge inward for first index above threshold
      2. Refine to the local max within a small window (max 15 positions)
      3. Verify the peak is a real transition: the Sobel must drop to
         below threshold within 20 positions after the peak

    Returns (peak_index, peak_value) or (None, max_value) if no peak found.
    """
    MAX_PEAK_WINDOW = 15   # max distance from trigger to peak
    DROPOFF_WINDOW = 20    # must see a drop within this many positions after peak

    # Skip index 0 (always 0 due to border), start at 1
    for i in range(1, len(strength)):
        if strength[i] >= threshold:
            # Found trigger. Find local max within a limited window
            peak_idx = i
            peak_val = strength[i]
            search_end = min(i + MAX_PEAK_WINDOW, len(strength))
            for j in range(i + 1, search_end):
                if strength[j] > peak_val:
                    peak_idx = j
                    peak_val = strength[j]
                elif strength[j] < threshold * 0.7:
                    break

            # Verify this is a real transition: Sobel should drop
            # after the peak (not stay elevated = card texture)
            drop_end = min(peak_idx + DROPOFF_WINDOW, len(strength))
            has_dropoff = False
            for j in range(peak_idx + 1, drop_end):
                if strength[j] < threshold * 0.6:
                    has_dropoff = True
                    break

            if has_dropoff:
                return peak_idx, peak_val
            else:
                # This looks like sustained texture, not a single edge.
                # Skip past this region and keep looking.
                # But also: if this is very early (within 5px of edge),
                # it's probably the image border itself, not sleeve.
                continue
    # No valid peak found
    return None, float(np.max(strength)) if len(strength) > 0 else 0.0


def find_card_boundary(gray, side):
    """Find where the actual card starts on a given side.

    Uses two complementary methods:
      1. Sobel edge detection — find the FIRST strong edge from outer edge
         inward (not the global max, which would pick internal card features)
      2. Brightness profile gradient — find the first row/col where the
         brightness changes sharply (transition from sleeve to card border)

    Returns the estimated number of sleeve pixels on this side.
    """
    h, w = gray.shape
    search_h = int(h * SEARCH_DEPTH)
    search_w = int(w * SEARCH_DEPTH)

    if side in ("top", "bottom"):
        # Horizontal Sobel for top/bottom edges
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        if side == "top":
            region = sobel[:search_h, :]
            # Mean absolute edge strength per row
            row_strength = np.array([np.mean(np.abs(region[r, :])) for r in range(search_h)])
            # Also compute brightness profile gradient
            row_means = np.array([np.mean(gray[r, :]) for r in range(search_h)])
        else:
            region = sobel[-search_h:, :]
            # Flip so index 0 = outermost row
            row_strength = np.array([np.mean(np.abs(region[search_h - 1 - r, :])) for r in range(search_h)])
            row_means = np.array([np.mean(gray[h - 1 - r, :]) for r in range(search_h)])

        profile_grad = np.abs(np.diff(row_means))

        # Method 1: first Sobel peak from outer edge inward
        sobel_pos, sobel_val = _find_first_sobel_peak(row_strength, SOBEL_THRESHOLD)

        # Method 2: first strong gradient in brightness profile
        grad_candidates = np.where(profile_grad > GRAD_THRESHOLD)[0]
        grad_pos = int(grad_candidates[0]) if len(grad_candidates) > 0 else 0

        # Use whichever method found a signal; prefer Sobel if found
        if sobel_pos is not None:
            boundary = sobel_pos
            method = "sobel"
        elif len(grad_candidates) > 0:
            boundary = grad_pos
            method = "gradient"
        else:
            boundary = 0
            method = "none"

        return {
            "side": side,
            "sleeve_px": boundary,
            "method": method,
            "sobel_peak_row": sobel_pos if sobel_pos is not None else -1,
            "sobel_peak_val": sobel_val,
            "grad_first_row": grad_pos,
            "profile_means": row_means[:30].tolist(),
            "profile_grad": profile_grad[:30].tolist(),
            "sobel_strength": row_strength[:30].tolist(),
        }

    else:  # left or right
        # Vertical Sobel for left/right edges
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

        if side == "left":
            region = sobel[:, :search_w]
            col_strength = np.array([np.mean(np.abs(region[:, c])) for c in range(search_w)])
            col_means = np.array([np.mean(gray[:, c]) for c in range(search_w)])
        else:
            region = sobel[:, -search_w:]
            col_strength = np.array([np.mean(np.abs(region[:, search_w - 1 - c])) for c in range(search_w)])
            col_means = np.array([np.mean(gray[:, w - 1 - c]) for c in range(search_w)])

        profile_grad = np.abs(np.diff(col_means))

        sobel_pos, sobel_val = _find_first_sobel_peak(col_strength, SOBEL_THRESHOLD)

        grad_candidates = np.where(profile_grad > GRAD_THRESHOLD)[0]
        grad_pos = int(grad_candidates[0]) if len(grad_candidates) > 0 else 0

        if sobel_pos is not None:
            boundary = sobel_pos
            method = "sobel"
        elif len(grad_candidates) > 0:
            boundary = grad_pos
            method = "gradient"
        else:
            boundary = 0
            method = "none"

        return {
            "side": side,
            "sleeve_px": boundary,
            "method": method,
            "sobel_peak_col": sobel_pos if sobel_pos is not None else -1,
            "sobel_peak_val": sobel_val,
            "grad_first_col": grad_pos,
            "profile_means": col_means[:30].tolist(),
            "profile_grad": profile_grad[:30].tolist(),
            "sobel_strength": col_strength[:30].tolist(),
        }


# ── Auto-trim function ──────────────────────────────────────────────────────

def auto_trim_sleeve(img):
    """Detect and remove binder sleeve edges from a card segment.

    For each edge (top/bottom/left/right), finds the card boundary by
    looking for brightness transitions and Sobel edge peaks. Crops to
    the card content only.

    Parameters
    ----------
    img : np.ndarray
        BGR card segment image.

    Returns
    -------
    tuple of (np.ndarray, dict)
        Cropped image and a dict with trim info per side.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    trim = {}
    for side in ("top", "bottom", "left", "right"):
        result = find_card_boundary(gray, side)
        trim[side] = result["sleeve_px"]

    # Apply a small safety margin: don't trim more than 12% on any side
    max_h = int(h * 0.12)
    max_w = int(w * 0.12)
    trim["top"] = min(trim["top"], max_h)
    trim["bottom"] = min(trim["bottom"], max_h)
    trim["left"] = min(trim["left"], max_w)
    trim["right"] = min(trim["right"], max_w)

    # Crop
    y1 = trim["top"]
    y2 = h - trim["bottom"]
    x1 = trim["left"]
    x2 = w - trim["right"]

    cropped = img[y1:y2, x1:x2]
    return cropped, trim


# ── Diagnostic runner ────────────────────────────────────────────────────────

def diagnose_segment(img_path):
    """Run full edge analysis on a single segment."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ERROR: could not read {img_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = gray.shape

    results = {"path": img_path, "shape": (h, w)}

    # Strip analysis (5% edges)
    results["strips"] = {}
    for side in ("top", "bottom", "left", "right"):
        results["strips"][side] = analyze_edge_strip(gray, hsv, side)

    # Boundary detection
    results["boundaries"] = {}
    for side in ("top", "bottom", "left", "right"):
        results["boundaries"][side] = find_card_boundary(gray, side)

    # Auto-trim
    cropped, trim_info = auto_trim_sleeve(img)
    results["trim"] = trim_info
    results["original_size"] = (h, w)
    results["trimmed_size"] = cropped.shape[:2]

    # Compute what the fixed 5% crop would have been
    fixed_crop = {
        "top": int(h * 0.05),
        "bottom": int(h * 0.05),
        "left": int(w * 0.05),
        "right": int(w * 0.05),
    }
    results["fixed_5pct"] = fixed_crop

    return results


def print_results(results):
    """Print diagnostic results for one segment."""
    name = os.path.basename(results["path"])
    h, w = results["shape"]

    print(f"\n{'=' * 70}")
    print(f"  {results['path']}  ({w}x{h})")
    print(f"{'=' * 70}")

    # Strip analysis
    print(f"\n  5% Strip Analysis:")
    print(f"  {'Side':<8} {'Bright':>7} {'Std':>5} {'Sat':>5} {'Dark?':>6}")
    print(f"  {'-'*35}")
    for side in ("top", "bottom", "left", "right"):
        s = results["strips"][side]
        dark_flag = "YES" if s["is_dark"] else ""
        print(f"  {side:<8} {s['bright_mean']:7.1f} {s['bright_std']:5.1f} {s['sat_mean']:5.1f} {dark_flag:>6}")

    # Boundary detection
    print(f"\n  Card Boundary Detection:")
    print(f"  {'Side':<8} {'Sleeve px':>10} {'Method':>10} {'Sobel peak':>12} {'Grad pos':>10}")
    print(f"  {'-'*55}")
    for side in ("top", "bottom", "left", "right"):
        b = results["boundaries"][side]
        spk = b.get("sobel_peak_row", b.get("sobel_peak_col", "?"))
        spv = b["sobel_peak_val"]
        gp = b.get("grad_first_row", b.get("grad_first_col", "?"))
        print(f"  {side:<8} {b['sleeve_px']:10d} {b['method']:>10} {spv:12.1f} {gp:>10}")

    # Trim comparison
    trim = results["trim"]
    fixed = results["fixed_5pct"]
    print(f"\n  Trim Comparison (pixels from edge):")
    print(f"  {'Side':<8} {'Auto':>6} {'Fixed 5%':>10} {'Delta':>7}")
    print(f"  {'-'*35}")
    total_auto = 0
    total_fixed = 0
    for side in ("top", "bottom", "left", "right"):
        a = trim[side]
        f = fixed[side]
        delta = a - f
        total_auto += a
        total_fixed += f
        better = "<< tighter" if a < f else (">> more" if a > f else "")
        print(f"  {side:<8} {a:6d} {f:10d} {delta:+7d}  {better}")

    th, tw = results["trimmed_size"]
    orig_area = h * w
    trim_area = th * tw
    fixed_h = h - 2 * fixed["top"]
    fixed_w = w - 2 * fixed["left"]
    fixed_area = fixed_h * fixed_w
    print(f"\n  Area: original={orig_area}  auto-trim={trim_area} ({100*trim_area/orig_area:.1f}%)  fixed-5%={fixed_area} ({100*fixed_area/orig_area:.1f}%)")


def save_comparison(img_path, output_dir):
    """Save original, auto-trimmed, and fixed-5%-trimmed side by side."""
    img = cv2.imread(img_path)
    if img is None:
        return

    h, w = img.shape[:2]

    # Auto-trim
    auto_cropped, trim = auto_trim_sleeve(img)

    # Fixed 5% trim
    t5 = int(h * 0.05)
    l5 = int(w * 0.05)
    fixed_cropped = img[t5:h-t5, l5:w-l5]

    # Resize all to same height for side-by-side
    target_h = 400
    def resize(im):
        scale = target_h / im.shape[0]
        new_w = int(im.shape[1] * scale)
        return cv2.resize(im, (new_w, target_h))

    orig_r = resize(img)
    auto_r = resize(auto_cropped)
    fixed_r = resize(fixed_cropped)

    # Draw trim lines on original
    orig_lined = img.copy()
    # Auto-trim lines (green)
    cv2.line(orig_lined, (0, trim["top"]), (w, trim["top"]), (0, 255, 0), 2)
    cv2.line(orig_lined, (0, h - trim["bottom"]), (w, h - trim["bottom"]), (0, 255, 0), 2)
    cv2.line(orig_lined, (trim["left"], 0), (trim["left"], h), (0, 255, 0), 2)
    cv2.line(orig_lined, (w - trim["right"], 0), (w - trim["right"], h), (0, 255, 0), 2)
    # Fixed 5% lines (red)
    cv2.line(orig_lined, (0, t5), (w, t5), (0, 0, 255), 1)
    cv2.line(orig_lined, (0, h - t5), (w, h - t5), (0, 0, 255), 1)
    cv2.line(orig_lined, (l5, 0), (l5, h), (0, 0, 255), 1)
    cv2.line(orig_lined, (w - l5, 0), (w - l5, h), (0, 0, 255), 1)
    orig_lined_r = resize(orig_lined)

    # Add labels
    cv2.putText(orig_lined_r, "green=auto red=5%", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(auto_r, "auto-trim", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(fixed_r, "fixed 5%", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Pad to same width
    max_w = max(orig_lined_r.shape[1], auto_r.shape[1], fixed_r.shape[1])
    def pad_w(im, target_w):
        if im.shape[1] < target_w:
            pad = np.zeros((im.shape[0], target_w - im.shape[1], 3), dtype=np.uint8)
            return np.hstack([im, pad])
        return im

    combo = np.hstack([
        pad_w(orig_lined_r, max_w),
        np.ones((target_h, 3, 3), dtype=np.uint8) * 128,  # separator
        pad_w(auto_r, max_w),
        np.ones((target_h, 3, 3), dtype=np.uint8) * 128,
        pad_w(fixed_r, max_w),
    ])

    basename = os.path.basename(os.path.dirname(img_path)) + "_" + os.path.basename(img_path)
    out_path = os.path.join(output_dir, basename)
    cv2.imwrite(out_path, combo)
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    improvements = []

    for seg_dir in SEGMENT_DIRS:
        page_name = os.path.basename(seg_dir)
        print(f"\n{'#' * 70}")
        print(f"  PAGE: {page_name}")
        print(f"{'#' * 70}")

        cards = sorted([f for f in os.listdir(seg_dir) if f.endswith(".png")])
        for card_file in cards:
            img_path = os.path.join(seg_dir, card_file)
            results = diagnose_segment(img_path)
            if results is None:
                continue

            all_results.append(results)
            print_results(results)

            # Save comparison image
            out_path = save_comparison(img_path, OUTPUT_DIR)
            if out_path:
                print(f"  Saved comparison: {out_path}")

            # Track improvements
            trim = results["trim"]
            fixed = results["fixed_5pct"]
            h, w = results["shape"]
            auto_total = sum(trim.values())
            fixed_total = sum(fixed.values())
            diff = auto_total - fixed_total
            improvements.append({
                "path": img_path,
                "auto_px": auto_total,
                "fixed_px": fixed_total,
                "diff": diff,
                "trim": trim,
                "fixed": fixed,
            })

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY: Auto-trim vs Fixed 5% crop")
    print(f"{'=' * 70}\n")

    # Cards where auto-trim crops LESS (preserves more card content)
    tighter = [x for x in improvements if x["diff"] < -5]
    # Cards where auto-trim crops MORE (removes more sleeve)
    more_trim = [x for x in improvements if x["diff"] > 5]
    # Cards where they're similar
    similar = [x for x in improvements if abs(x["diff"]) <= 5]

    print(f"  Total segments analyzed: {len(improvements)}")
    print(f"  Auto-trim crops LESS than 5% (preserves card): {len(tighter)}")
    print(f"  Auto-trim crops MORE than 5% (removes sleeve): {len(more_trim)}")
    print(f"  Similar (<5px total diff):                      {len(similar)}")

    if tighter:
        print(f"\n  Cards where auto-trim preserves more card content:")
        for x in sorted(tighter, key=lambda x: x["diff"]):
            name = os.path.basename(os.path.dirname(x["path"])) + "/" + os.path.basename(x["path"])
            print(f"    {name}: auto={x['auto_px']}px  fixed={x['fixed_px']}px  (saves {-x['diff']}px)")
            t = x["trim"]
            f = x["fixed"]
            for side in ("top", "bottom", "left", "right"):
                if abs(t[side] - f[side]) > 2:
                    print(f"      {side}: auto={t[side]}px vs fixed={f[side]}px")

    if more_trim:
        print(f"\n  Cards where auto-trim removes more sleeve:")
        for x in sorted(more_trim, key=lambda x: -x["diff"]):
            name = os.path.basename(os.path.dirname(x["path"])) + "/" + os.path.basename(x["path"])
            print(f"    {name}: auto={x['auto_px']}px  fixed={x['fixed_px']}px  (trims {x['diff']}px more)")
            t = x["trim"]
            f = x["fixed"]
            for side in ("top", "bottom", "left", "right"):
                if abs(t[side] - f[side]) > 2:
                    print(f"      {side}: auto={t[side]}px vs fixed={f[side]}px")

    # Per-side statistics
    print(f"\n  Per-side statistics (all segments):")
    print(f"  {'Side':<8} {'Auto mean':>10} {'Auto std':>10} {'Fixed':>8} {'Auto>Fixed':>12}")
    for side in ("top", "bottom", "left", "right"):
        auto_vals = [x["trim"][side] for x in improvements]
        fixed_val = improvements[0]["fixed"][side]
        auto_gt_fixed = sum(1 for v in auto_vals if v > fixed_val)
        print(f"  {side:<8} {np.mean(auto_vals):10.1f} {np.std(auto_vals):10.1f} {fixed_val:8d} {auto_gt_fixed:>12d}/{len(auto_vals)}")

    print(f"\n  Comparison images saved to: {OUTPUT_DIR}/")
    print(f"  View with: eog {OUTPUT_DIR}/ &")


if __name__ == "__main__":
    main()
