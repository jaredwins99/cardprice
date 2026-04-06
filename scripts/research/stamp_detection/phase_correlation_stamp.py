#!/usr/bin/env python3
"""
Phase correlation stamp detection via bilateral symmetry analysis.

Theory: Pokemon card artwork has rough bilateral symmetry. If we compare the
left half of the artwork against the right half using phase correlation,
stamped cards should show a mismatch in the stamp region (bottom-right)
that non-stamped cards don't.

Methods:
  1. Left-right symmetry: flip bottom-left quarter, correlate with bottom-right
  2. Top-bottom symmetry: compare top-right vs bottom-right
  3. Phase coherence map: spatial distribution of correlation breakdown

Runs on the 17-card binder ground truth dataset with known stamp labels.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path("/home/godli/cardprice")
INBOX = BASE / "data/inbox"
GT_PATH = BASE / "data/condition_training/stamps_real/binder_ground_truth.jsonl"
OUT_DIR = BASE / "data/condition_training/stamps_analysis/phase_correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth():
    """Load binder ground truth cards with stamp labels."""
    entries = []
    with open(GT_PATH) as f:
        for line in f:
            entry = json.loads(line.strip())
            img_path = INBOX / entry["image"]
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [WARN] Could not load: {img_path}")
                continue
            entries.append({
                "img": img,
                "stamped": entry["stamped"],
                "name": entry.get("card_name", "unknown"),
                "path": str(img_path),
                "variant": entry.get("variant", ""),
            })
    return entries


def crop_artwork(img):
    """Crop the artwork area (top portion, excluding name bar and bottom text)."""
    h, w = img.shape[:2]
    # Artwork is roughly y: 10-70% of card height, x: 5-95%
    y1, y2 = int(h * 0.10), int(h * 0.70)
    x1, x2 = int(w * 0.05), int(w * 0.95)
    return img[y1:y2, x1:x2]


def crop_bottom_left_quarter(artwork):
    """Bottom-left quarter of artwork (control, no stamp)."""
    h, w = artwork.shape[:2]
    return artwork[h // 2:, :w // 2]


def crop_bottom_right_quarter(artwork):
    """Bottom-right quarter of artwork (stamp region)."""
    h, w = artwork.shape[:2]
    return artwork[h // 2:, w // 2:]


def crop_top_right_quarter(artwork):
    """Top-right quarter of artwork (no stamp, reference)."""
    h, w = artwork.shape[:2]
    return artwork[:h // 2, w // 2:]


def crop_top_left_quarter(artwork):
    """Top-left quarter of artwork (no stamp, reference)."""
    h, w = artwork.shape[:2]
    return artwork[:h // 2, :w // 2]


def ensure_same_size(a, b):
    """Resize b to match a's dimensions."""
    h, w = a.shape[:2]
    if b.shape[:2] != (h, w):
        b = cv2.resize(b, (w, h))
    return a, b


def phase_correlate_gray(img_a, img_b):
    """Compute phase correlation between two grayscale images.

    Returns:
        peak_value: correlation peak (higher = more similar)
        shift: (dx, dy) translation offset
        response: the full phase correlation response map
    """
    img_a, img_b = ensure_same_size(img_a, img_b)

    # Convert to float
    fa = img_a.astype(np.float64)
    fb = img_b.astype(np.float64)

    # Apply Hanning window to reduce edge effects
    h, w = fa.shape
    hann_y = np.hanning(h)
    hann_x = np.hanning(w)
    window = np.outer(hann_y, hann_x)
    fa = fa * window
    fb = fb * window

    # FFT
    Fa = np.fft.fft2(fa)
    Fb = np.fft.fft2(fb)

    # Cross-power spectrum
    cross = Fa * np.conj(Fb)
    magnitude = np.abs(cross)
    magnitude[magnitude < 1e-10] = 1e-10
    cross_norm = cross / magnitude

    # Inverse FFT to get correlation
    response = np.fft.ifft2(cross_norm).real
    response = np.fft.fftshift(response)

    # Find peak
    peak_loc = np.unravel_index(np.argmax(response), response.shape)
    peak_value = response[peak_loc]

    # Shift relative to center
    cy, cx = h // 2, w // 2
    shift = (peak_loc[1] - cx, peak_loc[0] - cy)

    return peak_value, shift, response


def phase_coherence_map(img_a, img_b, block_size=16):
    """Compute local phase coherence map using block-wise phase correlation.

    Returns a map where each pixel value indicates the local phase correlation
    strength. Low values indicate where the two images differ (e.g., stamp).
    """
    img_a, img_b = ensure_same_size(img_a, img_b)
    h, w = img_a.shape

    # Compute block-wise correlation
    map_h = h // block_size
    map_w = w // block_size
    coherence = np.zeros((map_h, map_w), dtype=np.float64)

    for by in range(map_h):
        for bx in range(map_w):
            y1 = by * block_size
            y2 = y1 + block_size
            x1 = bx * block_size
            x2 = x1 + block_size

            block_a = img_a[y1:y2, x1:x2].astype(np.float64)
            block_b = img_b[y1:y2, x1:x2].astype(np.float64)

            # Normalized cross-correlation (simpler for small blocks)
            a_norm = block_a - np.mean(block_a)
            b_norm = block_b - np.mean(block_b)
            denom = np.sqrt(np.sum(a_norm**2) * np.sum(b_norm**2))
            if denom < 1e-10:
                coherence[by, bx] = 0.0
            else:
                coherence[by, bx] = np.sum(a_norm * b_norm) / denom

    return coherence


def ncc(img_a, img_b):
    """Normalized cross-correlation (global)."""
    img_a, img_b = ensure_same_size(img_a, img_b)
    a = img_a.astype(np.float64).ravel()
    b = img_b.astype(np.float64).ravel()
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom < 1e-10:
        return 0.0
    return float(np.sum(a * b) / denom)


def analyze_card(entry, idx):
    """Run all phase correlation analyses on a single card."""
    img = entry["img"]
    artwork = crop_artwork(img)
    artwork_gray = cv2.cvtColor(artwork, cv2.COLOR_BGR2GRAY)

    # Quarter crops
    bl = crop_bottom_left_quarter(artwork_gray)
    br = crop_bottom_right_quarter(artwork_gray)
    tr = crop_top_right_quarter(artwork_gray)
    tl = crop_top_left_quarter(artwork_gray)

    # Flip left for mirror comparison
    bl_flipped = cv2.flip(bl, 1)  # horizontal flip
    tl_flipped = cv2.flip(tl, 1)

    results = {
        "name": entry["name"],
        "stamped": entry["stamped"],
        "variant": entry["variant"],
    }

    # --- Method 1: Left-right symmetry (bottom half) ---
    # Flip bottom-left, compare with bottom-right
    bl_f, br_r = ensure_same_size(bl_flipped, br)
    peak_lr, shift_lr, response_lr = phase_correlate_gray(bl_f, br_r)
    ncc_lr = ncc(bl_f, br_r)
    results["lr_peak"] = peak_lr
    results["lr_shift"] = shift_lr
    results["lr_ncc"] = ncc_lr

    # --- Method 2: Top-bottom symmetry (right side) ---
    # Compare top-right vs bottom-right (vertical symmetry breaking)
    tr_r, br_r2 = ensure_same_size(tr, br)
    peak_tb, shift_tb, response_tb = phase_correlate_gray(tr_r, br_r2)
    ncc_tb = ncc(tr_r, br_r2)
    results["tb_peak"] = peak_tb
    results["tb_shift"] = shift_tb
    results["tb_ncc"] = ncc_tb

    # --- Method 3: Left-right symmetry (top half, for baseline) ---
    # Top half should have similar symmetry for both stamped and clean
    tl_f, tr_r2 = ensure_same_size(tl_flipped, tr)
    peak_lr_top, shift_lr_top, response_lr_top = phase_correlate_gray(tl_f, tr_r2)
    ncc_lr_top = ncc(tl_f, tr_r2)
    results["lr_top_peak"] = peak_lr_top
    results["lr_top_ncc"] = ncc_lr_top

    # --- Method 4: Symmetry ratio ---
    # Ratio of bottom LR symmetry to top LR symmetry
    # Stamps should reduce bottom symmetry relative to top
    results["lr_ratio"] = ncc_lr / (ncc_lr_top + 1e-10)

    # --- Method 5: Phase coherence maps ---
    bl_f_sized, br_sized = ensure_same_size(bl_flipped, br)
    coherence_lr = phase_coherence_map(bl_f_sized, br_sized, block_size=16)
    results["coherence_mean"] = float(np.mean(coherence_lr))
    results["coherence_std"] = float(np.std(coherence_lr))
    # Bottom-right of the coherence map is where the stamp would be
    ch, cw = coherence_lr.shape
    if ch > 1 and cw > 1:
        stamp_quadrant = coherence_lr[ch // 2:, cw // 2:]
        control_quadrant = coherence_lr[:ch // 2, :cw // 2]
        results["coherence_stamp_region"] = float(np.mean(stamp_quadrant)) if stamp_quadrant.size > 0 else 0.0
        results["coherence_control_region"] = float(np.mean(control_quadrant)) if control_quadrant.size > 0 else 0.0
        results["coherence_ratio"] = results["coherence_stamp_region"] / (results["coherence_control_region"] + 1e-10)
    else:
        results["coherence_stamp_region"] = 0.0
        results["coherence_control_region"] = 0.0
        results["coherence_ratio"] = 1.0

    # --- Method 6: Difference image analysis ---
    # Absolute difference between flipped-left and right
    diff = cv2.absdiff(bl_f_sized, br_sized)
    results["diff_mean"] = float(np.mean(diff))
    results["diff_std"] = float(np.std(diff))
    # Stamp region of the diff (bottom-right of the bottom-right quarter)
    dh, dw = diff.shape
    stamp_diff = diff[dh // 2:, dw // 2:]
    ctrl_diff = diff[:dh // 2, :dw // 2]
    results["diff_stamp_mean"] = float(np.mean(stamp_diff))
    results["diff_ctrl_mean"] = float(np.mean(ctrl_diff))
    results["diff_ratio"] = results["diff_stamp_mean"] / (results["diff_ctrl_mean"] + 1e-10)

    # Save visualization
    save_card_visualization(entry, idx, artwork_gray, bl_flipped, br,
                           response_lr, coherence_lr, diff)

    return results


def save_card_visualization(entry, idx, artwork_gray, bl_flipped, br,
                           response, coherence, diff):
    """Save a visualization of the phase correlation analysis for one card."""
    # Resize components to uniform sizes
    viz_h, viz_w = 120, 160

    def to_viz(img, sz=(viz_w, viz_h)):
        out = cv2.resize(img, sz)
        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        return out

    def normalize_float(img):
        mn, mx = img.min(), img.max()
        if mx - mn < 1e-10:
            return np.zeros_like(img, dtype=np.uint8)
        return ((img - mn) / (mx - mn) * 255).astype(np.uint8)

    # Build visualization row
    parts = []

    # 1. Full artwork
    parts.append(to_viz(artwork_gray))

    # 2. Flipped left
    bl_f_r, br_r = ensure_same_size(bl_flipped, br)
    parts.append(to_viz(bl_f_r))

    # 3. Right (stamp region)
    parts.append(to_viz(br_r))

    # 4. Phase correlation response
    resp_norm = normalize_float(response)
    parts.append(to_viz(resp_norm))

    # 5. Coherence map
    coh_norm = normalize_float(coherence)
    coh_color = cv2.applyColorMap(cv2.resize(coh_norm, (viz_w, viz_h)), cv2.COLORMAP_JET)
    parts.append(coh_color)

    # 6. Difference image
    parts.append(to_viz(normalize_float(diff.astype(np.float64))))

    row = np.hstack(parts)

    # Add label bar
    label_h = 30
    label_bar = np.zeros((label_h, row.shape[1], 3), dtype=np.uint8)
    stamp_str = "STAMPED" if entry["stamped"] else "CLEAN"
    color = (0, 0, 255) if entry["stamped"] else (0, 255, 0)
    text = f"{idx:02d} {entry['name']} [{stamp_str}] ({entry['variant']})"
    cv2.putText(label_bar, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Column labels
    col_labels = ["Artwork", "Flip-Left", "Right(stamp)", "PhaseCorr", "Coherence", "Diff"]
    col_bar = np.zeros((20, row.shape[1], 3), dtype=np.uint8)
    for i, lbl in enumerate(col_labels):
        x = i * viz_w + 5
        cv2.putText(col_bar, lbl, (x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    card_viz = np.vstack([label_bar, col_bar, row])

    out_path = OUT_DIR / f"card_{idx:02d}_{entry['name'].replace(' ', '_')}.png"
    cv2.imwrite(str(out_path), card_viz)
    return card_viz


def print_results_table(all_results):
    """Print a formatted results table."""
    print("\n" + "=" * 120)
    print("  Phase Correlation Stamp Detection Results")
    print("=" * 120)

    # Header
    print(f"  {'Card':<20} {'Stamp':>6} {'LR-NCC':>8} {'TB-NCC':>8} "
          f"{'LR-Top':>8} {'LRatio':>8} {'CohMn':>8} {'CohSR':>8} "
          f"{'DiffMn':>8} {'DiffR':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for r in all_results:
        stamp_str = "YES" if r["stamped"] else "no"
        print(f"  {r['name']:<20} {stamp_str:>6} "
              f"{r['lr_ncc']:>8.4f} {r['tb_ncc']:>8.4f} "
              f"{r['lr_top_ncc']:>8.4f} {r['lr_ratio']:>8.4f} "
              f"{r['coherence_mean']:>8.4f} {r['coherence_stamp_region']:>8.4f} "
              f"{r['diff_mean']:>8.2f} {r['diff_ratio']:>8.4f}")


def analyze_separation(all_results):
    """Analyze whether any metric separates stamped from clean."""
    stamped = [r for r in all_results if r["stamped"]]
    clean = [r for r in all_results if not r["stamped"]]

    if not stamped or not clean:
        print("\n  ERROR: Need both stamped and clean samples!")
        return

    metrics = [
        "lr_ncc", "tb_ncc", "lr_top_ncc", "lr_ratio",
        "coherence_mean", "coherence_std", "coherence_stamp_region",
        "coherence_control_region", "coherence_ratio",
        "diff_mean", "diff_std", "diff_stamp_mean", "diff_ctrl_mean", "diff_ratio",
        "lr_peak", "tb_peak", "lr_top_peak",
    ]

    print(f"\n{'='*90}")
    print(f"  Separation Analysis: {len(stamped)} stamped vs {len(clean)} clean")
    print(f"{'='*90}")
    print(f"  {'Metric':<28} {'Stamped':>16} {'Clean':>16} {'Sep':>8} {'Direction':>10}")
    print(f"  {'-'*28} {'-'*16} {'-'*16} {'-'*8} {'-'*10}")

    separations = []
    for metric in metrics:
        s_vals = np.array([r[metric] for r in stamped])
        c_vals = np.array([r[metric] for r in clean])
        s_mean, s_std = np.mean(s_vals), np.std(s_vals)
        c_mean, c_std = np.mean(c_vals), np.std(c_vals)
        sep = abs(s_mean - c_mean) / (s_std + c_std + 1e-10)
        direction = "stamp<clean" if s_mean < c_mean else "stamp>clean"
        separations.append((metric, sep, direction, s_mean, s_std, c_mean, c_std))
        print(f"  {metric:<28} {s_mean:>8.4f}+{s_std:<6.4f} "
              f"{c_mean:>8.4f}+{c_std:<6.4f} {sep:>7.3f} {direction:>10}")

    # Rank by separation
    separations.sort(key=lambda x: -x[1])
    print(f"\n  Top metrics by separation score:")
    for i, (name, sep, direction, sm, ss, cm, cs, ) in enumerate(separations[:5]):
        print(f"    {i+1}. {name}: sep={sep:.3f} ({direction})")

    # Try threshold classification for best metrics
    print(f"\n{'='*90}")
    print(f"  Threshold Classification (best metrics)")
    print(f"{'='*90}")

    for metric, sep, direction, sm, ss, cm, cs in separations[:8]:
        s_vals = [r[metric] for r in stamped]
        c_vals = [r[metric] for r in clean]
        all_vals = sorted(set(s_vals + c_vals))

        best_acc = 0.0
        best_thresh = 0.0
        best_dir = ">"

        for thresh in all_vals:
            # stamped > thresh
            tp = sum(1 for v in s_vals if v > thresh)
            tn = sum(1 for v in c_vals if v <= thresh)
            acc_gt = (tp + tn) / (len(s_vals) + len(c_vals))

            # stamped < thresh
            tp2 = sum(1 for v in s_vals if v < thresh)
            tn2 = sum(1 for v in c_vals if v >= thresh)
            acc_lt = (tp2 + tn2) / (len(s_vals) + len(c_vals))

            if acc_gt > best_acc:
                best_acc = acc_gt
                best_thresh = thresh
                best_dir = ">"
            if acc_lt > best_acc:
                best_acc = acc_lt
                best_thresh = thresh
                best_dir = "<"

        print(f"  {metric:<28} acc={best_acc:.1%}  thresh={best_thresh:.4f}  "
              f"dir=stamped{best_dir}thresh")

        # Show misclassifications
        for r in all_results:
            val = r[metric]
            if best_dir == ">":
                pred = val > best_thresh
            else:
                pred = val < best_thresh
            if pred != r["stamped"]:
                actual = "stamped" if r["stamped"] else "clean"
                predicted = "stamped" if pred else "clean"
                print(f"    MISS: {r['name']:<20} val={val:.4f} "
                      f"predicted={predicted} actual={actual}")


def save_combined_visualization(all_results, card_vizs):
    """Save a combined overview image."""
    if not card_vizs:
        return

    # Stack all card visualizations vertically
    combined = np.vstack(card_vizs)
    out_path = OUT_DIR / "combined_overview.png"
    cv2.imwrite(str(out_path), combined)
    print(f"\n  Saved combined overview: {out_path}")


def main():
    print("Phase Correlation Stamp Detection")
    print("=" * 70)
    print("Theory: Stamps break bilateral symmetry in bottom-right artwork region")
    print()

    # Load data
    entries = load_ground_truth()
    print(f"Loaded {len(entries)} cards from ground truth")
    n_stamped = sum(1 for e in entries if e["stamped"])
    n_clean = sum(1 for e in entries if not e["stamped"])
    print(f"  Stamped: {n_stamped}, Clean: {n_clean}")

    # Analyze each card
    all_results = []
    card_vizs = []
    for idx, entry in enumerate(entries):
        print(f"  [{idx+1}/{len(entries)}] {entry['name']} "
              f"({'STAMPED' if entry['stamped'] else 'clean'})")
        result = analyze_card(entry, idx)
        all_results.append(result)

    # Print results
    print_results_table(all_results)

    # Analyze separation
    analyze_separation(all_results)

    # Load saved per-card visualizations and combine
    card_viz_files = sorted(OUT_DIR.glob("card_*.png"))
    if card_viz_files:
        vizs = [cv2.imread(str(f)) for f in card_viz_files]
        # Ensure same width
        max_w = max(v.shape[1] for v in vizs)
        padded = []
        for v in vizs:
            if v.shape[1] < max_w:
                pad = np.zeros((v.shape[0], max_w - v.shape[1], 3), dtype=np.uint8)
                v = np.hstack([v, pad])
            padded.append(v)
        combined = np.vstack(padded)
        out_path = OUT_DIR / "combined_overview.png"
        cv2.imwrite(str(out_path), combined)
        print(f"\n  Saved combined overview: {out_path}")

    # Save raw results as JSON
    json_results = []
    for r in all_results:
        jr = {}
        for k, v in r.items():
            if k == "img":
                continue
            if isinstance(v, (np.integer,)):
                jr[k] = int(v)
            elif isinstance(v, (np.floating,)):
                jr[k] = float(v)
            elif isinstance(v, tuple):
                jr[k] = [int(x) if isinstance(x, (np.integer,)) else float(x) if isinstance(x, (np.floating,)) else x for x in v]
            else:
                jr[k] = v
        json_results.append(jr)
    out_json = OUT_DIR / "phase_correlation_results.json"
    with open(out_json, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved results: {out_json}")

    print(f"\n  All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
