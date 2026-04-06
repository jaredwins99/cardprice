#!/usr/bin/env python3
"""Diagnostic script to investigate bottom-row clipping on page 0.

Reproduces the grid fallback path from card_segmenter.py step by step,
printing every intermediate value so we can see exactly why cards 06/07/08
(bottom row) lose their weakness/resistance/retreat/artist credit.

Target image: data/inbox/page_20260228_174819.jpg
"""

import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Helpers copied from card_segmenter.py so we can instrument them
# ---------------------------------------------------------------------------

def _order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left has smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest x+y
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right has smallest x-y
    rect[3] = pts[np.argmax(d)]   # bottom-left has largest x-y
    return rect


def find_page_corners(image):
    """Find the binder page outline (largest rectangular contour)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    page_corners = None
    method_used = None

    for name, thresh_fn in [
        ("Canny", lambda g: cv2.Canny(g, 20, 60)),
        ("AdaptiveThresh", lambda g: cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 5)),
    ]:
        edges = thresh_fn(blurred)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(cnt)
            if area < h * w * 0.4:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.02, 0.04, 0.06, 0.08):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4:
                    page_corners = approx
                    method_used = f"{name} eps={eps}"
                    break
            if page_corners is not None:
                break
        if page_corners is not None:
            break

    return page_corners, method_used


def find_valleys(profile, n_cells, axis_len, axis_name=""):
    """Valley detection with full diagnostic output."""
    if n_cells <= 1:
        return [], {}

    kernel_size = max(3, int(axis_len * 0.02) | 1)
    smoothed = cv2.GaussianBlur(profile.reshape(-1, 1),
                                 (1, kernel_size), 0).flatten()
    expected_cell = axis_len / n_cells
    margin = int(expected_cell * 0.50)

    print(f"\n  [{axis_name}] axis_len={axis_len}, n_cells={n_cells}")
    print(f"  [{axis_name}] expected_cell={expected_cell:.1f}px, margin={margin}px")
    print(f"  [{axis_name}] search range: [{margin}, {len(smoothed) - margin}]")
    print(f"  [{axis_name}] EXCLUDED bottom zone: [{len(smoothed) - margin}, {len(smoothed)}] "
          f"({margin}px = {margin/axis_len*100:.1f}% of axis)")

    # Find all local minima
    minima_idx = []
    minima_val = []
    for i in range(margin, len(smoothed) - margin):
        if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
            minima_idx.append(i)
            minima_val.append(smoothed[i])

    print(f"  [{axis_name}] found {len(minima_idx)} local minima in search range")
    if minima_idx:
        print(f"  [{axis_name}] minima positions: {minima_idx}")

    if len(minima_idx) < n_cells - 1:
        print(f"  [{axis_name}] FAIL: only {len(minima_idx)} minima, need {n_cells - 1}")
        return None, {"smoothed": smoothed, "margin": margin, "expected_cell": expected_cell}

    # Score by depth
    neighbourhood = int(expected_cell * 0.3)
    scored = []
    for idx, val in zip(minima_idx, minima_val):
        lo = max(0, idx - neighbourhood)
        hi = min(len(smoothed), idx + neighbourhood)
        local_mean = smoothed[lo:hi].mean()
        depth = local_mean - val
        scored.append((idx, depth))

    scored.sort(key=lambda x: x[1], reverse=True)
    n_needed = n_cells - 1
    top_k = min(len(scored), max(n_needed * 3, 8))
    candidates = scored[:top_k]

    print(f"  [{axis_name}] top {top_k} candidates by depth:")
    for idx, depth in candidates:
        print(f"    pos={idx} ({idx/axis_len*100:.1f}%), depth={depth:.2f}")

    # Combinatorial search
    min_spacing = expected_cell * 0.4
    max_depth = candidates[0][1] if candidates[0][1] > 0 else 1.0
    best_combo = None
    best_score = -float("inf")
    combos_tried = 0
    combos_valid = 0

    for combo in combinations(range(len(candidates)), n_needed):
        combos_tried += 1
        idxs = sorted([candidates[c][0] for c in combo])
        depths = [candidates[c][1] for c in combo]

        valid = True
        for i in range(len(idxs) - 1):
            if idxs[i + 1] - idxs[i] < min_spacing:
                valid = False
                break
        if not valid:
            continue

        boundaries = [0] + idxs + [axis_len]
        cell_sizes = [boundaries[i + 1] - boundaries[i]
                      for i in range(len(boundaries) - 1)]
        out_of_range = False
        for cs in cell_sizes:
            if cs < expected_cell * 0.50 or cs > expected_cell * 1.70:
                out_of_range = True
                break
        if out_of_range:
            continue

        combos_valid += 1
        total_depth = sum(depths) / max_depth
        size_std = float(np.std(cell_sizes))
        size_penalty = size_std / expected_cell
        score = total_depth - 1.5 * size_penalty

        if score > best_score:
            best_score = score
            best_combo = idxs

    print(f"  [{axis_name}] combos tried={combos_tried}, valid={combos_valid}")
    if best_combo is None:
        print(f"  [{axis_name}] FAIL: no valid valley combination")
        return None, {"smoothed": smoothed, "margin": margin, "expected_cell": expected_cell}

    # Refine
    refine_radius = int(expected_cell * 0.04)
    refined = []
    for v in best_combo:
        lo = max(0, v - refine_radius)
        hi = min(axis_len, v + refine_radius + 1)
        local_min_offset = int(np.argmin(smoothed[lo:hi]))
        refined.append(lo + local_min_offset)

    print(f"  [{axis_name}] selected valleys (before refine): {best_combo}")
    print(f"  [{axis_name}] selected valleys (after refine):  {refined}")

    return refined, {"smoothed": smoothed, "margin": margin, "expected_cell": expected_cell}


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def main():
    img_path = Path("/home/godli/cardprice/data/inbox/page_20260228_174819.jpg")
    if not img_path.exists():
        print(f"ERROR: {img_path} not found")
        sys.exit(1)

    image = cv2.imread(str(img_path))
    h, w = image.shape[:2]
    print(f"=== Image: {img_path.name} ===")
    print(f"Original size: {w}x{h}")

    # Resize like segment_cards does
    max_dim = 4500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
        print(f"Resized to: {w}x{h} (scale={scale:.4f})")

    # Step 1: Find page corners
    print("\n" + "=" * 60)
    print("STEP 1: Page outline detection")
    print("=" * 60)

    page_corners, method = find_page_corners(image)
    if page_corners is None:
        print("No page outline found! Would use full image.")
        return

    print(f"Method: {method}")
    ordered = _order_points(page_corners.reshape(4, 2).astype(np.float32))
    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
    for label, pt in zip(labels, ordered):
        print(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})")

    # Compute page extent from detected corners
    top_y = min(ordered[0][1], ordered[1][1])
    bot_y = max(ordered[2][1], ordered[3][1])
    left_x = min(ordered[0][0], ordered[3][0])
    right_x = max(ordered[1][0], ordered[2][0])
    print(f"\nDetected page extent:")
    print(f"  Y range: {top_y:.1f} to {bot_y:.1f} (height={bot_y - top_y:.1f})")
    print(f"  X range: {left_x:.1f} to {right_x:.1f} (width={right_x - left_x:.1f})")
    print(f"  Image bounds: 0 to {h} (Y), 0 to {w} (X)")
    print(f"  Bottom gap: image_h - bot_y = {h - bot_y:.1f}px ({(h - bot_y)/h*100:.2f}%)")
    print(f"  Top gap:    top_y = {top_y:.1f}px ({top_y/h*100:.2f}%)")

    # Step 2: Expand corners (as segmenter does)
    print("\n" + "=" * 60)
    print("STEP 2: Corner expansion (4%)")
    print("=" * 60)

    centroid = ordered.mean(axis=0)
    expand_frac = 0.04
    expanded = centroid + (1.0 + expand_frac) * (ordered - centroid)
    print(f"Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})")

    for label, orig, exp in zip(labels, ordered, expanded):
        print(f"  {label}: ({orig[0]:.1f}, {orig[1]:.1f}) -> ({exp[0]:.1f}, {exp[1]:.1f})")

    # Clamp
    clamped = expanded.copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0, w - 1)
    clamped[:, 1] = np.clip(clamped[:, 1], 0, h - 1)

    clamped_any = not np.allclose(expanded, clamped)
    print(f"\nAfter clamping to image bounds:")
    for label, exp, clp in zip(labels, expanded, clamped):
        was_clamped = not np.allclose(exp, clp)
        flag = " ** CLAMPED **" if was_clamped else ""
        print(f"  {label}: ({clp[0]:.1f}, {clp[1]:.1f}){flag}")
    if clamped_any:
        print("  WARNING: Some corners were clamped to image bounds!")
        print("  This means the 4% expansion wanted to go beyond the image edge.")
        print("  The expansion is limited by the image boundary, so the bottom")
        print("  row gets less expansion than intended.")

    # Step 3: Destination size (from ORIGINAL corners, not expanded)
    print("\n" + "=" * 60)
    print("STEP 3: Destination rectangle (perspective warp target)")
    print("=" * 60)

    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bot = np.linalg.norm(ordered[2] - ordered[3])
    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    dst_w = int((width_top + width_bot) / 2)
    dst_h = int((height_left + height_right) / 2)

    print(f"Edge lengths (original corners):")
    print(f"  top edge:    {width_top:.1f}px")
    print(f"  bottom edge: {width_bot:.1f}px")
    print(f"  left edge:   {height_left:.1f}px")
    print(f"  right edge:  {height_right:.1f}px")
    print(f"Destination size: {dst_w}x{dst_h}")

    # KEY INSIGHT: The source quad is EXPANDED (larger area), but the
    # destination is sized from ORIGINAL corners (smaller).
    # So the warp maps a larger source area into a smaller rectangle.
    # Content near the edges of the expanded source gets "compressed".
    #
    # But actually -- this is correct for perspective correction.
    # The question is whether dst_h is tall enough.

    # Compute what the expanded source corners imply
    exp_width_top = np.linalg.norm(clamped[1] - clamped[0])
    exp_width_bot = np.linalg.norm(clamped[2] - clamped[3])
    exp_height_left = np.linalg.norm(clamped[3] - clamped[0])
    exp_height_right = np.linalg.norm(clamped[2] - clamped[1])
    exp_dst_w = int((exp_width_top + exp_width_bot) / 2)
    exp_dst_h = int((exp_height_left + exp_height_right) / 2)

    print(f"\nIf destination were sized from EXPANDED corners instead:")
    print(f"  Would be {exp_dst_w}x{exp_dst_h}")
    print(f"  Height difference: {exp_dst_h - dst_h}px ({(exp_dst_h - dst_h)/dst_h*100:.1f}%)")
    print(f"  This means the bottom {exp_dst_h - dst_h}px of content from the expanded source")
    print(f"  is being squished into a destination that's {dst_h}px tall instead of {exp_dst_h}px")

    # Step 4: Perform the warp and run valley detection
    print("\n" + "=" * 60)
    print("STEP 4: Perspective warp & valley detection")
    print("=" * 60)

    dst = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1],
    ], dtype=np.float32)
    M_warp = cv2.getPerspectiveTransform(clamped, dst)
    rectified = cv2.warpPerspective(image, M_warp, (dst_w, dst_h))

    page_h, page_w = rectified.shape[:2]
    print(f"Rectified page: {page_w}x{page_h}")

    page_is_landscape = page_w > page_h
    if page_is_landscape:
        img_rows, img_cols = 3, 3  # cols, rows swapped for landscape
        print(f"Landscape page detected, using {img_rows} img_rows x {img_cols} img_cols")
    else:
        img_rows, img_cols = 3, 3
        print(f"Portrait page, using {img_rows} rows x {img_cols} cols")

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    h_proj = gray.mean(axis=1).astype(np.float64)
    v_proj = gray.mean(axis=0).astype(np.float64)

    print("\n--- Horizontal (row) valleys ---")
    row_valleys, row_diag = find_valleys(h_proj, img_rows, page_h, axis_name="ROW")

    print("\n--- Vertical (column) valleys ---")
    col_valleys, col_diag = find_valleys(v_proj, img_cols, page_w, axis_name="COL")

    # Step 5: Compute boundaries
    print("\n" + "=" * 60)
    print("STEP 5: Cell boundaries & bottom row analysis")
    print("=" * 60)

    if row_valleys is not None:
        edges_row = [0] + row_valleys + [page_h]
        row_bounds = [(edges_row[i], edges_row[i + 1]) for i in range(len(edges_row) - 1)]
    else:
        cell_h = page_h / img_rows
        row_bounds = [(int(r * cell_h), int((r + 1) * cell_h)) for r in range(img_rows)]
        print("  Using uniform grid (valley detection failed)")

    if col_valleys is not None:
        edges_col = [0] + col_valleys + [page_w]
        col_bounds = [(edges_col[i], edges_col[i + 1]) for i in range(len(edges_col) - 1)]
    else:
        cell_w = page_w / img_cols
        col_bounds = [(int(c * cell_w), int((c + 1) * cell_w)) for c in range(img_cols)]
        print("  Using uniform grid for columns (valley detection failed)")

    print(f"\nRow boundaries:")
    for i, (s, e) in enumerate(row_bounds):
        size = e - s
        pct = size / page_h * 100
        print(f"  Row {i}: [{s}, {e}] size={size}px ({pct:.1f}%)")

    print(f"\nColumn boundaries:")
    for i, (s, e) in enumerate(col_bounds):
        size = e - s
        pct = size / page_w * 100
        print(f"  Col {i}: [{s}, {e}] size={size}px ({pct:.1f}%)")

    # Step 6: Analyze bottom row specifically
    print("\n" + "=" * 60)
    print("STEP 6: Bottom row detail")
    print("=" * 60)

    # The bottom row is the last row
    last_row_idx = len(row_bounds) - 1
    bot_start, bot_end = row_bounds[last_row_idx]
    bot_height = bot_end - bot_start

    # With padding applied (pad_frac=0.035 at page edges)
    pad_frac = 0.035
    pad_y_bot = int(bot_height * pad_frac)
    pad_y_top_for_last = 0  # interior boundary, no padding
    effective_bot_start = bot_start + pad_y_top_for_last
    effective_bot_end = bot_end - pad_y_bot

    print(f"Bottom row (row {last_row_idx}):")
    print(f"  Raw bounds: [{bot_start}, {bot_end}] = {bot_height}px")
    print(f"  Padding: top=0 (interior valley), bot={pad_y_bot}px (page edge)")
    print(f"  Effective: [{effective_bot_start}, {effective_bot_end}] = {effective_bot_end - effective_bot_start}px")
    print(f"  Page height: {page_h}px")
    print(f"  Bottom end vs page edge: {page_h - bot_end}px gap (should be 0)")

    # Compare row sizes
    print(f"\nRow size comparison:")
    sizes = [e - s for s, e in row_bounds]
    for i, sz in enumerate(sizes):
        pct_diff = (sz - sizes[0]) / sizes[0] * 100
        print(f"  Row {i}: {sz}px (diff from row 0: {pct_diff:+.1f}%)")

    # Expected card height in the rectified image
    expected_card_h = page_h / img_rows
    print(f"\nExpected cell height (uniform): {expected_card_h:.1f}px")
    print(f"Bottom row height: {bot_height}px ({bot_height/expected_card_h*100:.1f}% of expected)")

    # Step 7: Check where the bottom boundary valley sits
    print("\n" + "=" * 60)
    print("STEP 7: Profile analysis near bottom edge")
    print("=" * 60)

    if row_diag and "smoothed" in row_diag:
        smoothed = row_diag["smoothed"]
        margin = row_diag["margin"]
        expected_cell = row_diag["expected_cell"]

        # Show the bottom portion of the horizontal profile
        bottom_zone_start = max(0, page_h - int(expected_cell * 0.8))
        print(f"\nHorizontal profile (brightness) near bottom ({bottom_zone_start} to {page_h}):")
        print(f"  Margin exclusion zone starts at: {page_h - margin} "
              f"(last {margin}px = {margin/page_h*100:.1f}% excluded from valley search)")

        # Sample every ~20px
        step = max(1, (page_h - bottom_zone_start) // 30)
        for y in range(bottom_zone_start, page_h, step):
            val = smoothed[y]
            bar = "#" * int(val / 4)
            in_excl = " [EXCLUDED]" if y >= page_h - margin else ""
            print(f"  y={y:4d} ({y/page_h*100:5.1f}%): {val:6.1f} {bar}{in_excl}")

        # Check if there are dark valleys in the excluded zone
        print(f"\nLooking for potential valleys in EXCLUDED bottom zone [{page_h - margin}, {page_h}]:")
        excl_start = page_h - margin
        excl_profile = smoothed[excl_start:]
        if len(excl_profile) > 2:
            excl_min_idx = int(np.argmin(excl_profile))
            excl_min_val = excl_profile[excl_min_idx]
            global_min_idx = excl_start + excl_min_idx
            print(f"  Darkest point in excluded zone: y={global_min_idx}, brightness={excl_min_val:.1f}")
            # Compare with the detected valleys
            if row_valleys:
                for i, v in enumerate(row_valleys):
                    v_val = smoothed[v]
                    print(f"  Detected valley {i}: y={v}, brightness={v_val:.1f}")

    # Step 8: What would the bottom row look like with the page edge as boundary?
    print("\n" + "=" * 60)
    print("STEP 8: Impact assessment")
    print("=" * 60)

    if row_valleys:
        last_valley = row_valleys[-1]
        print(f"Last horizontal valley: y={last_valley} ({last_valley/page_h*100:.1f}% of page)")
        print(f"Page height: {page_h}")
        print(f"Bottom row: [{last_valley}, {page_h}] = {page_h - last_valley}px")

        # What if the bottom boundary were extended?
        # Cards are 63x88mm, ratio 0.716. If cell width is known:
        if col_bounds:
            avg_col_width = np.mean([e - s for s, e in col_bounds])
            expected_card_h_from_width = avg_col_width / 0.716
            print(f"\nAvg column width: {avg_col_width:.1f}px")
            print(f"Expected card height (from width/0.716): {expected_card_h_from_width:.1f}px")
            print(f"Bottom row actual height: {page_h - last_valley}px")
            print(f"Deficit: {expected_card_h_from_width - (page_h - last_valley):.1f}px "
                  f"({(1 - (page_h - last_valley)/expected_card_h_from_width)*100:.1f}%)")

            # Where does the card bottom ACTUALLY fall?
            # If each row's cards start at the valley + a small offset:
            mid_row_heights = [row_bounds[i][1] - row_bounds[i][0] for i in range(len(row_bounds) - 1)]
            if mid_row_heights:
                avg_mid_row_h = np.mean(mid_row_heights)
                print(f"\nAvg non-bottom-row height: {avg_mid_row_h:.1f}px")
                print(f"Bottom row height: {page_h - last_valley}px")
                print(f"Bottom row is {avg_mid_row_h - (page_h - last_valley):.1f}px shorter "
                      f"than middle rows ({(1 - (page_h - last_valley)/avg_mid_row_h)*100:.1f}%)")

    # Step 9: Save debug visualization
    print("\n" + "=" * 60)
    print("STEP 9: Saving debug visualization")
    print("=" * 60)

    debug_img = rectified.copy()

    # Draw row boundaries in red
    for s, e in row_bounds:
        cv2.line(debug_img, (0, s), (page_w, s), (0, 0, 255), 2)
        cv2.line(debug_img, (0, e), (page_w, e), (0, 0, 255), 2)

    # Draw column boundaries in blue
    for s, e in col_bounds:
        cv2.line(debug_img, (s, 0), (s, page_h), (255, 0, 0), 2)
        cv2.line(debug_img, (e, 0), (e, page_h), (255, 0, 0), 2)

    # Draw effective bottom row boundaries (with padding) in green
    cv2.line(debug_img, (0, effective_bot_start), (page_w, effective_bot_start), (0, 255, 0), 3)
    cv2.line(debug_img, (0, effective_bot_end), (page_w, effective_bot_end), (0, 255, 0), 3)

    # Label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(debug_img, "Bottom row effective bounds (green)", (10, 30),
                font, 0.8, (0, 255, 0), 2)
    cv2.putText(debug_img, f"Page h={page_h}, bot row=[{bot_start},{bot_end}]",
                (10, 60), font, 0.7, (0, 0, 255), 2)

    out_path = Path("/home/godli/cardprice/data/debug_bottom_row.jpg")
    cv2.imwrite(str(out_path), debug_img)
    print(f"Saved debug visualization to: {out_path}")

    # Also save the bottom row cells for visual inspection
    for ic in range(len(col_bounds)):
        rx1, rx2 = col_bounds[ic]
        ir = last_row_idx
        cell_h_px = bot_end - bot_start
        cell_w_px = rx2 - rx1
        pad_y_t = 0  # interior
        pad_y_b = int(cell_h_px * pad_frac)  # page edge
        pad_x_l = int(cell_w_px * pad_frac) if ic == 0 else 0
        pad_x_r = int(cell_w_px * pad_frac) if ic == len(col_bounds) - 1 else 0

        y1 = bot_start + pad_y_t
        y2 = bot_end - pad_y_b
        x1 = rx1 + pad_x_l
        x2 = rx2 - pad_x_r
        cell = rectified[y1:y2, x1:x2]

        ch, cw = cell.shape[:2]
        if cw > ch:
            cell = cv2.rotate(cell, cv2.ROTATE_90_COUNTERCLOCKWISE)

        card_idx = 6 + ic  # bottom row = cards 06, 07, 08
        cell_path = Path(f"/home/godli/cardprice/data/debug_bottom_card_{card_idx:02d}.jpg")
        cv2.imwrite(str(cell_path), cell)
        print(f"Saved bottom row cell {card_idx}: {cell_path} ({cell.shape[1]}x{cell.shape[0]})")

    # Step 10: Trace landscape index mapping for binder cards 06,07,08
    print("\n" + "=" * 60)
    print("STEP 10: Landscape index mapping for bottom binder row")
    print("=" * 60)

    # In _grid_fallback, for landscape pages with reverse_grid=True:
    #   ir = cols - 1 - bc   (image row from binder col)
    #   ic = rows - 1 - br   (image col from binder row)
    # Binder card numbering: br=0..2 (binder rows), bc=0..2 (binder cols)
    # Bottom binder row = br=2

    rows_binder, cols_binder = 3, 3
    print("Binder grid: 3 rows x 3 cols, landscape, reverse_grid=True")
    print("Card numbering: 0-8 in binder reading order (top-left to bottom-right)\n")

    for br in range(rows_binder):
        for bc in range(cols_binder):
            card_idx = br * cols_binder + bc
            # Reverse grid (as detected for this landscape page)
            ir = cols_binder - 1 - bc   # image row from binder col
            ic = rows_binder - 1 - br   # image col from binder row

            ry1, ry2 = row_bounds[ir]
            rx1, rx2 = col_bounds[ic]
            cell_h_px = ry2 - ry1
            cell_w_px = rx2 - rx1

            # Padding: page edges only
            pad_y_top = int(cell_h_px * pad_frac) if ir == 0 else 0
            pad_y_bot = int(cell_h_px * pad_frac) if ir == len(row_bounds) - 1 else 0
            pad_x_left = int(cell_w_px * pad_frac) if ic == 0 else 0
            pad_x_right = int(cell_w_px * pad_frac) if ic == len(col_bounds) - 1 else 0

            y1 = ry1 + pad_y_top
            y2 = ry2 - pad_y_bot
            x1 = rx1 + pad_x_left
            x2 = rx2 - pad_x_right

            cell_w_final = x2 - x1
            cell_h_final = y2 - y1
            is_landscape_cell = cell_w_final > cell_h_final

            # After rotation CCW: width becomes height, height becomes width
            # The cell's X range becomes the card's Y range (top-bottom)
            # The cell's Y range becomes the card's X range (left-right)
            # For CCW rotation: card_bottom = cell's x2 edge (right side of cell in image)

            marker = " <<< BOTTOM BINDER ROW" if br == 2 else ""
            print(f"  Card {card_idx:02d} (binder [{br},{bc}]): "
                  f"img_row={ir} img_col={ic} -> "
                  f"cell Y=[{y1},{y2}]({cell_h_final}px) "
                  f"X=[{x1},{x2}]({cell_w_final}px) "
                  f"landscape={is_landscape_cell}"
                  f"{'  pad_y_bot='+str(pad_y_bot) if pad_y_bot else ''}"
                  f"{'  pad_x_right='+str(pad_x_right) if pad_x_right else ''}"
                  f"{marker}")

            if br == 2:
                # After CCW rotation, the right edge of the cell (x2) becomes
                # the bottom of the card. So card bottom content comes from x2.
                # If ic = rows-1-br = 0, then rx1=0, x1 may have left padding
                # The CARD BOTTOM after CCW rotation = original image RIGHT edge of cell
                print(f"         After CCW rotation: card_top=img_y2={y2}, card_bot=img_y1={y1}")
                print(f"         card_left=img_x1={x1}, card_right=img_x2={x2}")
                print(f"         For card bottom (weakness/resistance): "
                      f"comes from img col_bounds[{ic}] right edge = {rx2}"
                      f"{'  (PAGE EDGE, padded by '+str(pad_x_right)+'px)' if ic == len(col_bounds)-1 else ''}"
                      f"{'  (PAGE EDGE col=0, padded left by '+str(pad_x_left)+'px)' if ic == 0 else ''}")

    # Step 11: The actual issue - which edge maps to card bottom?
    print("\n" + "=" * 60)
    print("STEP 11: Which image edge becomes card bottom after rotation?")
    print("=" * 60)

    # For CCW rotation (cv2.ROTATE_90_COUNTERCLOCKWISE):
    # Original (W x H) -> Rotated (H x W)
    # Original point (x, y) -> Rotated point (y, W-1-x)
    # So the BOTTOM of the rotated image (large y) corresponds to
    # small x in the original. I.e., the LEFT edge of the landscape cell.
    #
    # For binder bottom row (br=2) with reverse_grid=True:
    #   ic = rows-1-br = 0  -> col_bounds[0] = [0, 1243]
    # So x1=0 (or padded), x2=1243 (interior valley, no padding)
    # After CCW rotation: card bottom = original left edge (x=0 side)
    #                     card top = original right edge (x=1243 side)
    #
    # BUT WAIT: the bottom binder row maps to ic=0, which is the LEFTMOST
    # column in the image. The left edge is x=0 = page edge.
    # With pad_x_left applied, the effective left edge is padded inward.
    # After CCW rotation this padded left edge becomes the BOTTOM of the card.

    # Let's verify which rotation is actually used
    print("For landscape page with reverse_grid=True, cell_rot=CCW")
    print()
    print("CCW rotation mapping: original(x,y) -> rotated(y, W-1-x)")
    print("  original LEFT edge (small x)  -> rotated BOTTOM (large y)")
    print("  original RIGHT edge (large x) -> rotated TOP (small y)")
    print("  original TOP edge (small y)    -> rotated LEFT (small x)")
    print("  original BOTTOM edge (large y) -> rotated RIGHT (large x)")
    print()

    # Bottom binder row = ic=0
    ic_bottom_binder = 0
    rx1_bot, rx2_bot = col_bounds[ic_bottom_binder]
    cell_w_bot = rx2_bot - rx1_bot
    pad_x_left_bot = int(cell_w_bot * pad_frac)  # ic=0 is page edge
    effective_x1 = rx1_bot + pad_x_left_bot

    print(f"Bottom binder row maps to image col_bounds[{ic_bottom_binder}] = [{rx1_bot}, {rx2_bot}]")
    print(f"  Left edge (page boundary): x={rx1_bot}")
    print(f"  Left padding (pad_frac={pad_frac}): {pad_x_left_bot}px")
    print(f"  Effective left edge: x={effective_x1}")
    print(f"  After CCW rotation, this left edge (x={effective_x1}) becomes the CARD BOTTOM")
    print()
    print(f"  The card bottom is at x={effective_x1} in the rectified image.")
    print(f"  If the page corners were detected too far inward on the left,")
    print(f"  then x=0 in the rectified image is already past the true page edge,")
    print(f"  and the leftmost {effective_x1}px is ALSO trimmed by padding.")
    print()

    # Check: how much of the original image's left side is outside the warp source?
    # The expanded corners for the left side:
    print("Left-side corners in original image:")
    print(f"  top-left (expanded, clamped):    x={clamped[0][0]:.1f}")
    print(f"  bottom-left (expanded, clamped): x={clamped[3][0]:.1f}")
    print(f"  Original detected top-left:      x={ordered[0][0]:.1f}")
    print(f"  Original detected bottom-left:   x={ordered[3][0]:.1f}")
    print()

    # Now check: what's the actual content at x=0..50 in the rectified image?
    # It should be binder frame / sleeve edge, not card content
    left_strip = rectified[:, 0:100, :]
    left_strip_gray = cv2.cvtColor(left_strip, cv2.COLOR_BGR2GRAY)
    print(f"Rectified image left 100px strip: mean brightness={left_strip_gray.mean():.1f}")

    # And the right side (which becomes the TOP of cards in bottom binder row)
    right_strip = rectified[:, -100:, :]
    right_strip_gray = cv2.cvtColor(right_strip, cv2.COLOR_BGR2GRAY)
    print(f"Rectified image right 100px strip: mean brightness={right_strip_gray.mean():.1f}")

    # Save the left strip for visual inspection
    left_debug = rectified[:, 0:200, :]
    cv2.imwrite("/home/godli/cardprice/data/debug_left_strip.jpg", left_debug)
    print("Saved left 200px strip to data/debug_left_strip.jpg")

    # Step 12: The REAL test - compare with reference card
    print("\n" + "=" * 60)
    print("STEP 12: Asymmetry analysis - is the issue the warp or the valley?")
    print("=" * 60)

    # For the bottom binder row, cards are in image column 0 (ic=0).
    # The card BOTTOM (after CCW rotation) = image LEFT edge of column 0.
    # The card TOP (after CCW rotation) = image RIGHT edge of column 0 = col_bounds[0][1] = valley at x=1243.
    #
    # Column 0 width: 1243px. After padding left (43px), effective width = 1200px.
    # But a card's aspect ratio is 63/88 = 0.716, so if height after rotation = column width = 1200px,
    # then expected card width = 1200 * 0.716 = 859px.
    # The cell height (which becomes card width after rotation) comes from img_rows.
    # For cards 06/07/08: ir = 2,1,0 respectively.

    # Let's just look at card 06 specifically: br=2, bc=0 -> ir=2, ic=0
    print("\nCard 06 (Swampert) detailed trace:")
    print("  Binder position: row=2, col=0")
    print("  Image position: ir=2 (bottom row), ic=0 (left col)")
    ir06, ic06 = 2, 0
    ry1_06, ry2_06 = row_bounds[ir06]
    rx1_06, rx2_06 = col_bounds[ic06]
    cell_h_06 = ry2_06 - ry1_06
    cell_w_06 = rx2_06 - rx1_06

    # ir=2 is last row -> pad_y_bot applied
    # ic=0 is first col -> pad_x_left applied
    pad_y_top_06 = 0  # not first row
    pad_y_bot_06 = int(cell_h_06 * pad_frac)  # last row = page edge
    pad_x_left_06 = int(cell_w_06 * pad_frac)  # first col = page edge
    pad_x_right_06 = 0  # not last col

    y1_06 = ry1_06 + pad_y_top_06
    y2_06 = ry2_06 - pad_y_bot_06
    x1_06 = rx1_06 + pad_x_left_06
    x2_06 = rx2_06 - pad_x_right_06

    eff_h_06 = y2_06 - y1_06
    eff_w_06 = x2_06 - x1_06

    print(f"  Cell bounds: Y=[{ry1_06},{ry2_06}]={cell_h_06}px, X=[{rx1_06},{rx2_06}]={cell_w_06}px")
    print(f"  Padding: top={pad_y_top_06}, bot={pad_y_bot_06}, left={pad_x_left_06}, right={pad_x_right_06}")
    print(f"  Effective: Y=[{y1_06},{y2_06}]={eff_h_06}px, X=[{x1_06},{x2_06}]={eff_w_06}px")
    print(f"  Cell is landscape: {eff_w_06 > eff_h_06} -> will be rotated CCW")
    print()
    print(f"  After CCW rotation:")
    print(f"    Card width  = cell height (Y extent) = {eff_h_06}px")
    print(f"    Card height = cell width  (X extent) = {eff_w_06}px")
    print(f"    Card aspect (w/h) = {eff_h_06/eff_w_06:.3f} (expected ~0.716)")
    print(f"    Card BOTTOM comes from image LEFT edge: x={x1_06} (padded {pad_x_left_06}px from x=0)")
    print(f"    Card TOP comes from image x={x2_06} (interior valley, no padding)")
    print()

    # Compare with card 00 (top-left binder position)
    # br=0, bc=0 -> ir=2, ic=2 with reverse_grid
    print("Card 00 for comparison:")
    ir00, ic00 = 2, 2
    ry1_00, ry2_00 = row_bounds[ir00]
    rx1_00, rx2_00 = col_bounds[ic00]
    cell_h_00 = ry2_00 - ry1_00
    cell_w_00 = rx2_00 - rx1_00

    pad_y_top_00 = 0
    pad_y_bot_00 = int(cell_h_00 * pad_frac)  # ir=2 is last row
    pad_x_left_00 = 0  # ic=2 is not first col
    pad_x_right_00 = int(cell_w_00 * pad_frac)  # ic=2 is last col

    y1_00 = ry1_00 + pad_y_top_00
    y2_00 = ry2_00 - pad_y_bot_00
    x1_00 = rx1_00 + pad_x_left_00
    x2_00 = rx2_00 - pad_x_right_00

    eff_h_00 = y2_00 - y1_00
    eff_w_00 = x2_00 - x1_00

    print(f"  Binder pos: row=0, col=0 -> image ir={ir00}, ic={ic00}")
    print(f"  Cell bounds: Y=[{ry1_00},{ry2_00}]={cell_h_00}px, X=[{rx1_00},{rx2_00}]={cell_w_00}px")
    print(f"  Padding: top={pad_y_top_00}, bot={pad_y_bot_00}, left={pad_x_left_00}, right={pad_x_right_00}")
    print(f"  Effective: Y=[{y1_00},{y2_00}]={eff_h_00}px, X=[{x1_00},{x2_00}]={eff_w_00}px")
    print(f"  Card height after CCW rotation = X extent = {eff_w_00}px")
    print(f"  Card BOTTOM comes from image RIGHT edge: x={x2_00} (padded {pad_x_right_00}px from x={rx2_00})")
    print()

    # The KEY comparison
    print("=" * 60)
    print("KEY FINDING:")
    print("=" * 60)
    print(f"Card 06 (bottom binder row): card height after rotation = {eff_w_06}px")
    print(f"  Card bottom from: image x={x1_06} (LEFT page edge + {pad_x_left_06}px padding)")
    print(f"Card 00 (top binder row):    card height after rotation = {eff_w_00}px")
    print(f"  Card bottom from: image x={x2_00} (RIGHT page edge - {pad_x_right_00}px padding)")
    print(f"Height difference: {eff_w_06 - eff_w_00}px")
    print()

    # For the bottom binder row (ic=0), the card's full width in image X is
    # from x=0+padding to x=valley. The padding eats into the card bottom.
    # But for the top binder row (ic=2), the card's full width in image X is
    # from x=valley to x=page_w-padding. The padding eats into the card bottom too.
    # So both should be symmetric IF the page detection is symmetric.
    #
    # The REAL question: is the perspective warp cutting off content on the left
    # side that should be visible? I.e., does the original image have card content
    # to the LEFT of where the warp source quad begins?

    print("Left edge of warp source quad (clamped):")
    print(f"  top-left x = {clamped[0][0]:.1f}")
    print(f"  bottom-left x = {clamped[3][0]:.1f}")
    print(f"  Average = {(clamped[0][0] + clamped[3][0])/2:.1f}px into the image")
    print(f"  This means ~{(clamped[0][0] + clamped[3][0])/2:.0f}px of the original image's")
    print(f"  left side is included in the warp (everything from x~{min(clamped[0][0], clamped[3][0]):.0f} onward)")
    print()

    print("Right edge of warp source quad (clamped):")
    print(f"  top-right x = {clamped[1][0]:.1f}")
    print(f"  bottom-right x = {clamped[2][0]:.1f}")
    print(f"  Image width = {w}")
    print(f"  Gap from right edge = {w - (clamped[1][0] + clamped[2][0])/2:.1f}px")
    print()

    # Compute: in the ORIGINAL image, where does the warp quad place x=0 and x=page_w?
    # Inverse warp: rectified coords -> original image coords
    M_inv = cv2.getPerspectiveTransform(dst, clamped)
    # Bottom-left of rectified (card bottom for ic=0) = (0, page_h-1)
    # Map a few points near x=0 in rectified back to original image
    test_pts = np.array([
        [0, page_h // 2],          # left edge, middle
        [pad_x_left_06, page_h // 2],  # after padding
        [page_w - 1, page_h // 2],  # right edge, middle
        [page_w - 1 - pad_x_right_00, page_h // 2],  # after right padding
    ], dtype=np.float32).reshape(-1, 1, 2)
    orig_pts = cv2.perspectiveTransform(test_pts, M_inv)

    print("Mapping rectified X coordinates back to original image:")
    print(f"  Rectified x=0 (left page edge) -> original x={orig_pts[0][0][0]:.1f}")
    print(f"  Rectified x={pad_x_left_06} (after left pad) -> original x={orig_pts[1][0][0]:.1f}")
    print(f"  Rectified x={page_w-1} (right page edge) -> original x={orig_pts[2][0][0]:.1f}")
    print(f"  Rectified x={page_w-1-pad_x_right_00} (after right pad) -> original x={orig_pts[3][0][0]:.1f}")

    # =================================================================
    # FINAL ROOT CAUSE ANALYSIS
    # =================================================================
    print()
    print("=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)
    print()
    print("The bottom-row clipping is caused by ASYMMETRIC column widths in the")
    print("rectified image, combined with the landscape CCW rotation.")
    print()
    print("Column widths in rectified image:")
    for i, (s, e) in enumerate(col_bounds):
        print(f"  Col {i}: [{s}, {e}] = {e - s}px")
    print()
    print(f"Col 0 (bottom binder row after rotation): {col_bounds[0][1] - col_bounds[0][0]}px")
    print(f"Col 2 (top binder row after rotation):    {col_bounds[2][1] - col_bounds[2][0]}px")
    print(f"Difference: {(col_bounds[2][1] - col_bounds[2][0]) - (col_bounds[0][1] - col_bounds[0][0])}px")
    print()
    print("After CCW rotation, each column's X-extent becomes the card's height.")
    print(f"Card 06 height (from col 0): {eff_w_06}px")
    print(f"Card 00 height (from col 2): {eff_w_00}px")
    print(f"Card 06 is {eff_w_00 - eff_w_06}px shorter ({(eff_w_00 - eff_w_06)/eff_w_00*100:.1f}%)")
    print()
    print("WHY col 0 is narrower:")
    print("  The perspective warp destination size is computed from the ORIGINAL")
    print("  (unexpanded) page corners. The source quad is the EXPANDED corners.")
    print(f"  Original page left edges: TL x={ordered[0][0]:.0f}, BL x={ordered[3][0]:.0f}")
    print(f"  Expanded (clamped) left:  TL x={clamped[0][0]:.0f}, BL x={clamped[3][0]:.0f}")
    print(f"  The warp maps the expanded source (wider) into the original-sized")
    print(f"  destination (narrower). This COMPRESSES all content horizontally.")
    print(f"  The compression is ~{(exp_dst_w - dst_w)/exp_dst_w*100:.1f}% overall, but it's")
    print(f"  ASYMMETRIC -- content near the edges (where expansion happened)")
    print(f"  gets compressed more than the center.")
    print()
    print("  Additionally, the first column valley is at x={v1}, but the page")
    print(f"  left edge is at x=0. After {pad_x_left_06}px padding, the usable")
    print(f"  width is only {eff_w_06}px vs {eff_w_00}px for col 2.".format(
        v1=col_bounds[0][1]))
    print()
    print("  The key asymmetry: col 2 spans [{s2}, {e2}] = {w2}px (includes page".format(
        s2=col_bounds[2][0], e2=col_bounds[2][1], w2=col_bounds[2][1]-col_bounds[2][0]))
    print(f"  right edge which is far from center), while col 0 spans")
    print(f"  [{s0}, {e0}] = {w0}px. The valleys at x={col_bounds[0][1]} and".format(
        s0=col_bounds[0][0], e0=col_bounds[0][1], w0=col_bounds[0][1]-col_bounds[0][0]))
    print(f"  x={col_bounds[1][1]} are NOT centered: they're shifted LEFT,")
    print(f"  making col 0 the narrowest and col 2 the widest.")
    print()
    print("CONTRIBUTING FACTORS (in order of impact):")
    print("  1. MAIN: Destination rect height computed from original corners")
    print(f"     while source uses expanded corners. Dst width={dst_w} but")
    print(f"     expanded source implies {exp_dst_w}. This loses {exp_dst_w-dst_w}px")
    print(f"     ({(exp_dst_w-dst_w)/exp_dst_w*100:.1f}%) of horizontal content.")
    print(f"  2. SECONDARY: Column valleys are not centered -- col 0 is {col_bounds[0][1]-col_bounds[0][0]}px")
    print(f"     vs col 2 is {col_bounds[2][1]-col_bounds[2][0]}px (real physical asymmetry in page).")
    print(f"  3. MINOR: Edge padding (pad_frac={pad_frac}) removes {pad_x_left_06}px")
    print(f"     from col 0 left edge and {pad_x_right_00}px from col 2 right edge.")
    print()
    print("POTENTIAL FIXES:")
    print("  A. Compute destination size from EXPANDED (clamped) corners instead of")
    print("     original corners. This preserves the extra content from expansion.")
    print("  B. Reduce/skip edge padding for the edge that maps to card BOTTOM after")
    print("     landscape rotation (currently padding trims card content, not binder frame).")
    print("  C. Use the column width ratio to detect asymmetry and extend the narrow")
    print("     column boundary outward to match the wider column.")


if __name__ == "__main__":
    main()
