#!/usr/bin/env python3
"""Diagnostic script for fire page segmentation.

Reproduces the grid fallback pipeline on page_20260307_020047.jpg,
dumps the horizontal/vertical projection profiles, detected valleys,
and saves the rectified page + all 9 segments to /tmp/fire_page_debug/.
"""

import sys
from pathlib import Path
from itertools import combinations

import cv2
import numpy as np

OUT_DIR = Path("/tmp/fire_page_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT = Path("data/inbox/page_20260307_020047.jpg")

CARD_OUTPUT_W = 1008
CARD_OUTPUT_H = 1530


def order_points(pts):
    """Order 4 points: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def find_page_corners(image):
    """Find the binder page outline corners."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    for thresh_fn in [
        lambda g: cv2.Canny(g, 20, 60),
        lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 51, 5),
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
                    return approx
    return None


def rectify_page(image, page_corners):
    """Perspective warp the page with expand_frac=0.02."""
    h, w = image.shape[:2]
    ordered = order_points(page_corners.reshape(4, 2).astype(np.float32))

    centroid = ordered.mean(axis=0)
    expand_frac = 0.02
    ordered_expanded = centroid + (1.0 + expand_frac) * (ordered - centroid)

    pad_needed = int(max(w, h) * expand_frac) + 10
    padded_image = cv2.copyMakeBorder(
        image, pad_needed, pad_needed, pad_needed, pad_needed,
        cv2.BORDER_REPLICATE,
    )
    ordered_expanded += pad_needed

    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bot = np.linalg.norm(ordered[2] - ordered[3])
    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    dst_w = int((width_top + width_bot) / 2)
    dst_h = int((height_left + height_right) / 2)

    dst = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1],
    ], dtype=np.float32)
    M_warp = cv2.getPerspectiveTransform(ordered_expanded, dst)
    rectified = cv2.warpPerspective(padded_image, M_warp, (dst_w, dst_h))
    return rectified, ordered


def find_valleys_detailed(profile, n_cells, axis_len, axis_name=""):
    """Reproduce _find_grid_lines valley detection with full diagnostics."""
    if n_cells <= 1:
        return [], {}

    kernel_size = max(3, int(axis_len * 0.02) | 1)
    smoothed = cv2.GaussianBlur(profile.reshape(-1, 1),
                                 (1, kernel_size), 0).flatten()
    expected_cell = axis_len / n_cells
    margin = int(expected_cell * 0.50)

    print(f"\n{'='*60}")
    print(f"  Valley detection for {axis_name}")
    print(f"{'='*60}")
    print(f"  axis_len={axis_len}  n_cells={n_cells}  expected_cell={expected_cell:.1f}")
    print(f"  kernel_size={kernel_size}  margin={margin}")

    # Find all local minima
    minima_idx = []
    minima_val = []
    for i in range(margin, len(smoothed) - margin):
        if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
            minima_idx.append(i)
            minima_val.append(smoothed[i])

    print(f"  Found {len(minima_idx)} local minima in range [{margin}, {axis_len - margin}]")

    # Score each minimum
    neighbourhood = int(expected_cell * 0.3)
    scored = []
    for idx, val in zip(minima_idx, minima_val):
        lo = max(0, idx - neighbourhood)
        hi = min(len(smoothed), idx + neighbourhood)
        local_mean = smoothed[lo:hi].mean()
        depth = local_mean - val
        scored.append((idx, depth, val))

    # Show all minima sorted by depth
    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  All minima (sorted by depth):")
    for i, (idx, depth, val) in enumerate(scored[:15]):
        pct = idx / axis_len * 100
        print(f"    #{i+1}: pos={idx} ({pct:.1f}%)  depth={depth:.2f}  brightness={val:.1f}")

    # Top candidates
    n_needed = n_cells - 1
    top_k = min(len(scored), max(n_needed * 3, 8))
    candidates = [(idx, depth) for idx, depth, _ in scored[:top_k]]

    # Find best combination
    min_spacing = expected_cell * 0.4
    max_depth = candidates[0][1] if candidates[0][1] > 0 else 1.0

    best_combo = None
    best_score = -float("inf")
    all_combos = []

    for combo in combinations(range(len(candidates)), n_needed):
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

        total_depth = sum(depths) / max_depth
        size_std = float(np.std(cell_sizes))
        size_penalty = size_std / expected_cell
        score = total_depth - 1.5 * size_penalty

        all_combos.append((idxs, cell_sizes, score, total_depth, size_penalty))

        if score > best_score:
            best_score = score
            best_combo = idxs

    # Show top combos
    all_combos.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  Top valley combinations (of {len(all_combos)} valid):")
    for i, (idxs, sizes, score, tdepth, spenalty) in enumerate(all_combos[:5]):
        bounds = [(0, idxs[0])] + [(idxs[j], idxs[j+1]) for j in range(len(idxs)-1)] + [(idxs[-1], axis_len)]
        print(f"    #{i+1}: valleys={idxs}  sizes={sizes}  score={score:.3f}  "
              f"depth={tdepth:.3f}  penalty={spenalty:.3f}")
        print(f"          bounds={bounds}")

    if best_combo is not None:
        # Refinement
        refine_radius = int(expected_cell * 0.04)
        refined = []
        for v in best_combo:
            lo = max(0, v - refine_radius)
            hi = min(axis_len, v + refine_radius + 1)
            local_min_offset = int(np.argmin(smoothed[lo:hi]))
            refined.append(lo + local_min_offset)
        print(f"\n  SELECTED: {best_combo} -> refined: {refined}")

        edges = [0] + refined + [axis_len]
        final_bounds = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        final_sizes = [e - s for s, e in final_bounds]
        print(f"  Final bounds: {final_bounds}")
        print(f"  Final sizes: {final_sizes}")
        print(f"  Size ratios: {[s / min(final_sizes) for s in final_sizes]}")

        return refined, {
            "smoothed": smoothed,
            "minima": scored,
            "candidates": candidates,
            "all_combos": all_combos,
        }
    else:
        print("\n  NO VALID COMBINATION FOUND")
        return None, {"smoothed": smoothed, "minima": scored}


def save_projection_plot(profile, smoothed, valleys, axis_name, filename):
    """Save a visual plot of the projection profile as an image."""
    h_plot = 400
    w_plot = len(profile)

    # Normalize to 0-1
    pmin, pmax = profile.min(), profile.max()
    norm_p = (profile - pmin) / (pmax - pmin + 1e-6)
    norm_s = (smoothed - pmin) / (pmax - pmin + 1e-6)

    img = np.ones((h_plot, w_plot, 3), dtype=np.uint8) * 255

    # Draw raw profile in light gray
    for x in range(w_plot - 1):
        y1 = int((1 - norm_p[x]) * (h_plot - 1))
        y2 = int((1 - norm_p[x + 1]) * (h_plot - 1))
        cv2.line(img, (x, y1), (x + 1, y2), (200, 200, 200), 1)

    # Draw smoothed profile in blue
    for x in range(w_plot - 1):
        y1 = int((1 - norm_s[x]) * (h_plot - 1))
        y2 = int((1 - norm_s[x + 1]) * (h_plot - 1))
        cv2.line(img, (x, y1), (x + 1, y2), (255, 0, 0), 2)

    # Draw valleys in red
    if valleys:
        for v in valleys:
            cv2.line(img, (v, 0), (v, h_plot), (0, 0, 255), 2)
            cv2.putText(img, str(v), (v + 5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(str(OUT_DIR / filename), img)
    print(f"  Saved projection plot: {filename}")


def main():
    image = cv2.imread(str(INPUT))
    if image is None:
        print(f"ERROR: Cannot load {INPUT}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"Input image: {w}x{h}")

    # Step 1: Find page corners
    page_corners = find_page_corners(image)
    if page_corners is None:
        print("ERROR: No page corners found")
        sys.exit(1)

    print(f"Page corners found: {page_corners.reshape(4, 2).tolist()}")

    # Step 2: Rectify
    rectified, ordered = rectify_page(image, page_corners)
    rh, rw = rectified.shape[:2]
    print(f"Rectified page: {rw}x{rh}")
    cv2.imwrite(str(OUT_DIR / "rectified.jpg"), rectified)
    print(f"Saved: rectified.jpg")

    # Step 3: Determine orientation
    page_is_landscape = rw > rh
    print(f"Landscape: {page_is_landscape}")

    if page_is_landscape:
        img_rows, img_cols = 3, 3  # cols, rows swapped
    else:
        img_rows, img_cols = 3, 3

    # Step 4: Projection profiles
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    h_proj = gray.mean(axis=1).astype(np.float64)
    v_proj = gray.mean(axis=0).astype(np.float64)

    print(f"\nProjection profile stats:")
    print(f"  H-proj (rows): len={len(h_proj)}, min={h_proj.min():.1f}, max={h_proj.max():.1f}")
    print(f"  V-proj (cols): len={len(v_proj)}, min={v_proj.min():.1f}, max={v_proj.max():.1f}")

    # Step 5: Valley detection with diagnostics
    row_valleys, row_info = find_valleys_detailed(h_proj, img_rows, rh, "ROWS (horizontal projection)")
    col_valleys, col_info = find_valleys_detailed(v_proj, img_cols, rw, "COLS (vertical projection)")

    # Save projection plots
    if "smoothed" in row_info:
        save_projection_plot(h_proj, row_info["smoothed"], row_valleys, "rows",
                           "h_projection.png")
    if "smoothed" in col_info:
        save_projection_plot(v_proj, col_info["smoothed"], col_valleys, "cols",
                           "v_projection.png")

    # Step 6: Build boundaries
    if row_valleys and col_valleys:
        row_edges = [0] + row_valleys + [rh]
        col_edges = [0] + col_valleys + [rw]
    else:
        print("\nValley detection failed, using uniform grid")
        cell_h = rh / img_rows
        cell_w = rw / img_cols
        row_edges = [int(r * cell_h) for r in range(img_rows + 1)]
        col_edges = [int(c * cell_w) for c in range(img_cols + 1)]

    row_bounds = [(row_edges[i], row_edges[i + 1]) for i in range(len(row_edges) - 1)]
    col_bounds = [(col_edges[i], col_edges[i + 1]) for i in range(len(col_edges) - 1)]

    print(f"\nFinal row_bounds: {row_bounds}")
    print(f"Final col_bounds: {col_bounds}")
    print(f"Row sizes: {[e - s for s, e in row_bounds]}")
    print(f"Col sizes: {[e - s for s, e in col_bounds]}")

    # Step 7: Draw grid lines on rectified image
    annotated = rectified.copy()
    for v in (row_valleys or []):
        cv2.line(annotated, (0, v), (rw, v), (0, 0, 255), 3)
        cv2.putText(annotated, f"y={v}", (10, v - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    for v in (col_valleys or []):
        cv2.line(annotated, (v, 0), (v, rh), (0, 255, 0), 3)
        cv2.putText(annotated, f"x={v}", (v + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(str(OUT_DIR / "rectified_annotated.jpg"), annotated)
    print(f"Saved: rectified_annotated.jpg")

    # Step 8: Extract and save segments
    pad_frac = 0.035
    interior_frac = pad_frac * 0.4

    # Determine traversal order (assume landscape CCW = reverse for now)
    if page_is_landscape:
        cell_rot = cv2.ROTATE_90_COUNTERCLOCKWISE
        reverse_grid = True
    else:
        cell_rot = None
        reverse_grid = False

    for br in range(3):
        for bc in range(3):
            if page_is_landscape:
                if reverse_grid:
                    ir = 3 - 1 - bc
                    ic = 3 - 1 - br
                else:
                    ir = bc
                    ic = br
            else:
                ir, ic = br, bc

            ry1, ry2 = row_bounds[ir]
            rx1, rx2 = col_bounds[ic]
            cell_h_px = ry2 - ry1
            cell_w_px = rx2 - rx1

            pad_y_top = int(cell_h_px * pad_frac) if ir == 0 else int(cell_h_px * interior_frac)
            pad_y_bot = int(cell_h_px * pad_frac) if ir == len(row_bounds) - 1 else int(cell_h_px * interior_frac)
            pad_x_left = int(cell_w_px * pad_frac) if ic == 0 else int(cell_w_px * interior_frac)
            pad_x_right = int(cell_w_px * pad_frac) if ic == len(col_bounds) - 1 else int(cell_w_px * interior_frac)

            y1 = ry1 + pad_y_top
            y2 = ry2 - pad_y_bot
            x1 = rx1 + pad_x_left
            x2 = rx2 - pad_x_right
            cell = rectified[y1:y2, x1:x2]

            # Rotate if landscape
            ch, cw = cell.shape[:2]
            if cw > ch and cell_rot is not None:
                cell = cv2.rotate(cell, cell_rot)

            # Resize to standard output
            cell = cv2.resize(cell, (CARD_OUTPUT_W, CARD_OUTPUT_H))

            slot = br * 3 + bc
            fname = f"segment_{slot}_r{br}c{bc}.jpg"
            cv2.imwrite(str(OUT_DIR / fname), cell)
            print(f"  Saved segment {slot} (binder r{br}c{bc}) <- grid cell ir={ir},ic={ic}  "
                  f"region=[{y1}:{y2}, {x1}:{x2}]  raw_size={x2-x1}x{y2-y1}")


if __name__ == "__main__":
    main()
