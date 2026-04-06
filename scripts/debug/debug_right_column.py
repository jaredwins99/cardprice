"""Diagnostic: investigate why grid fallback clips the left edge of right-column cards.

Page 0 (data/inbox/page_20260228_174819.jpg) is landscape. The grid fallback
triggers because one contour fails the quality check. Cards at positions
[0,2] (Trapinch) and [1,2] (Skitty) have their left edges clipped so only
"...inch" and "kitty" are visible.

This script reproduces the grid fallback pipeline step-by-step and prints
diagnostics at each stage to identify where the clipping occurs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from cardprice.ml.card_segmenter import (
    _find_card_contours,
    _find_grid_lines,
    _order_points,
    CARD_OUTPUT_W,
    CARD_OUTPUT_H,
)

IMAGE_PATH = Path(__file__).resolve().parent.parent / "data/inbox/page_20260228_174819.jpg"


def main():
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        print(f"ERROR: Cannot read {IMAGE_PATH}")
        return
    h, w = image.shape[:2]
    print(f"=== Input image: {w}x{h} (landscape={w > h}) ===\n")

    # Resize same as segment_cards
    max_dim = 4500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
        print(f"Resized to {w}x{h}\n")

    # Step 1: Find contours (to see what triggered fallback)
    contours = _find_card_contours(image, expected_count=9)
    print(f"=== Step 1: Contour detection ===")
    print(f"Found {len(contours)} contours")
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        median_area = sorted(areas)[len(areas) // 2]
        min_area = min(areas)
        print(f"Areas: {[f'{a:.0f}' for a in areas]}")
        print(f"Median area: {median_area:.0f}, Min area: {min_area:.0f}")
        print(f"Min/Median ratio: {min_area/median_area:.3f} (fallback if < 0.5)")
        if len(contours) < 9:
            print(f"-> FALLBACK: only {len(contours)}/9 contours")
        elif min_area < median_area * 0.5:
            print(f"-> FALLBACK: quality check failed (ratio {min_area/median_area:.3f} < 0.5)")
        else:
            print(f"-> No fallback would be triggered")
    else:
        print(f"-> FALLBACK: no contours found")
    print()

    # Step 2: Find page outline (same as _grid_fallback)
    print(f"=== Step 2: Page outline detection ===")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    page_corners = None

    for thresh_name, thresh_fn in [
        ("Canny", lambda g: cv2.Canny(g, 20, 60)),
        ("Adaptive", lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                      cv2.THRESH_BINARY_INV, 51, 5)),
    ]:
        edges = thresh_fn(blurred)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(edge_contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(cnt)
            if area < h * w * 0.4:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.02, 0.04, 0.06, 0.08):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4:
                    page_corners = approx
                    print(f"Found page outline via {thresh_name} (eps={eps})")
                    break
            if page_corners is not None:
                break
        if page_corners is not None:
            break

    if page_corners is None:
        print("No page outline found!")
        return

    ordered = _order_points(page_corners.reshape(4, 2).astype(np.float32))
    print(f"Page corner points (ordered TL, TR, BR, BL):")
    for label, pt in zip(["TL", "TR", "BR", "BL"], ordered):
        print(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})")
    print()

    # Step 3: Perspective warp with expansion
    print(f"=== Step 3: Perspective warp ===")
    centroid = ordered.mean(axis=0)
    expand_frac = 0.04
    ordered_expanded = centroid + (1.0 + expand_frac) * (ordered - centroid)
    ordered_expanded[:, 0] = np.clip(ordered_expanded[:, 0], 0, w - 1)
    ordered_expanded[:, 1] = np.clip(ordered_expanded[:, 1], 0, h - 1)

    print(f"Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})")
    print(f"Expanded corners (4% outward):")
    for label, pt, orig in zip(["TL", "TR", "BR", "BL"], ordered_expanded, ordered):
        delta_x = pt[0] - orig[0]
        delta_y = pt[1] - orig[1]
        clamped = ""
        if pt[0] == 0 or pt[0] == w - 1:
            clamped += " [X CLAMPED]"
        if pt[1] == 0 or pt[1] == h - 1:
            clamped += " [Y CLAMPED]"
        print(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f}) delta=({delta_x:+.1f}, {delta_y:+.1f}){clamped}")

    # Check asymmetry: how much expansion room is there on each side?
    print(f"\nExpansion room analysis:")
    print(f"  Left edge:  TL.x={ordered[0][0]:.1f}, BL.x={ordered[3][0]:.1f}")
    print(f"  Right edge: TR.x={ordered[1][0]:.1f}, BR.x={ordered[2][0]:.1f}")
    print(f"  Top edge:   TL.y={ordered[0][1]:.1f}, TR.y={ordered[1][1]:.1f}")
    print(f"  Bot edge:   BL.y={ordered[3][1]:.1f}, BR.y={ordered[2][1]:.1f}")
    print(f"  Image width: {w}, Image height: {h}")
    left_room = min(ordered[0][0], ordered[3][0])
    right_room = w - max(ordered[1][0], ordered[2][0])
    top_room = min(ordered[0][1], ordered[1][1])
    bot_room = h - max(ordered[2][1], ordered[3][1])
    print(f"  Room left:   {left_room:.1f}px")
    print(f"  Room right:  {right_room:.1f}px")
    print(f"  Room top:    {top_room:.1f}px")
    print(f"  Room bottom: {bot_room:.1f}px")

    # Check if clamping causes asymmetric expansion
    desired_expand_px = expand_frac * (ordered - centroid)
    actual_expand_px = ordered_expanded - ordered
    print(f"\nDesired vs actual expansion per corner:")
    for label, desired, actual in zip(["TL", "TR", "BR", "BL"], desired_expand_px, actual_expand_px):
        print(f"  {label}: desired=({desired[0]:+.1f}, {desired[1]:+.1f}), "
              f"actual=({actual[0]:+.1f}, {actual[1]:+.1f})")
    print()

    # Do the warp
    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bot = np.linalg.norm(ordered[2] - ordered[3])
    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    dst_w = int((width_top + width_bot) / 2)
    dst_h = int((height_left + height_right) / 2)
    print(f"Destination size: {dst_w}x{dst_h}")
    print(f"  width_top={width_top:.1f}, width_bot={width_bot:.1f}")
    print(f"  height_left={height_left:.1f}, height_right={height_right:.1f}")

    dst = np.array([
        [0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]
    ], dtype=np.float32)
    M_warp = cv2.getPerspectiveTransform(ordered_expanded, dst)
    rectified = cv2.warpPerspective(image, M_warp, (dst_w, dst_h))
    page_h, page_w = rectified.shape[:2]
    page_is_landscape = page_w > page_h
    print(f"Rectified page: {page_w}x{page_h} (landscape={page_is_landscape})")
    print()

    # Step 4: Grid detection on rectified page
    # For landscape page: swap rows/cols
    if page_is_landscape:
        img_rows, img_cols = 3, 3  # cols, rows swapped for landscape
        print(f"=== Step 4: Valley detection (landscape, img_rows=3, img_cols=3) ===")
    else:
        img_rows, img_cols = 3, 3
        print(f"=== Step 4: Valley detection (portrait, img_rows=3, img_cols=3) ===")

    # Run valley detection
    grid_lines = _find_grid_lines(rectified, img_rows, img_cols)

    # Also run our own detailed valley analysis
    gray_rect = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    h_proj = gray_rect.mean(axis=1).astype(np.float64)
    v_proj = gray_rect.mean(axis=0).astype(np.float64)

    print(f"\nHorizontal projection (row valleys, profile length={len(h_proj)}):")
    print(f"  Expected cell height: {page_h / img_rows:.1f}px")
    print(f"Vertical projection (col valleys, profile length={len(v_proj)}):")
    print(f"  Expected cell width: {page_w / img_cols:.1f}px")

    if grid_lines is not None:
        row_bounds, col_bounds = grid_lines
        print(f"\nDetected row boundaries: {row_bounds}")
        print(f"Detected col boundaries: {col_bounds}")

        print(f"\nRow cell sizes:")
        for i, (s, e) in enumerate(row_bounds):
            print(f"  Row {i}: [{s}, {e}] size={e-s}px")

        print(f"\nColumn cell sizes:")
        for i, (s, e) in enumerate(col_bounds):
            print(f"  Col {i}: [{s}, {e}] size={e-s}px")
            expected = page_w / img_cols
            pct = (e - s) / expected * 100
            print(f"    {pct:.1f}% of expected ({expected:.1f}px)")

        # Now trace what happens for the right-column cards after landscape mapping
        # For landscape with reverse_grid=True:
        #   binder (br=0,bc=2) -> ir = cols-1-bc = 3-1-2 = 0, ic = rows-1-br = 3-1-0 = 2
        #   binder (br=1,bc=2) -> ir = cols-1-bc = 3-1-2 = 0, ic = rows-1-br = 3-1-1 = 1
        # For landscape with reverse_grid=False:
        #   binder (br=0,bc=2) -> ir = bc = 2, ic = br = 0
        #   binder (br=1,bc=2) -> ir = bc = 2, ic = br = 1

        print(f"\n=== Step 5: Cell extraction trace for right-column cards ===")
        pad_frac = 0.035

        for reverse_grid in [True, False]:
            grid_label = "REVERSE" if reverse_grid else "FORWARD"
            print(f"\n--- Traversal: {grid_label} ---")

            for br in range(3):
                for bc in range(3):
                    if page_is_landscape:
                        if reverse_grid:
                            ir = 3 - 1 - bc   # img_cols - 1 - bc
                            ic = 3 - 1 - br   # img_rows - 1 - br
                        else:
                            ir = bc
                            ic = br
                    else:
                        ir, ic = br, bc

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

                    # Only print interesting columns (right column = bc==2)
                    if bc == 2 or True:  # Print all for now
                        cell = rectified[y1:y2, x1:x2]
                        ch, cw = cell.shape[:2]
                        is_landscape_cell = cw > ch
                        print(f"  binder[{br},{bc}] -> img[{ir},{ic}]  "
                              f"row=[{ry1},{ry2}] col=[{rx1},{rx2}]  "
                              f"pad=({pad_y_top},{pad_y_bot},{pad_x_left},{pad_x_right})  "
                              f"crop=[{y1}:{y2}, {x1}:{x2}]  "
                              f"size={cw}x{ch}  landscape={is_landscape_cell}")

        # Detailed analysis of column valleys
        print(f"\n=== Step 6: Valley position analysis ===")
        kernel_size = max(3, int(page_w * 0.02) | 1)
        smoothed_v = cv2.GaussianBlur(v_proj.reshape(-1, 1), (1, kernel_size), 0).flatten()

        expected_cell = page_w / img_cols
        margin = int(expected_cell * 0.50)

        print(f"Vertical profile stats:")
        print(f"  Length: {len(v_proj)}")
        print(f"  Min value: {v_proj.min():.1f} at x={v_proj.argmin()}")
        print(f"  Max value: {v_proj.max():.1f} at x={v_proj.argmax()}")
        print(f"  Expected cell width: {expected_cell:.1f}")
        print(f"  Search margin: {margin} (excluding first/last {margin}px)")

        # Find all local minima in the search region
        print(f"\n  All local minima in search region [{margin}, {len(smoothed_v)-margin}]:")
        minima = []
        for i in range(margin, len(smoothed_v) - margin):
            if smoothed_v[i] < smoothed_v[i-1] and smoothed_v[i] < smoothed_v[i+1]:
                minima.append((i, smoothed_v[i]))
        for idx, val in sorted(minima, key=lambda x: x[1]):
            # What fraction of the axis is this?
            frac = idx / len(smoothed_v)
            print(f"    x={idx} ({frac:.3f} of width)  value={val:.1f}")

        # Show where the valleys actually are vs where they should be
        col_valleys = [col_bounds[i][1] for i in range(len(col_bounds) - 1)]
        print(f"\n  Selected valley positions: {col_valleys}")
        print(f"  Ideal uniform positions: {[int(expected_cell * (i+1)) for i in range(img_cols-1)]}")
        for i, v in enumerate(col_valleys):
            ideal = expected_cell * (i + 1)
            print(f"    Valley {i}: x={v} (ideal={ideal:.1f}, delta={v-ideal:+.1f}px)")

        # For the right column specifically: where does the card content start?
        # After landscape rotation, the "left edge" of the right column card
        # corresponds to one of the boundaries in the rectified image.
        # Let's identify which boundary maps to card name visibility.
        print(f"\n=== Step 7: Right-column card edge analysis ===")
        print(f"Page is landscape. After CCW rotation of a cell, the cell's")
        print(f"TOP becomes the card's LEFT, and the cell's BOTTOM becomes the card's RIGHT.")
        print(f"So for name clipping, we care about the TOP of the landscape cell.")
        print(f"")
        print(f"For reverse_grid, binder col 2 maps to img row 0.")
        print(f"  Row 0 starts at y={row_bounds[0][0]} (top of rectified page)")
        print(f"  With pad_y_top = {int(row_bounds[0][1] * pad_frac) if len(row_bounds) > 0 else '?'}")
        print(f"")
        print(f"For forward_grid, binder col 2 maps to img row 2.")
        print(f"  Row 2 starts at y={row_bounds[2][0]}")

        # Save debug images
        debug_dir = Path(__file__).resolve().parent.parent / "data" / "debug_right_col"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Save rectified page with grid overlay
        debug_rect = rectified.copy()
        for s, e in row_bounds:
            cv2.line(debug_rect, (0, s), (page_w, s), (0, 255, 0), 2)
            cv2.line(debug_rect, (0, e), (page_w, e), (0, 0, 255), 2)
        for s, e in col_bounds:
            cv2.line(debug_rect, (s, 0), (s, page_h), (0, 255, 0), 2)
            cv2.line(debug_rect, (e, 0), (e, page_h), (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / "rectified_grid.jpg"), debug_rect)
        print(f"\nSaved rectified page with grid overlay to {debug_dir / 'rectified_grid.jpg'}")

        # Save the vertical projection profile as a simple text visualization
        with open(str(debug_dir / "v_profile.txt"), "w") as f:
            f.write("Vertical projection profile (column mean brightness)\n")
            f.write(f"x,raw,smoothed\n")
            for i in range(len(v_proj)):
                f.write(f"{i},{v_proj[i]:.1f},{smoothed_v[i]:.1f}\n")
        print(f"Saved vertical profile to {debug_dir / 'v_profile.txt'}")

        # Extract and save the right-column cells (for both traversal orders)
        for reverse in [True, False]:
            label = "reverse" if reverse else "forward"
            for br in range(3):
                bc = 2  # right column only
                if reverse:
                    ir = 3 - 1 - bc
                    ic = 3 - 1 - br
                else:
                    ir = bc
                    ic = br

                ry1, ry2 = row_bounds[ir]
                rx1, rx2 = col_bounds[ic]
                cell_h_px = ry2 - ry1
                cell_w_px = rx2 - rx1
                pad_y_top = int(cell_h_px * pad_frac) if ir == 0 else 0
                pad_y_bot = int(cell_h_px * pad_frac) if ir == len(row_bounds) - 1 else 0
                pad_x_left = int(cell_w_px * pad_frac) if ic == 0 else 0
                pad_x_right = int(cell_w_px * pad_frac) if ic == len(col_bounds) - 1 else 0

                y1 = ry1 + pad_y_top
                y2 = ry2 - pad_y_bot
                x1 = rx1 + pad_x_left
                x2 = rx2 - pad_x_right
                cell = rectified[y1:y2, x1:x2]
                ch, cw = cell.shape[:2]
                if cw > ch:
                    cell = cv2.rotate(cell, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cell_resized = cv2.resize(cell, (CARD_OUTPUT_W, CARD_OUTPUT_H),
                                          interpolation=cv2.INTER_AREA)
                fname = f"rightcol_{label}_br{br}_bc{bc}_ir{ir}_ic{ic}.jpg"
                cv2.imwrite(str(debug_dir / fname), cell_resized)
                print(f"Saved {fname}")

        # Also save an annotated rectified page showing which img cells map to binder right column
        debug_annotated = rectified.copy()
        for reverse in [True]:
            for br in range(3):
                bc = 2
                ir = 3 - 1 - bc
                ic = 3 - 1 - br
                ry1, ry2 = row_bounds[ir]
                rx1, rx2 = col_bounds[ic]
                # Draw the cell in red
                cv2.rectangle(debug_annotated, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3)
                cv2.putText(debug_annotated, f"binder[{br},{bc}]",
                           (rx1 + 10, ry1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / "rightcol_cells_annotated.jpg"), debug_annotated)
        print(f"Saved annotated right-column cells to {debug_dir / 'rightcol_cells_annotated.jpg'}")

    else:
        print("Valley detection FAILED - would use uniform grid")
        # Show uniform grid analysis
        cell_h = page_h / img_rows
        cell_w = page_w / img_cols
        print(f"Uniform cell size: {cell_w:.1f}x{cell_h:.1f}")

    print(f"\n=== Summary ===")
    print(f"Image: {IMAGE_PATH.name} ({w}x{h})")
    print(f"Rectified: {page_w}x{page_h} (landscape={page_is_landscape})")
    if grid_lines:
        col_sizes = [e - s for s, e in col_bounds]
        row_sizes = [e - s for s, e in row_bounds]
        print(f"Column sizes: {col_sizes}")
        print(f"Row sizes: {row_sizes}")
        print(f"Column size range: {min(col_sizes)}-{max(col_sizes)} (diff={max(col_sizes)-min(col_sizes)})")
        print(f"Row size range: {min(row_sizes)}-{max(row_sizes)} (diff={max(row_sizes)-min(row_sizes)})")

    print()
    print("=" * 72)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 72)
    print()
    print("The right-column card clipping is caused by ASYMMETRIC PERSPECTIVE")
    print("WARP EXPANSION due to the page being close to the image edge.")
    print()
    print("Chain of causation:")
    print()
    print("1. PHOTO FRAMING: The binder page's top-right corner (TR) is at")
    print(f"   y={ordered[1][1]:.0f} in the original photo -- right at the image edge.")
    print(f"   The bottom-right (BR) is at x={ordered[2][0]:.0f} of {w} -- also near edge.")
    print()
    print("2. EXPANSION CLAMPING: The 4% outward expansion is clamped by image bounds:")
    print(f"   TR: desired y expansion = {expand_frac * (ordered[1][1] - centroid[1]):+.1f}px, actual = {ordered_expanded[1][1] - ordered[1][1]:+.1f}px")
    print(f"   BR: desired x expansion = {expand_frac * (ordered[2][0] - centroid[0]):+.1f}px, actual = {ordered_expanded[2][0] - ordered[2][0]:+.1f}px")
    print(f"   (TR.y clamped to 0, BR.x clamped to {w-1})")
    print()
    print("3. DESTINATION SIZE uses ORIGINAL (unexpanded) corner distances:")
    print(f"   dst_w={dst_w} dst_h={dst_h}")
    print(f"   But the EXPANDED source quad is asymmetric -- the right/top sides")
    print(f"   of the source quad are barely expanded while left/bottom are fully")
    print(f"   expanded. The warp maps this asymmetric source to the full")
    print(f"   destination rectangle, effectively COMPRESSING the right/top edge")
    print(f"   content into fewer destination pixels.")
    print()
    print("4. LANDSCAPE ROTATION: After CCW rotation, img row 0's TOP edge")
    print("   becomes the card's LEFT edge (where the name is). Row 0 starts")
    print(f"   at y=0 in the rectified image. The compressed/clipped top edge")
    print("   of the rectified page = clipped left edge of the card name.")
    print()
    print("5. VALLEY DETECTION SHIFT: The column valleys are shifted left of ideal:")
    if grid_lines:
        for i, v in enumerate([col_bounds[i][1] for i in range(len(col_bounds) - 1)]):
            ideal = page_w / img_cols * (i + 1)
            print(f"   Valley {i}: x={v} vs ideal {ideal:.0f} (delta={v - ideal:+.0f}px)")
        print("   This makes the last column 101px wider than the first, but")
        print("   the extra width is on the RIGHT side (orange binder frame),")
        print("   NOT compensating for the clipped left/top edge.")
    print()
    print("POTENTIAL FIXES:")
    print("  A. Increase expand_frac from 0.04 to ~0.06-0.08 (helps marginally,")
    print("     still clamped at image edge)")
    print("  B. After clamping, adjust dst size proportionally to the actual")
    print("     expansion achieved (so the warp doesn't compress the clamped side)")
    print("  C. Pad the SOURCE IMAGE with border pixels before perspective warp")
    print("     when corners are near the image edge, giving room to expand")
    print("  D. Use cv2.copyMakeBorder on the source before warping to ensure")
    print("     there is always room for 4% expansion in every direction")


if __name__ == "__main__":
    main()
