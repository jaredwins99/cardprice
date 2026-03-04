"""Test valley-based grid vs uniform grid on all 3 binder page photos.

Usage:
    python scripts/test_valley_grid.py
"""
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("grid_test")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cardprice.ml.card_segmenter import (
    _find_card_contours, _find_grid_lines, _grid_fallback,
    CARD_OUTPUT_W, CARD_OUTPUT_H,
)

TEST_IMAGES = [
    "data/inbox/page_20260228_174819.jpg",
    "data/inbox/page_20260228_195512.jpg",
    "data/inbox/page_20260228_202134.jpg",
]


def compute_card_content_score(cell_img):
    """Score how well a cell captures a card (higher laplacian = more detail,
    lower border edges = less binder material bleeding in)."""
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Variance of Laplacian (focus/detail measure) for center 60%
    margin_y = int(h * 0.2)
    margin_x = int(w * 0.2)
    center = gray[margin_y:h - margin_y, margin_x:w - margin_x]
    laplacian_var = cv2.Laplacian(center, cv2.CV_64F).var()

    # Edge density in border strips (top/bottom/left/right 10%)
    border_top = gray[:int(h * 0.1), :]
    border_bot = gray[int(h * 0.9):, :]
    border_left = gray[:, :int(w * 0.1)]
    border_right = gray[:, int(w * 0.9):]

    border_edges = sum(
        cv2.Canny(b, 50, 150).mean()
        for b in [border_top, border_bot, border_left, border_right]
    ) / 4

    # Color saturation in center (cards are colorful, binder is gray/orange)
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    center_sat = hsv[margin_y:h - margin_y, margin_x:w - margin_x, 1].mean()

    return {
        "laplacian_var": laplacian_var,
        "border_edge_density": border_edges,
        "center_saturation": center_sat,
    }


def detect_page_region(image):
    """Find the binder page bounding box (same logic as _grid_fallback)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ih, iw = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    page_corners = None

    for thresh_fn in [
        lambda g: cv2.Canny(g, 20, 60),
        lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 51, 5),
    ]:
        edges = thresh_fn(blurred)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(cnt)
            if area < ih * iw * 0.4:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.02, 0.04, 0.06, 0.08):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4:
                    page_corners = approx
                    break
            if page_corners is not None:
                break
        if page_corners is not None:
            break

    if page_corners is not None:
        bx, by, bw, bh = cv2.boundingRect(page_corners)
        bx = max(0, bx)
        by = max(0, by)
        bx2 = min(iw, bx + bw)
        by2 = min(ih, by + bh)
        return image[by:by2, bx:bx2]
    return image


def run_grid_comparison(image_path):
    """Run both uniform and valley grid on an image, compare results."""
    print("=" * 60)
    print(f"Testing: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ERROR: Could not read {image_path}")
        return None

    h, w = image.shape[:2]
    max_dim = 3000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)

    contours = _find_card_contours(image)
    print(f"  Contour detection found {len(contours)} cards")

    rectified = detect_page_region(image)
    page_h, page_w = rectified.shape[:2]
    page_is_landscape = page_w > page_h
    img_rows, img_cols = (3, 3) if not page_is_landscape else (3, 3)

    print(f"  Page: {page_w}x{page_h}, landscape={page_is_landscape}")

    # Uniform grid
    cell_h = page_h / img_rows
    cell_w = page_w / img_cols
    uniform_bounds_r = [(int(r * cell_h), int((r + 1) * cell_h)) for r in range(img_rows)]
    uniform_bounds_c = [(int(c * cell_w), int((c + 1) * cell_w)) for c in range(img_cols)]

    # Valley grid
    valley_result = _find_grid_lines(rectified, img_rows, img_cols)

    if valley_result is None:
        print("  Valley detection FAILED -- would use uniform fallback")
        valley_bounds_r = uniform_bounds_r
        valley_bounds_c = uniform_bounds_c
        valley_succeeded = False
    else:
        valley_bounds_r, valley_bounds_c = valley_result
        valley_succeeded = True

    # Compare cell quality
    pad_frac = 0.06

    print(f"\n  {'Cell':<6} {'Method':<8} {'LaplVar':>8} {'BdrEdge':>8} {'CtrSat':>7} {'CellSize'}")
    print("  " + "-" * 62)

    summary = {}
    for method_name, rb, cb in [("UNIFORM", uniform_bounds_r, uniform_bounds_c),
                                 ("VALLEY", valley_bounds_r, valley_bounds_c)]:
        scores = []
        for ir in range(img_rows):
            for ic in range(img_cols):
                ry1, ry2 = rb[ir]
                rx1, rx2 = cb[ic]
                ch = ry2 - ry1
                cw = rx2 - rx1
                py = int(ch * pad_frac)
                px = int(cw * pad_frac)
                cell = rectified[ry1 + py:ry2 - py, rx1 + px:rx2 - px]

                if cell.size == 0:
                    continue

                cell_resized = cv2.resize(cell, (CARD_OUTPUT_W, CARD_OUTPUT_H),
                                          interpolation=cv2.INTER_AREA)
                sc = compute_card_content_score(cell_resized)
                cell_idx = ir * img_cols + ic
                print(f"  [{cell_idx}]  {method_name:<8} {sc['laplacian_var']:8.1f} "
                      f"{sc['border_edge_density']:8.1f} {sc['center_saturation']:7.1f} "
                      f"{cw}x{ch}")
                scores.append(sc)

        if scores:
            avg_lap = np.mean([s["laplacian_var"] for s in scores])
            avg_bdr = np.mean([s["border_edge_density"] for s in scores])
            avg_sat = np.mean([s["center_saturation"] for s in scores])
            print(f"  AVG   {method_name:<8} {avg_lap:8.1f} {avg_bdr:8.1f} {avg_sat:7.1f}")
            summary[method_name] = {"laplacian": avg_lap, "border": avg_bdr, "saturation": avg_sat}

    # Show boundary differences
    if valley_succeeded:
        print(f"\n  Row boundaries comparison:")
        for i in range(img_rows):
            u_s, u_e = uniform_bounds_r[i]
            v_s, v_e = valley_bounds_r[i]
            diff_s = v_s - u_s
            diff_e = v_e - u_e
            print(f"    Row {i}: uniform=({u_s},{u_e}) valley=({v_s},{v_e}) "
                  f"delta_start={diff_s:+d} delta_end={diff_e:+d}")

        print(f"\n  Col boundaries comparison:")
        for i in range(img_cols):
            u_s, u_e = uniform_bounds_c[i]
            v_s, v_e = valley_bounds_c[i]
            diff_s = v_s - u_s
            diff_e = v_e - u_e
            print(f"    Col {i}: uniform=({u_s},{u_e}) valley=({v_s},{v_e}) "
                  f"delta_start={diff_s:+d} delta_end={diff_e:+d}")

    print()
    return valley_succeeded, summary


def main():
    print("=" * 60)
    print("VALLEY-BASED GRID vs UNIFORM GRID COMPARISON")
    print("=" * 60)

    base = Path(__file__).resolve().parent.parent
    results = []
    for img_rel in TEST_IMAGES:
        img_path = base / img_rel
        if not img_path.exists():
            print(f"  SKIP: {img_path} not found")
            continue
        result = run_grid_comparison(str(img_path))
        results.append((img_path.name, result))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results:
        if result is None:
            print(f"  {name}: ERROR")
        else:
            succeeded, summary = result
            status = "VALLEY" if succeeded else "UNIFORM (valley failed)"
            print(f"  {name}: {status}")
            if "VALLEY" in summary and "UNIFORM" in summary:
                u = summary["UNIFORM"]
                v = summary["VALLEY"]
                lap_diff = v["laplacian"] - u["laplacian"]
                bdr_diff = v["border"] - u["border"]
                sat_diff = v["saturation"] - u["saturation"]
                print(f"    Laplacian variance: {u['laplacian']:.1f} -> {v['laplacian']:.1f} ({lap_diff:+.1f})")
                print(f"    Border edge density: {u['border']:.1f} -> {v['border']:.1f} ({bdr_diff:+.1f})")
                print(f"    Center saturation: {u['saturation']:.1f} -> {v['saturation']:.1f} ({sat_diff:+.1f})")
                # Higher laplacian = more detail (good)
                # Lower border = less binder bleed (good)
                # Higher saturation = more card content (good)
                improvements = 0
                if lap_diff > 0:
                    improvements += 1
                if bdr_diff < 0:
                    improvements += 1
                if sat_diff > 0:
                    improvements += 1
                print(f"    Improvement: {improvements}/3 metrics better with valley grid")


if __name__ == "__main__":
    main()
