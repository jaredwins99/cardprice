"""Debug valley detection profiles to understand why detection fails."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

images = [
    "data/inbox/page_20260228_174819.jpg",
    "data/inbox/page_20260228_195512.jpg",
    "data/inbox/page_20260228_202134.jpg",
]

base = Path(__file__).resolve().parent.parent

for img_rel in images:
    img_path = base / img_rel
    image = cv2.imread(str(img_path))
    h, w = image.shape[:2]
    max_dim = 3000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Detect page region
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
        bx, by = max(0, bx), max(0, by)
        bx2, by2 = min(iw, bx+bw), min(ih, by+bh)
        rectified = image[by:by2, bx:bx2]
    else:
        rectified = image

    page_h, page_w = rectified.shape[:2]
    is_landscape = page_w > page_h
    img_rows = img_cols = 3

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    h_proj = gray.mean(axis=1).astype(np.float64)
    v_proj = gray.mean(axis=0).astype(np.float64)

    print(f"\n{'='*60}")
    print(f"Image: {img_rel}")
    print(f"  Rectified: {page_w}x{page_h}, landscape={is_landscape}")

    for axis_name, profile, n_cells, axis_len in [
        ("ROWS (h_proj)", h_proj, img_rows, page_h),
        ("COLS (v_proj)", v_proj, img_cols, page_w),
    ]:
        kernel_size = max(3, int(axis_len * 0.02) | 1)
        smoothed = cv2.GaussianBlur(profile.reshape(-1, 1), (1, kernel_size), 0).flatten()
        expected_cell = axis_len / n_cells
        margin = int(axis_len * 0.03)

        # Find minima
        minima = []
        for i in range(margin, len(smoothed) - margin):
            if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
                minima.append((i, smoothed[i]))

        print(f"\n  {axis_name}: len={axis_len}, expected_cell={expected_cell:.0f}")
        print(f"    Smoothing kernel={kernel_size}, margin={margin}")
        print(f"    Profile range: {smoothed.min():.1f} - {smoothed.max():.1f}")
        print(f"    Number of local minima: {len(minima)}")

        # Score by depth
        neighbourhood = int(expected_cell * 0.3)
        scored = []
        for idx, val in minima:
            lo = max(0, idx - neighbourhood)
            hi = min(len(smoothed), idx + neighbourhood)
            local_mean = smoothed[lo:hi].mean()
            depth = local_mean - val
            scored.append((idx, depth, val))

        scored.sort(key=lambda x: x[1], reverse=True)
        print(f"    Top 10 minima by depth:")
        for j, (idx, depth, val) in enumerate(scored[:10]):
            pct = idx / axis_len * 100
            print(f"      #{j}: pos={idx} ({pct:.1f}%) val={val:.1f} depth={depth:.2f}")

        # Check greedy selection
        min_spacing = expected_cell * 0.4
        selected = []
        for idx, depth, val in scored:
            if all(abs(idx - s) >= min_spacing for s in selected):
                selected.append(idx)
            if len(selected) == n_cells - 1:
                break
        selected.sort()
        print(f"    Selected valleys: {selected} (need {n_cells - 1})")

        if len(selected) == n_cells - 1:
            # Check sanity
            boundaries_test = [0] + selected + [axis_len]
            print(f"    Cell sizes: {[boundaries_test[i+1] - boundaries_test[i] for i in range(len(boundaries_test)-1)]}")
            print(f"    Expected cell: {expected_cell:.0f}, min={expected_cell*0.3:.0f}, max={expected_cell*2:.0f}")
            for i in range(len(boundaries_test) - 1):
                cs = boundaries_test[i+1] - boundaries_test[i]
                if cs < expected_cell * 0.30 or cs > expected_cell * 2.0:
                    print(f"    FAILED sanity: cell {i} size {cs}")

        # Show expected valley positions
        print(f"    Expected valley positions: {[int(expected_cell * (k+1)) for k in range(n_cells-1)]}")
