"""Save visual grid comparison images for all 3 binder photos."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from cardprice.ml.card_segmenter import _find_grid_lines

base = Path(__file__).resolve().parent.parent / "data" / "inbox"
outdir = Path(__file__).resolve().parent.parent / "data"

for name in ["page_20260228_174819.jpg", "page_20260228_195512.jpg", "page_20260228_202134.jpg"]:
    img_path = base / name
    if not img_path.exists():
        continue
    image = cv2.imread(str(img_path))
    h, w = image.shape[:2]
    if max(h, w) > 3000:
        scale = 3000 / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

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
        rectified = image[max(0, by):min(ih, by + bh), max(0, bx):min(iw, bx + bw)]
    else:
        rectified = image

    page_h, page_w = rectified.shape[:2]
    img_rows = img_cols = 3

    debug_u = rectified.copy()
    debug_v = rectified.copy()

    for i in range(1, 3):
        y = int(page_h * i / 3)
        x = int(page_w * i / 3)
        cv2.line(debug_u, (0, y), (page_w, y), (0, 255, 0), 3)
        cv2.line(debug_u, (x, 0), (x, page_h), (0, 255, 0), 3)
    cv2.putText(debug_u, "UNIFORM", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

    result = _find_grid_lines(rectified, img_rows, img_cols)
    if result:
        rb, cb = result
        for s, e in rb[1:]:
            cv2.line(debug_v, (0, s), (page_w, s), (0, 0, 255), 3)
        for s, e in cb[1:]:
            cv2.line(debug_v, (s, 0), (s, page_h), (0, 0, 255), 3)
    cv2.putText(debug_v, "VALLEY", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

    combined = np.hstack([debug_u, debug_v])
    out = outdir / f"grid_comparison_{img_path.stem}.jpg"
    cv2.imwrite(str(out), combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"Saved: {out}")

print("Done")
