#!/usr/bin/env python3
"""Extract energy type symbol ROI from all 27 test card segments.

Strategy: First detect the card border within the segment (segments contain
variable amounts of orange binder sleeve), then crop the type symbol region
relative to the detected card area.

The type symbol on Pokemon cards is located:
  - Right after the HP value, in the top-right of the card art area
  - Roughly at 88-95% of card width from left, 5-10% of card height from top

Saves crops to data/eval/type_symbol_crops/ and logs mean BGR/HSV values.
"""

import json
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_JSON = ROOT / "data" / "eval" / "binder_eval.json"
OUTPUT_DIR = ROOT / "data" / "eval" / "type_symbol_crops"


def find_card_bounds(img):
    """Detect the card rectangle within the segment.

    The binder sleeve is orange/warm-colored. The card itself has a distinct
    border. We detect the card by finding the largest non-orange region.

    Returns (x1, y1, x2, y2) of the card content area.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Find left and right card edges by scanning horizontal lines
    right_edges = []
    left_edges = []

    for y_frac in [0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        y = int(y_frac * h)
        row_hsv = hsv[y, :, :]

        # Orange sleeve: H=3-28, S>80, V>100
        is_orange = (
            (row_hsv[:, 0] >= 3) & (row_hsv[:, 0] <= 28) &
            (row_hsv[:, 1] > 80) & (row_hsv[:, 2] > 100)
        )
        not_orange = ~is_orange
        if np.any(not_orange):
            right_edges.append(int(np.max(np.where(not_orange))))
            left_edges.append(int(np.min(np.where(not_orange))))

    if right_edges:
        right_x = int(np.percentile(right_edges, 80))
        left_x = int(np.percentile(left_edges, 20))
    else:
        right_x = w - 1
        left_x = 0

    # Find top and bottom card edges
    top_edges = []
    bottom_edges = []
    for x_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        x = int(x_frac * w)
        col_hsv = hsv[:, x, :]
        is_orange = (
            (col_hsv[:, 0] >= 3) & (col_hsv[:, 0] <= 28) &
            (col_hsv[:, 1] > 80) & (col_hsv[:, 2] > 100)
        )
        not_orange = ~is_orange
        if np.any(not_orange):
            top_edges.append(int(np.min(np.where(not_orange))))
            bottom_edges.append(int(np.max(np.where(not_orange))))

    if top_edges:
        top_y = int(np.percentile(top_edges, 20))
        bottom_y = int(np.percentile(bottom_edges, 80))
    else:
        top_y = 0
        bottom_y = h - 1

    return left_x, top_y, right_x, bottom_y


def extract_type_symbol(img, card_bounds):
    """Extract the type symbol region relative to detected card bounds.

    Returns dict with multiple crop variants.
    """
    cx1, cy1, cx2, cy2 = card_bounds
    card_w = cx2 - cx1
    card_h = cy2 - cy1
    h, w = img.shape[:2]

    def safe_crop(x1, y1, x2, y2):
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        return img[y1:y2, x1:x2]

    crops = {}

    # Wide HP+symbol region: rightmost 45% of card width, top 9% of card height
    crops["wide"] = safe_crop(
        cx1 + int(card_w * 0.55), cy1,
        cx2, cy1 + int(card_h * 0.09)
    )

    # Tight symbol: ~88-97% of card width, 2.5-7.5% of card height
    crops["symbol"] = safe_crop(
        cx1 + int(card_w * 0.88), cy1 + int(card_h * 0.025),
        cx1 + int(card_w * 0.97), cy1 + int(card_h * 0.075)
    )

    # Wider symbol region: 78-98% x, 1.5-9% y (catches more era variation)
    crops["symbol_wide"] = safe_crop(
        cx1 + int(card_w * 0.78), cy1 + int(card_h * 0.015),
        cx1 + int(card_w * 0.98), cy1 + int(card_h * 0.09)
    )

    return crops


def analyze_crop(crop):
    """Return mean BGR and HSV for a crop."""
    if crop.size == 0:
        return [0, 0, 0], [0, 0, 0]
    bgr = crop.mean(axis=(0, 1))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).mean(axis=(0, 1))
    return [round(float(v), 1) for v in bgr], [round(float(v), 1) for v in hsv]


def main():
    with open(EVAL_JSON) as f:
        data = json.load(f)

    # Create output directories
    for name in ["wide", "symbol", "symbol_wide"]:
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)

    results = []

    for page_idx, page in enumerate(data["pages"]):
        seg_dir = ROOT / page["segments_dir"]
        for card in page["cards"]:
            seg_name = card["segment"]
            card_num = int(seg_name.replace("card_", "").replace(".png", ""))
            card_path = seg_dir / seg_name
            card_id = card.get("card_id")
            name = card.get("name", "?")

            if not card_path.exists():
                print(f"  MISSING: {card_path}")
                continue

            img = cv2.imread(str(card_path))
            if img is None:
                print(f"  FAILED to read: {card_path}")
                continue

            h, w = img.shape[:2]

            # Detect card bounds within segment
            bounds = find_card_bounds(img)
            bx1, by1, bx2, by2 = bounds

            # Extract crops
            crops = extract_type_symbol(img, bounds)

            result = {
                "page": page_idx,
                "card": card_num,
                "name": name,
                "card_id": card_id,
                "dims": f"{w}x{h}",
                "card_bounds": list(bounds),
                "card_size": f"{bx2-bx1}x{by2-by1}",
            }

            for crop_name, crop in crops.items():
                fname = f"p{page_idx}_c{card_num:02d}.png"
                if crop.size > 0:
                    cv2.imwrite(str(OUTPUT_DIR / crop_name / fname), crop)
                    bgr, hsv = analyze_crop(crop)
                    result[f"{crop_name}_bgr"] = bgr
                    result[f"{crop_name}_hsv"] = hsv
                    result[f"{crop_name}_shape"] = f"{crop.shape[1]}x{crop.shape[0]}"
                else:
                    result[f"{crop_name}_bgr"] = [0, 0, 0]
                    result[f"{crop_name}_hsv"] = [0, 0, 0]
                    result[f"{crop_name}_shape"] = "0x0"

            results.append(result)

    # Print summary table
    print(f"\n{'Page':>4} {'Card':>4} {'Name':<20} {'Card ID':<22} "
          f"{'Bounds':<25} {'CardSz':<12} "
          f"{'SymWide BGR':<28} {'SymWide HSV':<28}")
    print("-" * 170)

    for r in results:
        def fmt(key):
            v = r.get(key, [0, 0, 0])
            return f"({v[0]:5.1f},{v[1]:5.1f},{v[2]:5.1f})"
        b = r["card_bounds"]
        bounds_str = f"({b[0]:3d},{b[1]:3d},{b[2]:3d},{b[3]:3d})"
        print(f"{r['page']:>4} {r['card']:>4} {r['name']:<20} {str(r['card_id']):<22} "
              f"{bounds_str:<25} {r['card_size']:<12} "
              f"{fmt('symbol_wide_bgr'):<28} {fmt('symbol_wide_hsv'):<28}")

    # Save JSON results
    out_json = OUTPUT_DIR / "color_analysis.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} crops to {OUTPUT_DIR}")
    print(f"Color analysis saved to {out_json}")


if __name__ == "__main__":
    main()
