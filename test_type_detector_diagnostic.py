#!/usr/bin/env python
"""Diagnostic analysis of type_detector sampling and classification."""

import logging
from pathlib import Path
import cv2
import numpy as np
from cardprice.ml.type_detector import (
    _sample_border_region,
    _classify_pixels,
    detect_type,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

base_dir = Path("/home/godli/cardprice/data/inbox/page_20260228_202134_cards")

print("=" * 90)
print("TYPE DETECTOR DIAGNOSTIC - SAMPLING AND CLASSIFICATION")
print("=" * 90)
print()

# Analyze one card in detail
card_id = "card_01"
image_path = base_dir / f"{card_id}.png"

if image_path.exists():
    img = cv2.imread(str(image_path))
    if img is not None:
        h, w = img.shape[:2]
        print(f"Image dimensions: {w} x {h} pixels")
        print()

        # Sample the border region
        border_pixels_bgr = _sample_border_region(img)
        print(f"Border pixels sampled: {len(border_pixels_bgr)}")
        print()

        # Convert to HSV
        border_pixels_bgr_3d = border_pixels_bgr.reshape(-1, 1, 3)
        border_pixels_hsv = cv2.cvtColor(border_pixels_bgr_3d, cv2.COLOR_BGR2HSV)
        border_pixels_hsv = border_pixels_hsv.reshape(-1, 3)

        # Analyze HSV distribution
        h_chan = border_pixels_hsv[:, 0].astype(np.float32)
        s_chan = border_pixels_hsv[:, 1].astype(np.float32)
        v_chan = border_pixels_hsv[:, 2].astype(np.float32)

        print("HSV Statistics:")
        print(f"  Hue:        mean={h_chan.mean():.1f}, min={h_chan.min():.1f}, max={h_chan.max():.1f}")
        print(f"  Saturation: mean={s_chan.mean():.1f}, min={s_chan.min():.1f}, max={s_chan.max():.1f}")
        print(f"  Value:      mean={v_chan.mean():.1f}, min={v_chan.min():.1f}, max={v_chan.max():.1f}")
        print()

        # Count pixels by category
        low_sat = s_chan < 40  # _SAT_THRESHOLD
        print(f"Low saturation pixels (S < 40): {low_sat.sum()} / {len(h_chan)} ({100*low_sat.sum()/len(h_chan):.1f}%)")

        # Classify and show votes
        votes = _classify_pixels(border_pixels_hsv)
        print()
        print("Type votes (by pixel count):")
        total_votes = sum(votes.values())
        for type_name in sorted(votes.keys(), key=lambda x: votes[x], reverse=True):
            count = votes[type_name]
            pct = 100 * count / total_votes if total_votes > 0 else 0
            print(f"  {type_name:12s}: {int(count):6d} pixels ({pct:5.1f}%)")

        print()
        print("Final detection:")
        results = detect_type(image_path, top_n=3)
        for type_name, conf in results:
            print(f"  {type_name:12s}: {conf:.1%}")

        print()
        print("OBSERVATION:")
        print("  The binder sleeve (orange) is dominating the sampling region.")
        print("  Binder sleeves are typically high saturation orange, which can be")
        print("  classified as Fire or Fighting depending on exact hue.")
        print("  However, the card's actual type color is buried beneath the sleeve.")
        print()
        print("RECOMMENDATION:")
        print("  1. Crop cards more tightly to exclude the binder sleeve edge")
        print("  2. Adjust sampling regions to avoid sleeve bleed (expand left, shrink right)")
        print("  3. Add pre-processing to detect and mask out sleeve regions")
        print("  4. Use additional features (e.g., card border patterns) for type detection")
