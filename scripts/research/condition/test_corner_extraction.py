#!/usr/bin/env python3
"""Test corner ROI extraction from the corner_classifier module.

Extracts all 4 corners from 9 cards (first eval page), saves them as PNGs,
reports dimensions, and tests grade_corners() behavior with no checkpoint.
"""

import json
import logging
import sys
from pathlib import Path

import cv2

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cardprice.ml.corner_classifier import (
    CARD_H,
    CARD_W,
    CORNER_NAMES,
    ROI_H,
    ROI_W,
    extract_corner_rois,
    grade_corners,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Load eval config - use first page (9 cards)
    eval_path = ROOT / "data" / "eval" / "binder_eval.json"
    with open(eval_path) as f:
        eval_data = json.load(f)

    page = eval_data["pages"][0]
    segments_dir = ROOT / page["segments_dir"]
    output_dir = ROOT / "data" / "eval" / "corner_crops"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Segments dir: {segments_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Reference ROI size at {CARD_W}x{CARD_H}: {ROI_W}x{ROI_H}")
    print()

    issues = []
    all_ok = True

    for card_info in page["cards"]:
        seg_name = card_info["segment"]
        card_name = card_info["name"]
        seg_path = segments_dir / seg_name
        stem = seg_path.stem  # e.g. "card_00"

        if not seg_path.exists():
            print(f"MISSING: {seg_path}")
            issues.append(f"{seg_name}: file not found")
            continue

        # Read image to check dimensions
        img = cv2.imread(str(seg_path))
        h, w = img.shape[:2]
        print(f"--- {stem} ({card_name}) --- segment: {w}x{h}")

        # Calculate expected scaled ROI dimensions
        scale_x = w / CARD_W
        scale_y = h / CARD_H
        expected_rw = max(int(ROI_W * scale_x), 16)
        expected_rh = max(int(ROI_H * scale_y), 16)
        print(f"    Scale factors: x={scale_x:.4f}  y={scale_y:.4f}")
        print(f"    Expected ROI:  {expected_rw}x{expected_rh}")

        # Extract corners
        rois = extract_corner_rois(seg_path)

        for corner_name in CORNER_NAMES:
            roi = rois[corner_name]
            rh, rw = roi.shape[:2]
            out_path = output_dir / f"{stem}_{corner_name}.png"
            cv2.imwrite(str(out_path), roi)

            # Validate dimensions
            ok = (rw == expected_rw and rh == expected_rh)
            status = "OK" if ok else "MISMATCH"
            if not ok:
                all_ok = False
                issues.append(f"{stem}_{corner_name}: got {rw}x{rh}, expected {expected_rw}x{expected_rh}")

            print(f"    {corner_name:14s}: {rw:4d}x{rh:4d}  [{status}]  -> {out_path.name}")

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_corners = len(page["cards"]) * 4
    print(f"Total corners extracted: {total_corners}")
    print(f"Output dir: {output_dir}")

    if issues:
        print(f"\nISSUES ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("All dimensions match expected scaled values.")

    # Check: at 1008x1530 (our actual segment size), what's the ROI?
    print(f"\nROI size check for 1008x1530 segments:")
    print(f"  scale_x = 1008/{CARD_W} = {1008/CARD_W:.4f}")
    print(f"  scale_y = 1530/{CARD_H} = {1530/CARD_H:.4f}")
    print(f"  ROI_W = int({ROI_W} * {1008/CARD_W:.4f}) = {int(ROI_W * 1008/CARD_W)}")
    print(f"  ROI_H = int({ROI_H} * {1530/CARD_H:.4f}) = {int(ROI_H * 1530/CARD_H)}")

    # Test grade_corners() with no checkpoint
    print("\n" + "=" * 60)
    print("TESTING grade_corners() WITH NO CHECKPOINT")
    print("=" * 60)
    test_seg = segments_dir / page["cards"][0]["segment"]
    try:
        # Reset the module-level singleton so it reloads
        import cardprice.ml.corner_classifier as cc
        cc._model = None
        cc._device = None

        result = grade_corners(str(test_seg))
        print(f"grade_corners() returned successfully (no crash).")
        print(f"  overall_grade:      {result['overall_grade']}")
        print(f"  overall_confidence: {result['overall_confidence']}")
        for cname, cdata in result["corners"].items():
            print(f"  {cname:14s}: {cdata['grade']:10s} (conf={cdata['confidence']:.4f})")
        print("\nNote: With random classifier head, grades are meaningless but the")
        print("function handles the missing checkpoint gracefully (warning + random predictions).")
    except FileNotFoundError as e:
        print(f"FileNotFoundError (expected if no checkpoint): {e}")
        print("The function does NOT gracefully handle missing checkpoints.")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
