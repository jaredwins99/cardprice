#!/usr/bin/env python3
"""Test EX-era stamp detection via DINOv2 region comparison.

Tests on:
1. ex15 Dragon Frontiers page (mix of stamped and normal cards)
2. Poochyena/ex4 page (all normal, no stamps expected)
"""

import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
from pathlib import Path

# Ground truth for ex15 page (page_20260305_094228_cards)
# Based on visual inspection of scans:
#   Stamped (reverse holo with DRAGON FRONTIERS foil): card_00, card_02, card_05, card_08
#   Normal (no stamp): card_01, card_03, card_04, card_06, card_07
EX15_PAGE = "data/inbox/page_20260305_094228_cards"
EX15_GROUND_TRUTH = {
    "card_00": {"card_id": "ex15-44/normal", "name": "Chikorita",   "stamped": True},
    "card_01": {"card_id": "ex15-26/normal", "name": "Bayleef",     "stamped": False},
    "card_02": {"card_id": "ex15-4/normal",  "name": "Meganium",    "stamped": True},
    "card_03": {"card_id": "ex15-67/normal", "name": "Totodile",    "stamped": False},
    "card_04": {"card_id": "ex15-27/normal", "name": "Croconaw",    "stamped": False},
    "card_05": {"card_id": "ex15-2/normal",  "name": "Feraligatr",  "stamped": True},
    "card_06": {"card_id": "ex15-45/normal", "name": "Cyndaquil",   "stamped": False},
    "card_07": {"card_id": "ex15-36/normal", "name": "Quilava",     "stamped": False},
    "card_08": {"card_id": "ex15-12/normal", "name": "Typhlosion",  "stamped": True},
}

# Poochyena page (page_20260320_223702_cards) - all ex4, no stamps
POOCHYENA_PAGE = "data/inbox/page_20260320_223702_cards"
POOCHYENA_GROUND_TRUTH = {
    "card_00": {"card_id": "ex1-64/normal",  "name": "Poochyena",            "stamped": False},
    "card_01": {"card_id": "ex1-64/normal",  "name": "Poochyena (holo)",     "stamped": False},
    "card_02": {"card_id": "ex4-30/normal",  "name": "T.Aqua's Mightyena",   "stamped": False},
    "card_03": {"card_id": "ex4-54/normal",  "name": "T.Aqua's Poochyena",   "stamped": False},
    "card_04": {"card_id": "ex4-55/normal",  "name": "T.Aqua's Poochyena",   "stamped": False},
    "card_05": {"card_id": "ex4-15/normal",  "name": "T.Aqua's Mightyena",   "stamped": False},
    "card_06": {"card_id": "ex4-65/normal",  "name": "T.Magma's Poochyena",  "stamped": False},
    "card_07": {"card_id": "ex4-66/normal",  "name": "T.Magma's Poochyena",  "stamped": False},
    "card_08": {"card_id": "ex4-21/normal",  "name": "T.Magma's Mightyena",  "stamped": False},
}


def test_page(page_dir: str, ground_truth: dict, page_name: str):
    """Test stamp detection on a page of cards."""
    from cardprice.ml.stamp_detection import _check_ex_stamp_dino

    print(f"\n{'='*70}")
    print(f"Testing: {page_name}")
    print(f"{'='*70}")

    stamped_sims = []
    normal_sims = []
    correct = 0
    total = 0
    errors = []

    for card_file, gt in sorted(ground_truth.items()):
        card_path = Path(page_dir) / f"{card_file}.png"
        if not card_path.exists():
            print(f"  SKIP {card_file}: file not found")
            continue

        img = cv2.imread(str(card_path))
        if img is None:
            print(f"  SKIP {card_file}: could not read")
            continue

        card_id = gt["card_id"]
        set_id = card_id.split("-")[0]  # e.g. "ex15" from "ex15-44/normal"

        result = _check_ex_stamp_dino(img, card_id, set_id)

        if result is None:
            print(f"  SKIP {card_file} ({gt['name']}): no reference image for {card_id}")
            continue

        stamp_sim = result.get("stamp_similarity", result.get("similarity", -1))
        ctrl_sim = result.get("control_similarity", -1)
        diff = result.get("differential", -1)
        detected = result["detected"]
        expected = gt["stamped"]
        is_correct = detected == expected
        total += 1

        if expected:
            stamped_sims.append(diff)
        else:
            normal_sims.append(diff)

        if is_correct:
            correct += 1
            status = "OK"
        else:
            status = "WRONG"
            errors.append((card_file, gt["name"], expected, detected, diff))

        label = "STAMPED" if expected else "normal"
        det_label = "DETECTED" if detected else "clean"
        print(f"  {status:5s} {card_file} ({gt['name']:20s}) "
              f"stamp_sim={stamp_sim:.4f}  ctrl_sim={ctrl_sim:.4f}  "
              f"diff={diff:+.4f}  expected={label:7s}  got={det_label}")

    print(f"\n  Results: {correct}/{total} correct")

    if stamped_sims:
        print(f"  Stamped differentials:  min={min(stamped_sims):.4f}  "
              f"max={max(stamped_sims):.4f}  mean={np.mean(stamped_sims):.4f}")
    if normal_sims:
        print(f"  Normal differentials:   min={min(normal_sims):.4f}  "
              f"max={max(normal_sims):.4f}  mean={np.mean(normal_sims):.4f}")

    if stamped_sims and normal_sims:
        gap = min(stamped_sims) - max(normal_sims)
        print(f"  Gap (min_stamped - max_normal): {gap:.4f}")
        if gap > 0:
            print(f"  Clean separation! Threshold anywhere in "
                  f"[{max(normal_sims):.4f}, {min(stamped_sims):.4f}]")
        else:
            print(f"  WARNING: Overlap between stamped and normal ranges!")

    if errors:
        print(f"\n  Errors:")
        for card_file, name, expected, detected, diff in errors:
            print(f"    {card_file} ({name}): expected={'stamped' if expected else 'normal'}, "
                  f"got={'stamped' if detected else 'normal'}, diff={diff:+.4f}")

    return correct, total, stamped_sims, normal_sims


def main():
    print("EX-era Stamp Detection via DINOv2 Region Comparison")
    print("=" * 70)

    all_correct = 0
    all_total = 0
    all_stamped = []
    all_normal = []

    # Test ex15 page
    c, t, s, n = test_page(EX15_PAGE, EX15_GROUND_TRUTH,
                           "EX Dragon Frontiers (ex15) - mixed stamped/normal")
    all_correct += c
    all_total += t
    all_stamped.extend(s)
    all_normal.extend(n)

    # Note: Poochyena page (ex1/ex4) cards would never reach the EX stamp
    # check because ex1 and ex4 are not in _EX_STAMPED_SETS (only ex7-ex16).
    # Testing them here would be misleading. Skipping.

    # Summary
    print(f"\n{'='*70}")
    print(f"OVERALL: {all_correct}/{all_total} correct "
          f"({100*all_correct/all_total:.1f}%)" if all_total > 0 else "No tests run")

    if all_stamped:
        print(f"All stamped diffs: {[f'{x:+.4f}' for x in sorted(all_stamped)]}")
    if all_normal:
        print(f"All normal diffs:  {[f'{x:+.4f}' for x in sorted(all_normal)]}")

    if all_stamped and all_normal:
        gap = min(all_stamped) - max(all_normal)
        mid = (max(all_normal) + min(all_stamped)) / 2
        print(f"Suggested threshold: {mid:.4f}  (gap={gap:.4f})")
        if gap > 0:
            print(f"Clean separation! Threshold range: "
                  f"[{max(all_normal):.4f}, {min(all_stamped):.4f}]")


if __name__ == "__main__":
    main()
