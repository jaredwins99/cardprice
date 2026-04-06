#!/usr/bin/env python3
"""Test variant_detector on page 1 (index 1) card segments from binder_eval.json.

Loads page 1 segments, runs detect_variant() on each, looks up the ground-truth
card's era and allowed variants, and prints a summary table flagging any
incorrect detections.
"""

import json
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

from cardprice.ml.variant_detector import detect_variant, get_valid_variants
from cardprice.ml.era_detector import get_card_era, get_era_name


def main():
    eval_path = ROOT / "data" / "eval" / "binder_eval.json"
    with open(eval_path) as f:
        eval_data = json.load(f)

    page = eval_data["pages"][1]  # page 1 (0-indexed)
    segments_dir = ROOT / page["segments_dir"]

    print(f"Page 1: {page['image']}")
    print(f"Segments: {segments_dir}")
    print()

    # Table header
    hdr = (
        f"{'Pos':<6} {'Card ID':<24} {'Name':<14} "
        f"{'GT Variant':<20} {'Detected':<20} "
        f"{'Era':<4} {'Era Name':<28} {'Allowed Variants':<50} {'Status'}"
    )
    print(hdr)
    print("-" * len(hdr))

    correct = 0
    total = 0
    errors = []

    for card in page["cards"]:
        card_id = card["card_id"]
        if card_id is None:
            continue  # skip empty slots

        segment_path = segments_dir / card["segment"]
        if not segment_path.exists():
            print(f"  MISSING: {segment_path}")
            continue

        # Ground truth variant from card_id (after the '/')
        gt_variant = card_id.split("/")[1] if "/" in card_id else "normal"

        # Also check if eval JSON has an explicit variant override
        # (some entries have "variant" field with actual physical variant)
        gt_physical = card.get("variant")
        if gt_physical:
            # Normalize eval JSON variant names to our format
            variant_map = {
                "holofoil": "holofoil",
                "reverseHolofoil": "reverse_holofoil",
                "normal": "normal",
                "1stEdition": "1st_edition",
            }
            gt_physical = variant_map.get(gt_physical, gt_physical)

        # Run detection
        detected = detect_variant(str(segment_path))

        # Era info
        era = get_card_era(card_id)
        era_name = get_era_name(era)

        # Get set prefix for allowed variants
        bare_id = card_id.split("/")[0]
        set_id = bare_id.rsplit("-", 1)[0]
        allowed = get_valid_variants(set_id, era)
        allowed_str = ", ".join(sorted(allowed))

        # Determine correctness
        # Compare against physical variant if available, otherwise gt from card_id
        expected = gt_physical if gt_physical else gt_variant
        pos_str = f"[{card['position'][0]},{card['position'][1]}]"
        is_correct = (detected == expected)
        status = "OK" if is_correct else "WRONG"

        if is_correct:
            correct += 1
        else:
            errors.append({
                "pos": pos_str,
                "card_id": card_id,
                "name": card["name"],
                "expected": expected,
                "detected": detected,
            })

        total += 1

        display_gt = gt_physical if gt_physical else gt_variant
        print(
            f"{pos_str:<6} {card_id:<24} {card['name']:<14} "
            f"{display_gt:<20} {detected:<20} "
            f"{era:<4} {era_name:<28} {allowed_str:<50} {status}"
        )

    print()
    print(f"Results: {correct}/{total} correct ({100*correct/total:.1f}%)" if total else "No cards")
    print()

    if errors:
        print("INCORRECT DETECTIONS:")
        for e in errors:
            print(f"  {e['pos']} {e['card_id']} ({e['name']}): "
                  f"expected={e['expected']}, detected={e['detected']}")
    else:
        print("All detections correct!")


if __name__ == "__main__":
    main()
