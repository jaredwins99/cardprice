#!/usr/bin/env python3
"""Test variant_detector on page 0 binder eval segments.

Loads page 0 from data/eval/binder_eval.json, runs detect_variant() on each
card segment, and compares against expected variant ("normal" for all page 0
cards -- they are in binder sleeves under overhead lighting with no holo effect).

Also runs get_card_era() on each ground truth card_id and shows era-allowed
variants for context.
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cardprice.ml.variant_detector import detect_variant, get_valid_variants
from cardprice.ml.era_detector import get_card_era, get_era_name


def main():
    eval_path = PROJECT_ROOT / "data" / "eval" / "binder_eval.json"
    with open(eval_path) as f:
        eval_data = json.load(f)

    page = eval_data["pages"][0]
    segments_dir = PROJECT_ROOT / page["segments_dir"]

    print(f"Page 0: {page['image']}")
    print(f"Segments dir: {segments_dir}")
    print()

    # Table header
    header = (
        f"{'card_id':<25} {'detected':<20} {'expected':<12} "
        f"{'era':<5} {'era_name':<30} {'era_allowed_variants':<50} {'status'}"
    )
    print(header)
    print("-" * len(header))

    correct = 0
    total = 0

    for card in page["cards"]:
        card_id = card["card_id"]
        if card_id is None:
            # Skip empty slots
            continue

        segment_path = segments_dir / card["segment"]
        if not segment_path.exists():
            print(f"MISSING: {segment_path}")
            continue

        # Expected variant: all page 0 cards should be "normal"
        expected = "normal"

        # Detect variant
        detected = detect_variant(str(segment_path))

        # Get era info
        era = get_card_era(card_id)
        era_name = get_era_name(era)
        set_id = card_id.split("/")[0].rsplit("-", 1)[0]
        allowed = sorted(get_valid_variants(set_id, era))

        # Check correctness
        is_correct = detected == expected
        status = "OK" if is_correct else "FAIL"
        if is_correct:
            correct += 1
        total += 1

        print(
            f"{card_id:<25} {detected:<20} {expected:<12} "
            f"{era:<5} {era_name:<30} {', '.join(allowed):<50} {status}"
        )

    print()
    print(f"Result: {correct}/{total} correct ({100*correct/total:.1f}%)")
    if correct < total:
        print("WARNING: Some cards detected as non-normal. These are binder "
              "sleeve scans under overhead lighting -- all should be 'normal'.")


if __name__ == "__main__":
    main()
