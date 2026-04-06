#!/usr/bin/env python3
"""Test variant_detector on page 2 card segments from binder_eval.json.

Loads page 2 (index 2) segments, runs detect_variant() on each,
looks up the ground-truth era via get_card_era(), and prints a
comparison table flagging any incorrect detections.
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

    # Page 2 is index 2
    page = eval_data["pages"][2]
    segments_dir = ROOT / page["segments_dir"]
    cards = page["cards"]

    print(f"Page 2: {page.get('notes', '')}")
    print(f"Segments dir: {segments_dir}")
    print()

    # Table header
    hdr = (
        f"{'Pos':<6} {'card_id':<22} {'name':<16} "
        f"{'GT variant':<18} {'detected':<18} {'era':<5} "
        f"{'era_allowed_variants':<45} {'flag'}"
    )
    print(hdr)
    print("-" * len(hdr))

    correct = 0
    total = 0
    skipped = 0

    for card in cards:
        pos = f"{card['position'][0]},{card['position'][1]}"
        card_id = card.get("card_id")
        name = card.get("name", "")
        # Ground truth variant from eval JSON (key "variant"), falling back
        # to the variant suffix in card_id (after "/")
        gt_variant_raw = card.get("variant")
        if gt_variant_raw is None and card_id and "/" in card_id:
            gt_variant_raw = card_id.split("/", 1)[1]

        segment_path = segments_dir / card["segment"]

        # Skip null card_id (empty slots)
        if card_id is None:
            print(
                f"{pos:<6} {'(empty)':<22} {name:<16} "
                f"{'N/A':<18} {'SKIPPED':<18} {'--':<5} "
                f"{'--':<45} {'--'}"
            )
            skipped += 1
            continue

        # Normalize GT variant names to match detector output
        gt_norm_map = {
            "normal": "normal",
            "holofoil": "holofoil",
            "reverseHolofoil": "reverse_holofoil",
            "reverse_holofoil": "reverse_holofoil",
            "1st_edition": "1st_edition",
            "1st_edition_holofoil": "1st_edition",
        }
        gt_variant = gt_norm_map.get(gt_variant_raw, gt_variant_raw) if gt_variant_raw else "?"

        # Run detector
        if not segment_path.exists():
            detected = "FILE_NOT_FOUND"
        else:
            try:
                detected = detect_variant(str(segment_path))
            except Exception as e:
                detected = f"ERROR:{e}"

        # Look up era
        era = get_card_era(card_id)
        era_name = get_era_name(era)

        # Get set prefix for valid variants lookup
        bare_id = card_id.split("/")[0]
        set_id = bare_id.rsplit("-", 1)[0]
        allowed = get_valid_variants(set_id, era)
        allowed_str = ", ".join(sorted(allowed))

        # Flag logic
        flag = ""
        total += 1

        if gt_variant == "?":
            flag = "NO_GT"
        elif detected == gt_variant:
            flag = "OK"
            correct += 1
        else:
            flag = "WRONG"

        # Also flag if detected variant is not in the era's allowed set
        if detected not in ("FILE_NOT_FOUND",) and not detected.startswith("ERROR:"):
            if detected not in allowed:
                flag += " ERA_INVALID"

        print(
            f"{pos:<6} {card_id:<22} {name:<16} "
            f"{gt_variant:<18} {detected:<18} {era:<5} "
            f"{allowed_str:<45} {flag}"
        )

    print()
    print(f"Results: {correct}/{total} correct, {skipped} skipped")
    if total > 0:
        print(f"Accuracy: {correct/total*100:.1f}%")


if __name__ == "__main__":
    main()
