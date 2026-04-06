#!/usr/bin/env python3
"""Test variant_detector.detect_variant() on all eval binder segments.

Loads binder_eval.json, runs detect_variant_detailed() on each segment,
and reports results.

Page 0 & Page 1: all normal cards (EX-era and WotC e-Card era).
Page 2: mixed -- some holofoil, some reverse_holofoil, one card back.

Also tests era_detector functions to verify era assignment for each card.
"""

import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardprice.ml.variant_detector import detect_variant_detailed, get_valid_variants
from cardprice.ml.era_detector import get_card_era, get_era_name


def main():
    eval_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "eval", "binder_eval.json")
    with open(eval_path) as f:
        eval_data = json.load(f)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    results = []
    false_positives = []
    era_results = []

    total_time = 0.0

    for page_idx, page in enumerate(eval_data["pages"]):
        segments_dir = os.path.join(project_root, page["segments_dir"])
        print(f"\n{'='*80}")
        print(f"Page {page_idx}: {page['image']}")
        print(f"{'='*80}")

        for card in page["cards"]:
            seg_path = os.path.join(segments_dir, card["segment"])
            name = card["name"]
            card_id = card.get("card_id") or "empty"
            expected_physical = card.get("variant", "normal")  # physical variant from eval

            if not os.path.exists(seg_path):
                print(f"  SKIP {card['segment']} -- file not found")
                continue

            # --- Era detection test ---
            if card_id != "empty":
                era = get_card_era(card_id)
                era_name = get_era_name(era)
                set_id = card_id.split("/")[0].rsplit("-", 1)[0]
                valid_variants = get_valid_variants(set_id, era)
            else:
                era = 0
                era_name = "N/A (empty)"
                set_id = ""
                valid_variants = set()

            era_results.append({
                "card_id": card_id,
                "name": name,
                "era": era,
                "era_name": era_name,
                "set_id": set_id,
                "valid_variants": valid_variants,
            })

            # --- Variant detection test ---
            t0 = time.time()
            detail = detect_variant_detailed(seg_path, era=era, card_id=card_id)
            elapsed = time.time() - t0
            total_time += elapsed
            detected = detail["variant"]

            # Determine if this is a false positive
            # Page 2 cards have physical variant annotations -- detecting holo
            # on those is CORRECT, not a false positive
            is_fp = False
            if card_id == "empty":
                # Card back -- should be normal
                if detected != "normal":
                    is_fp = True
            elif expected_physical == "normal" and detected != "normal":
                is_fp = True

            status = "FALSE POSITIVE" if is_fp else "ok"
            if expected_physical != "normal" and detected != "normal":
                status = "ok (holo expected)"
            if expected_physical != "normal" and detected == "normal":
                status = "MISSED HOLO"

            results.append({
                "page": page_idx,
                "segment": card["segment"],
                "name": name,
                "card_id": card_id,
                "physical_variant": expected_physical,
                "detected": detected,
                "is_false_positive": is_fp,
                "art_combined": detail["art_combined_score"],
                "border_combined": detail["border_combined_score"],
                "art_noise": detail["art_spatial_noise"],
                "border_noise": detail["border_spatial_noise"],
                "has_stamp": detail["has_1st_edition_stamp"],
                "is_full_art": detail["is_full_art"],
                "is_reverse_holo": detail["is_reverse_holo"],
                "gold_result": detail["gold_rare_result"],
                "elapsed": elapsed,
            })

            if is_fp:
                false_positives.append(results[-1])

            print(f"  {card['segment']:12s} {name:20s} -> {str(detected):20s} "
                  f"art={detail['art_combined_score']:6.1f} "
                  f"border={detail['border_combined_score']:6.1f} "
                  f"stamp={detail['has_1st_edition_stamp']}  "
                  f"era={era}({era_name[:10]})  "
                  f"[{status}] ({elapsed:.2f}s)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("VARIANT DETECTION SUMMARY")
    print(f"{'='*80}")
    total = len(results)
    fp_count = len(false_positives)
    print(f"Total segments tested: {total}")
    print(f"Total time: {total_time:.1f}s (avg {total_time/max(total,1):.2f}s/card)")
    print(f"False positives: {fp_count}")

    if false_positives:
        print(f"\nFALSE POSITIVE DETAILS:")
        for fp in false_positives:
            print(f"  Page {fp['page']} {fp['segment']} {fp['name']}: "
                  f"detected={fp['detected']} (expected normal)")
            print(f"    art_combined={fp['art_combined']:.1f} "
                  f"border_combined={fp['border_combined']:.1f} "
                  f"art_noise={fp['art_noise']:.1f} "
                  f"border_noise={fp['border_noise']:.1f}")
            print(f"    stamp={fp['has_stamp']} full_art={fp['is_full_art']} "
                  f"rev_holo={fp['is_reverse_holo']} gold={fp['gold_result']}")
    else:
        print("\nNo false positives -- all non-holo cards correctly classified as normal.")

    # Variant distribution
    from collections import Counter
    dist = Counter(r["detected"] for r in results)
    print(f"\nVariant distribution: {dict(dist)}")

    # Holo detection on page 2 (should detect holos)
    page2_holos = [r for r in results if r["page"] == 2 and r["physical_variant"] != "normal"]
    if page2_holos:
        print(f"\nPage 2 holo detection (expected holofoil/reverseHolofoil):")
        detected_correct = 0
        for r in page2_holos:
            match = r["detected"] in ("holofoil", "reverse_holofoil")
            if r["physical_variant"] == "reverseHolofoil":
                match = r["detected"] == "reverse_holofoil"
            elif r["physical_variant"] == "holofoil":
                match = r["detected"] == "holofoil"
            status = "CORRECT" if match else f"WRONG (got {r['detected']})"
            if match:
                detected_correct += 1
            print(f"  {r['segment']} {r['name']:15s}: expected={r['physical_variant']:18s} "
                  f"detected={r['detected']:18s} [{status}]")
        print(f"  Holo detection: {detected_correct}/{len(page2_holos)} correct")

    # Confidence distribution (art_combined and border_combined scores)
    print(f"\nScore distribution for 'normal' cards (pages 0-1):")
    normal_cards = [r for r in results
                    if r["physical_variant"] == "normal" and r["card_id"] != "empty"]
    if normal_cards:
        art_scores = [r["art_combined"] for r in normal_cards]
        border_scores = [r["border_combined"] for r in normal_cards]
        print(f"  Art combined:    min={min(art_scores):.1f}  max={max(art_scores):.1f}  "
              f"avg={sum(art_scores)/len(art_scores):.1f}")
        print(f"  Border combined: min={min(border_scores):.1f}  max={max(border_scores):.1f}  "
              f"avg={sum(border_scores)/len(border_scores):.1f}")

    # =========================================================================
    # ERA DETECTION SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("ERA DETECTION SUMMARY")
    print(f"{'='*80}")
    era_errors = 0
    for er in era_results:
        if er["card_id"] == "empty":
            continue
        # Expected eras based on set prefix
        expected_era = None
        sid = er["set_id"]
        if sid.startswith("ex"):
            expected_era = 2
        elif sid.startswith("ecard"):
            expected_era = 1
        elif sid.startswith("dp"):
            expected_era = 3
        elif sid.startswith("pl"):
            expected_era = 3
        elif sid.startswith("tk"):
            expected_era = 2

        if expected_era is not None and er["era"] != expected_era:
            print(f"  ERA MISMATCH: {er['card_id']} {er['name']}: "
                  f"got era {er['era']} expected {expected_era}")
            era_errors += 1
        else:
            print(f"  {er['card_id']:25s} {er['name']:20s} -> era {er['era']} "
                  f"({er['era_name']}) variants={sorted(er['valid_variants'])}")

    if era_errors == 0:
        print(f"\n  All {len([e for e in era_results if e['card_id'] != 'empty'])} "
              f"cards have correct era assignments.")
    else:
        print(f"\n  {era_errors} era mismatches found!")

    print()
    return 1 if false_positives else 0


if __name__ == "__main__":
    sys.exit(main())
