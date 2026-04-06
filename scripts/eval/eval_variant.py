#!/usr/bin/env python3
"""Evaluate variant_detector on the 27 binder eval segments.

Reports:
  - Per-card: detected variant, holographic analysis scores, 1st edition flag
  - False positive rates by variant type (all test cards are "normal")
  - Comparison: detect_variant() with and without era parameter
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cardprice.ml.variant_detector import detect_variant, detect_variant_detailed
from cardprice.ml.era_detector import get_card_era, get_era_name


def main():
    eval_path = PROJECT_ROOT / "data" / "eval" / "binder_eval.json"
    with open(eval_path) as f:
        eval_data = json.load(f)

    results = []
    for page_idx, page in enumerate(eval_data["pages"]):
        segments_dir = PROJECT_ROOT / page["segments_dir"]
        for card in page["cards"]:
            card_id = card.get("card_id")
            if card_id is None:
                continue  # skip empty slot

            segment_path = segments_dir / card["segment"]
            if not segment_path.exists():
                print(f"WARNING: segment not found: {segment_path}")
                continue

            # Get era from card_id
            era = get_card_era(card_id)
            era_name = get_era_name(era)

            # Run detailed detection WITHOUT era
            detail_no_era = detect_variant_detailed(str(segment_path))

            # Run detailed detection WITH era
            detail_with_era = detect_variant_detailed(str(segment_path), era=era, card_id=card_id)

            # Ground truth variant (from eval json, default "normal")
            gt_variant = card.get("variant", "normal")

            results.append({
                "page": page_idx,
                "position": card["position"],
                "name": card["name"],
                "card_id": card_id,
                "era": era,
                "era_name": era_name,
                "gt_variant": gt_variant,
                "segment_path": str(segment_path),
                "no_era": detail_no_era,
                "with_era": detail_with_era,
            })

    # Print detailed per-card results
    print("=" * 120)
    print("VARIANT DETECTOR EVALUATION -- 27 Binder Eval Segments")
    print("=" * 120)

    for r in results:
        no_era = r["no_era"]
        with_era = r["with_era"]
        gt = r["gt_variant"]

        # Determine correctness
        no_era_correct = no_era["variant"] == gt
        with_era_correct = with_era["variant"] == gt

        no_era_mark = "OK" if no_era_correct else "FP"
        with_era_mark = "OK" if with_era_correct else "FP"

        print(f"\n--- Page {r['page']}, Pos {r['position']} | {r['name']} ({r['card_id']}) | Era {r['era']} ({r['era_name']}) ---")
        print(f"  Ground truth variant: {gt}")
        print(f"  Detected (no era):    {no_era['variant']:20s}  [{no_era_mark}]")
        print(f"  Detected (with era):  {with_era['variant']:20s}  [{with_era_mark}]")
        print(f"  Holographic analysis:")
        print(f"    Art:    hue_spread={no_era['art_hue_spread']:3d}  spatial_noise={no_era['art_spatial_noise']:.2f}  combined={no_era['art_combined_score']:.2f}  sat_std={no_era['art_saturation_std']:.1f}")
        print(f"    Border: hue_spread={no_era['border_hue_spread']:3d}  spatial_noise={no_era['border_spatial_noise']:.2f}  combined={no_era['border_combined_score']:.2f}  sat_std={no_era['border_saturation_std']:.1f}")
        print(f"  1st Edition stamp:    {no_era['has_1st_edition_stamp']}")
        print(f"  Full art:             {no_era['is_full_art']}")
        print(f"  Gold/rainbow rare:    {no_era['gold_rare_result']}")
        if no_era.get("is_shadowless") is not None:
            print(f"  Shadowless:           {no_era['is_shadowless']}  (right_grad={no_era['shadow_right_grad']:.2f}, bottom_grad={no_era['shadow_bottom_grad']:.2f}, combined={no_era['shadow_combined']:.2f})")

    # Summary statistics
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    total = len(results)

    # Count by ground truth variant
    gt_counts = {}
    for r in results:
        gt = r["gt_variant"]
        gt_counts[gt] = gt_counts.get(gt, 0) + 1
    print(f"\nGround truth distribution: {gt_counts}")

    # Accuracy per mode
    for mode_label, mode_key in [("No era", "no_era"), ("With era", "with_era")]:
        correct = sum(1 for r in results if r[mode_key]["variant"] == r["gt_variant"])
        print(f"\n{mode_label}: {correct}/{total} correct ({100*correct/total:.1f}%)")

        # False positive breakdown (detected != ground truth)
        fp_by_type = {}
        fp_details = []
        for r in results:
            detected = r[mode_key]["variant"]
            gt = r["gt_variant"]
            if detected != gt:
                fp_by_type[detected] = fp_by_type.get(detected, 0) + 1
                fp_details.append(f"    {r['name']} ({r['card_id']}): gt={gt}, detected={detected}")

        if fp_by_type:
            print(f"  False detections by type: {fp_by_type}")
            for d in fp_details:
                print(d)
        else:
            print(f"  No false detections!")

    # Cards that triggered holo-like scores (even if below threshold)
    print("\n" + "=" * 120)
    print("HOLOGRAPHIC SCORE RANKING (all cards, by art_combined + border_combined)")
    print("=" * 120)
    print(f"{'Name':<25s} {'Card ID':<20s} {'GT Var':<18s} {'Art Comb':>9s} {'Bdr Comb':>9s} {'Max':>9s} {'Detected':<20s}")
    print("-" * 120)

    # Sort by max combined score descending
    ranked = sorted(results, key=lambda r: max(r["no_era"]["art_combined_score"], r["no_era"]["border_combined_score"]), reverse=True)
    for r in ranked:
        ne = r["no_era"]
        max_score = max(ne["art_combined_score"], ne["border_combined_score"])
        mark = "" if ne["variant"] == r["gt_variant"] else " <-- WRONG"
        print(f"{r['name']:<25s} {r['card_id']:<20s} {r['gt_variant']:<18s} {ne['art_combined_score']:9.2f} {ne['border_combined_score']:9.2f} {max_score:9.2f} {ne['variant']:<20s}{mark}")

    # 1st edition false positive details
    print("\n" + "=" * 120)
    print("1st EDITION STAMP DETECTIONS")
    print("=" * 120)
    stamp_cards = [r for r in results if r["no_era"]["has_1st_edition_stamp"]]
    if stamp_cards:
        for r in stamp_cards:
            print(f"  FALSE POSITIVE: {r['name']} ({r['card_id']})")
    else:
        print("  No 1st edition stamps detected (correct -- none in test set)")

    # Full art false positives
    print("\n" + "=" * 120)
    print("FULL ART DETECTIONS")
    print("=" * 120)
    fa_cards = [r for r in results if r["no_era"]["is_full_art"]]
    if fa_cards:
        for r in fa_cards:
            print(f"  FALSE POSITIVE: {r['name']} ({r['card_id']})")
    else:
        print("  No full art detected (correct -- none in test set)")

    # Gold/rainbow rare false positives
    print("\n" + "=" * 120)
    print("GOLD/RAINBOW RARE DETECTIONS")
    print("=" * 120)
    gr_cards = [r for r in results if r["no_era"]["gold_rare_result"] is not None]
    if gr_cards:
        for r in gr_cards:
            print(f"  FALSE POSITIVE: {r['name']} ({r['card_id']}) -- detected as {r['no_era']['gold_rare_result']}")
    else:
        print("  No gold/rainbow detections (correct -- none in test set)")


if __name__ == "__main__":
    main()
