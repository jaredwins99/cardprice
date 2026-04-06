#!/usr/bin/env python3
"""Test edge_whitening module on all binder page segments from binder_eval.json.

Runs measure_edge_whitening() on all ~27 segments across 3 binder pages and
reports per-card metrics to validate behavior at 1008x1530 resolution.
"""

import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cardprice.ml.edge_whitening import measure_edge_whitening


def main():
    eval_path = ROOT / "data" / "eval" / "binder_eval.json"
    with open(eval_path) as f:
        eval_data = json.load(f)

    # Header
    print(f"{'Page':>6} {'Pos':>5} {'Name':<25} {'Shape':>12} "
          f"{'Overall':>9} {'Worst Edge':>11} {'Worst Ratio':>12} "
          f"{'MaxRun':>7} {'Clusters':>9} {'Condition':>10}")
    print("-" * 120)

    all_results = []

    for page_idx, page in enumerate(eval_data["pages"]):
        seg_dir = ROOT / page["segments_dir"]
        for card in page["cards"]:
            seg_path = seg_dir / card["segment"]
            name = card["name"]
            pos = f"{card['position'][0]},{card['position'][1]}"

            try:
                result = measure_edge_whitening(str(seg_path))
            except Exception as e:
                print(f"  Page {page_idx} {pos:>5} {name:<25} ERROR: {e}")
                continue

            shape_str = f"{result['image_shape'][1]}x{result['image_shape'][0]}"
            print(f"  P{page_idx}   {pos:>5} {name:<25} {shape_str:>12} "
                  f"{result['overall_ratio']:>9.6f} {result['worst_edge']:>11} "
                  f"{result['worst_ratio']:>12.6f} {result['max_white_run']:>7} "
                  f"{result['cluster_count']:>9} {result['tcg_condition']:>10}")

            all_results.append({
                "page": page_idx,
                "pos": pos,
                "name": name,
                "card_id": card.get("card_id"),
                **result,
            })

    # Summary statistics
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    ratios = [r["overall_ratio"] for r in all_results]
    conditions = {}
    for r in all_results:
        cond = r["tcg_condition"]
        conditions.setdefault(cond, []).append(r["name"])

    print(f"\nTotal segments analyzed: {len(all_results)}")
    print(f"Overall ratio range: {min(ratios):.6f} - {max(ratios):.6f}")
    print(f"Mean overall ratio: {sum(ratios)/len(ratios):.6f}")
    print(f"Median overall ratio: {sorted(ratios)[len(ratios)//2]:.6f}")

    print("\nCondition distribution:")
    for cond in ["NM", "LP", "MP", "HP"]:
        cards = conditions.get(cond, [])
        print(f"  {cond}: {len(cards)} cards")
        for c in cards:
            print(f"       - {c}")

    # Flag suspicious results
    print("\n" + "-" * 120)
    print("ANALYSIS: Potential false positives (LP or worse)")
    print("-" * 120)
    flagged = [r for r in all_results if r["tcg_condition"] not in ("NM",)]
    if not flagged:
        print("  None - all cards classified as NM")
    else:
        for r in flagged:
            print(f"\n  {r['name']} (Page {r['page']}, pos {r['pos']}):")
            print(f"    overall_ratio={r['overall_ratio']:.6f}, "
                  f"worst_edge={r['worst_edge']} ({r['worst_ratio']:.6f})")
            print(f"    max_white_run={r['max_white_run']}, clusters={r['cluster_count']}")
            print(f"    condition={r['tcg_condition']} ({r['condition_label']})")
            # Per-edge detail
            for side in ("top", "bottom", "left", "right"):
                e = r["edges"][side]
                print(f"      {side:>6}: ratio={e['whitening_ratio']:.6f}, "
                      f"max_run={e['max_white_run']}, clusters={e['cluster_count']}, "
                      f"mean_L={e['mean_lightness']}")

    # Check for binder sleeve glare patterns
    print("\n" + "-" * 120)
    print("ANALYSIS: Edge-by-edge hotspots (any edge > 0.003)")
    print("-" * 120)
    any_hotspot = False
    for r in all_results:
        for side in ("top", "bottom", "left", "right"):
            e = r["edges"][side]
            if e["whitening_ratio"] > 0.003:
                any_hotspot = True
                print(f"  {r['name']:25s} P{r['page']} {r['pos']:>5} "
                      f"{side:>6} ratio={e['whitening_ratio']:.6f} "
                      f"run={e['max_white_run']} clusters={e['cluster_count']} "
                      f"mean_L={e['mean_lightness']}")
    if not any_hotspot:
        print("  None - all individual edges below 0.003")

    print("\n" + "-" * 120)
    print("RESOLUTION NOTE")
    print("-" * 120)
    print("  These segments are 1008x1530 from binder page scans.")
    print("  The edge_whitening module strip_width=30 scales to ~30px at this resolution.")
    print("  At this resolution, subtle whitening may be below detection threshold.")
    print(f"  Consider: if ALL cards are NM, the thresholds may be too lenient for this resolution,")
    print(f"  OR these cards genuinely have minimal edge wear visible at scan resolution.")


if __name__ == "__main__":
    main()
