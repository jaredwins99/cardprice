#!/usr/bin/env python3
"""Test surface_detector.py on binder page segments vs reference images.

Runs detect_surface_defects() on 6 segment/reference pairs from the eval
dataset and prints defect_score, defect_count, mean_similarity, condition
grade, and the worst patch locations.

Usage:
    python -m scripts.test_surface_detector
    # or
    python scripts/test_surface_detector.py
"""

import json
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cardprice.ml.surface_detector import (
    detect_surface_defects,
    estimate_condition,
    extract_patch_tokens,
    render_heatmap,
)


def card_id_to_ref_path(card_id: str) -> Path:
    """Convert card_id like 'ex15-92/normal' to reference image path."""
    base, variant = card_id.split("/")
    # set_id is everything before the last dash-number
    # e.g. ex15-92 -> set_id=ex15, ecard3-80 -> set_id=ecard3
    parts = base.rsplit("-", 1)
    set_id = parts[0]
    return PROJECT_ROOT / "data" / "card_images" / set_id / f"{base}_{variant}.png"


# 6 test cases spanning different eras, card types, and binder pages
TEST_CASES = [
    # Page 0 (EX delta species / EX era)
    {
        "name": "Flygon ex delta (EX era, holo)",
        "segment": "data/inbox/page_20260228_174819_cards_hires/card_00.png",
        "card_id": "ex15-92/normal",
    },
    {
        "name": "Delcatty ex (EX era)",
        "segment": "data/inbox/page_20260228_174819_cards_hires/card_04.png",
        "card_id": "ex14-91/normal",
    },
    # Page 1 (e-card era)
    {
        "name": "Natu (e-Card era)",
        "segment": "data/inbox/page_20260228_195512_cards_hires/card_00.png",
        "card_id": "ecard3-80/normal",
    },
    {
        "name": "Raticate (e-Card era)",
        "segment": "data/inbox/page_20260228_195512_cards_hires/card_07.png",
        "card_id": "ecard3-89/normal",
    },
    # Page 2 (DP/Platinum era, holos)
    {
        "name": "Suicune (DP era, holo)",
        "segment": "data/inbox/page_20260228_202134_cards_hires/card_07.png",
        "card_id": "dp3-19/normal",
    },
    {
        "name": "Staraptor (DP era, reverse holo)",
        "segment": "data/inbox/page_20260228_202134_cards_hires/card_08.png",
        "card_id": "dp1-16/normal",
    },
]


def main():
    print("=" * 80)
    print("Surface Detector Test — Binder Segments vs Reference Images")
    print("=" * 80)
    print()

    # Pre-extract reference patches for speed (batch not needed for 6 images)
    results = []

    for tc in TEST_CASES:
        seg_path = PROJECT_ROOT / tc["segment"]
        ref_path = card_id_to_ref_path(tc["card_id"])

        if not seg_path.exists():
            print(f"SKIP {tc['name']}: segment not found at {seg_path}")
            continue
        if not ref_path.exists():
            print(f"SKIP {tc['name']}: reference not found at {ref_path}")
            continue

        print(f"--- {tc['name']} ---")
        print(f"  Segment:   {seg_path}")
        print(f"  Reference: {ref_path}")

        t0 = time.time()
        result = detect_surface_defects(str(seg_path), str(ref_path))
        elapsed = time.time() - t0

        cond = estimate_condition(result)

        print(f"  defect_score:    {result['defect_score']:.4f}")
        print(f"  defect_count:    {result['defect_count']}/{256} patches flagged")
        print(f"  defect_ratio:    {result['defect_ratio']:.3f}")
        print(f"  mean_similarity: {result['mean_similarity']:.4f}")
        print(f"  min_similarity:  {result['min_similarity']:.4f}")
        print(f"  condition:       {cond['grade']} ({cond['grade_abbrev']}) "
              f"[confidence: {cond['confidence']}]")
        print(f"  time:            {elapsed:.2f}s")

        # Show worst 5 patches
        if result["defect_patches"]:
            worst = result["defect_patches"][:5]
            print(f"  worst patches:   {', '.join(f'({r},{c})={s:.3f}' for r,c,s in worst)}")

        # Save heatmap (skip if matplotlib API incompatible)
        try:
            heatmap_dir = PROJECT_ROOT / "data" / "eval" / "surface_heatmaps"
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            card_label = tc["card_id"].replace("/", "_")
            heatmap_path = heatmap_dir / f"heatmap_{card_label}.png"
            render_heatmap(
                result["anomaly_map"],
                output_path=str(heatmap_path),
                title=f"{tc['name']} (score={result['defect_score']:.3f})",
                patch_threshold=result["patch_threshold"],
            )
            print(f"  heatmap saved:   {heatmap_path}")
        except Exception as e:
            print(f"  heatmap skipped: {e}")
        print()

        results.append({
            "name": tc["name"],
            "card_id": tc["card_id"],
            "defect_score": result["defect_score"],
            "defect_count": result["defect_count"],
            "defect_ratio": result["defect_ratio"],
            "mean_similarity": result["mean_similarity"],
            "min_similarity": result["min_similarity"],
            "grade": cond["grade"],
            "grade_abbrev": cond["grade_abbrev"],
            "confidence": cond["confidence"],
        })

    # Summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Card':<35} {'Score':>6} {'Flagged':>8} {'MeanSim':>8} {'MinSim':>8} {'Grade':>5}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<35} {r['defect_score']:>6.3f} "
              f"{r['defect_count']:>4}/256 "
              f"{r['mean_similarity']:>8.4f} {r['min_similarity']:>8.4f} "
              f"{r['grade_abbrev']:>5}")
    print()

    # Threshold sweep: re-run with various thresholds to find sweet spot
    print()
    print("=" * 80)
    print("THRESHOLD SWEEP")
    print("=" * 80)
    thresholds_to_try = [0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25]
    print(f"{'Card':<30}", end="")
    for t in thresholds_to_try:
        print(f" t={t:.2f}", end="")
    print()
    print("-" * 100)

    for tc in TEST_CASES:
        seg_path = PROJECT_ROOT / tc["segment"]
        ref_path = card_id_to_ref_path(tc["card_id"])
        if not seg_path.exists() or not ref_path.exists():
            continue

        # Extract once, reuse
        q_patches = extract_patch_tokens(str(seg_path))
        r_patches = extract_patch_tokens(str(ref_path))

        label = tc["name"][:29]
        print(f"{label:<30}", end="")
        for t in thresholds_to_try:
            res = detect_surface_defects(
                str(seg_path), str(ref_path),
                patch_threshold=t,
                query_patches=q_patches,
                ref_patches=r_patches,
            )
            print(f" {res['defect_count']:>3}/{256}", end="")
        print()
    print()
    print("Flagged patches at each threshold (lower threshold = more lenient)")
    print()

    # Analysis
    scores = [r["defect_score"] for r in results]
    mean_sims = [r["mean_similarity"] for r in results]
    if scores:
        print(f"Score range:          {min(scores):.3f} - {max(scores):.3f}")
        print(f"Mean similarity range: {min(mean_sims):.4f} - {max(mean_sims):.4f}")
        print()

        # Reasonableness check
        all_nm = all(r["grade_abbrev"] == "NM" for r in results)
        all_dmg = all(r["grade_abbrev"] == "DMG" for r in results)
        high_scores = [r for r in results if r["defect_score"] > 0.5]

        if all_dmg:
            print("WARNING: All cards graded as Damaged. Threshold is likely too strict.")
            print("  Consider raising patch_threshold from 0.85 to ~0.75 or 0.70.")
        elif len(high_scores) > len(results) // 2:
            print("WARNING: Most cards have high defect scores. Threshold may be too strict.")
            print("  Binder scans differ from pristine references due to lighting/angle.")
            print("  Consider lowering patch_threshold to accommodate scan artifacts.")
        elif all_nm:
            print("NOTE: All cards graded Near Mint. Threshold may be too lenient,")
            print("  or these cards are genuinely in good condition.")
        else:
            print("Grades look reasonable -- a mix of conditions as expected for binder scans.")

    # Diagnosis and recommendations
    print()
    print("=" * 80)
    print("DIAGNOSIS AND RECOMMENDATIONS")
    print("=" * 80)
    print("""
The current approach (DINOv2 patch-level cosine similarity) is fundamentally
limited for binder page scans vs digital reference images because:

1. HOLOGRAPHIC CARDS (mean_sim 0.20-0.43): Holofoil patterns reflect light
   differently in every photo. The digital reference shows a flat scan, while
   the binder photo captures glare/rainbow patterns. Patch similarity drops to
   near-random levels (0.20-0.43 vs ~0.43 for completely different cards).

2. NON-HOLO CARDS (mean_sim 0.68-0.73): Better but still low. Binder sleeve
   reflections, warm lighting color cast, and angle distortion reduce similarity.
   At threshold=0.35, only 12-22/256 patches flagged -- more reasonable.

3. EDGE/BORDER PATCHES: The 16x16 grid means border patches include sleeve
   edges, adjacent cards, or binder background -- these always differ from the
   clean reference border.

SUGGESTED FIXES:
- For binder scans: patch_threshold should be ~0.30-0.35 for non-holo cards.
- Holo cards need a different approach entirely (e.g., compare only text/border
  regions, or use a style-invariant feature space).
- Consider masking border patches (rows 0,15 and cols 0,15) since those always
  include non-card content.
- Best use case: comparing two PHOTOS of the same card (before/after shipping),
  not photo-vs-digital-reference.
- Alternative: train a lightweight condition classifier on labeled examples
  rather than relying on patch similarity to a single reference.
""")


    # Save JSON results
    json_path = PROJECT_ROOT / "data" / "eval" / "surface_detector_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
