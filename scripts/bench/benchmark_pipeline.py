#!/usr/bin/env python3
"""Rigorous pipeline benchmark: measures real wall-clock time for each stage.

Usage:
    python scripts/bench/benchmark_pipeline.py                    # benchmark 3 pages
    python scripts/bench/benchmark_pipeline.py data/inbox/page_20260228_174819.jpg  # specific image
    python scripts/bench/benchmark_pipeline.py --all              # all inbox pages

Outputs a table with per-stage timings and totals. No estimates, no lies.
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def benchmark_page(image_path: str) -> dict:
    """Benchmark the full pipeline for one binder page image.

    Returns dict with per-stage wall-clock timings in seconds.
    """
    from cardprice.ml.card_segmenter import segment_cards
    from cardprice.ml import identify_page_v2
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    result = {"image": os.path.basename(image_path), "stages": {}}

    # Stage 1: Segmentation
    t0 = time.time()
    cards = segment_cards(image_path)
    t_seg = time.time() - t0
    result["stages"]["1_segmentation"] = round(t_seg, 2)
    result["n_cards"] = len(cards)

    if not cards:
        result["stages"]["total"] = round(t_seg, 2)
        result["error"] = "No cards segmented"
        return result

    # segment_cards returns paths (PosixPath or str)
    card_paths = [str(c) for c in cards]

    # Stage 2: Identification (includes OCR, DINOv2, attacks, page context)
    engine = create_engine("postgresql+psycopg2://godli@/cardprice")
    with Session(engine) as session:
        t0 = time.time()
        page_results = identify_page_v2(card_paths, session=session, detect_variants=False)
        t_id = time.time() - t0

    result["stages"]["2_identification"] = round(t_id, 2)
    result["stages"]["total"] = round(t_seg + t_id, 2)

    # Per-card results
    result["cards"] = []
    for i, r in enumerate(page_results):
        result["cards"].append({
            "idx": i,
            "name": r.get("card_name", r.get("name", "?")),
            "confidence": round(r.get("confidence", 0), 3),
            "method": r.get("method", "?"),
        })

    avg_conf = sum(c["confidence"] for c in result["cards"]) / max(len(result["cards"]), 1)
    result["avg_confidence"] = round(avg_conf, 3)

    return result


def print_results(results: list[dict]):
    """Print benchmark results as a formatted table."""
    print("\n" + "=" * 80)
    print("PIPELINE BENCHMARK RESULTS")
    print("=" * 80)

    for r in results:
        stages = r["stages"]
        print(f"\n--- {r['image']} ({r.get('n_cards', 0)} cards) ---")
        print(f"  Segmentation:    {stages.get('1_segmentation', '?'):>6.2f}s")
        print(f"  Identification:  {stages.get('2_identification', '?'):>6.2f}s")
        print(f"  TOTAL:           {stages.get('total', '?'):>6.2f}s")
        print(f"  Avg confidence:  {r.get('avg_confidence', 0):.3f}")
        if r.get("error"):
            print(f"  ERROR: {r['error']}")
        if r.get("cards"):
            for c in r["cards"]:
                flag = "✓" if c["confidence"] >= 0.8 else "?" if c["confidence"] >= 0.5 else "✗"
                print(f"    {flag} [{c['idx']}] {c['name']:<30s} conf={c['confidence']:.3f}  method={c['method']}")

    # Summary
    print("\n" + "=" * 80)
    totals = [r["stages"].get("total", 0) for r in results]
    segs = [r["stages"].get("1_segmentation", 0) for r in results]
    ids = [r["stages"].get("2_identification", 0) for r in results]
    n = len(results)
    print(f"SUMMARY ({n} pages)")
    print(f"  Avg segmentation:   {sum(segs)/n:>6.2f}s")
    print(f"  Avg identification: {sum(ids)/n:>6.2f}s")
    print(f"  Avg total:          {sum(totals)/n:>6.2f}s")
    print(f"  Min/Max total:      {min(totals):.2f}s / {max(totals):.2f}s")
    print("=" * 80)

    # Save raw JSON
    outpath = ROOT / "data" / "eval" / "benchmark_results.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {outpath}")


def main():
    args = sys.argv[1:]

    if args and args[0] == "--all":
        images = sorted(Path("data/inbox").glob("page_*.jpg"))
    elif args:
        images = [Path(a) for a in args]
    else:
        # Default: 3 diverse pages
        candidates = sorted(Path("data/inbox").glob("page_*.jpg"))
        images = candidates[:3] if len(candidates) >= 3 else candidates

    if not images:
        print("No images found. Provide paths or use --all.")
        sys.exit(1)

    print(f"Benchmarking {len(images)} page(s)...")
    print(f"Images: {[str(p) for p in images]}")

    results = []
    for img in images:
        print(f"\n>>> Processing {img.name}...")
        r = benchmark_page(str(img))
        results.append(r)
        print(f"    Done in {r['stages'].get('total', '?')}s")

    print_results(results)


if __name__ == "__main__":
    main()
