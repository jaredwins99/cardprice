#!/usr/bin/env python3
"""Quick test: run 1 page to verify accuracy and timing."""
import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cardprice.ml import identify_page_v2

with open("data/eval/binder_eval.json") as f:
    eval_data = json.load(f)

for pi, page in enumerate(eval_data["pages"]):
    seg_dir = Path(page["segments_dir"])
    paths = [str(seg_dir / c["segment"]) for c in page["cards"]]
    gt_ids = [c["card_id"] for c in page["cards"]]

    t0 = time.time()
    results = identify_page_v2(paths)
    elapsed = time.time() - t0

    correct = 0
    for j, (r, gt) in enumerate(zip(results, gt_ids)):
        match = r.get("card_id") == gt
        if match: correct += 1
        print(f"  p{pi} card {j}: {'OK' if match else 'MISS'} got={r.get('card_id')} expected={gt} conf={r.get('confidence', 0):.2f} method={r.get('method')}")

    print(f"\n  Page {pi}: {correct}/{len(gt_ids)} in {elapsed:.1f}s\n")
