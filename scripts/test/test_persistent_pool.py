#!/usr/bin/env python3
"""Test persistent ProcessPool: run eval twice to measure cold vs warm timing."""
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cardprice.ml import identify_page_v2


def run_eval():
    with open("data/eval/binder_eval.json") as f:
        eval_data = json.load(f)

    total_correct = 0
    total_cards = 0

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
            if match:
                correct += 1
            status = "OK" if match else "MISS"
            print(f"  p{pi} card {j}: {status} got={r.get('card_id')} expected={gt} conf={r.get('confidence', 0):.2f} method={r.get('method')}")

        print(f"  Page {pi}: {correct}/{len(gt_ids)} in {elapsed:.1f}s")
        total_correct += correct
        total_cards += len(gt_ids)

    return total_correct, total_cards


if __name__ == "__main__":
    print("=== Run 1 (cold pool) ===")
    t1 = time.time()
    c1, n1 = run_eval()
    e1 = time.time() - t1
    print(f"\nRun 1: {c1}/{n1} = {c1/n1*100:.1f}% in {e1:.1f}s\n")

    print("=== Run 2 (warm pool) ===")
    t2 = time.time()
    c2, n2 = run_eval()
    e2 = time.time() - t2
    print(f"\nRun 2: {c2}/{n2} = {c2/n2*100:.1f}% in {e2:.1f}s")

    print(f"\nSpeedup: {e1:.1f}s -> {e2:.1f}s ({(1 - e2/e1)*100:.0f}% faster)")
    print(f"Accuracy: {c1}/{n1} = {c1/n1*100:.1f}%")

    if c1 < 25:
        print(f"\nWARNING: Accuracy dropped below 96.2% (25/26)!")
        sys.exit(1)
