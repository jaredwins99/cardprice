#!/usr/bin/env python3
"""Evaluate type_detector accuracy on the binder eval set.

Loads ground truth types from data/card_names.json, runs type detection on
each segment in data/eval/binder_eval.json, and reports per-card results.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cardprice.ml.type_detector import detect_type


def load_ground_truth() -> dict:
    """Return {card_id_base: [type1, ...]} from card_names.json."""
    with open(ROOT / "data" / "card_names.json") as f:
        rows = json.load(f)
    gt = {}
    for row in rows:
        card_id = row[0]  # e.g. "ex15-92/normal"
        types = row[4] if len(row) > 4 else []
        base = card_id.split("/")[0]  # e.g. "ex15-92"
        gt[base] = types
    return gt


def main():
    with open(ROOT / "data" / "eval" / "binder_eval.json") as f:
        eval_data = json.load(f)

    gt = load_ground_truth()

    correct = 0
    total = 0
    skipped = 0
    results = []

    for page in eval_data["pages"]:
        seg_dir = ROOT / page["segments_dir"]
        for card in page["cards"]:
            card_id = card.get("card_id")
            name = card.get("name", "?")
            seg_path = seg_dir / card["segment"]

            # Skip empty slots
            if card_id is None:
                skipped += 1
                results.append({
                    "name": name,
                    "card_id": None,
                    "expected": "N/A",
                    "detected": "N/A",
                    "correct": "SKIP",
                })
                continue

            base = card_id.split("/")[0]
            expected_types = gt.get(base, [])
            if not expected_types:
                # No type info in ground truth
                skipped += 1
                results.append({
                    "name": name,
                    "card_id": card_id,
                    "expected": "UNKNOWN",
                    "detected": "?",
                    "correct": "SKIP",
                })
                continue

            # Run type detection
            preds = detect_type(str(seg_path), top_n=3)
            detected = preds[0][0] if preds else "?"
            conf = preds[0][1] if preds else 0.0

            # Check if detected type is in the card's type list
            match = detected in expected_types
            total += 1
            if match:
                correct += 1

            results.append({
                "name": name,
                "card_id": card_id,
                "expected": ", ".join(expected_types),
                "detected": detected,
                "confidence": f"{conf:.0%}",
                "alts": ", ".join(f"{n} {c:.0%}" for n, c in preds[1:]),
                "correct": "OK" if match else "WRONG",
            })

    # Print results
    print("=" * 90)
    print("TYPE DETECTOR EVALUATION")
    print("=" * 90)
    print()
    print(f"{'Name':<22} {'Expected':<14} {'Detected':<14} {'Conf':>6}  {'Result':<6}  Alts")
    print("-" * 90)

    for r in results:
        if r["correct"] == "SKIP":
            print(f"{r['name']:<22} {'---':<14} {'---':<14} {'':>6}  SKIP")
            continue
        print(
            f"{r['name']:<22} "
            f"{r['expected']:<14} "
            f"{r['detected']:<14} "
            f"{r.get('confidence', ''):>6}  "
            f"{r['correct']:<6}  "
            f"{r.get('alts', '')}"
        )

    print("-" * 90)
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}" if total else "No cards evaluated")
    if skipped:
        print(f"Skipped:  {skipped} (empty slots or no type info)")
    print()

    # Summary by type
    from collections import Counter
    type_correct = Counter()
    type_total = Counter()
    for r in results:
        if r["correct"] in ("OK", "WRONG"):
            exp = r["expected"]
            type_total[exp] += 1
            if r["correct"] == "OK":
                type_correct[exp] += 1

    print("Per-type breakdown:")
    for t in sorted(type_total.keys()):
        c = type_correct[t]
        tot = type_total[t]
        print(f"  {t:<14} {c}/{tot} = {c/tot:.0%}")


if __name__ == "__main__":
    main()
