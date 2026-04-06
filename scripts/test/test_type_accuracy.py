#!/usr/bin/env python3
"""Baseline accuracy test for color_detector and type_detector on eval segments.

Loads ground truth from data/eval/binder_eval.json, looks up each card's
Pokemon type from data/card_names.json (no DB required), then runs both
detectors on the hi-res segments and reports accuracy + confusion patterns.

Usage:
    python scripts/test_type_accuracy.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cardprice.ml.color_detector import detect_color_type
from cardprice.ml.type_detector import detect_type


def load_eval_cards() -> list[dict]:
    """Load eval cards with their ground-truth types.

    Uses binder_eval.json for segment paths and card IDs, and
    card_names.json for type lookups (no DB needed).

    card_names.json format: [card_id, name, set_id, hp, types_list]
    where types_list is the CARD type (may differ from species type
    for delta species cards).
    """
    eval_path = PROJECT_ROOT / "data" / "eval" / "binder_eval.json"
    names_path = PROJECT_ROOT / "data" / "card_names.json"

    with open(eval_path) as f:
        eval_data = json.load(f)

    # Build card_id -> types lookup from card_names.json
    with open(names_path) as f:
        card_names = json.load(f)
    type_lookup = {}
    for entry in card_names:
        card_id = entry[0]  # e.g. "ex15-92/normal"
        types = entry[4]    # e.g. ["Psychic"]
        if types:
            base_id = card_id.split("/")[0]
            type_lookup[base_id] = types

    cards = []
    for page in eval_data["pages"]:
        segments_dir = PROJECT_ROOT / page["segments_dir"]
        for card in page["cards"]:
            card_id = card.get("card_id")
            if card_id is None:
                continue  # empty slot

            base_id = card_id.split("/")[0]
            types = type_lookup.get(base_id, [])
            if not types:
                print(f"WARNING: no type data for {card_id} ({card['name']})")
                continue

            segment_path = segments_dir / card["segment"]
            if not segment_path.exists():
                print(f"WARNING: segment not found: {segment_path}")
                continue

            card_type = types[0]

            cards.append({
                "card_id": card_id,
                "name": card["name"],
                "segment_path": str(segment_path),
                "card_type": card_type,
                "page": Path(page["image"]).stem,
            })

    return cards


def normalize_type(t: str) -> str:
    """Normalize type names between detectors.

    type_detector uses 'Dark', card_names.json uses 'Darkness'.
    Standardize to the card_names.json convention.
    """
    mapping = {
        "Dark": "Darkness",
    }
    return mapping.get(t, t)


def run_detector(cards: list[dict], detector_fn, detector_name: str) -> dict:
    """Run a detector on all cards and compute accuracy metrics."""
    correct = 0
    total = 0
    top3_correct = 0
    results = []
    confusion = defaultdict(Counter)  # expected -> predicted -> count

    for card in cards:
        total += 1
        try:
            preds = detector_fn(card["segment_path"], top_n=3)
        except Exception as e:
            results.append({
                **card,
                "predicted": f"ERROR: {e}",
                "confidence": 0.0,
                "correct": False,
                "top3_correct": False,
                "all_preds": [],
            })
            confusion[card["card_type"]]["ERROR"] += 1
            continue

        predicted = normalize_type(preds[0][0])
        conf = preds[0][1]
        expected = card["card_type"]
        is_correct = predicted == expected

        # Check if correct type appears in top 3
        top3_types = [normalize_type(p[0]) for p in preds]
        in_top3 = expected in top3_types

        if is_correct:
            correct += 1
        if in_top3:
            top3_correct += 1

        confusion[expected][predicted] += 1

        results.append({
            **card,
            "predicted": predicted,
            "confidence": conf,
            "correct": is_correct,
            "top3_correct": in_top3,
            "all_preds": [(normalize_type(p[0]), round(p[1], 3)) for p in preds],
        })

    accuracy = correct / total if total > 0 else 0.0
    top3_accuracy = top3_correct / total if total > 0 else 0.0

    return {
        "detector": detector_name,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "top3_correct": top3_correct,
        "top3_accuracy": top3_accuracy,
        "results": results,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def print_report(report: dict):
    """Print a formatted accuracy report."""
    print()
    print("=" * 100)
    print(f"  {report['detector']}")
    print(f"  Top-1 Accuracy: {report['correct']}/{report['total']} = {report['accuracy']:.1%}")
    print(f"  Top-3 Accuracy: {report['top3_correct']}/{report['total']} = {report['top3_accuracy']:.1%}")
    print("=" * 100)

    # Per-card results grouped by page
    print(f"\n{'Status':<6} {'Card':<28} {'Expected':<12} {'Predicted':<12} {'Conf':>5}  All Predictions")
    print("-" * 100)

    current_page = None
    for r in report["results"]:
        if r["page"] != current_page:
            current_page = r["page"]
            print(f"\n  --- {current_page} ---")

        marker = "OK" if r["correct"] else "MISS"
        name = r["name"][:26]
        preds_str = ", ".join(f"{t} {c:.0%}" for t, c in r["all_preds"])
        print(
            f"[{marker:4s}] {name:<28} {r['card_type']:<12} {r['predicted']:<12} "
            f"{r['confidence']:>4.0%}  {preds_str}"
        )

    # Confusion matrix
    print(f"\nConfusion Matrix ({report['detector']}):")
    print("-" * 60)
    print(f"{'Expected':<12} -> Predicted distribution")
    for expected in sorted(report["confusion"].keys()):
        preds = report["confusion"][expected]
        total_for_type = sum(preds.values())
        pred_str = ", ".join(
            f"{t}: {c}/{total_for_type}"
            for t, c in sorted(preds.items(), key=lambda x: -x[1])
        )
        print(f"  {expected:<12} -> {pred_str}")

    # Failure details
    failures = [r for r in report["results"] if not r["correct"]]
    if failures:
        print(f"\nFailure Details ({len(failures)} misses):")
        print("-" * 60)
        for r in failures:
            in_top3 = "(in top-3)" if r["top3_correct"] else "(NOT in top-3)"
            preds_str = ", ".join(f"{t} {c:.0%}" for t, c in r["all_preds"])
            print(f"  {r['name']:<25} expected={r['card_type']:<12} got={r['predicted']:<12} {in_top3}")
            print(f"    predictions: {preds_str}")
            print(f"    card_id: {r['card_id']}")

    # Per-type accuracy
    print(f"\nPer-Type Accuracy ({report['detector']}):")
    print("-" * 40)
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in report["results"]:
        type_stats[r["card_type"]]["total"] += 1
        if r["correct"]:
            type_stats[r["card_type"]]["correct"] += 1
    for t in sorted(type_stats.keys()):
        s = type_stats[t]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        print(f"  {t:<12} {s['correct']}/{s['total']} = {acc:.0%}")


def main():
    print("Loading eval cards from binder_eval.json + card_names.json...")
    cards = load_eval_cards()
    print(f"Found {len(cards)} cards with type information.\n")

    if not cards:
        print("ERROR: No cards found. Check paths.")
        sys.exit(1)

    # Show type distribution
    type_counts = Counter(c["card_type"] for c in cards)
    print(f"Type distribution: {dict(sorted(type_counts.items()))}")

    # Run both detectors
    print("\nRunning color_detector.detect_color_type() on all segments...")
    color_report = run_detector(cards, detect_color_type, "color_detector (K-means HSV)")

    print("Running type_detector.detect_type() on all segments...")
    type_report = run_detector(cards, detect_type, "type_detector (pixel voting)")

    # Print reports
    print_report(color_report)
    print_report(type_report)

    # Summary comparison
    print()
    print("=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Detector':<35} {'Top-1':>8} {'Top-3':>8}")
    print("-" * 60)
    for r in [color_report, type_report]:
        print(f"{r['detector']:<35} {r['accuracy']:>7.1%} {r['top3_accuracy']:>7.1%}")
    print()

    # Save results
    output_path = PROJECT_ROOT / "data" / "eval" / "type_detector_baseline.json"
    output = {
        "description": "Baseline type detection accuracy on eval binder segments",
        "color_detector": {
            "accuracy": color_report["accuracy"],
            "top3_accuracy": color_report["top3_accuracy"],
            "correct": color_report["correct"],
            "total": color_report["total"],
            "confusion": color_report["confusion"],
            "results": [
                {
                    "card_id": r["card_id"],
                    "name": r["name"],
                    "card_type": r["card_type"],
                    "predicted": r["predicted"],
                    "confidence": r["confidence"],
                    "correct": r["correct"],
                    "top3_correct": r["top3_correct"],
                    "all_preds": r["all_preds"],
                }
                for r in color_report["results"]
            ],
        },
        "type_detector": {
            "accuracy": type_report["accuracy"],
            "top3_accuracy": type_report["top3_accuracy"],
            "correct": type_report["correct"],
            "total": type_report["total"],
            "confusion": type_report["confusion"],
            "results": [
                {
                    "card_id": r["card_id"],
                    "name": r["name"],
                    "card_type": r["card_type"],
                    "predicted": r["predicted"],
                    "confidence": r["confidence"],
                    "correct": r["correct"],
                    "top3_correct": r["top3_correct"],
                    "all_preds": r["all_preds"],
                }
                for r in type_report["results"]
            ],
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
