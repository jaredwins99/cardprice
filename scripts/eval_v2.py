#!/usr/bin/env python3
"""Evaluate the v2 card identification pipeline against ground truth.

Ground truth: data/eval/binder_eval.json (3 pages, 27 cards)
Card segments: data/inbox/ subdirectories
Pipeline: cardprice.ml.identify_card_v2() and identify_page_v2()

Reports:
  - Overall accuracy (exact card_id match)
  - Per-page accuracy
  - Per-method accuracy breakdown
  - Name-level accuracy (correct Pokemon name even if wrong set/variant)
  - Detailed per-card results saved to data/eval/v2_eval_results.json
"""

import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_v2")


def load_ground_truth(eval_path: Path) -> list[dict]:
    """Load binder_eval.json and flatten to a list of card entries.

    Each entry has: segment_path, card_id, name, page_index, position.
    """
    with open(eval_path) as f:
        data = json.load(f)

    cards = []
    for page_idx, page in enumerate(data["pages"]):
        segments_dir = PROJECT_ROOT / page["segments_dir"]
        for card in page["cards"]:
            segment_path = segments_dir / card["segment"]
            cards.append({
                "segment_path": str(segment_path),
                "card_id": card["card_id"],  # may be None for empty slots
                "name": card["name"],
                "page_index": page_idx,
                "position": card["position"],
                "variant": card.get("variant"),
                "segments_dir": str(segments_dir),
            })
    return cards


def extract_name_from_card_id(card_id: str) -> str:
    """Extract a rough name identifier from card_id for name-level comparison.

    We can't do this well without DB, so we just compare the base card code.
    """
    if not card_id:
        return ""
    return card_id.split("/")[0]  # e.g. "ex15-92"


def extract_set_from_card_id(card_id: str) -> str:
    """Extract set ID from card_id like 'base1-4/normal' -> 'base1'."""
    if not card_id:
        return ""
    base = card_id.split("/")[0]
    parts = base.rsplit("-", 1)
    return parts[0] if len(parts) == 2 else base


def run_eval():
    """Run the v2 eval pipeline."""
    eval_path = PROJECT_ROOT / "data" / "eval" / "binder_eval.json"
    output_path = PROJECT_ROOT / "data" / "eval" / "v2_eval_results.json"

    if not eval_path.exists():
        logger.error("Ground truth not found: %s", eval_path)
        sys.exit(1)

    # Load ground truth
    gt_cards = load_ground_truth(eval_path)
    logger.info("Loaded %d ground truth cards from %s", len(gt_cards), eval_path)

    # Import v2 pipeline
    from cardprice.ml import identify_card_v2

    # Get a DB session
    session = None
    try:
        from cardprice.db.session import SessionLocal
        session = SessionLocal()
        logger.info("DB session acquired")
    except Exception as e:
        logger.warning("Could not get DB session (will limit functionality): %s", e)

    # Run identification on each card
    results = []
    total_time = 0.0
    for i, gt in enumerate(gt_cards):
        segment_path = gt["segment_path"]
        expected_id = gt["card_id"]
        expected_name = gt["name"]

        if expected_id is None:
            # Empty slot - skip
            logger.info(
                "[%2d/%d] SKIP empty slot at page %d pos %s",
                i + 1, len(gt_cards), gt["page_index"], gt["position"],
            )
            results.append({
                "index": i,
                "page_index": gt["page_index"],
                "position": gt["position"],
                "segment_path": segment_path,
                "expected_card_id": None,
                "expected_name": expected_name,
                "predicted_card_id": None,
                "predicted_method": "skip",
                "predicted_confidence": 0.0,
                "predicted_explanation": "Empty slot, skipped",
                "exact_match": True,  # null == null
                "name_match": True,
                "set_match": True,
                "time_seconds": 0.0,
            })
            continue

        if not Path(segment_path).exists():
            logger.warning("[%2d/%d] Segment not found: %s", i + 1, len(gt_cards), segment_path)
            results.append({
                "index": i,
                "page_index": gt["page_index"],
                "position": gt["position"],
                "segment_path": segment_path,
                "expected_card_id": expected_id,
                "expected_name": expected_name,
                "predicted_card_id": None,
                "predicted_method": "missing_segment",
                "predicted_confidence": 0.0,
                "predicted_explanation": "Segment file not found",
                "exact_match": False,
                "name_match": False,
                "set_match": False,
                "time_seconds": 0.0,
            })
            continue

        # Run v2 pipeline
        logger.info(
            "[%2d/%d] Processing: %s (expected: %s / %s)",
            i + 1, len(gt_cards),
            Path(segment_path).name, expected_id, expected_name,
        )

        t0 = time.time()
        try:
            result = identify_card_v2(segment_path, session=session)
        except Exception as e:
            logger.error("  identify_card_v2 failed: %s", e, exc_info=True)
            result = {
                "card_id": None,
                "confidence": 0.0,
                "method": "error",
                "explanation": str(e),
                "raw_response": {},
            }
        elapsed = time.time() - t0
        total_time += elapsed

        pred_id = result.get("card_id")
        pred_method = result.get("method", "?")
        pred_conf = result.get("confidence", 0.0)
        pred_expl = result.get("explanation", "")

        # Exact card_id match
        exact = (pred_id == expected_id)

        # Name-level match: same base card code (ignoring variant)
        name_match = (
            extract_name_from_card_id(pred_id) == extract_name_from_card_id(expected_id)
            if pred_id and expected_id else False
        )

        # Set-level match
        set_match = (
            extract_set_from_card_id(pred_id) == extract_set_from_card_id(expected_id)
            if pred_id and expected_id else False
        )

        marker = "OK" if exact else ("NAME" if name_match else "MISS")
        logger.info(
            "  [%4s] predicted=%s (conf=%.3f, method=%s, %.1fs)",
            marker, pred_id, pred_conf, pred_method, elapsed,
        )

        # Serialize raw_response for JSON (handle non-serializable types)
        raw = result.get("raw_response", {})
        try:
            json.dumps(raw)
        except (TypeError, ValueError):
            raw = str(raw)

        results.append({
            "index": i,
            "page_index": gt["page_index"],
            "position": gt["position"],
            "segment_path": segment_path,
            "expected_card_id": expected_id,
            "expected_name": expected_name,
            "predicted_card_id": pred_id,
            "predicted_method": pred_method,
            "predicted_confidence": pred_conf,
            "predicted_explanation": pred_expl,
            "exact_match": exact,
            "name_match": name_match,
            "set_match": set_match,
            "time_seconds": round(elapsed, 2),
            "raw_response": raw,
        })

    # -----------------------------------------------------------------------
    # Compute summary statistics
    # -----------------------------------------------------------------------

    # Filter out empty slots for accuracy calculations
    scored = [r for r in results if r["expected_card_id"] is not None]
    n_scored = len(scored)

    # Overall accuracy
    n_exact = sum(1 for r in scored if r["exact_match"])
    n_name = sum(1 for r in scored if r["name_match"])
    n_set = sum(1 for r in scored if r["set_match"])
    overall_accuracy = n_exact / n_scored if n_scored else 0.0
    name_accuracy = n_name / n_scored if n_scored else 0.0
    set_accuracy = n_set / n_scored if n_scored else 0.0

    # Per-page accuracy
    page_stats = defaultdict(lambda: {"total": 0, "exact": 0, "name": 0, "set": 0})
    for r in scored:
        pg = r["page_index"]
        page_stats[pg]["total"] += 1
        if r["exact_match"]:
            page_stats[pg]["exact"] += 1
        if r["name_match"]:
            page_stats[pg]["name"] += 1
        if r["set_match"]:
            page_stats[pg]["set"] += 1

    per_page = {}
    for pg, stats in sorted(page_stats.items()):
        per_page[f"page_{pg}"] = {
            "total": stats["total"],
            "exact_correct": stats["exact"],
            "exact_accuracy": round(stats["exact"] / stats["total"], 3) if stats["total"] else 0,
            "name_correct": stats["name"],
            "name_accuracy": round(stats["name"] / stats["total"], 3) if stats["total"] else 0,
        }

    # Per-method accuracy
    method_stats = defaultdict(lambda: {"total": 0, "exact": 0, "name": 0})
    for r in scored:
        m = r["predicted_method"] or "none"
        method_stats[m]["total"] += 1
        if r["exact_match"]:
            method_stats[m]["exact"] += 1
        if r["name_match"]:
            method_stats[m]["name"] += 1

    per_method = {}
    for m, stats in sorted(method_stats.items()):
        per_method[m] = {
            "total": stats["total"],
            "exact_correct": stats["exact"],
            "exact_accuracy": round(stats["exact"] / stats["total"], 3) if stats["total"] else 0,
            "name_correct": stats["name"],
            "name_accuracy": round(stats["name"] / stats["total"], 3) if stats["total"] else 0,
        }

    # Average confidence
    avg_conf = (
        sum(r["predicted_confidence"] for r in scored) / n_scored
        if n_scored else 0.0
    )

    # Confidence by correctness
    correct_confs = [r["predicted_confidence"] for r in scored if r["exact_match"]]
    wrong_confs = [r["predicted_confidence"] for r in scored if not r["exact_match"]]
    avg_conf_correct = sum(correct_confs) / len(correct_confs) if correct_confs else 0.0
    avg_conf_wrong = sum(wrong_confs) / len(wrong_confs) if wrong_confs else 0.0

    summary = {
        "total_cards": len(gt_cards),
        "scored_cards": n_scored,
        "skipped_empty": len(gt_cards) - n_scored,
        "exact_correct": n_exact,
        "exact_accuracy": round(overall_accuracy, 3),
        "name_correct": n_name,
        "name_accuracy": round(name_accuracy, 3),
        "set_correct": n_set,
        "set_accuracy": round(set_accuracy, 3),
        "avg_confidence": round(avg_conf, 3),
        "avg_confidence_correct": round(avg_conf_correct, 3),
        "avg_confidence_wrong": round(avg_conf_wrong, 3),
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_card": round(total_time / n_scored, 1) if n_scored else 0,
        "per_page": per_page,
        "per_method": per_method,
    }

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output_data = {
        "summary": summary,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info("Results saved to %s", output_path)

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("V2 PIPELINE EVALUATION RESULTS")
    print("=" * 80)

    print(f"\nCards scored: {n_scored} (+ {len(gt_cards) - n_scored} empty slots skipped)")
    print(f"Total time:  {total_time:.1f}s ({total_time / n_scored:.1f}s/card)" if n_scored else "")

    print(f"\n{'Metric':<25} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print("-" * 55)
    print(f"{'Exact card_id match':<25} {n_exact:>8} {n_scored:>6} {overall_accuracy:>10.1%}")
    print(f"{'Name-level match':<25} {n_name:>8} {n_scored:>6} {name_accuracy:>10.1%}")
    print(f"{'Set-level match':<25} {n_set:>8} {n_scored:>6} {set_accuracy:>10.1%}")

    print(f"\n{'Confidence':<25} {'Avg':>10}")
    print("-" * 40)
    print(f"{'Overall':<25} {avg_conf:>10.3f}")
    print(f"{'Correct predictions':<25} {avg_conf_correct:>10.3f}")
    print(f"{'Wrong predictions':<25} {avg_conf_wrong:>10.3f}")

    print(f"\nPer-page accuracy:")
    print(f"  {'Page':<10} {'Exact':>8} {'Name':>8} {'Total':>6}")
    print("  " + "-" * 36)
    for pg_name, pg_data in sorted(per_page.items()):
        print(
            f"  {pg_name:<10} "
            f"{pg_data['exact_correct']}/{pg_data['total']:>2} ({pg_data['exact_accuracy']:.0%})  "
            f"{pg_data['name_correct']}/{pg_data['total']:>2} ({pg_data['name_accuracy']:.0%})  "
            f"{pg_data['total']:>4}"
        )

    print(f"\nPer-method breakdown:")
    print(f"  {'Method':<35} {'Exact':>8} {'Name':>8} {'Count':>6}")
    print("  " + "-" * 60)
    for m, m_data in sorted(per_method.items(), key=lambda x: -x[1]["total"]):
        print(
            f"  {m:<35} "
            f"{m_data['exact_correct']}/{m_data['total']:>2} ({m_data['exact_accuracy']:.0%})  "
            f"{m_data['name_correct']}/{m_data['total']:>2} ({m_data['name_accuracy']:.0%})  "
            f"{m_data['total']:>4}"
        )

    # Print detailed card-by-card results
    print(f"\nDetailed results:")
    print(f"  {'#':<3} {'Pg':>2} {'Segment':<16} {'Expected':<24} {'Predicted':<24} {'Method':<25} {'Conf':>5} {'Time':>5} {'Status':<5}")
    print("  " + "-" * 125)
    for r in results:
        if r["expected_card_id"] is None:
            status = "SKIP"
        elif r["exact_match"]:
            status = "OK"
        elif r["name_match"]:
            status = "NAME"
        else:
            status = "MISS"

        seg_name = Path(r["segment_path"]).name
        exp_id = r["expected_card_id"] or "(empty)"
        pred_id = r["predicted_card_id"] or "(none)"

        print(
            f"  {r['index']:<3} {r['page_index']:>2} {seg_name:<16} "
            f"{exp_id:<24} {pred_id:<24} "
            f"{r['predicted_method']:<25} "
            f"{r['predicted_confidence']:>5.3f} "
            f"{r['time_seconds']:>5.1f}s "
            f"{status:<5}"
        )

    print("\n" + "=" * 80)

    return summary


if __name__ == "__main__":
    run_eval()
