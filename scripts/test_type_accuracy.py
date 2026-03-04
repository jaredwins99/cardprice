#!/usr/bin/env python3
"""Evaluate type_detector accuracy on all eval binder cards.

For each card in binder_eval.json:
  1. Run detect_type() on the segment image
  2. Look up ground-truth types from dim_cards -> dim_pokemon
  3. Check top-1 and top-3 accuracy

Results saved to data/eval/type_results.json
"""

import json
import logging
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cardprice.db.session import engine
from cardprice.ml.type_detector import detect_type
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_eval_cards():
    """Load all cards from binder_eval.json, skipping nulls."""
    eval_path = ROOT / "data" / "eval" / "binder_eval.json"
    with open(eval_path) as f:
        data = json.load(f)

    cards = []
    for page in data["pages"]:
        seg_dir = ROOT / page["segments_dir"]
        for card in page["cards"]:
            if card["card_id"] is None:
                continue  # skip empty slots
            cards.append({
                "card_id": card["card_id"],
                "name": card["name"],
                "segment_path": str(seg_dir / card["segment"]),
            })
    return cards


def lookup_ground_truth(card_ids):
    """Look up Pokemon types from DB for a list of card_ids.

    Returns dict: card_id -> list of type strings (e.g. ["Grass", "Poison"]).
    """
    result = {}
    with engine.connect() as conn:
        for cid in card_ids:
            row = conn.execute(text("""
                SELECT p.types
                FROM dim_cards c
                JOIN dim_pokemon p ON c.pokemon_id = p.pokemon_id
                WHERE c.card_id = :card_id
            """), {"card_id": cid}).fetchone()
            if row and row[0]:
                result[cid] = row[0]  # TEXT[] returned as list
            else:
                # Card might be a Trainer/Energy with no pokemon_id
                result[cid] = []
    return result


def main():
    cards = load_eval_cards()
    logger.info("Loaded %d eval cards", len(cards))

    # Get ground truth types
    card_ids = list(set(c["card_id"] for c in cards))
    gt_types = lookup_ground_truth(card_ids)

    results = []
    top1_correct = 0
    top3_correct = 0
    total_evaluated = 0

    for card in cards:
        cid = card["card_id"]
        name = card["name"]
        seg_path = card["segment_path"]

        gt = gt_types.get(cid, [])
        if not gt:
            logger.warning("No ground truth types for %s (%s) - skipping", cid, name)
            results.append({
                "card_id": cid,
                "name": name,
                "ground_truth": [],
                "predictions": [],
                "top1_correct": None,
                "top3_correct": None,
                "note": "No pokemon types in DB (Trainer/Energy/etc)",
            })
            continue

        # Run type detection
        try:
            preds = detect_type(seg_path, top_n=5)
        except Exception as e:
            logger.error("detect_type failed for %s: %s", seg_path, e)
            results.append({
                "card_id": cid,
                "name": name,
                "ground_truth": gt,
                "predictions": [],
                "top1_correct": False,
                "top3_correct": False,
                "note": f"Error: {e}",
            })
            total_evaluated += 1
            continue

        pred_types = [p[0] for p in preds]
        pred_confs = [{"type": p[0], "confidence": round(p[1], 4)} for p in preds]

        # Check: is any ground truth type in top-1?
        t1 = pred_types[0] in gt if pred_types else False
        # Check: is any ground truth type in top-3?
        t3 = any(pt in gt for pt in pred_types[:3])

        if t1:
            top1_correct += 1
        if t3:
            top3_correct += 1
        total_evaluated += 1

        results.append({
            "card_id": cid,
            "name": name,
            "ground_truth": gt,
            "predictions": pred_confs,
            "top1_correct": t1,
            "top3_correct": t3,
        })

        status = "OK" if t1 else ("top3" if t3 else "MISS")
        logger.info(
            "[%s] %-25s  gt=%-20s  pred=%-12s (%.0f%%)  %s",
            status, name, ",".join(gt), pred_types[0] if pred_types else "?",
            preds[0][1] * 100 if preds else 0,
            f"alts: {','.join(pred_types[1:3])}" if len(pred_types) > 1 else "",
        )

    # Summary
    logger.info("=" * 70)
    logger.info("RESULTS: %d cards evaluated", total_evaluated)
    logger.info("Top-1 accuracy: %d/%d = %.1f%%", top1_correct, total_evaluated,
                100 * top1_correct / total_evaluated if total_evaluated else 0)
    logger.info("Top-3 accuracy: %d/%d = %.1f%%", top3_correct, total_evaluated,
                100 * top3_correct / total_evaluated if total_evaluated else 0)

    # Save results
    output = {
        "total_cards": len(cards),
        "evaluated": total_evaluated,
        "skipped": len(cards) - total_evaluated,
        "top1_correct": top1_correct,
        "top1_accuracy": round(top1_correct / total_evaluated, 4) if total_evaluated else 0,
        "top3_correct": top3_correct,
        "top3_accuracy": round(top3_correct / total_evaluated, 4) if total_evaluated else 0,
        "per_card": results,
    }

    out_path = ROOT / "data" / "eval" / "type_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
