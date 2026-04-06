#!/usr/bin/env python3
"""Evaluate ref_matcher on all 27 eval cards using ground truth attributes.

Tests two modes:
1. Name-only: match_by_reference(image, pokemon_name=gt_name)
2. Name+HP: match_by_reference(image, pokemon_name=gt_name, hp=gt_hp)

This measures the CEILING of reference matching — if name identification
were perfect, how good would the final card identification be?

Results saved to data/eval/ref_matcher_results.json
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from cardprice.db.session import SessionLocal
from cardprice.ml.ref_matcher import match_by_reference, get_candidate_card_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_eval_data():
    eval_path = PROJECT_ROOT / "data" / "eval" / "binder_eval.json"
    with open(eval_path) as f:
        return json.load(f)


def get_hp_from_db(card_id: str, session) -> int | None:
    """Look up HP for a card_id from dim_cards."""
    row = session.execute(
        text("SELECT hp FROM dim_cards WHERE card_id = :cid"),
        {"cid": card_id},
    ).fetchone()
    if row and row[0]:
        return int(row[0])
    return None


def run_eval():
    eval_data = load_eval_data()
    session = SessionLocal()

    results = []
    name_only_correct = 0
    name_hp_correct = 0
    total = 0
    skipped = 0

    for page in eval_data["pages"]:
        segments_dir = PROJECT_ROOT / page["segments_dir"]
        for card_entry in page["cards"]:
            gt_card_id = card_entry["card_id"]
            gt_name = card_entry["name"]

            # Skip empty slots
            if gt_card_id is None:
                logger.info("Skipping empty slot: %s", gt_name)
                skipped += 1
                continue

            segment_path = segments_dir / card_entry["segment"]
            if not segment_path.is_file():
                logger.warning("Segment not found: %s", segment_path)
                skipped += 1
                continue

            total += 1
            gt_hp = get_hp_from_db(gt_card_id, session)

            # Count candidates for context
            candidates_name = get_candidate_card_ids(gt_name, session=session)
            candidates_name_hp = get_candidate_card_ids(gt_name, hp=gt_hp, session=session) if gt_hp else candidates_name

            logger.info(
                "=== Card %d: %s (GT: %s, HP: %s) ===",
                total, gt_name, gt_card_id, gt_hp,
            )
            logger.info("  Candidates (name only): %d", len(candidates_name))
            logger.info("  Candidates (name+HP): %d", len(candidates_name_hp))

            # --- Test 1: Name only ---
            t0 = time.time()
            result_name, score_name = match_by_reference(
                str(segment_path),
                pokemon_name=gt_name,
                session=session,
            )
            t_name = time.time() - t0
            name_match = result_name == gt_card_id
            if name_match:
                name_only_correct += 1
            logger.info(
                "  Name-only: %s (score=%.4f, time=%.2fs) %s",
                result_name, score_name, t_name,
                "CORRECT" if name_match else f"WRONG (gt={gt_card_id})",
            )

            # --- Test 2: Name + HP ---
            t0 = time.time()
            if gt_hp:
                result_hp, score_hp = match_by_reference(
                    str(segment_path),
                    pokemon_name=gt_name,
                    hp=gt_hp,
                    session=session,
                )
            else:
                result_hp, score_hp = result_name, score_name
            t_hp = time.time() - t0
            hp_match = result_hp == gt_card_id
            if hp_match:
                name_hp_correct += 1
            logger.info(
                "  Name+HP:   %s (score=%.4f, time=%.2fs) %s",
                result_hp, score_hp, t_hp,
                "CORRECT" if hp_match else f"WRONG (gt={gt_card_id})",
            )

            # Record result
            results.append({
                "segment": str(segment_path),
                "gt_card_id": gt_card_id,
                "gt_name": gt_name,
                "gt_hp": gt_hp,
                "candidates_name_only": len(candidates_name),
                "candidates_name_hp": len(candidates_name_hp),
                "name_only": {
                    "result": result_name,
                    "score": round(score_name, 4),
                    "correct": name_match,
                    "time_s": round(t_name, 2),
                },
                "name_hp": {
                    "result": result_hp,
                    "score": round(score_hp, 4),
                    "correct": hp_match,
                    "time_s": round(t_hp, 2),
                },
            })

    session.close()

    # Summary
    summary = {
        "total": total,
        "skipped": skipped,
        "name_only_correct": name_only_correct,
        "name_only_accuracy": round(name_only_correct / total, 4) if total else 0,
        "name_hp_correct": name_hp_correct,
        "name_hp_accuracy": round(name_hp_correct / total, 4) if total else 0,
    }

    output = {
        "description": "ref_matcher evaluation with ground truth name/HP (oracle ceiling test)",
        "summary": summary,
        "results": results,
    }

    out_path = PROJECT_ROOT / "data" / "eval" / "ref_matcher_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("REF MATCHER EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total cards evaluated: {total} (skipped: {skipped})")
    print(f"Name-only accuracy:    {name_only_correct}/{total} = {summary['name_only_accuracy']:.1%}")
    print(f"Name+HP accuracy:      {name_hp_correct}/{total} = {summary['name_hp_accuracy']:.1%}")
    print(f"\nResults saved to: {out_path}")

    # Print failure details
    failures_name = [r for r in results if not r["name_only"]["correct"]]
    failures_hp = [r for r in results if not r["name_hp"]["correct"]]

    if failures_name:
        print(f"\n--- Name-only failures ({len(failures_name)}) ---")
        for r in failures_name:
            print(f"  {r['gt_name']} (GT: {r['gt_card_id']}, Got: {r['name_only']['result']}, "
                  f"Score: {r['name_only']['score']:.4f}, Candidates: {r['candidates_name_only']})")

    if failures_hp:
        print(f"\n--- Name+HP failures ({len(failures_hp)}) ---")
        for r in failures_hp:
            print(f"  {r['gt_name']} (GT: {r['gt_card_id']}, Got: {r['name_hp']['result']}, "
                  f"Score: {r['name_hp']['score']:.4f}, Candidates: {r['candidates_name_hp']})")


if __name__ == "__main__":
    run_eval()
