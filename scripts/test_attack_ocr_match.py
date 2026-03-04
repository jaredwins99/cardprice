#!/usr/bin/env python3
"""Test attack OCR disambiguation on the 27 eval binder cards.

For each card in data/eval/binder_eval.json:
1. Run extract_attack_names() to OCR the attack region
2. Run identify_by_attacks() to get candidate card IDs
3. Check if the ground-truth card_id is in the results
4. Also test narrow_candidates() with a simulated candidate set

Reports:
- Per-card: OCR fragments, matched attacks, rank of correct card
- Summary: recall@1, recall@5, mean reciprocal rank
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_JSON = PROJECT_ROOT / "data" / "eval" / "binder_eval.json"
ATTACK_INDEX = PROJECT_ROOT / "data" / "attack_index.pkl"


def main():
    t0 = time.time()

    # Load eval data
    with open(EVAL_JSON) as f:
        eval_data = json.load(f)

    # Load attack index for ground truth lookups
    with open(ATTACK_INDEX, "rb") as f:
        attack_index = pickle.load(f)
    card_to_attacks = attack_index["card_to_attacks"]
    atk_to_cards = attack_index["attack_to_cards"]

    from cardprice.ml.attack_ocr import (
        extract_attack_names,
        identify_by_attacks,
        narrow_candidates,
        fuzzy_match_attacks,
    )

    results = []
    total_evaluated = 0
    total_skipped = 0
    correct_at_1 = 0
    correct_at_5 = 0
    reciprocal_ranks = []
    attacks_found_total = 0
    attacks_expected_total = 0

    # Also test narrowing with simulated candidate sets
    narrow_correct_at_1 = 0
    narrow_evaluated = 0

    for page_idx, page in enumerate(eval_data["pages"]):
        segments_dir = page["segments_dir"]
        print(f"\n{'='*70}")
        print(f"Page {page_idx + 1}: {page.get('image', 'unknown')}")
        print(f"{'='*70}")

        for card_info in page["cards"]:
            card_id = card_info["card_id"]
            name = card_info["name"]
            segment = card_info["segment"]

            # Skip empty slots
            if card_id is None:
                print(f"  SKIP {name}: empty slot")
                total_skipped += 1
                continue

            img_path = PROJECT_ROOT / segments_dir / segment
            if not img_path.exists():
                print(f"  SKIP {name}: segment not found at {img_path}")
                total_skipped += 1
                continue

            # Look up expected attacks
            expected_attacks = card_to_attacks.get(card_id, [])
            if not expected_attacks:
                base_id = card_id.split("/")[0]
                for k, v in card_to_attacks.items():
                    if k.startswith(base_id + "/"):
                        expected_attacks = v
                        break

            print(f"\n  --- {name} ({card_id}) ---")
            print(f"  Expected attacks: {expected_attacks}")

            # Step 1: Extract attack names from OCR
            ocr_candidates = extract_attack_names(str(img_path))
            ocr_texts = [t for t, _ in ocr_candidates]
            print(f"  OCR candidates:   {ocr_texts}")

            # Step 2: Check which expected attacks were found via fuzzy match
            if expected_attacks and ocr_candidates:
                for exp_atk in expected_attacks:
                    attacks_expected_total += 1
                    best_score = 0.0
                    best_ocr = ""
                    for ocr_t, _ in ocr_candidates:
                        from difflib import SequenceMatcher
                        score = SequenceMatcher(
                            None, ocr_t.lower(), exp_atk.lower()
                        ).ratio()
                        if score > best_score:
                            best_score = score
                            best_ocr = ocr_t
                    found = best_score >= 0.60
                    if found:
                        attacks_found_total += 1
                    flag = "FOUND" if found else "MISS "
                    print(
                        f"    [{flag}] '{exp_atk}' ~ '{best_ocr}' "
                        f"(ratio={best_score:.2f})"
                    )
            elif expected_attacks:
                attacks_expected_total += len(expected_attacks)
                print(f"    No OCR candidates extracted")

            # Step 3: Full identification (open search)
            ranked = identify_by_attacks(str(img_path))
            total_evaluated += 1

            # Find rank of correct card
            rank = None
            for i, (cid, score) in enumerate(ranked):
                if cid == card_id:
                    rank = i + 1
                    break

            if rank == 1:
                correct_at_1 += 1
                correct_at_5 += 1
                reciprocal_ranks.append(1.0)
                print(f"  RESULT: CORRECT @1 (score={ranked[0][1]:.3f})")
            elif rank is not None and rank <= 5:
                correct_at_5 += 1
                reciprocal_ranks.append(1.0 / rank)
                print(
                    f"  RESULT: found @{rank} (score={ranked[rank-1][1]:.3f}), "
                    f"top was {ranked[0][0]} ({ranked[0][1]:.3f})"
                )
            elif rank is not None:
                reciprocal_ranks.append(1.0 / rank)
                print(
                    f"  RESULT: found @{rank} (score={ranked[rank-1][1]:.3f})"
                )
            elif ranked:
                reciprocal_ranks.append(0.0)
                print(
                    f"  RESULT: NOT FOUND in {len(ranked)} candidates. "
                    f"Top: {ranked[:3]}"
                )
            else:
                reciprocal_ranks.append(0.0)
                print(f"  RESULT: no candidates returned")

            # Step 4: Narrowing test
            # Build a simulated candidate set: correct card + 9 random cards
            # with the same attack names (realistic disambiguation scenario)
            if expected_attacks:
                # Get all cards sharing any attack with the correct card
                sibling_cards = set()
                for atk in expected_attacks:
                    for cid in atk_to_cards.get(atk, []):
                        sibling_cards.add(cid)
                sibling_cards.add(card_id)
                sibling_list = list(sibling_cards)[:20]  # cap at 20

                if len(sibling_list) > 1:
                    narrow_evaluated += 1
                    narrow_ranked = narrow_candidates(
                        str(img_path), sibling_list
                    )
                    if narrow_ranked and narrow_ranked[0][0] == card_id:
                        narrow_correct_at_1 += 1
                        print(
                            f"  NARROW: CORRECT @1 from {len(sibling_list)} "
                            f"candidates (score={narrow_ranked[0][1]:.3f})"
                        )
                    elif narrow_ranked:
                        narrow_rank = None
                        for i, (cid, score) in enumerate(narrow_ranked):
                            if cid == card_id:
                                narrow_rank = i + 1
                                break
                        if narrow_rank:
                            print(
                                f"  NARROW: found @{narrow_rank} from "
                                f"{len(sibling_list)} candidates"
                            )
                        else:
                            print(
                                f"  NARROW: NOT FOUND in "
                                f"{len(sibling_list)} candidates"
                            )
                    else:
                        print(f"  NARROW: no results")

            result = {
                "card_id": card_id,
                "name": name,
                "segment": segment,
                "expected_attacks": expected_attacks,
                "ocr_candidates": ocr_texts,
                "rank": rank,
                "num_candidates": len(ranked),
                "top_5": ranked[:5] if ranked else [],
            }
            results.append(result)

    elapsed = time.time() - t0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

    attack_recall = (
        attacks_found_total / attacks_expected_total
        if attacks_expected_total > 0
        else 0
    )

    print(f"\n{'='*70}")
    print("ATTACK OCR MATCH SUMMARY")
    print(f"{'='*70}")
    print(f"Cards evaluated:     {total_evaluated}")
    print(f"Cards skipped:       {total_skipped}")
    print()
    print(f"Attack recall:       {attacks_found_total}/{attacks_expected_total} "
          f"({attack_recall:.1%})")
    print()
    print(f"Open search:")
    print(f"  Correct @1:        {correct_at_1}/{total_evaluated} "
          f"({correct_at_1/total_evaluated:.1%})" if total_evaluated else "N/A")
    print(f"  Correct @5:        {correct_at_5}/{total_evaluated} "
          f"({correct_at_5/total_evaluated:.1%})" if total_evaluated else "N/A")
    print(f"  MRR:               {mrr:.3f}")
    print()
    if narrow_evaluated > 0:
        print(f"Narrowing (sibling candidate sets):")
        print(f"  Correct @1:        {narrow_correct_at_1}/{narrow_evaluated} "
              f"({narrow_correct_at_1/narrow_evaluated:.1%})")
    print()
    print(f"Time elapsed:        {elapsed:.1f}s")

    # Save results
    out_path = PROJECT_ROOT / "data" / "eval" / "attack_ocr_match_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_evaluated": total_evaluated,
        "total_skipped": total_skipped,
        "attack_recall": round(attack_recall, 4),
        "correct_at_1": correct_at_1,
        "correct_at_5": correct_at_5,
        "mrr": round(mrr, 4),
        "narrow_correct_at_1": narrow_correct_at_1,
        "narrow_evaluated": narrow_evaluated,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_card": results}, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
