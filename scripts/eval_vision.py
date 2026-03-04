#!/usr/bin/env python3
"""Evaluate identify_page_vision_first() against binder_eval.json ground truth.

Must run as a background process (claude -p subprocesses block in foreground).
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "data" / "eval" / "vision_eval.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("eval_vision")


def run_eval():
    with open(ROOT / "data" / "eval" / "binder_eval.json") as f:
        gt = json.load(f)

    from cardprice.ml import identify_page_vision_first
    from cardprice.db.session import SessionLocal

    sess = SessionLocal()
    total = 0
    correct = 0
    results_detail = []

    for page_idx, page in enumerate(gt["pages"]):
        seg_dir = ROOT / page["segments_dir"]
        cards = page["cards"]
        image_paths = [str(seg_dir / c["segment"]) for c in cards]

        logger.info("=== Page %d: %s ===", page_idx + 1, seg_dir.name)
        t0 = time.time()
        results = identify_page_vision_first(image_paths, session=sess)
        elapsed = time.time() - t0
        logger.info("Page %d took %.1fs", page_idx + 1, elapsed)

        for i, (card_gt, result) in enumerate(zip(cards, results)):
            gt_id = card_gt["card_id"]
            gt_name = card_gt["name"]
            pred_id = result.get("card_id")
            pred_conf = result.get("confidence", 0)
            method = result.get("method", result.get("explanation", "?"))

            # For null ground truth (empty slot), match if pred is also null
            if gt_id is None:
                is_correct = pred_id is None
            else:
                # Match on base card_id (strip variant)
                gt_base = gt_id.split("/")[0]
                pred_base = (pred_id or "").split("/")[0]
                is_correct = gt_base == pred_base

            total += 1
            if is_correct:
                correct += 1

            status = "OK" if is_correct else "MISS"
            logger.info(
                "  [%s] %s: gt=%s pred=%s (conf=%.2f, %s)",
                status, gt_name, gt_id, pred_id, pred_conf, method,
            )

            # Capture vision sub-results for analysis
            vision_info = {}
            raw = result.get("raw_response", {})
            if raw and "vision_result" in raw:
                vr = raw["vision_result"]
                vision_info = {
                    "vision_name": vr.get("pokemon_name"),
                    "vision_attacks": vr.get("attacks", []),
                    "vision_number": vr.get("card_number"),
                    "vision_era": vr.get("era"),
                    "vision_hp": vr.get("hp"),
                }

            results_detail.append({
                "page": page_idx + 1,
                "position": card_gt["position"],
                "gt_name": gt_name,
                "gt_id": gt_id,
                "pred_id": pred_id,
                "confidence": pred_conf,
                "method": method,
                "correct": is_correct,
                **vision_info,
            })

    sess.close()

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("OVERALL: %d/%d correct (%.1f%%)", correct, total,
                100 * correct / total if total else 0)
    logger.info("=" * 60)

    # Per-page
    for p in range(len(gt["pages"])):
        page_results = [r for r in results_detail if r["page"] == p + 1]
        page_correct = sum(1 for r in page_results if r["correct"])
        logger.info("  Page %d: %d/%d", p + 1, page_correct, len(page_results))

    # Per-method
    from collections import Counter
    methods = Counter()
    method_correct = Counter()
    for r in results_detail:
        m = r["method"]
        methods[m] += 1
        if r["correct"]:
            method_correct[m] += 1
    logger.info("\nBy method:")
    for m, count in methods.most_common():
        logger.info("  %s: %d/%d", m, method_correct[m], count)

    # Save detailed results
    out_path = ROOT / "data" / "eval" / "vision_eval_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0,
            "details": results_detail,
        }, f, indent=2)
    logger.info("\nDetailed results saved to %s", out_path)


if __name__ == "__main__":
    run_eval()
