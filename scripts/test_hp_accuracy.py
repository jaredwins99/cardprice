#!/usr/bin/env python3
"""Evaluate HP detection accuracy on all 27 eval binder cards.

Two approaches tested:
1. detect_hp() from cardprice.ml.hp_detector (multi-crop, multi-preprocess cascade)
2. Simple EasyOCR on the top-right corner, looking for any number

Results saved to data/eval/hp_results.json.
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Project root
ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("hp_accuracy")

# ---------------------------------------------------------------------------
# Load eval dataset
# ---------------------------------------------------------------------------
EVAL_PATH = ROOT / "data" / "eval" / "binder_eval.json"
with open(EVAL_PATH) as f:
    eval_data = json.load(f)

# Collect all cards with their segment paths and ground truth IDs
cards = []
for page in eval_data["pages"]:
    seg_dir = ROOT / page["segments_dir"]
    for card in page["cards"]:
        if card["card_id"] is None:
            continue  # skip empty slots
        seg_path = seg_dir / card["segment"]
        cards.append({
            "card_id": card["card_id"],
            "name": card["name"],
            "segment": str(seg_path),
            "page_image": page["image"],
        })

logger.info("Loaded %d cards from eval dataset", len(cards))

# ---------------------------------------------------------------------------
# Look up ground truth HP from database
# ---------------------------------------------------------------------------
from sqlalchemy import create_engine, text

engine = create_engine("postgresql+psycopg2://godli@/cardprice")
gt_hp_map = {}
with engine.connect() as conn:
    card_ids = [c["card_id"] for c in cards]
    for cid in set(card_ids):
        r = conn.execute(text("SELECT hp FROM dim_cards WHERE card_id = :cid"), {"cid": cid})
        row = r.fetchone()
        gt_hp_map[cid] = row[0] if row else None

logger.info("Ground truth HP lookup: %d unique cards, %d with HP values",
            len(set(card_ids)), sum(1 for v in gt_hp_map.values() if v is not None))

# ---------------------------------------------------------------------------
# Approach 1: detect_hp() from hp_detector
# ---------------------------------------------------------------------------
from cardprice.ml.hp_detector import detect_hp

logger.info("=" * 70)
logger.info("APPROACH 1: detect_hp() (full HP detector cascade)")
logger.info("=" * 70)

results = []
for card in cards:
    gt_hp = gt_hp_map.get(card["card_id"])
    seg = card["segment"]
    if not os.path.exists(seg):
        logger.warning("Segment not found: %s", seg)
        results.append({
            "card_id": card["card_id"],
            "name": card["name"],
            "segment": seg,
            "gt_hp": gt_hp,
            "detected_hp": None,
            "simple_hp": None,
            "status": "missing_segment",
        })
        continue

    t0 = time.time()
    detected = detect_hp(seg)
    elapsed = time.time() - t0

    if gt_hp is None:
        status = "no_gt"
    elif detected is None:
        status = "miss"
    elif detected == gt_hp:
        status = "exact"
    elif abs(detected - gt_hp) <= 10:
        status = "within_10"
    else:
        status = "wrong"

    logger.info("%-25s gt=%-4s det=%-4s  %s  (%.2fs) [%s]",
                card["name"], gt_hp, detected, status, elapsed, Path(seg).name)

    results.append({
        "card_id": card["card_id"],
        "name": card["name"],
        "segment": seg,
        "gt_hp": gt_hp,
        "detected_hp": detected,
        "simple_hp": None,  # filled in approach 2
        "status": status,
    })

# ---------------------------------------------------------------------------
# Approach 2: Simple EasyOCR on top-right corner
# ---------------------------------------------------------------------------
logger.info("")
logger.info("=" * 70)
logger.info("APPROACH 2: Simple EasyOCR on top-right 40%% x top 15%%")
logger.info("=" * 70)

try:
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
except ImportError:
    reader = None
    logger.error("EasyOCR not available, skipping approach 2")


def simple_hp_ocr(image_path: str) -> int | None:
    """Read top-right corner with EasyOCR, return first plausible HP number."""
    if reader is None:
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    # Crop top-right: right 40%, top 15%
    crop = img[0:int(h * 0.15), int(w * 0.60):w]
    if crop.size == 0:
        return None

    # Upscale 3x for better OCR
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    ocr_results = reader.readtext(crop)
    texts = [(r[1], float(r[2])) for r in ocr_results]

    # Look for any 2-3 digit number that's a multiple of 10
    for text, conf in texts:
        nums = re.findall(r'\d{2,3}', text)
        for n in nums:
            val = int(n)
            if 30 <= val <= 400 and val % 10 == 0:
                return val
    return None


for i, card in enumerate(cards):
    seg = card["segment"]
    if not os.path.exists(seg):
        continue

    t0 = time.time()
    simple = simple_hp_ocr(seg)
    elapsed = time.time() - t0
    results[i]["simple_hp"] = simple

    gt_hp = results[i]["gt_hp"]
    if gt_hp is None:
        s_status = "no_gt"
    elif simple is None:
        s_status = "miss"
    elif simple == gt_hp:
        s_status = "exact"
    elif abs(simple - gt_hp) <= 10:
        s_status = "within_10"
    else:
        s_status = "wrong"
    results[i]["simple_status"] = s_status

    logger.info("%-25s gt=%-4s simple=%-4s  %s  (%.2fs)",
                card["name"], gt_hp, simple, s_status, elapsed)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
logger.info("")
logger.info("=" * 70)
logger.info("SUMMARY")
logger.info("=" * 70)

# Filter to cards with ground truth HP
with_gt = [r for r in results if r["gt_hp"] is not None]
total = len(with_gt)

# Approach 1 stats
a1_exact = sum(1 for r in with_gt if r["status"] == "exact")
a1_within10 = sum(1 for r in with_gt if r["status"] in ("exact", "within_10"))
a1_miss = sum(1 for r in with_gt if r["status"] == "miss")
a1_wrong = sum(1 for r in with_gt if r["status"] == "wrong")

# Approach 2 stats
a2_exact = sum(1 for r in with_gt if r.get("simple_status") == "exact")
a2_within10 = sum(1 for r in with_gt if r.get("simple_status") in ("exact", "within_10"))
a2_miss = sum(1 for r in with_gt if r.get("simple_status") == "miss")
a2_wrong = sum(1 for r in with_gt if r.get("simple_status") == "wrong")

pct = lambda n: f"{n}/{total} ({100*n/total:.1f}%)" if total else "0/0"

logger.info("")
logger.info("Approach 1: detect_hp() (full cascade)")
logger.info("  Exact match:   %s", pct(a1_exact))
logger.info("  Within 10:     %s", pct(a1_within10))
logger.info("  Total miss:    %s", pct(a1_miss))
logger.info("  Wrong:         %s", pct(a1_wrong))

logger.info("")
logger.info("Approach 2: Simple EasyOCR top-right corner")
logger.info("  Exact match:   %s", pct(a2_exact))
logger.info("  Within 10:     %s", pct(a2_within10))
logger.info("  Total miss:    %s", pct(a2_miss))
logger.info("  Wrong:         %s", pct(a2_wrong))

# Detailed mismatches
logger.info("")
logger.info("DETAILED MISMATCHES (approach 1 - detect_hp):")
for r in with_gt:
    if r["status"] not in ("exact",):
        logger.info("  %-25s gt=%-4s det=%-4s  (%s)", r["name"], r["gt_hp"], r["detected_hp"], r["status"])

logger.info("")
logger.info("DETAILED MISMATCHES (approach 2 - simple OCR):")
for r in with_gt:
    if r.get("simple_status") not in ("exact",):
        logger.info("  %-25s gt=%-4s simple=%-4s  (%s)", r["name"], r["gt_hp"], r["simple_hp"], r.get("simple_status"))

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
output = {
    "description": "HP detection accuracy evaluation on binder_eval.json cards",
    "total_cards": len(cards),
    "cards_with_gt_hp": total,
    "approach_1_detect_hp": {
        "exact_match": a1_exact,
        "within_10": a1_within10,
        "miss": a1_miss,
        "wrong": a1_wrong,
        "accuracy_exact_pct": round(100 * a1_exact / total, 1) if total else 0,
        "accuracy_within10_pct": round(100 * a1_within10 / total, 1) if total else 0,
    },
    "approach_2_simple_ocr": {
        "exact_match": a2_exact,
        "within_10": a2_within10,
        "miss": a2_miss,
        "wrong": a2_wrong,
        "accuracy_exact_pct": round(100 * a2_exact / total, 1) if total else 0,
        "accuracy_within10_pct": round(100 * a2_within10 / total, 1) if total else 0,
    },
    "per_card": results,
}

out_path = ROOT / "data" / "eval" / "hp_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

logger.info("")
logger.info("Results saved to %s", out_path)
logger.info("Done.")
