#!/usr/bin/env python3
"""Evaluate DINOv2 top-N name voting as a card identification signal.

For each card in binder_eval.json:
1. Run DINOv2 FAISS search (top-50)
2. Look up the name of each result from dim_cards
3. Compute the plurality name at top-5, top-10, top-20, top-50
4. Check if the plurality name matches the ground truth

Saves results to data/eval/dino_name_results.json.
"""

import json
import logging
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

# Project root
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------

def load_name_lookup():
    """Build card_id -> name mapping from dim_cards."""
    from sqlalchemy import text
    from cardprice.db.session import SessionLocal

    session = SessionLocal()
    try:
        rows = session.execute(
            text("SELECT card_id, name FROM dim_cards")
        ).fetchall()
        lookup = {r[0]: r[1] for r in rows}
        logger.info("Loaded %d card names from dim_cards.", len(lookup))
        return lookup
    finally:
        session.close()


def load_faiss_index():
    """Load FAISS index and card_id mapping."""
    index_path = "data/dino_index.faiss"
    mapping_path = "data/dino_card_ids.pkl"

    index = faiss.read_index(index_path)
    with open(mapping_path, "rb") as f:
        card_ids = pickle.load(f)

    logger.info("Loaded FAISS index with %d vectors.", index.ntotal)
    return index, card_ids


def load_eval_data():
    """Load binder_eval.json and flatten to a list of card entries."""
    with open("data/eval/binder_eval.json") as f:
        data = json.load(f)

    cards = []
    for page in data["pages"]:
        segments_dir = page["segments_dir"]
        for card in page["cards"]:
            if card["card_id"] is None:
                continue  # skip empty slots
            cards.append({
                "segment_path": os.path.join(segments_dir, card["segment"]),
                "card_id": card["card_id"],
                "name": card["name"],
            })
    logger.info("Loaded %d eval cards (excluding empty slots).", len(cards))
    return cards


# ---------------------------------------------------------------------------
# Name voting logic
# ---------------------------------------------------------------------------

def plurality_name(names):
    """Return the most common name and its count."""
    if not names:
        return None, 0
    counter = Counter(names)
    name, count = counter.most_common(1)[0]
    return name, count


def normalize_name(name):
    """Normalize a card name for comparison (strip delta symbols, lowercase)."""
    if name is None:
        return ""
    # Replace delta symbol variants
    n = name.replace("δ", "delta").replace("Δ", "delta").replace(" δ", " delta")
    # Strip "ex" suffix variations for matching
    return n.strip().lower()


def names_match(gt_name, predicted_name):
    """Check if ground truth name matches predicted name (fuzzy)."""
    gt = normalize_name(gt_name)
    pred = normalize_name(predicted_name)
    if not gt or not pred:
        return False
    # Exact match after normalization
    if gt == pred:
        return True
    # Check if one contains the other (handles "Flygon ex δ" vs "Flygon ex delta")
    if gt in pred or pred in gt:
        return True
    return False


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

TOP_K = 50
THRESHOLDS = [5, 10, 20, 50]


def evaluate():
    t0 = time.time()

    # Load everything
    name_lookup = load_name_lookup()
    faiss_index, faiss_card_ids = load_faiss_index()
    eval_cards = load_eval_data()

    # Import dino embedding extraction
    from cardprice.ml.dino_matcher import extract_embedding

    results = []
    summary = {k: {"correct": 0, "total": 0} for k in THRESHOLDS}

    for i, card in enumerate(eval_cards):
        seg_path = card["segment_path"]
        gt_name = card["name"]
        gt_card_id = card["card_id"]

        logger.info("[%d/%d] Processing %s (GT: %s)", i + 1, len(eval_cards), seg_path, gt_name)

        if not os.path.exists(seg_path):
            logger.warning("  Segment not found: %s — skipping", seg_path)
            continue

        # Extract embedding and search FAISS
        query = extract_embedding(seg_path).reshape(1, -1)
        k = min(TOP_K, faiss_index.ntotal)
        scores, indices = faiss_index.search(query, k)

        # Resolve card_ids and names for all top-K results
        top_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            cid = faiss_card_ids[idx]
            # card_id in FAISS might be "set/card/variant", look up base card_id
            name = name_lookup.get(cid, None)
            if name is None:
                # Try stripping variant: "ex15-92/normal" -> look up as-is
                # The FAISS mapping already includes variant, same as dim_cards card_id
                pass
            top_results.append({
                "rank": len(top_results) + 1,
                "card_id": cid,
                "name": name,
                "score": float(score),
            })

        # Compute plurality name at each threshold
        card_result = {
            "segment": seg_path,
            "gt_card_id": gt_card_id,
            "gt_name": gt_name,
            "top1_card_id": top_results[0]["card_id"] if top_results else None,
            "top1_name": top_results[0]["name"] if top_results else None,
            "top1_score": top_results[0]["score"] if top_results else 0.0,
            "thresholds": {},
        }

        for t in THRESHOLDS:
            names_in_range = [r["name"] for r in top_results[:t] if r["name"] is not None]
            pname, pcount = plurality_name(names_in_range)
            match = names_match(gt_name, pname) if pname else False

            card_result["thresholds"][str(t)] = {
                "plurality_name": pname,
                "plurality_count": pcount,
                "total_with_names": len(names_in_range),
                "fraction": pcount / len(names_in_range) if names_in_range else 0,
                "match": match,
                "top_names": Counter(names_in_range).most_common(5),
            }

            summary[t]["total"] += 1
            if match:
                summary[t]["correct"] += 1

        results.append(card_result)

        # Log quick summary for this card
        t5 = card_result["thresholds"]["5"]
        t50 = card_result["thresholds"]["50"]
        logger.info(
            "  top-5: %s (%d/%d) %s | top-50: %s (%d/%d) %s",
            t5["plurality_name"], t5["plurality_count"], t5["total_with_names"],
            "MATCH" if t5["match"] else "MISS",
            t50["plurality_name"], t50["plurality_count"], t50["total_with_names"],
            "MATCH" if t50["match"] else "MISS",
        )

    elapsed = time.time() - t0

    # Build final output
    output = {
        "description": "DINOv2 top-N name voting evaluation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "num_cards": len(results),
        "summary": {},
        "per_card": results,
    }

    for t in THRESHOLDS:
        s = summary[t]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        output["summary"][f"top_{t}"] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": round(acc, 4),
        }
        logger.info(
            "Top-%d plurality name accuracy: %d/%d = %.1f%%",
            t, s["correct"], s["total"], acc * 100,
        )

    # Save results
    out_path = "data/eval/dino_name_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to %s", out_path)
    logger.info("Total time: %.1fs", elapsed)


if __name__ == "__main__":
    evaluate()
