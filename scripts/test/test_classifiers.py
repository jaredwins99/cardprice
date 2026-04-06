#!/usr/bin/env python3
"""Test cheap classifiers (OCR name, type detector, HP detector) on the eval dataset.

For each card in the eval dataset, extract classifier signals and check
accuracy against ground truth. Also query the DB to see how many candidates
each signal combination narrows to.

Results are saved to data/eval/classifier_test.json.

Usage:
    python scripts/test_classifiers.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("classifier_test")

EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "binder_eval.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "eval" / "classifier_test.json"


def load_eval_data():
    """Load the binder eval JSON and flatten to a list of card dicts with image paths."""
    with open(EVAL_PATH) as f:
        data = json.load(f)

    cards = []
    for page in data["pages"]:
        segments_dir = PROJECT_ROOT / page["segments_dir"]
        for card in page["cards"]:
            # Skip empty slots
            if card.get("card_id") is None:
                continue
            image_path = segments_dir / card["segment"]
            cards.append({
                "image_path": str(image_path),
                "card_id": card["card_id"],
                "name": card["name"],
                "segment": card["segment"],
                "page_image": page["image"],
            })
    return cards


def get_ground_truth_from_db(card_id):
    """Look up ground truth HP, types from the database for a card_id."""
    from cardprice.db.session import engine
    from sqlalchemy import text

    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT dc.name, dc.hp, dc.set_id, dc.supertype,
                       dp.types
                FROM dim_cards dc
                LEFT JOIN dim_pokemon dp ON dp.pokemon_id = dc.pokemon_id
                WHERE dc.card_id = :cid
            """),
            {"cid": card_id},
        ).fetchone()

    if row is None:
        return None

    return {
        "db_name": row[0],
        "db_hp": row[1],
        "db_set_id": row[2],
        "db_supertype": row[3],
        "db_types": list(row[4]) if row[4] else [],
    }


def count_candidates_by_signals(name_query, hp_value, type_name):
    """Query the DB to count how many cards match various signal combinations.

    Returns dict with candidate counts for:
    - name_only: cards matching the name (case-insensitive ILIKE)
    - name_hp: cards matching name AND hp
    - name_type: cards matching name AND pokemon type
    - name_hp_type: cards matching all three
    - hp_only: cards with this HP
    - type_only: cards with this type
    - hp_type: cards with this HP and type
    """
    from cardprice.db.session import engine
    from sqlalchemy import text

    counts = {}

    if not name_query:
        counts["name_only"] = None
        counts["name_hp"] = None
        counts["name_type"] = None
        counts["name_hp_type"] = None
    else:
        # Fuzzy name matching: use ILIKE with the cleaned OCR name
        # This approximates fuzzy matching -- real fuzzy would use fuzzywuzzy
        name_pattern = f"%{name_query}%"

        with engine.connect() as conn:
            # Name only (case-insensitive substring)
            r = conn.execute(
                text("SELECT COUNT(*) FROM dim_cards WHERE LOWER(name) LIKE LOWER(:pat)"),
                {"pat": name_pattern},
            ).scalar()
            counts["name_only"] = r

            # Name + HP
            if hp_value is not None:
                r = conn.execute(
                    text("SELECT COUNT(*) FROM dim_cards WHERE LOWER(name) LIKE LOWER(:pat) AND hp = :hp"),
                    {"pat": name_pattern, "hp": hp_value},
                ).scalar()
                counts["name_hp"] = r
            else:
                counts["name_hp"] = None

            # Name + Type
            if type_name:
                r = conn.execute(
                    text("""
                        SELECT COUNT(*) FROM dim_cards dc
                        JOIN dim_pokemon dp ON dp.pokemon_id = dc.pokemon_id
                        WHERE LOWER(dc.name) LIKE LOWER(:pat)
                          AND :typ = ANY(dp.types)
                    """),
                    {"pat": name_pattern, "typ": type_name},
                ).scalar()
                counts["name_type"] = r
            else:
                counts["name_type"] = None

            # Name + HP + Type
            if hp_value is not None and type_name:
                r = conn.execute(
                    text("""
                        SELECT COUNT(*) FROM dim_cards dc
                        JOIN dim_pokemon dp ON dp.pokemon_id = dc.pokemon_id
                        WHERE LOWER(dc.name) LIKE LOWER(:pat)
                          AND dc.hp = :hp
                          AND :typ = ANY(dp.types)
                    """),
                    {"pat": name_pattern, "hp": hp_value, "typ": type_name},
                ).scalar()
                counts["name_hp_type"] = r
            else:
                counts["name_hp_type"] = None

    # Signal-only counts (no name)
    with engine.connect() as conn:
        if hp_value is not None:
            r = conn.execute(
                text("SELECT COUNT(*) FROM dim_cards WHERE hp = :hp"),
                {"hp": hp_value},
            ).scalar()
            counts["hp_only"] = r
        else:
            counts["hp_only"] = None

        if type_name:
            r = conn.execute(
                text("""
                    SELECT COUNT(*) FROM dim_cards dc
                    JOIN dim_pokemon dp ON dp.pokemon_id = dc.pokemon_id
                    WHERE :typ = ANY(dp.types)
                """),
                {"typ": type_name},
            ).scalar()
            counts["type_only"] = r
        else:
            counts["type_only"] = None

        if hp_value is not None and type_name:
            r = conn.execute(
                text("""
                    SELECT COUNT(*) FROM dim_cards dc
                    JOIN dim_pokemon dp ON dp.pokemon_id = dc.pokemon_id
                    WHERE dc.hp = :hp AND :typ = ANY(dp.types)
                """),
                {"hp": hp_value, "typ": type_name},
            ).scalar()
            counts["hp_type"] = r
        else:
            counts["hp_type"] = None

    return counts


def count_exact_name_candidates(exact_name):
    """Count how many cards in the DB share this exact name (case-insensitive).

    This is the most useful narrowing metric: if OCR reads the correct name,
    how many cards have that same name across all sets/variants?
    """
    from cardprice.db.session import engine
    from sqlalchemy import text

    with engine.connect() as conn:
        r = conn.execute(
            text("SELECT COUNT(*) FROM dim_cards WHERE LOWER(name) = LOWER(:n)"),
            {"n": exact_name},
        ).scalar()
    return r


def run_classifiers_on_card(image_path):
    """Run all cheap classifiers on a single card image.

    Returns dict with all extracted signals and timing.
    """
    results = {}

    # 1. OCR card name (from ocr_matcher -- the one used in the cascade)
    t0 = time.time()
    try:
        from cardprice.ml.ocr_matcher import extract_card_name, _clean_ocr_text
        raw_text, ocr_conf = extract_card_name(image_path)
        cleaned_name = _clean_ocr_text(raw_text) if raw_text else ""
        results["ocr_name_raw"] = raw_text
        results["ocr_name_cleaned"] = cleaned_name
        results["ocr_name_confidence"] = round(ocr_conf, 3)
    except Exception as e:
        logger.warning("OCR name failed for %s: %s", image_path, e)
        results["ocr_name_raw"] = None
        results["ocr_name_cleaned"] = None
        results["ocr_name_confidence"] = 0.0
        results["ocr_name_error"] = str(e)
    results["ocr_name_time"] = round(time.time() - t0, 3)

    # 2. HP detection (from hp_detector)
    t0 = time.time()
    try:
        from cardprice.ml.hp_detector import detect_hp
        hp_value = detect_hp(image_path)
        results["hp_detected"] = hp_value
    except Exception as e:
        logger.warning("HP detection failed for %s: %s", image_path, e)
        results["hp_detected"] = None
        results["hp_error"] = str(e)
    results["hp_time"] = round(time.time() - t0, 3)

    # 3. Type detection (from type_detector)
    t0 = time.time()
    try:
        from cardprice.ml.type_detector import detect_type
        type_preds = detect_type(image_path, top_n=3)
        results["type_predictions"] = [(name, round(conf, 3)) for name, conf in type_preds]
        results["type_top1"] = type_preds[0][0] if type_preds else None
        results["type_top1_conf"] = round(type_preds[0][1], 3) if type_preds else 0.0
    except Exception as e:
        logger.warning("Type detection failed for %s: %s", image_path, e)
        results["type_predictions"] = []
        results["type_top1"] = None
        results["type_top1_conf"] = 0.0
        results["type_error"] = str(e)
    results["type_time"] = round(time.time() - t0, 3)

    # 4. HP detector's card name reading (separate from ocr_matcher)
    t0 = time.time()
    try:
        from cardprice.ml.hp_detector import detect_card_name
        hp_det_name = detect_card_name(image_path)
        results["hp_det_name"] = hp_det_name
    except Exception as e:
        logger.warning("HP detector name failed for %s: %s", image_path, e)
        results["hp_det_name"] = None
        results["hp_det_name_error"] = str(e)
    results["hp_det_name_time"] = round(time.time() - t0, 3)

    # 5. Damage values (attack damage from hp_detector)
    t0 = time.time()
    try:
        from cardprice.ml.hp_detector import detect_damage
        damages = detect_damage(image_path)
        results["damage_values"] = damages
    except Exception as e:
        logger.warning("Damage detection failed for %s: %s", image_path, e)
        results["damage_values"] = []
        results["damage_error"] = str(e)
    results["damage_time"] = round(time.time() - t0, 3)

    # 6. Attack names from OCR (read the attack region text)
    t0 = time.time()
    try:
        from cardprice.ml.hp_detector import _crop_region, _ATTACK_REGION, _ocr_easyocr
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            attack_crop = _crop_region(img, _ATTACK_REGION)
            attack_texts = _ocr_easyocr(attack_crop)
            results["attack_ocr_texts"] = [(t, round(c, 3)) for t, c in attack_texts]
        else:
            results["attack_ocr_texts"] = []
    except Exception as e:
        logger.warning("Attack OCR failed for %s: %s", image_path, e)
        results["attack_ocr_texts"] = []
        results["attack_ocr_error"] = str(e)
    results["attack_ocr_time"] = round(time.time() - t0, 3)

    return results


def evaluate_name_match(ocr_name, ground_truth_name):
    """Check if OCR name matches ground truth using fuzzy matching.

    Returns dict with match info.
    """
    if not ocr_name or not ground_truth_name:
        return {
            "name_match": False,
            "fuzzy_score": 0,
            "match_type": "no_ocr" if not ocr_name else "no_gt",
        }

    # Strip delta/ex suffixes for comparison (OCR often misses these)
    import re
    gt_clean = re.sub(r"\s*(ex|EX|δ|GX|V|VMAX|VSTAR)\s*$", "", ground_truth_name).strip()
    ocr_clean = ocr_name.strip()

    # Exact match (case-insensitive)
    if ocr_clean.lower() == gt_clean.lower():
        return {"name_match": True, "fuzzy_score": 100, "match_type": "exact"}

    # Try fuzzy matching
    try:
        from thefuzz import fuzz
    except ImportError:
        try:
            from fuzzywuzzy import fuzz
        except ImportError:
            # Manual simple ratio
            if gt_clean.lower() in ocr_clean.lower() or ocr_clean.lower() in gt_clean.lower():
                return {"name_match": True, "fuzzy_score": 80, "match_type": "substring"}
            return {"name_match": False, "fuzzy_score": 0, "match_type": "no_fuzzy_lib"}

    # Use token_set_ratio which handles word order and partial matches well
    score = fuzz.token_set_ratio(ocr_clean.lower(), gt_clean.lower())
    partial_score = fuzz.partial_ratio(ocr_clean.lower(), gt_clean.lower())
    best_score = max(score, partial_score)

    return {
        "name_match": best_score >= 70,
        "fuzzy_score": best_score,
        "match_type": "fuzzy_exact" if best_score >= 90 else "fuzzy_partial" if best_score >= 70 else "no_match",
    }


def main():
    logger.info("Loading eval data from %s", EVAL_PATH)
    cards = load_eval_data()
    logger.info("Loaded %d eval cards (excluding empty slots)", len(cards))

    results = []
    summary = {
        "total_cards": len(cards),
        "ocr_name_correct": 0,
        "ocr_name_partial": 0,  # fuzzy >= 70 but < 90
        "ocr_name_fail": 0,
        "hp_correct": 0,
        "hp_detected_wrong": 0,
        "hp_not_detected": 0,
        "type_correct": 0,
        "type_wrong": 0,
        "type_not_detected": 0,
        "hp_det_name_correct": 0,
    }

    for i, card in enumerate(cards):
        logger.info("--- Card %d/%d: %s (%s) ---", i + 1, len(cards), card["name"], card["card_id"])

        # Get ground truth from DB
        gt = get_ground_truth_from_db(card["card_id"])
        if gt is None:
            logger.warning("Card %s not found in DB, skipping", card["card_id"])
            continue

        # Run classifiers
        signals = run_classifiers_on_card(card["image_path"])

        # Evaluate OCR name
        name_eval = evaluate_name_match(signals["ocr_name_cleaned"], card["name"])
        if name_eval["fuzzy_score"] >= 90:
            summary["ocr_name_correct"] += 1
        elif name_eval["fuzzy_score"] >= 70:
            summary["ocr_name_partial"] += 1
        else:
            summary["ocr_name_fail"] += 1

        # Evaluate HP
        gt_hp = gt["db_hp"]
        detected_hp = signals["hp_detected"]
        if gt_hp is not None and detected_hp is not None:
            if detected_hp == gt_hp:
                summary["hp_correct"] += 1
                hp_correct = True
            else:
                summary["hp_detected_wrong"] += 1
                hp_correct = False
        elif gt_hp is not None and detected_hp is None:
            summary["hp_not_detected"] += 1
            hp_correct = False
        else:
            # GT has no HP (Trainer/Energy) -- skip
            hp_correct = None

        # Evaluate type
        gt_types = gt["db_types"]
        detected_type = signals["type_top1"]
        if gt_types and detected_type:
            type_correct = detected_type in gt_types
            if type_correct:
                summary["type_correct"] += 1
            else:
                summary["type_wrong"] += 1
        elif gt_types and not detected_type:
            summary["type_not_detected"] += 1
            type_correct = False
        else:
            type_correct = None

        # Evaluate hp_detector's name reading
        hp_det_name_eval = evaluate_name_match(signals.get("hp_det_name"), card["name"])
        if hp_det_name_eval["fuzzy_score"] >= 70:
            summary["hp_det_name_correct"] += 1

        # Count DB candidates with various signal combos
        candidate_counts = count_candidates_by_signals(
            signals["ocr_name_cleaned"] if name_eval["fuzzy_score"] >= 70 else None,
            detected_hp,
            detected_type if type_correct else None,  # only count if type is right
        )

        # Count exact name candidates (using ground truth name, to measure
        # the theoretical best case if OCR were perfect)
        exact_name_candidates = count_exact_name_candidates(card["name"])

        # Also count using the OCR name (to measure actual narrowing)
        ocr_name_candidates = count_exact_name_candidates(
            signals["ocr_name_cleaned"]
        ) if signals["ocr_name_cleaned"] else None

        # Build result entry
        entry = {
            "card_id": card["card_id"],
            "name": card["name"],
            "segment": card["segment"],
            "ground_truth": gt,
            "signals": signals,
            "evaluation": {
                "name_eval": name_eval,
                "hp_correct": hp_correct,
                "hp_gt": gt_hp,
                "hp_detected": detected_hp,
                "type_correct": type_correct,
                "type_gt": gt_types,
                "type_detected": detected_type,
                "hp_det_name_eval": hp_det_name_eval,
            },
            "candidate_counts": candidate_counts,
            "exact_name_candidates": exact_name_candidates,
            "ocr_name_candidates": ocr_name_candidates,
        }
        results.append(entry)

        # Log summary for this card
        logger.info(
            "  OCR name: %r -> %s (fuzzy=%d)",
            signals["ocr_name_cleaned"],
            name_eval["match_type"],
            name_eval["fuzzy_score"],
        )
        logger.info(
            "  HP: detected=%s gt=%s correct=%s",
            detected_hp, gt_hp, hp_correct,
        )
        logger.info(
            "  Type: detected=%s gt=%s correct=%s",
            detected_type, gt_types, type_correct,
        )
        logger.info(
            "  DB candidates: name_only=%s name+hp=%s name+type=%s all=%s | exact_name=%s",
            candidate_counts.get("name_only"),
            candidate_counts.get("name_hp"),
            candidate_counts.get("name_type"),
            candidate_counts.get("name_hp_type"),
            exact_name_candidates,
        )

    # Build final output
    output = {
        "eval_file": str(EVAL_PATH),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": summary,
        "accuracy": {
            "ocr_name_rate": round(summary["ocr_name_correct"] / summary["total_cards"], 3)
            if summary["total_cards"] > 0 else 0,
            "ocr_name_partial_rate": round(
                (summary["ocr_name_correct"] + summary["ocr_name_partial"]) / summary["total_cards"], 3
            ) if summary["total_cards"] > 0 else 0,
            "hp_rate": round(
                summary["hp_correct"] / (summary["hp_correct"] + summary["hp_detected_wrong"] + summary["hp_not_detected"]), 3
            ) if (summary["hp_correct"] + summary["hp_detected_wrong"] + summary["hp_not_detected"]) > 0 else 0,
            "type_rate": round(
                summary["type_correct"] / (summary["type_correct"] + summary["type_wrong"]), 3
            ) if (summary["type_correct"] + summary["type_wrong"]) > 0 else 0,
        },
        "cards": results,
    }

    # Compute average candidate narrowing stats
    name_only_counts = [c["candidate_counts"]["name_only"] for c in results if c["candidate_counts"]["name_only"] is not None]
    name_hp_counts = [c["candidate_counts"]["name_hp"] for c in results if c["candidate_counts"]["name_hp"] is not None]
    name_type_counts = [c["candidate_counts"]["name_type"] for c in results if c["candidate_counts"]["name_type"] is not None]
    all_counts = [c["candidate_counts"]["name_hp_type"] for c in results if c["candidate_counts"]["name_hp_type"] is not None]
    exact_counts = [c["exact_name_candidates"] for c in results]

    output["narrowing_stats"] = {
        "avg_exact_name_candidates": round(sum(exact_counts) / len(exact_counts), 1) if exact_counts else None,
        "avg_ocr_name_candidates": round(
            sum(c for c in [r["ocr_name_candidates"] for r in results] if c is not None)
            / len([c for c in [r["ocr_name_candidates"] for r in results] if c is not None]), 1
        ) if any(r["ocr_name_candidates"] is not None for r in results) else None,
        "avg_name_only": round(sum(name_only_counts) / len(name_only_counts), 1) if name_only_counts else None,
        "avg_name_hp": round(sum(name_hp_counts) / len(name_hp_counts), 1) if name_hp_counts else None,
        "avg_name_type": round(sum(name_type_counts) / len(name_type_counts), 1) if name_type_counts else None,
        "avg_name_hp_type": round(sum(all_counts) / len(all_counts), 1) if all_counts else None,
    }

    # Ensure output dir exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("Total cards: %d", summary["total_cards"])
    logger.info("OCR name correct (>=90): %d/%d (%.0f%%)",
                summary["ocr_name_correct"], summary["total_cards"],
                100 * summary["ocr_name_correct"] / max(1, summary["total_cards"]))
    logger.info("OCR name partial (70-89): %d/%d",
                summary["ocr_name_partial"], summary["total_cards"])
    logger.info("OCR name fail (<70): %d/%d",
                summary["ocr_name_fail"], summary["total_cards"])
    logger.info("HP correct: %d, wrong: %d, not detected: %d",
                summary["hp_correct"], summary["hp_detected_wrong"], summary["hp_not_detected"])
    logger.info("Type correct: %d, wrong: %d, not detected: %d",
                summary["type_correct"], summary["type_wrong"], summary["type_not_detected"])
    logger.info("HP detector name correct (>=70): %d/%d",
                summary["hp_det_name_correct"], summary["total_cards"])
    logger.info("")
    logger.info("Candidate narrowing (averages):")
    for k, v in output["narrowing_stats"].items():
        logger.info("  %s: %s", k, v)
    logger.info("")
    logger.info("Results saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
