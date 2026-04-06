"""Scan result logger — appends each card identification to a CSV for tracking accuracy over time.

Usage:
    from cardprice.scan_logger import log_scan_result, get_accuracy_summary

    # After identifying a card:
    log_scan_result(page_id="page_20260305_094228_cards", position=0, result_dict={
        "card_id": "ex15-26/normal",
        "confidence": 0.96,
        "method": "v2_card_number",
        "detected_variant": "normal",
        "raw_response": {"ocr_name": "Bayleef", "ocr_confidence": 0.91},
    })

    # Get running stats:
    summary = get_accuracy_summary()
    # => {"total": 108, "with_ground_truth": 95, "correct": 93, "accuracy": 0.979, ...}
"""

import csv
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "scan_results.csv"

COLUMNS = [
    "timestamp",
    "page_id",
    "position",
    "card_id_predicted",
    "card_name",
    "confidence",
    "method",
    "ocr_name",
    "ocr_conf",
    "variant_detected",
    "variant_correct",   # nullable — filled manually or via ground truth
    "is_correct",        # nullable — filled when ground truth exists
]

_lock = threading.Lock()

# In-memory index: (page_id, position) -> row index in _rows
_rows: list[dict] = []
_index: dict[tuple[str, int], int] = {}
_ground_truth: dict[str, dict] = {}  # page_id -> {position_int: card_id}
_loaded = False


def _load_ground_truth():
    """Load ground truth data from data/ground_truth.json."""
    global _ground_truth
    gt_path = Path(__file__).resolve().parent.parent / "data" / "ground_truth.json"
    if not gt_path.exists():
        logger.warning("Ground truth file not found: %s", gt_path)
        return
    try:
        with open(gt_path) as f:
            data = json.load(f)
        pages = data.get("pages", {})
        for page_id, page_data in pages.items():
            cards = {}
            for key, val in page_data.items():
                if key.startswith("card_") and isinstance(val, dict) and "card_id" in val:
                    # card_00 -> position 0, card_01 -> position 1, etc.
                    try:
                        pos = int(key.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    cards[pos] = val["card_id"]
            if cards:
                _ground_truth[page_id] = cards
        logger.info("Loaded ground truth: %d pages, %d cards",
                     len(_ground_truth),
                     sum(len(v) for v in _ground_truth.values()))
    except Exception as e:
        logger.error("Failed to load ground truth: %s", e)


def _load_existing_csv():
    """Load existing CSV into memory on first access."""
    global _rows, _index, _loaded
    if _loaded:
        return
    _loaded = True

    _load_ground_truth()

    if not CSV_PATH.exists():
        logger.info("No existing scan_results.csv — starting fresh")
        return
    try:
        with open(CSV_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize types
                try:
                    row["position"] = int(row["position"])
                except (ValueError, TypeError):
                    row["position"] = 0
                try:
                    row["confidence"] = float(row["confidence"]) if row.get("confidence") else None
                except (ValueError, TypeError):
                    row["confidence"] = None
                try:
                    row["ocr_conf"] = float(row["ocr_conf"]) if row.get("ocr_conf") else None
                except (ValueError, TypeError):
                    row["ocr_conf"] = None

                key = (row["page_id"], row["position"])
                if key in _index:
                    # Update existing row (dedup)
                    _rows[_index[key]] = row
                else:
                    _index[key] = len(_rows)
                    _rows.append(row)
        logger.info("Loaded %d scan results from CSV", len(_rows))
    except Exception as e:
        logger.error("Failed to load scan_results.csv: %s", e)


def _check_correctness(page_id: str, position: int, predicted_card_id: str) -> bool | None:
    """Check predicted card_id against ground truth. Returns None if no GT available."""
    gt_page = _ground_truth.get(page_id)
    if gt_page is None:
        return None
    gt_card_id = gt_page.get(position)
    if gt_card_id is None:
        return None
    # Compare base card IDs (strip variant suffix for comparison)
    # Ground truth uses format like "neo1-53/normal", prediction may match
    pred_base = predicted_card_id.split("/")[0] if predicted_card_id else ""
    gt_base = gt_card_id.split("/")[0] if gt_card_id else ""
    return pred_base == gt_base


def _flush_csv():
    """Write all rows to CSV (overwrites file). Caller must hold _lock."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_rows)


def log_scan_result(page_id: str, position: int, result_dict: dict) -> dict:
    """Log a single card scan result.

    Args:
        page_id: Page identifier (e.g. "page_20260305_094228_cards").
        position: Card position on the page (0-8 for 3x3 grid).
        result_dict: Result from identify_card_v2 or the card_data dict from
            the server's scan-page handler. Expected keys:
            - card_id: predicted card ID
            - confidence: identification confidence
            - method: identification method
            - detected_variant: variant type ("normal", "reverse_holo", etc.)
            - raw_response: dict with ocr_name, ocr_confidence, etc.
            Also accepts flattened keys (card_name, ocr_name, ocr_conf) from
            the server response.

    Returns:
        The row dict that was written.
    """
    with _lock:
        _load_existing_csv()

        raw = result_dict.get("raw_response", {}) or {}
        card_id = result_dict.get("card_id")

        # Extract OCR info — try raw_response first, then top-level keys
        ocr_name = raw.get("ocr_name") or result_dict.get("ocr_name")
        ocr_conf = raw.get("ocr_confidence") or raw.get("ocr_conf") or result_dict.get("ocr_conf")

        # Card name — try top-level (from server enrichment) or fallback to ocr_name
        card_name = result_dict.get("card_name") or ocr_name

        # Variant detection
        variant = result_dict.get("detected_variant", "normal")

        # Check against ground truth
        is_correct = _check_correctness(page_id, position, card_id) if card_id else None

        # Check variant correctness against ground truth if available
        variant_correct = None  # No variant ground truth yet — left nullable

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "page_id": page_id,
            "position": position,
            "card_id_predicted": card_id,
            "card_name": card_name,
            "confidence": round(float(ocr_conf), 4) if ocr_conf is not None else result_dict.get("confidence"),
            "method": result_dict.get("method"),
            "ocr_name": ocr_name,
            "ocr_conf": round(float(ocr_conf), 4) if ocr_conf is not None else None,
            "variant_detected": variant,
            "variant_correct": variant_correct,
            "is_correct": is_correct,
        }

        key = (page_id, position)
        if key in _index:
            # Update existing row (dedup)
            _rows[_index[key]] = row
            logger.debug("Updated scan result: %s pos %d -> %s", page_id, position, card_id)
        else:
            _index[key] = len(_rows)
            _rows.append(row)
            logger.debug("Logged scan result: %s pos %d -> %s", page_id, position, card_id)

        _flush_csv()
        return row


def get_accuracy_summary() -> dict:
    """Compute running accuracy statistics from logged scan results.

    Returns dict with:
        total: total number of logged scans
        with_ground_truth: scans that have ground truth comparison
        correct: number of correct identifications
        incorrect: number of incorrect identifications
        accuracy: correct / with_ground_truth (or None if no GT)
        by_method: {method: {total, correct, accuracy}}
        by_page: {page_id: {total, correct, accuracy}}
    """
    with _lock:
        _load_existing_csv()

        total = len(_rows)
        gt_rows = [r for r in _rows if r.get("is_correct") is not None]
        correct = sum(1 for r in gt_rows if r["is_correct"] is True or r["is_correct"] == "True")
        incorrect = len(gt_rows) - correct

        # Breakdown by method
        by_method: dict[str, dict] = {}
        for r in gt_rows:
            m = r.get("method", "unknown")
            if m not in by_method:
                by_method[m] = {"total": 0, "correct": 0}
            by_method[m]["total"] += 1
            if r["is_correct"] is True or r["is_correct"] == "True":
                by_method[m]["correct"] += 1
        for v in by_method.values():
            v["accuracy"] = round(v["correct"] / v["total"], 4) if v["total"] > 0 else None

        # Breakdown by page
        by_page: dict[str, dict] = {}
        for r in gt_rows:
            p = r.get("page_id", "unknown")
            if p not in by_page:
                by_page[p] = {"total": 0, "correct": 0}
            by_page[p]["total"] += 1
            if r["is_correct"] is True or r["is_correct"] == "True":
                by_page[p]["correct"] += 1
        for v in by_page.values():
            v["accuracy"] = round(v["correct"] / v["total"], 4) if v["total"] > 0 else None

        return {
            "total": total,
            "with_ground_truth": len(gt_rows),
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": round(correct / len(gt_rows), 4) if gt_rows else None,
            "by_method": by_method,
            "by_page": by_page,
            "by_card": get_card_ledger(),
        }


def get_card_ledger() -> list[dict]:
    """Get per-card accuracy ledger across all scan attempts.

    Groups all scans by the ground truth card_id (what the card actually is),
    tracks how many times we scanned it and how often we got it right.

    Returns a list sorted by accuracy (worst first), each entry:
        card_id: the actual card identity (from ground truth)
        card_name: display name
        scans: total scan attempts of this card
        correct: how many were correctly identified
        accuracy: correct / scans (0.0 - 1.0)
        predictions: list of {predicted, confidence, page_id, correct}
    """
    _load_existing_csv()

    # Build ledger keyed by ground truth card_id
    ledger: dict[str, dict] = {}

    for r in _rows:
        page_id = r.get("page_id", "")
        position = r.get("position", 0)
        if isinstance(position, str):
            try:
                position = int(position)
            except ValueError:
                continue

        # Look up what this card actually is from ground truth
        gt_page = _ground_truth.get(page_id)
        if not gt_page:
            continue
        actual_id = gt_page.get(position)
        if not actual_id:
            continue

        actual_base = actual_id.split("/")[0] if actual_id else ""
        predicted_id = r.get("card_id_predicted", "")
        predicted_base = predicted_id.split("/")[0] if predicted_id else ""
        is_correct = actual_base == predicted_base

        if actual_id not in ledger:
            ledger[actual_id] = {
                "card_id": actual_id,
                "card_name": r.get("card_name") or actual_id,
                "scans": 0,
                "correct": 0,
                "predictions": [],
            }

        entry = ledger[actual_id]
        entry["scans"] += 1
        if is_correct:
            entry["correct"] += 1
        entry["predictions"].append({
            "predicted": predicted_id,
            "confidence": r.get("confidence"),
            "page_id": page_id,
            "correct": is_correct,
        })

    # Compute accuracy and sort worst-first
    result = []
    for entry in ledger.values():
        entry["accuracy"] = round(entry["correct"] / entry["scans"], 4) if entry["scans"] > 0 else 0.0
        result.append(entry)

    result.sort(key=lambda x: (x["accuracy"], -x["scans"]))
    return result
