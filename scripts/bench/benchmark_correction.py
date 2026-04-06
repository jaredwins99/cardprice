#!/usr/bin/env python3
"""Benchmark card identification accuracy with and without perspective correction.

Runs identify_card_v2 on every ground truth card twice:
  1. Without any image correction (baseline)
  2. With CLAHE-based image correction applied

Reports accuracy, confidence changes, and per-page breakdown.

Usage:
  python scripts/bench/benchmark_correction.py
"""

import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
INBOX_DIR = DATA_DIR / "inbox"

# Duplicate scans — use first scan only
DUPLICATE_PAGES = {
    "page_20260305_094637_cards",
    "page_20260305_094749_cards",
    "page_20260307_014621_cards",
    "page_20260307_014711_cards",
    "page_20260307_121117_cards",
}


def load_card_names_lookup():
    """Build card_id -> name lookup from card_names.json and DB."""
    lookup = {}
    cn_path = DATA_DIR / "card_names.json"
    if cn_path.exists():
        with open(cn_path) as f:
            for row in json.load(f):
                lookup[row[0]] = row[1]
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine("postgresql+psycopg2://godli@/cardprice")
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT card_id, name FROM dim_cards"))
            for r in rows:
                lookup[r[0]] = r[1]
    except Exception:
        pass
    return lookup


def fuzzy_name_match(predicted, expected):
    """Check if predicted name fuzzy-matches expected name."""
    if not predicted or not expected:
        return False
    p = predicted.lower().strip()
    e = expected.lower().strip()
    if p == e:
        return True
    if p in e or e in p:
        return True
    return SequenceMatcher(None, p, e).ratio() >= 0.75


def apply_correction(img):
    """Apply CLAHE-based image correction (contrast enhancement on L channel).

    If cardprice.ml.card_corrector exists, use it instead.
    """
    try:
        from cardprice.ml.card_corrector import correct_card
        return correct_card(img)
    except (ImportError, AttributeError):
        pass

    # Fallback: CLAHE on L channel of LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def is_correct(result, gt_card, name_lookup):
    """Check if identification result matches ground truth."""
    pred_id = result.get("card_id")
    gt_id = gt_card.get("card_id")
    gt_name = gt_card.get("name", "")

    if not pred_id:
        return False

    # Exact card_id match (strip variant suffix for comparison)
    pred_base = pred_id.split("/")[0] if pred_id else ""
    gt_base = gt_id.split("/")[0] if gt_id else ""
    if pred_base and gt_base and pred_base == gt_base:
        return True

    # Fuzzy name match as fallback (some ground truth card_ids may differ)
    pred_name = name_lookup.get(pred_id, "")
    if fuzzy_name_match(pred_name, gt_name):
        return True

    return False


def build_test_cards():
    """Load ground truth and return list of (page_dir, slot_key, gt_card, img_path)."""
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    cards = []
    for page_dir, page_data in gt["pages"].items():
        if page_dir in DUPLICATE_PAGES:
            continue
        if isinstance(page_data, str):
            continue

        for slot_key in sorted(k for k in page_data if k.startswith("card_")):
            card = page_data[slot_key]
            if card.get("empty_slot"):
                continue
            if not card.get("card_id") and not card.get("name"):
                continue

            img_path = INBOX_DIR / page_dir / f"{slot_key}.png"
            if not img_path.exists():
                continue

            cards.append((page_dir, slot_key, card, str(img_path)))

    return cards


def run_benchmark():
    """Run the full benchmark comparing corrected vs uncorrected identification."""
    from cardprice.ml import identify_card_v2, _scan_cache

    name_lookup = load_card_names_lookup()
    test_cards = build_test_cards()

    print(f"Loaded {len(test_cards)} ground truth cards across "
          f"{len(set(c[0] for c in test_cards))} pages")
    print()

    results_without = []
    results_with = []
    correction_times = []

    for i, (page_dir, slot_key, gt_card, img_path) in enumerate(test_cards):
        gt_name = gt_card.get("name", "?")
        gt_id = gt_card.get("card_id", "?")
        short_page = page_dir.replace("page_", "").replace("_cards", "")

        print(f"[{i+1}/{len(test_cards)}] {short_page}/{slot_key}: {gt_name} ({gt_id})")

        # --- Run WITHOUT correction ---
        _scan_cache.clear()
        t0 = time.time()
        result_without = identify_card_v2(img_path, detect_variants=False)
        time_without = time.time() - t0

        correct_without = is_correct(result_without, gt_card, name_lookup)
        conf_without = result_without.get("confidence", 0)
        method_without = result_without.get("method", "?")

        results_without.append({
            "page": page_dir,
            "slot": slot_key,
            "gt_name": gt_name,
            "gt_id": gt_id,
            "pred_id": result_without.get("card_id"),
            "confidence": conf_without,
            "method": method_without,
            "correct": correct_without,
            "time": time_without,
        })

        # --- Apply correction and run WITH correction ---
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: could not read image {img_path}")
            results_with.append(results_without[-1].copy())
            continue

        t_corr_start = time.time()
        corrected = apply_correction(img)
        t_corr = time.time() - t_corr_start
        correction_times.append(t_corr)

        # Save corrected image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, corrected)

        _scan_cache.clear()
        t0 = time.time()
        result_with = identify_card_v2(tmp_path, detect_variants=False)
        time_with = time.time() - t0

        os.unlink(tmp_path)

        correct_with = is_correct(result_with, gt_card, name_lookup)
        conf_with = result_with.get("confidence", 0)
        method_with = result_with.get("method", "?")

        results_with.append({
            "page": page_dir,
            "slot": slot_key,
            "gt_name": gt_name,
            "gt_id": gt_id,
            "pred_id": result_with.get("card_id"),
            "confidence": conf_with,
            "method": method_with,
            "correct": correct_with,
            "time": time_with,
        })

        # Status indicator
        status = ""
        if correct_without and correct_with:
            status = "  both correct"
        elif not correct_without and correct_with:
            status = "  IMPROVED (wrong->right)"
        elif correct_without and not correct_with:
            status = "  REGRESSED (right->wrong)"
        else:
            status = "  both wrong"

        conf_delta = conf_with - conf_without
        pred_without = result_without.get('card_id') or 'None'
        pred_with = result_with.get('card_id') or 'None'
        print(f"  without: {pred_without:30s} "
              f"conf={conf_without:.3f} ({method_without})")
        print(f"  with:    {pred_with:30s} "
              f"conf={conf_with:.3f} ({method_with})")
        print(f"  {status}  conf_delta={conf_delta:+.3f}")
        print()

    # === Generate Report ===
    print()
    print("=" * 60)
    print("=== Perspective Correction Benchmark ===")
    print("=" * 60)

    total = len(test_cards)
    correct_without_count = sum(1 for r in results_without if r["correct"])
    correct_with_count = sum(1 for r in results_with if r["correct"])

    print(f"Cards tested: {total}")
    print(f"Without correction: {correct_without_count}/{total} "
          f"({100*correct_without_count/total:.1f}%)")
    print(f"With correction:    {correct_with_count}/{total} "
          f"({100*correct_with_count/total:.1f}%)")
    print()

    # Categorize changes
    improved = []
    regressed = []
    both_right = []
    both_wrong = []

    for rw, rc in zip(results_without, results_with):
        if not rw["correct"] and rc["correct"]:
            improved.append((rw, rc))
        elif rw["correct"] and not rc["correct"]:
            regressed.append((rw, rc))
        elif rw["correct"] and rc["correct"]:
            both_right.append((rw, rc))
        else:
            both_wrong.append((rw, rc))

    print(f"Improved  (wrong->right): {len(improved)} cards")
    print(f"Regressed (right->wrong): {len(regressed)} cards")
    print(f"Unchanged (both right):   {len(both_right)} cards")
    print(f"Unchanged (both wrong):   {len(both_wrong)} cards")
    print()

    # Confidence stats
    confs_without = [r["confidence"] for r in results_without if r["confidence"]]
    confs_with = [r["confidence"] for r in results_with if r["confidence"]]

    if confs_without:
        print("Average confidence:")
        print(f"  Without: {sum(confs_without)/len(confs_without):.3f}")
        print(f"  With:    {sum(confs_with)/len(confs_with):.3f}")
        print()

    # Confidence on correct identifications only
    confs_correct_without = [r["confidence"] for r in results_without
                             if r["correct"] and r["confidence"]]
    confs_correct_with = [r["confidence"] for r in results_with
                          if r["correct"] and r["confidence"]]
    if confs_correct_without:
        print("Average confidence (correct IDs only):")
        print(f"  Without: {sum(confs_correct_without)/len(confs_correct_without):.3f}")
        print(f"  With:    {sum(confs_correct_with)/len(confs_correct_with):.3f}")
        print()

    # Timing
    times_without = [r["time"] for r in results_without]
    times_with = [r["time"] for r in results_with]
    print("Timing:")
    print(f"  Avg per card (without correction): {sum(times_without)/len(times_without):.2f}s")
    print(f"  Avg per card (with correction):    {sum(times_with)/len(times_with):.2f}s")
    if correction_times:
        print(f"  Avg correction overhead:           {sum(correction_times)/len(correction_times)*1000:.1f}ms")
    print(f"  Total (without): {sum(times_without):.1f}s")
    print(f"  Total (with):    {sum(times_with):.1f}s")
    print()

    # Per-page breakdown
    pages = sorted(set(r["page"] for r in results_without))
    print("Per-page breakdown:")
    for page in pages:
        page_rw = [r for r in results_without if r["page"] == page]
        page_rc = [r for r in results_with if r["page"] == page]
        n = len(page_rw)
        c_before = sum(1 for r in page_rw if r["correct"])
        c_after = sum(1 for r in page_rc if r["correct"])
        short = page.replace("page_", "").replace("_cards", "")
        delta = c_after - c_before
        delta_str = f" ({delta:+d})" if delta != 0 else ""
        print(f"  {short}: before={c_before}/{n}, after={c_after}/{n}{delta_str}")
    print()

    # Detail: improved cards
    if improved:
        print("--- Improved cards (wrong -> right) ---")
        for rw, rc in improved:
            short = rw["page"].replace("page_", "").replace("_cards", "")
            print(f"  {short}/{rw['slot']}: {rw['gt_name']}")
            print(f"    before: {rw['pred_id']} (conf={rw['confidence']:.3f})")
            print(f"    after:  {rc['pred_id']} (conf={rc['confidence']:.3f})")
        print()

    # Detail: regressed cards
    if regressed:
        print("--- Regressed cards (right -> wrong) ---")
        for rw, rc in regressed:
            short = rw["page"].replace("page_", "").replace("_cards", "")
            print(f"  {short}/{rw['slot']}: {rw['gt_name']}")
            print(f"    before: {rw['pred_id']} (conf={rw['confidence']:.3f})")
            print(f"    after:  {rc['pred_id']} (conf={rc['confidence']:.3f})")
        print()

    # Detail: both wrong
    if both_wrong:
        print("--- Both wrong ---")
        for rw, rc in both_wrong:
            short = rw["page"].replace("page_", "").replace("_cards", "")
            print(f"  {short}/{rw['slot']}: {rw['gt_name']} (gt={rw['gt_id']})")
            print(f"    before: {rw['pred_id']} (conf={rw['confidence']:.3f})")
            print(f"    after:  {rc['pred_id']} (conf={rc['confidence']:.3f})")
        print()

    # Method distribution
    print("Method distribution:")
    methods_without = defaultdict(int)
    methods_with = defaultdict(int)
    for r in results_without:
        methods_without[r["method"]] += 1
    for r in results_with:
        methods_with[r["method"]] += 1
    all_methods = sorted(set(methods_without) | set(methods_with))
    print(f"  {'Method':<30s} {'Without':>8s} {'With':>8s}")
    for m in all_methods:
        print(f"  {m:<30s} {methods_without.get(m,0):>8d} {methods_with.get(m,0):>8d}")


if __name__ == "__main__":
    run_benchmark()
