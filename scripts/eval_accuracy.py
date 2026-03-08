#!/usr/bin/env python3
"""Evaluate card identification accuracy of identify_page_v2 against ground truth."""

import json
import os
import sys
import time
from difflib import SequenceMatcher

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.json")
INBOX_DIR = os.path.join(DATA_DIR, "inbox")


def load_card_names_lookup():
    """Build card_id -> name lookup from card_names.json and DB."""
    lookup = {}
    # From card_names.json
    cn_path = os.path.join(DATA_DIR, "card_names.json")
    if os.path.exists(cn_path):
        with open(cn_path) as f:
            for row in json.load(f):
                # [card_id, name, set_id, ...]
                lookup[row[0]] = row[1]
    # Also try DB
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine("postgresql+psycopg2://godli@/cardprice")
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT card_id, name FROM dim_cards")).fetchall()
            for r in rows:
                # card_id in DB may not have /variant suffix
                lookup[r[0]] = r[1]
    except Exception:
        pass
    return lookup


def fuzzy_name_match(predicted: str, expected: str) -> bool:
    """Check if predicted name fuzzy-matches expected name."""
    if not predicted or not expected:
        return False
    p = predicted.lower().strip()
    e = expected.lower().strip()
    # Exact match
    if p == e:
        return True
    # One contains the other
    if p in e or e in p:
        return True
    # SequenceMatcher ratio
    ratio = SequenceMatcher(None, p, e).ratio()
    return ratio >= 0.75


def get_name_from_result(result, card_names_lookup):
    """Extract Pokemon name from a pipeline result dict."""
    if not result:
        return None
    # Check raw_response signals for name_used
    raw = result.get("raw_response", {})
    signals = raw.get("signals", {})
    name_used = signals.get("name_used")
    if name_used:
        return name_used
    # Look up card_id in our name lookup
    card_id = result.get("card_id")
    if card_id:
        # Try exact match
        if card_id in card_names_lookup:
            return card_names_lookup[card_id]
        # Try without variant suffix
        base_id = card_id.split("/")[0] if "/" in card_id else card_id
        # Search for any variant
        for cid, name in card_names_lookup.items():
            if cid.startswith(base_id + "/") or cid == base_id:
                return name
    return card_id  # fallback: return card_id itself


def main():
    print("Loading ground truth...")
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    pages = gt["pages"]
    print(f"Found {len(pages)} pages in ground truth\n")

    print("Loading card names lookup...")
    card_names_lookup = load_card_names_lookup()
    print(f"  {len(card_names_lookup)} card names loaded\n")

    # Import pipeline
    print("Importing ML pipeline (this may take a moment)...")
    t0 = time.time()
    from cardprice.ml import identify_page_v2
    print(f"  Import done in {time.time() - t0:.1f}s\n")

    # Create DB session
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    engine = create_engine("postgresql+psycopg2://godli@/cardprice")

    total_correct = 0
    total_cards = 0
    total_pages_perfect = 0
    all_failures = []

    for page_idx, (page_name, page_gt) in enumerate(pages.items()):
        page_dir = os.path.join(INBOX_DIR, page_name)
        if not os.path.isdir(page_dir):
            print(f"SKIP {page_name}: directory not found")
            continue

        # Collect card image paths
        card_paths = []
        gt_cards = {}
        for key, val in page_gt.items():
            if key.startswith("card_") and isinstance(val, dict):
                card_num = key  # e.g. "card_00"
                img_path = os.path.join(page_dir, f"{card_num}.png")
                if os.path.exists(img_path):
                    card_paths.append(img_path)
                    gt_cards[len(card_paths) - 1] = val
                else:
                    print(f"  WARNING: {img_path} not found")

        if not card_paths:
            print(f"SKIP {page_name}: no card images found")
            continue

        desc = page_gt.get("description", "")
        print(f"[{page_idx+1}/{len(pages)}] {page_name} ({len(card_paths)} cards) - {desc}")

        # Run pipeline
        t_start = time.time()
        with Session(engine) as session:
            results = identify_page_v2(card_paths, session=session)
        elapsed = time.time() - t_start
        print(f"  Pipeline: {elapsed:.1f}s ({elapsed/len(card_paths):.1f}s/card)")

        # Compare results
        page_correct = 0
        page_failures = []
        for i, (result, (gt_idx, gt_card)) in enumerate(zip(results, gt_cards.items())):
            expected_name = gt_card["name"]
            predicted_name = get_name_from_result(result, card_names_lookup)
            card_id = result.get("card_id", "???") if result else "???"
            confidence = result.get("confidence", 0) if result else 0
            method = result.get("method", "???") if result else "???"

            if fuzzy_name_match(predicted_name, expected_name):
                page_correct += 1
            else:
                failure = {
                    "page": page_name,
                    "slot": f"card_{i:02d}",
                    "expected": expected_name,
                    "predicted": predicted_name,
                    "card_id": card_id,
                    "confidence": confidence,
                    "method": method,
                    "gt_hp": gt_card.get("hp"),
                    "gt_era": gt_card.get("era", ""),
                }
                page_failures.append(failure)

        total_correct += page_correct
        total_cards += len(card_paths)
        if page_correct == len(card_paths):
            total_pages_perfect += 1
            print(f"  Result: {page_correct}/{len(card_paths)} PERFECT")
        else:
            print(f"  Result: {page_correct}/{len(card_paths)} ({len(page_failures)} failures)")
            for f in page_failures:
                print(f"    FAIL {f['slot']}: expected={f['expected']!r}, "
                      f"got={f['predicted']!r} (card_id={f['card_id']}, "
                      f"conf={f['confidence']:.3f}, method={f['method']})")
        all_failures.extend(page_failures)
        print()

    # Summary
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Cards correct:  {total_correct}/{total_cards} ({100*total_correct/total_cards:.1f}%)")
    print(f"Perfect pages:  {total_pages_perfect}/{len(pages)}")
    print(f"Total failures: {len(all_failures)}")
    print()

    if all_failures:
        print("ALL FAILURES:")
        print("-" * 70)
        for i, f in enumerate(all_failures, 1):
            print(f"{i:3d}. [{f['page']}] {f['slot']}")
            print(f"     Expected: {f['expected']!r} (HP={f['gt_hp']}, era={f['gt_era']})")
            print(f"     Got:      {f['predicted']!r} (card_id={f['card_id']}, "
                  f"conf={f['confidence']:.3f}, method={f['method']})")
            print()


if __name__ == "__main__":
    main()
