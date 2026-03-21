#!/usr/bin/env python3
"""Evaluate card identification accuracy using identify_page_v2.

For each ground truth page:
  1. Collect card image paths
  2. Run identify_page_v2 (full pipeline with page context)
  3. Compare predicted card_id against ground truth card_id/name

Usage:
  python scripts/eval_dino_global.py                    # all unique pages
  python scripts/eval_dino_global.py page_name          # single page
  python scripts/eval_dino_global.py --run-all           # orchestrator mode
"""

import gc
import json
import os
import subprocess
import sys
import time
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.json")
INBOX_DIR = os.path.join(DATA_DIR, "inbox")

# Duplicate scans of the same physical page — skip these
DUPLICATE_PAGES = {
    "page_20260305_094637_cards",
    "page_20260305_094749_cards",
    "page_20260307_014621_cards",
    "page_20260307_014711_cards",
    "page_20260307_121117_cards",
}


def fuzzy_name_match(predicted, expected):
    if not predicted or not expected:
        return False
    p = predicted.lower().strip()
    e = expected.lower().strip()
    if p == e:
        return True
    if p in e or e in p:
        return True
    return SequenceMatcher(None, p, e).ratio() >= 0.75


def load_card_names_lookup():
    lookup = {}
    cn_path = os.path.join(DATA_DIR, "card_names.json")
    if os.path.exists(cn_path):
        with open(cn_path) as f:
            for row in json.load(f):
                lookup[row[0]] = row[1]
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine("postgresql+psycopg2://godli@/cardprice")
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT card_id, name FROM dim_cards")).fetchall()
            for r in rows:
                lookup[r[0]] = r[1]
    except Exception:
        pass
    return lookup


def _lookup_name(card_id, lookup):
    """Look up card name from card_id, trying full ID then base ID."""
    if not card_id:
        return ""
    if card_id in lookup:
        return lookup[card_id]
    base = card_id.split("/")[0] if "/" in card_id else card_id
    if base in lookup:
        return lookup[base]
    for cid, name in lookup.items():
        if cid.startswith(base + "/") or cid.startswith(base + "-"):
            return name
    return card_id


def card_id_match(predicted_id, expected_id):
    """Check if two card_ids match, ignoring /normal suffix differences."""
    if not predicted_id or not expected_id:
        return False
    def normalize(cid):
        if cid.endswith("/normal"):
            return cid[:-7]
        return cid
    return normalize(predicted_id) == normalize(expected_id)


def run_single_page(page_name):
    """Run identify_page_v2 on a single page and print results as JSON to stdout."""
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    page_gt = gt["pages"].get(page_name)
    if not page_gt:
        print(json.dumps({"error": f"Page {page_name} not in ground truth"}))
        return

    page_dir = os.path.join(INBOX_DIR, page_name)
    if not os.path.isdir(page_dir):
        print(json.dumps({"error": f"Page dir {page_dir} not found"}))
        return

    card_keys = sorted(
        [k for k in page_gt if k.startswith("card_") and isinstance(page_gt[k], dict)],
        key=lambda k: int(k.split("_")[1])
    )
    card_paths = []
    gt_entries = []
    for key in card_keys:
        img_path = os.path.join(page_dir, f"{key}.png")
        if os.path.exists(img_path):
            card_paths.append(img_path)
            gt_entries.append(page_gt[key])

    if not card_paths:
        print(json.dumps({"error": "No card images found"}))
        return

    from cardprice.ml import identify_page_v2
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    engine = create_engine("postgresql+psycopg2://godli@/cardprice")

    t_page = time.time()
    with Session(engine) as session:
        page_results = identify_page_v2(card_paths, session=session)
    page_time = time.time() - t_page

    # Output results as JSON (one line, parseable by orchestrator)
    output = {
        "page": page_name,
        "time": round(page_time, 1),
        "results": []
    }
    for i, (gt_entry, result) in enumerate(zip(gt_entries, page_results)):
        output["results"].append({
            "card": f"card_{i:02d}",
            "gt_name": gt_entry.get("name", ""),
            "gt_id": gt_entry.get("card_id", ""),
            "empty_slot": gt_entry.get("empty_slot", False),
            "predicted_id": result.get("card_id", "") or "",
            "method": result.get("method", "unknown") or "unknown",
            "confidence": result.get("confidence", 0.0) or 0.0,
        })

    # Print the JSON result on a special marker line so orchestrator can find it
    print(f"EVAL_RESULT:{json.dumps(output)}")


def run_all():
    """Orchestrator: run each page in a separate subprocess to avoid OOM."""
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    pages = gt["pages"]
    sorted_page_names = sorted(pages.keys())
    card_names_lookup = load_card_names_lookup()

    total_correct = 0
    total_cards = 0
    results_by_method = {}
    correct_by_method = {}
    all_errors = []
    total_time = 0

    script_path = os.path.abspath(__file__)

    for page_name in sorted_page_names:
        if page_name in DUPLICATE_PAGES:
            continue

        page_dir = os.path.join(INBOX_DIR, page_name)
        if not os.path.isdir(page_dir):
            continue

        page_gt = pages[page_name]
        desc = page_gt.get("description", "")
        print(f"\n{'='*70}")
        print(f"{page_name} - {desc}")
        print("="*70)

        # Run in subprocess to isolate memory
        result = subprocess.run(
            [sys.executable, script_path, "--single", page_name],
            capture_output=True, text=True, timeout=600
        )

        # Parse result from stdout
        page_data = None
        for line in result.stdout.splitlines():
            if line.startswith("EVAL_RESULT:"):
                page_data = json.loads(line[len("EVAL_RESULT:"):])
                break

        if not page_data:
            print(f"  FAILED: subprocess returned no results")
            if result.stderr:
                # Show last few lines of stderr
                stderr_lines = result.stderr.strip().splitlines()
                for line in stderr_lines[-3:]:
                    print(f"  stderr: {line}")
            continue

        total_time += page_data.get("time", 0)
        print(f"  identify_page_v2 completed in {page_data['time']}s")

        for card_result in page_data["results"]:
            if card_result.get("empty_slot"):
                continue
            expected_name = card_result["gt_name"]
            if not expected_name:
                continue

            expected_id = card_result["gt_id"]
            predicted_id = card_result["predicted_id"]
            predicted_name = _lookup_name(predicted_id, card_names_lookup)
            method = card_result["method"]
            confidence = card_result["confidence"]

            total_cards += 1
            results_by_method[method] = results_by_method.get(method, 0) + 1

            if expected_id:
                correct = card_id_match(predicted_id, expected_id)
            else:
                correct = fuzzy_name_match(predicted_name, expected_name)

            if correct:
                total_correct += 1
                correct_by_method[method] = correct_by_method.get(method, 0) + 1

            status = "OK" if correct else ("WRONG" if predicted_id else "MISS")

            print(f"  {card_result['card']} {status:<5} exp={expected_name:<25} "
                  f"got={predicted_name:<25} method={method:<15} "
                  f"conf={confidence:.3f}  id={predicted_id}")

            if not correct:
                all_errors.append({
                    "page": page_name,
                    "card": card_result["card"],
                    "expected_name": expected_name,
                    "expected_id": expected_id,
                    "predicted_name": predicted_name,
                    "predicted_id": predicted_id,
                    "method": method,
                    "confidence": confidence,
                })

    print(f"\n{'='*70}")
    if total_cards == 0:
        print("No cards evaluated!")
        return
    print(f"OVERALL: {total_correct}/{total_cards} ({100*total_correct/total_cards:.1f}%)")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"\nBy method:")
    for method in sorted(results_by_method.keys()):
        n = results_by_method[method]
        c = correct_by_method.get(method, 0)
        print(f"  {method:<20} {c}/{n} correct")

    if all_errors:
        print(f"\nErrors ({len(all_errors)}):")
        for e in all_errors:
            print(f"  {e['page']}/{e['card']}: "
                  f"exp={e['expected_name']} ({e['expected_id']}) "
                  f"got={e['predicted_name']} ({e['predicted_id']}) "
                  f"method={e['method']} conf={e['confidence']:.3f}")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--single":
        # Subprocess mode: run single page, output JSON
        run_single_page(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--run-all":
        run_all()
    elif len(sys.argv) == 1:
        # Default: orchestrator mode
        run_all()
    else:
        # Single page filter (legacy usage)
        # Run as orchestrator but filter to matching pages
        # For simplicity, just run the single page directly
        run_single_page(sys.argv[1])


if __name__ == "__main__":
    main()
