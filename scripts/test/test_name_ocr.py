#!/usr/bin/env python3
"""Test Pokemon name OCR detection on the eval dataset.

Tests detect_pokemon_name() on all 27 card segments from binder_eval.json.
Reports per-card results and overall accuracy.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

EVAL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'eval', 'binder_eval.json')
BASE = os.path.join(os.path.dirname(__file__), '..')


def normalize_name(name: str) -> str:
    """Normalize a Pokemon name for comparison.

    Strips delta symbols, 'ex' suffixes, and normalizes case/whitespace.
    """
    import re
    name = name.strip()
    # Remove delta symbol and variants
    name = name.replace('\u03b4', '').replace('δ', '')
    # Remove ' ex' suffix for comparison (we just want the base name)
    name = re.sub(r'\s+ex$', '', name, flags=re.IGNORECASE)
    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name.lower()


def main():
    debug = '--debug' in sys.argv

    with open(EVAL_PATH) as f:
        eval_data = json.load(f)

    from cardprice.ml.ocr_matcher import detect_pokemon_name

    total = 0
    correct = 0
    name_detected = 0
    results = []

    t0 = time.time()

    for page in eval_data['pages']:
        seg_dir = page['segments_dir']
        page_name = os.path.basename(seg_dir)
        print(f"\n=== {page_name} ===")

        for card in page['cards']:
            if not card['card_id']:
                print(f"  {card['segment']:12s} SKIP (empty slot)")
                continue

            seg_path = os.path.join(BASE, seg_dir, card['segment'])
            if not os.path.exists(seg_path):
                print(f"  {card['segment']:12s} SKIP (file not found)")
                continue

            total += 1
            expected = card['name']
            expected_norm = normalize_name(expected)

            detected_name, confidence = detect_pokemon_name(seg_path, debug=debug)

            if detected_name is not None:
                name_detected += 1
                detected_norm = normalize_name(detected_name)
                # Check if the base Pokemon name matches
                # Handle cases like "Latias ex" -> detected "Latias"
                # or "Flygon ex delta" -> detected "Flygon"
                match = (detected_norm == expected_norm
                         or detected_norm in expected_norm
                         or expected_norm in detected_norm
                         or expected_norm.startswith(detected_norm)
                         or detected_norm.startswith(expected_norm))
                if match:
                    correct += 1
                    status = "OK"
                else:
                    status = "WRONG"

                print(f"  {card['segment']:12s} {status:5s} "
                      f"expected={expected:20s} detected={detected_name:20s} "
                      f"(conf={confidence:.2f})")
            else:
                print(f"  {card['segment']:12s} MISS  "
                      f"expected={expected:20s} detected=None")

            results.append({
                'segment': card['segment'],
                'expected': expected,
                'detected': detected_name,
                'confidence': confidence,
                'correct': detected_name is not None and (
                    normalize_name(detected_name) == expected_norm
                    or normalize_name(detected_name) in expected_norm
                    or expected_norm in normalize_name(detected_name or '')
                ),
            })

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"RESULTS: {correct}/{total} correct ({100*correct/total:.0f}%)")
    print(f"  Names detected: {name_detected}/{total} ({100*name_detected/total:.0f}%)")
    print(f"  Names correct:  {correct}/{total} ({100*correct/total:.0f}%)")
    if name_detected > 0:
        print(f"  Precision (correct/detected): {correct}/{name_detected} "
              f"({100*correct/name_detected:.0f}%)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total:.1f}s/card)")
    print(f"{'='*60}")

    # Categorize failures
    misses = [r for r in results if r['detected'] is None and not r['correct']]
    wrong = [r for r in results if r['detected'] is not None and not r['correct']]

    if wrong:
        print(f"\nWrong detections ({len(wrong)}):")
        for r in wrong:
            print(f"  {r['segment']:12s} expected={r['expected']:20s} "
                  f"got={r['detected']}")

    if misses:
        print(f"\nMissed detections ({len(misses)}):")
        for r in misses:
            print(f"  {r['segment']:12s} expected={r['expected']}")

    # Per-page breakdown
    print(f"\nPer-page breakdown:")
    page_idx = 0
    for page in eval_data['pages']:
        cards = [c for c in page['cards'] if c['card_id']]
        page_results = results[page_idx:page_idx + len(cards)]
        page_correct = sum(1 for r in page_results if r['correct'])
        page_detected = sum(1 for r in page_results if r['detected'] is not None)
        page_name = os.path.basename(page['segments_dir'])
        print(f"  {page_name}: {page_correct}/{len(cards)} correct, "
              f"{page_detected}/{len(cards)} detected")
        page_idx += len(cards)


if __name__ == '__main__':
    main()
