#!/usr/bin/env python3
"""Validate _is_card_back() detection against binder eval segments and reference images.

Tests:
  1. All 27 binder segments from binder_eval.json (3 pages x 9 cards)
     - Page 2 card 0 (card_id=null) is the only expected card back
  2. Regular card reference images should NOT be detected as backs
  3. Blue Water-type edge cases (Blastoise, Gyarados, Lapras, Articuno)
     should NOT trigger false positives despite heavy blue artwork
"""

import json
import os
import sys

# Allow running from repo root or scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cardprice.ml import _is_card_back

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_binder_eval():
    """Load all segments from binder_eval.json with expected is_card_back labels."""
    path = os.path.join(ROOT, "data", "eval", "binder_eval.json")
    with open(path) as f:
        data = json.load(f)

    entries = []
    for page in data["pages"]:
        seg_dir = os.path.join(ROOT, page["segments_dir"])
        for card in page["cards"]:
            seg_path = os.path.join(seg_dir, card["segment"])
            # card_id is null for the card back
            expected_back = card["card_id"] is None
            entries.append({
                "path": seg_path,
                "expected_back": expected_back,
                "label": card["name"],
            })
    return entries


def reference_image_entries():
    """A handful of normal card reference images -- should never be card backs."""
    card_dir = os.path.join(ROOT, "data", "card_images")
    samples = [
        # Arbitrary non-blue cards
        ("base1/base1-1_normal.png", "Alakazam (base1)"),
        ("base1/base1-4_normal.png", "Charizard (base1)"),
        ("base1/base1-15_normal.png", "Venusaur (base1)"),
        ("base1/base1-16_normal.png", "Zapdos (base1)"),
        # Green / grass
        ("base1/base1-11_normal.png", "Nidoking (base1)"),
    ]
    entries = []
    for rel, label in samples:
        p = os.path.join(card_dir, rel)
        if os.path.exists(p):
            entries.append({"path": p, "expected_back": False, "label": f"[ref] {label}"})
    return entries


def blue_edge_case_entries():
    """Blue Water-type cards -- the hardest negatives for a blue-dominance detector."""
    card_dir = os.path.join(ROOT, "data", "card_images")
    blue_cards = [
        ("base1/base1-2_normal.png", "Blastoise (base1)"),
        ("base1/base1-6_normal.png", "Gyarados (base1)"),
        ("base2/base2-10_normal.png", "Lapras (base2/Fossil)"),
        ("base2/base2-3_normal.png", "Articuno (base2/Fossil)"),
    ]
    entries = []
    for rel, label in blue_cards:
        p = os.path.join(card_dir, rel)
        if os.path.exists(p):
            entries.append({"path": p, "expected_back": False, "label": f"[blue] {label}"})
        else:
            print(f"  WARN: missing {p}, skipping")
    return entries


def main():
    all_entries = []

    print("Loading binder eval segments...")
    binder = load_binder_eval()
    all_entries.extend(binder)
    print(f"  {len(binder)} binder segments")

    print("Loading reference card images...")
    refs = reference_image_entries()
    all_entries.extend(refs)
    print(f"  {len(refs)} reference images")

    print("Loading blue edge-case cards...")
    blues = blue_edge_case_entries()
    all_entries.extend(blues)
    print(f"  {len(blues)} blue edge cases")

    print()
    header = f"{'image_path':<75} {'is_back':>7} {'expected':>8} {'correct':>7}"
    print(header)
    print("-" * len(header))

    correct = 0
    total = 0
    false_positives = []
    false_negatives = []

    for entry in all_entries:
        path = entry["path"]
        expected = entry["expected_back"]
        label = entry["label"]

        if not os.path.exists(path):
            short = os.path.relpath(path, ROOT)
            print(f"{short:<75} {'MISSING':>7} {'':>8} {'':>7}")
            continue

        result = _is_card_back(path)
        ok = result == expected
        total += 1
        if ok:
            correct += 1
        else:
            if result and not expected:
                false_positives.append((path, label))
            else:
                false_negatives.append((path, label))

        short = os.path.relpath(path, ROOT)
        tag = "OK" if ok else "FAIL"
        print(f"{short:<75} {str(result):>7} {str(expected):>8} {tag:>7}")

    print()
    print(f"Results: {correct}/{total} correct ({100*correct/total:.1f}%)")
    if false_positives:
        print(f"\nFalse positives ({len(false_positives)}) -- wrongly flagged as card back:")
        for p, label in false_positives:
            print(f"  {label}: {os.path.relpath(p, ROOT)}")
    if false_negatives:
        print(f"\nFalse negatives ({len(false_negatives)}) -- missed actual card back:")
        for p, label in false_negatives:
            print(f"  {label}: {os.path.relpath(p, ROOT)}")
    if not false_positives and not false_negatives:
        print("\nAll tests passed -- no false positives or false negatives.")

    sys.exit(0 if correct == total else 1)


if __name__ == "__main__":
    main()
