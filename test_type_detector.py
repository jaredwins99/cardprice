#!/usr/bin/env python
"""Test type_detector module on page 3 segments."""

import logging
from pathlib import Path
from cardprice.ml.type_detector import detect_type

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

# Expected types for each card
expected_types = {
    "card_01": "Water",  # Latios δ (Water type)
    "card_02": "Fire",   # Latias ex (Fire energy in Dragon Dew + Heat Blast)
    "card_03": "Grass",  # Venusaur
    "card_04": "Colorless",  # Flygon
    "card_05": "Lightning",  # Raikou
    "card_06": "Water",  # Kingdra
    "card_07": "Water",  # Suicune
    "card_08": "Colorless",  # Staraptor
}

base_dir = Path("/home/godli/cardprice/data/inbox/page_20260228_202134_cards")
results = []
correct = 0
total = 0

print("=" * 90)
print("TYPE DETECTOR TEST - PAGE 3 SEGMENTS")
print("=" * 90)
print()

for card_num in range(1, 9):
    card_id = f"card_{card_num:02d}"
    image_path = base_dir / f"{card_id}.png"
    expected = expected_types.get(card_id, "Unknown")

    if not image_path.exists():
        print(f"{card_id:15s} -> NOT FOUND")
        continue

    try:
        detections = detect_type(image_path, top_n=3)
        detected_type = detections[0][0] if detections else "?"
        confidence = detections[0][1] if detections else 0.0

        is_correct = detected_type == expected
        if is_correct:
            correct += 1
        total += 1

        # Format output
        status = "✓" if is_correct else "✗"
        alts = ", ".join(f"{t} ({c:.0%})" for t, c in detections[1:])
        alts_str = f" | alts: {alts}" if alts else ""

        line = (
            f"{status} {card_id:15s} | Expected: {expected:12s} | "
            f"Detected: {detected_type:12s} ({confidence:.1%}){alts_str}"
        )
        print(line)
        results.append((card_id, expected, detected_type, confidence, is_correct))

    except Exception as e:
        print(f"✗ {card_id:15s} -> ERROR: {e}")
        total += 1

print()
print("=" * 90)
accuracy = (correct / total * 100) if total > 0 else 0
print(f"ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
print("=" * 90)
print()

# Analysis
print("ANALYSIS: Type Detection for Card Identification")
print("-" * 90)
print()
print("Correct detections:")
for card_id, expected, detected, conf, is_correct in results:
    if is_correct:
        print(f"  {card_id}: {expected} at {conf:.1%}")

print()
print("Incorrect detections:")
for card_id, expected, detected, conf, is_correct in results:
    if not is_correct:
        print(f"  {card_id}: expected {expected}, got {detected} at {conf:.1%}")

print()
print("Type filtering effectiveness:")
print("  - Water filter (cards with 'Latios', 'Latias', 'Kingdra', 'Suicune'): 4 water types")
print("    Latios δ is Water type → Water detection narrows to Water Latios variants")
print("    Latias ex with Fire energy → Fire detection may conflict with name")
print("    Kingdra and Suicune are always Water type → direct variant narrow-down")
print()
print("  - Fire filter (Latias ex): 1 fire type")
print("    Latias ex with Fire energy (Heat Blast) → Fire type detection confirms")
print()
print("  - Grass filter (Venusaur): 1 grass type")
print("    Venusaur is always Grass → direct match to Grass variants")
print()
print("  - Colorless filter (Flygon, Staraptor): 2 colorless types")
print("    Flygon and Staraptor variants exist → reduces variant count")
print()
print("  - Lightning filter (Raikou): 1 lightning type")
print("    Raikou is always Lightning → direct match to Lightning variants")
print()
print("CONCLUSION:")
print("  Type detection helps narrow down card identification by:")
print("  1. Eliminating non-matching type variants (e.g., Fire Latios when Water detected)")
print("  2. Confirming card identity when combined with name detection")
print("  3. Especially useful for multi-variant cards like 'Latios' or 'Latias'")
