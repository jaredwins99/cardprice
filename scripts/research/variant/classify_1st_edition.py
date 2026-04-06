#!/usr/bin/env python3
"""
Classify archive.org card scans as 1st edition or unlimited.

The 1st edition stamp appears on the left side of the card, below the artwork frame.
For Pokemon cards: approximately at x=8-15%, y=55-62% of card height.
For Energy cards: approximately at x=85-92%, y=5-12% (top right).
For Trainer cards: no stamp area below artwork on left side.

Strategy: Look for the dark "Edition 1" stamp in the expected region.
The stamp is a small circle with "1" and text. In unlimited cards,
this region is blank/plain colored.

We use pixel darkness analysis: the 1st edition stamp creates a dark
cluster in an otherwise light region.
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

BASE_DIR = Path("/home/godli/cardprice/data/condition_training/ground_truth_variants")
ARCHIVE_DIR = BASE_DIR / "archive_raw"
FIRST_ED_DIR = BASE_DIR / "1st_edition"
UNLIMITED_DIR = BASE_DIR / "unlimited"


def detect_1st_edition_stamp(img_path: str) -> dict:
    """
    Detect if a card has a 1st edition stamp.

    Returns dict with:
        - has_stamp: bool
        - confidence: float (0-1)
        - stamp_region_darkness: float
        - card_number: str (from bottom of card if readable)
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    arr = np.array(img)

    # The 1st edition stamp location for Pokemon/Trainer cards:
    # Left side, below artwork frame, roughly:
    # x: 5-18% of width, y: 53-63% of height
    stamp_x1 = int(w * 0.05)
    stamp_x2 = int(w * 0.18)
    stamp_y1 = int(h * 0.53)
    stamp_y2 = int(h * 0.63)

    stamp_region = arr[stamp_y1:stamp_y2, stamp_x1:stamp_x2]

    # Also check energy card stamp location (top right)
    energy_x1 = int(w * 0.82)
    energy_x2 = int(w * 0.95)
    energy_y1 = int(h * 0.04)
    energy_y2 = int(h * 0.12)
    energy_region = arr[energy_y1:energy_y2, energy_x1:energy_x2]

    # Convert to grayscale for analysis
    stamp_gray = np.mean(stamp_region, axis=2)
    energy_gray = np.mean(energy_region, axis=2)

    # 1st edition stamp is dark (~black) pixels in a mostly light region
    # Count very dark pixels (< 80 brightness)
    dark_threshold = 80

    stamp_dark_ratio = np.mean(stamp_gray < dark_threshold)
    energy_dark_ratio = np.mean(energy_gray < dark_threshold)

    # Also check for the specific stamp shape: small dark cluster
    # surrounded by lighter pixels
    stamp_mean = np.mean(stamp_gray)
    stamp_std = np.std(stamp_gray)
    energy_mean = np.mean(energy_gray)
    energy_std = np.std(energy_gray)

    # A stamp creates high contrast (high std) with significant dark pixels
    # Unlimited cards have uniform coloring in this region (low std)

    pokemon_stamp_score = stamp_dark_ratio * 3 + (stamp_std / 80)
    energy_stamp_score = energy_dark_ratio * 3 + (energy_std / 80)

    best_score = max(pokemon_stamp_score, energy_stamp_score)
    location = "pokemon" if pokemon_stamp_score > energy_stamp_score else "energy"

    return {
        "has_stamp": best_score > 0.15,
        "confidence": min(best_score / 0.3, 1.0),
        "score": best_score,
        "pokemon_score": pokemon_stamp_score,
        "energy_score": energy_stamp_score,
        "location": location,
        "stamp_dark_ratio": stamp_dark_ratio,
        "stamp_std": stamp_std,
    }


def classify_set(set_id: str, verify_samples: bool = False):
    """Classify all images in a set as 1st edition or unlimited."""
    set_dir = ARCHIVE_DIR / set_id
    if not set_dir.exists():
        print(f"Set {set_id} not found at {set_dir}")
        return

    files = sorted([f for f in os.listdir(set_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"\n=== {set_id}: {len(files)} images ===")

    results = {"1st_edition": [], "unlimited": [], "uncertain": []}

    for fname in files:
        fpath = set_dir / fname
        try:
            result = detect_1st_edition_stamp(str(fpath))
            if result["has_stamp"] and result["confidence"] > 0.5:
                edition = "1st_edition"
            elif not result["has_stamp"] and result["confidence"] < 0.3:
                edition = "unlimited"
            else:
                edition = "uncertain"

            results[edition].append((fname, result))
        except Exception as e:
            print(f"  Error processing {fname}: {e}")

    print(f"  1st Edition: {len(results['1st_edition'])}")
    print(f"  Unlimited:   {len(results['unlimited'])}")
    print(f"  Uncertain:   {len(results['uncertain'])}")

    # Show some samples
    for edition, items in results.items():
        if items:
            print(f"\n  Sample {edition}:")
            for fname, r in items[:3]:
                print(f"    {fname}: score={r['score']:.3f} conf={r['confidence']:.3f} "
                      f"pokemon={r['pokemon_score']:.3f} energy={r['energy_score']:.3f}")

    return results


def main():
    sets = ["base1", "base2", "base3", "base5", "gym1", "gym2",
            "neo1", "neo2", "neo3", "neo4"]

    all_results = {}
    for set_id in sets:
        results = classify_set(set_id)
        if results:
            all_results[set_id] = results

    # Summary
    print("\n\n=== SUMMARY ===")
    total_1st = total_unl = total_unc = 0
    for set_id, results in all_results.items():
        n1 = len(results["1st_edition"])
        nu = len(results["unlimited"])
        nc = len(results["uncertain"])
        total_1st += n1
        total_unl += nu
        total_unc += nc
        print(f"  {set_id:8s}: 1st={n1:3d}  unl={nu:3d}  unc={nc:3d}")
    print(f"  {'TOTAL':8s}: 1st={total_1st:3d}  unl={total_unl:3d}  unc={total_unc:3d}")


if __name__ == "__main__":
    main()
