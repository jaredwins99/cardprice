#!/usr/bin/env python3
"""Quick test script for the unified condition estimator.

Usage:
    python scripts/test_condition_estimate.py <image_path> [card_id]

Examples:
    python scripts/test_condition_estimate.py data/inbox/page_cards/card_01.png
    python scripts/test_condition_estimate.py photo.jpg base1-4
    python scripts/test_condition_estimate.py photo.jpg --ref data/card_images/base1/base1-4_normal.png
"""

import argparse
import json
import logging
import sys
import time

# Add project root to path
sys.path.insert(0, ".")

from cardprice.ml.condition_estimator import estimate_condition


def main():
    parser = argparse.ArgumentParser(description="Test card condition estimation")
    parser.add_argument("image_path", help="Path to card photo to assess")
    parser.add_argument("card_id", nargs="?", default=None,
                        help="Card ID for reference lookup (e.g. base1-4)")
    parser.add_argument("--ref", default=None,
                        help="Explicit path to reference image")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed debug output")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    print(f"Image: {args.image_path}")
    if args.card_id:
        print(f"Card ID: {args.card_id}")
    if args.ref:
        print(f"Reference: {args.ref}")
    print()

    t0 = time.time()
    result = estimate_condition(
        image_path=args.image_path,
        card_id=args.card_id,
        ref_image_path=args.ref,
    )
    elapsed = time.time() - t0

    # Print summary
    print(f"{'=' * 50}")
    print(f"  OVERALL GRADE:  {result['overall_grade']}")
    print(f"  OVERALL SCORE:  {result['overall_score']}/10")
    print(f"  CONFIDENCE:     {result['confidence']}")
    print(f"{'=' * 50}")

    print(f"\nSub-grades:")
    for key, val in result["sub_grades"].items():
        label = f"  {key:>12s}:"
        if val is not None:
            bar = "#" * int(val) + "." * (10 - int(val))
            print(f"{label}  {val:>4.1f}/10  [{bar}]")
        else:
            print(f"{label}  --  (not available)")

    if result["defects"]:
        print(f"\nDefect patches: {len(result['defects'])} flagged")
        for d in result["defects"][:5]:
            print(f"  patch ({d['row']},{d['col']}): similarity={d['similarity']}")
        if len(result["defects"]) > 5:
            print(f"  ... and {len(result['defects']) - 5} more")

    if args.verbose and result["details"]:
        print(f"\nDetails:")
        # Filter out non-serializable items
        print(json.dumps(result["details"], indent=2, default=str))

    print(f"\nCompleted in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
