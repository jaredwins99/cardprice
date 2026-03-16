#!/usr/bin/env python3
"""Test script for Nyckel card condition pseudo-labeling.

Usage:
    # Single image
    python scripts/test_nyckel.py path/to/card.jpg

    # Multiple images
    python scripts/test_nyckel.py card1.jpg card2.jpg card3.jpg

    # With verbose output (all label confidences)
    python scripts/test_nyckel.py -v path/to/card.jpg

    # With custom function ID (e.g. your own trained classifier)
    python scripts/test_nyckel.py --function-id my-card-condition card.jpg

    # Dry run: just validate setup without calling API
    python scripts/test_nyckel.py --dry-run card.jpg

Required environment variables:
    NYCKEL_CLIENT_ID      - OAuth2 client ID from Nyckel dashboard
    NYCKEL_CLIENT_SECRET  - OAuth2 client secret from Nyckel dashboard

Optional:
    NYCKEL_FUNCTION_ID    - Override default pretrained classifier ID
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_setup():
    """Verify environment variables are set."""
    client_id = os.environ.get("NYCKEL_CLIENT_ID", "")
    client_secret = os.environ.get("NYCKEL_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        print("ERROR: Missing Nyckel credentials.\n")
        print("Setup steps:")
        print("  1. Sign up at https://www.nyckel.com/signup")
        print("  2. Go to Settings > API Credentials")
        print("  3. Create a new credential pair")
        print("  4. Export them:")
        print('     export NYCKEL_CLIENT_ID="your_client_id"')
        print('     export NYCKEL_CLIENT_SECRET="your_client_secret"')
        print()
        print("NOTE: Nyckel has NO pretrained Pokemon card condition classifier.")
        print("The default uses 'beanie-baby-condition' as the closest match.")
        print("For best results, train a custom classifier on Nyckel's platform")
        print("with Pokemon card images labeled NM/LP/MP/HP/DMG, then set:")
        print('  export NYCKEL_FUNCTION_ID="your-custom-function-id"')
        return False

    func_id = os.environ.get("NYCKEL_FUNCTION_ID", "beanie-baby-condition")
    print(f"Credentials: OK")
    print(f"Function ID: {func_id}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test Nyckel card condition prediction"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Path(s) to card image file(s)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all label confidences",
    )
    parser.add_argument(
        "--function-id",
        default=None,
        help="Override Nyckel function ID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without calling API",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output raw JSON",
    )
    args = parser.parse_args()

    # Validate images exist
    for img in args.images:
        if not Path(img).is_file():
            print(f"ERROR: Image not found: {img}")
            sys.exit(1)

    # Check setup
    if not check_setup():
        sys.exit(1)

    if args.dry_run:
        print("\nDry run: setup OK, skipping API calls.")
        sys.exit(0)

    # Import after setup check so missing deps show a clear error
    from cardprice.ml.nyckel_labeler import (
        predict_condition,
        NyckelAuthError,
        NyckelError,
    )

    print()
    results = []
    for img_path in args.images:
        print(f"--- {img_path} ---")
        try:
            result = predict_condition(
                img_path,
                function_id=args.function_id,
                return_all_labels=args.verbose,
            )
            results.append(result)

            if args.json_output:
                # Strip nyckel_raw for cleaner output unless verbose
                output = dict(result)
                if not args.verbose:
                    output.pop("nyckel_raw", None)
                print(json.dumps(output, indent=2))
            else:
                print(f"  TCG Grade:    {result['predicted_label']}")
                print(f"  Confidence:   {result['confidence']:.2%}")
                print(f"  Nyckel Label: {result['nyckel_label']}")
                if result.get("low_confidence"):
                    print("  WARNING: Low confidence prediction")

                if args.verbose and "all_labels" in result:
                    print("  All labels:")
                    for lc in result["all_labels"]:
                        bar = "#" * int(lc["confidence"] * 40)
                        print(
                            f"    {lc['nyckel_label']:25s} "
                            f"-> {lc['tcg_label']:3s}  "
                            f"{lc['confidence']:6.2%}  {bar}"
                        )

        except NyckelAuthError as e:
            print(f"  AUTH ERROR: {e}")
            sys.exit(1)
        except NyckelError as e:
            print(f"  API ERROR: {e}")
            results.append({"error": str(e), "image_path": img_path})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"error": str(e), "image_path": img_path})

        print()

    # Summary for batch
    if len(args.images) > 1:
        successful = [r for r in results if "error" not in r]
        print(f"=== Summary: {len(successful)}/{len(args.images)} successful ===")
        if successful:
            from collections import Counter
            grades = Counter(r["predicted_label"] for r in successful)
            for grade in ["NM", "LP", "MP", "HP", "DMG"]:
                count = grades.get(grade, 0)
                if count:
                    print(f"  {grade}: {count}")


if __name__ == "__main__":
    main()
