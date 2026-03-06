#!/usr/bin/env python3
"""Scrape eBay for graded Pokemon card sold listings.

Convenience wrapper around scrapers.ebay_graded.GradedCardScraper that
collects training data for the corner classifier.

Examples:
    # Scrape 100 PSA 9 listings
    python scripts/scrape_graded_cards.py --grade "PSA 9" --count 100

    # Scrape 50 BGS 9.5 listings (will extract per-corner sub-grades)
    python scripts/scrape_graded_cards.py --grade "BGS 9.5" --count 50

    # Scrape PSA 10 with 3 pages max
    python scripts/scrape_graded_cards.py --grade "PSA 10" --count 100 --pages 3

    # Scrape all BGS grades (best for corner training data)
    python scripts/scrape_graded_cards.py --grade "BGS all" --count 100

    # Check progress
    python scripts/scrape_graded_cards.py --stats

    # Skip corner extraction, just download images
    python scripts/scrape_graded_cards.py --grade "PSA 10" --count 50 --no-corners

Output directory structure:
    data/condition_training/
        corners/Gem/*.png         -- corner ROIs for training
        corners/Mint/*.png
        corners/Light/*.png
        corners/Moderate/*.png
        corners/Heavy/*.png
        images/psa_10/            -- raw listing photos
        images/bgs_9_5/
        listings/psa_10.jsonl     -- listing metadata
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scrapers.ebay_graded import GradedCardScraper


def parse_grade_spec(grade_spec: str) -> tuple[str, str]:
    """Parse a grade specification like 'PSA 9' or 'BGS 9.5'.

    Returns (authority, grade) tuple.

    Accepts formats:
        "PSA 9"     -> ("PSA", "9")
        "BGS 9.5"   -> ("BGS", "9.5")
        "CGC 10"    -> ("CGC", "10")
        "PSA all"   -> ("PSA", "all")
        "BGS all"   -> ("BGS", "all")
        "9"         -> ("PSA", "9")   (default authority)
        "10"        -> ("PSA", "10")

    Raises ValueError on unrecognized format.
    """
    parts = grade_spec.strip().split(None, 1)

    if len(parts) == 2:
        authority = parts[0].upper()
        grade = parts[1].strip()
        if authority not in ("PSA", "BGS", "CGC", "SGC"):
            raise ValueError(
                f"Unknown grading authority: {authority}. "
                f"Expected PSA, BGS, or CGC."
            )
        return authority, grade

    if len(parts) == 1:
        val = parts[0]
        if val.upper() in ("PSA", "BGS", "CGC"):
            raise ValueError(
                f"Please specify a grade: '{val} 10' or '{val} all'"
            )
        # Bare number -> default to PSA
        return "PSA", val

    raise ValueError(f"Cannot parse grade spec: {grade_spec!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape eBay graded Pokemon cards for corner classifier training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --grade "PSA 9" --count 100
  %(prog)s --grade "BGS 9.5" --count 50
  %(prog)s --grade "BGS all" --count 100
  %(prog)s --stats
        """,
    )
    parser.add_argument(
        "--grade", "-g",
        type=str, default=None,
        help=(
            'Grade to scrape, e.g. "PSA 9", "BGS 9.5", "CGC 10". '
            'Use "PSA all" or "BGS all" for all grades.'
        ),
    )
    parser.add_argument(
        "--count", "-c",
        type=int, default=100,
        help="Number of listings to scrape per grade tier (default: 100).",
    )
    parser.add_argument(
        "--pages", "-p",
        type=int, default=5,
        help="Max search result pages per grade (default: 5, 60 items each).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default="data/condition_training",
        help="Output directory (default: data/condition_training).",
    )
    parser.add_argument(
        "--no-corners",
        action="store_true",
        help="Skip corner ROI extraction (just download images).",
    )
    parser.add_argument(
        "--no-listing-images",
        action="store_true",
        help="Skip individual listing page visits (thumbnails only).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print current scraping statistics and exit.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Stats mode: show stats for all authorities and exit
    if args.stats:
        for auth in ("PSA", "BGS", "CGC"):
            scraper = GradedCardScraper(
                authority=auth,
                output_dir=args.output,
                extract_corners=False,
                fetch_listing_images=False,
            )
            stats = scraper.get_stats()
            if stats["total_listings"] > 0 or stats["corner_class_counts"]:
                print(f"\n=== {auth} ===")
                print(json.dumps(stats, indent=2))

        # Also show corner class totals
        corners_dir = Path(args.output) / "corners"
        if corners_dir.exists():
            print("\n=== Corner ROI totals ===")
            total = 0
            for cls_dir in sorted(corners_dir.iterdir()):
                if cls_dir.is_dir():
                    count = sum(
                        1 for f in cls_dir.iterdir()
                        if f.suffix in (".png", ".jpg")
                    )
                    print(f"  {cls_dir.name}: {count}")
                    total += count
            print(f"  Total: {total}")
        return

    # Require --grade for scraping
    if not args.grade:
        parser.print_help()
        print('\nSpecify --grade (e.g. --grade "PSA 9") or --stats.')
        sys.exit(1)

    # Parse grade specification
    try:
        authority, grade = parse_grade_spec(args.grade)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Scraping {authority} grade {grade} sold listings from eBay")
    print(f"  Target: {args.count} listings per grade")
    print(f"  Max pages: {args.pages}")
    print(f"  Output: {args.output}")
    print(f"  Corner extraction: {'off' if args.no_corners else 'on'}")
    print(f"  Listing images: {'off' if args.no_listing_images else 'on'}")
    print()

    scraper = GradedCardScraper(
        authority=authority,
        output_dir=args.output,
        extract_corners=not args.no_corners,
        fetch_listing_images=not args.no_listing_images,
    )

    if grade.lower() == "all":
        summary = scraper.scrape_all_grades(max_pages_per_grade=args.pages)
        print(f"\nScrape summary for {authority}:")
        for g, count in summary.items():
            print(f"  Grade {g}: {count} listings")
        total = sum(summary.values())
        print(
            f"\n  Total: {total} listings, "
            f"{scraper.progress.total_images} images, "
            f"{scraper.progress.total_corners} corner ROIs"
        )
    else:
        count = scraper.scrape_grade(
            grade, max_pages=args.pages, max_listings=args.count
        )
        print(
            f"\nScraped {count} new {authority} {grade} listings "
            f"({scraper.progress.total_images} images, "
            f"{scraper.progress.total_corners} corner ROIs)"
        )

    # Print final stats
    stats = scraper.get_stats()
    if stats["corner_class_counts"]:
        print("\nCorner ROIs by class:")
        for cls, cnt in sorted(stats["corner_class_counts"].items()):
            print(f"  {cls}: {cnt}")

    print(f"\nErrors: {stats['errors']}")


if __name__ == "__main__":
    main()
