#!/usr/bin/env python3
"""CLI script to scrape eBay sold listings and ingest into fact_sales.

Usage:
    # Scrape by card name
    python scripts/scrape_ebay_sales.py --card-name "Charizard Base Set"

    # Scrape by set name (scrapes top cards in that set)
    python scripts/scrape_ebay_sales.py --set-name "Base Set"

    # Control pages and rate limiting
    python scripts/scrape_ebay_sales.py --card-name "Pikachu" --max-pages 5

    # Dry run (scrape + parse but don't insert)
    python scripts/scrape_ebay_sales.py --card-name "Charizard Base Set" --dry-run

    # Scrape all WotC era sets
    python scripts/scrape_ebay_sales.py --wotc

    # Scrape current era sets
    python scripts/scrape_ebay_sales.py --current

    # Resume interrupted batch (skips already-scraped queries via log)
    python scripts/scrape_ebay_sales.py --set-name "Base Set" --resume
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cardprice.scrapers.ebay import scrape_sold_listings
from cardprice.scrapers.ebay_title_parser import parse_title
from cardprice.scrapers.ebay_matcher import match_listing, ingest_ebay_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scrape_ebay_sales")

# ---------------------------------------------------------------------------
# Progress / resume tracking
# ---------------------------------------------------------------------------

PROGRESS_DIR = PROJECT_ROOT / "data" / "ebay_progress"


def _progress_file() -> Path:
    """Return path to progress tracking file."""
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    return PROGRESS_DIR / "completed_queries.json"


def _load_completed() -> set[str]:
    """Load set of already-completed query strings."""
    pf = _progress_file()
    if pf.exists():
        try:
            return set(json.loads(pf.read_text()))
        except (json.JSONDecodeError, TypeError):
            return set()
    return set()


def _mark_completed(query: str):
    """Mark a query as completed in the progress file."""
    completed = _load_completed()
    completed.add(query)
    _progress_file().write_text(json.dumps(sorted(completed), indent=2))


# ---------------------------------------------------------------------------
# Set-based query generation
# ---------------------------------------------------------------------------

# WotC era sets (Base Set through Neo Destiny + Legendary Collection + e-reader)
WOTC_SETS = [
    "Base Set", "Jungle", "Fossil", "Base Set 2", "Team Rocket",
    "Gym Heroes", "Gym Challenge",
    "Neo Genesis", "Neo Discovery", "Neo Revelation", "Neo Destiny",
    "Legendary Collection",
    "Expedition", "Aquapolis", "Skyridge",
]

# Current era sets (Scarlet & Violet block)
CURRENT_SETS = [
    "Scarlet & Violet", "Paldea Evolved", "Obsidian Flames",
    "Paradox Rift", "Paldean Fates", "Temporal Forces",
    "Twilight Masquerade", "Shrouding Storm", "Surging Sparks",
    "Prismatic Evolutions", "Journey Together", "151",
]

# High-value Pokemon names to search within a set
TOP_POKEMON = [
    "Charizard", "Blastoise", "Venusaur", "Pikachu", "Mewtwo", "Mew",
    "Lugia", "Ho-Oh", "Gengar", "Dragonite", "Alakazam", "Gyarados",
    "Espeon", "Umbreon", "Tyranitar", "Celebi",
]

# For current sets, different high-value targets
CURRENT_TOP_POKEMON = [
    "Charizard", "Pikachu", "Mewtwo", "Mew", "Umbreon", "Eevee",
    "Gardevoir", "Miraidon", "Koraidon", "Rayquaza", "Lugia",
    "Gengar", "Gyarados", "Arcanine",
]


def _generate_set_queries(set_name: str, pokemon_list: list[str] | None = None) -> list[str]:
    """Generate search queries for a set: set-wide + per top Pokemon."""
    queries = [f"Pokemon {set_name}"]
    poke_list = pokemon_list or TOP_POKEMON
    for poke in poke_list:
        queries.append(f"{poke} {set_name} Pokemon")
    return queries


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def scrape_and_ingest(query: str, max_pages: int = 3, dry_run: bool = False) -> dict:
    """Scrape eBay for a query and ingest results into fact_sales.

    Returns summary stats dict.
    """
    logger.info("=" * 60)
    logger.info("QUERY: %s (max_pages=%d, dry_run=%s)", query, max_pages, dry_run)
    logger.info("=" * 60)

    # Step 1: Scrape
    listings = scrape_sold_listings(query, max_pages=max_pages)
    logger.info("Scraped %d listings", len(listings))

    if not listings:
        return {"total": 0, "matched": 0, "inserted": 0, "skipped": 0, "errors": 0,
                "query": query}

    # Step 2: Parse titles and log summary
    graded_count = 0
    raw_count = 0
    for listing in listings:
        parsed = parse_title(listing.get("title", ""))
        if parsed["is_graded"]:
            graded_count += 1
        else:
            raw_count += 1

    logger.info("Listing breakdown: %d raw, %d graded", raw_count, graded_count)

    if dry_run:
        # In dry-run mode, parse and match but don't insert
        from cardprice.db.session import SessionLocal
        from cardprice.scrapers.ebay_matcher import _load_candidates

        session = SessionLocal()
        try:
            candidates = _load_candidates(session)
            matched = 0
            for listing in listings:
                parsed = parse_title(listing.get("title", ""))
                card_id, confidence = match_listing(parsed, session, candidates)
                status = "MATCH" if card_id else "NO MATCH"
                logger.info(
                    "  [%s] %.2f  %s  ->  %s",
                    status, confidence,
                    listing.get("title", "")[:70],
                    card_id or "-",
                )
                if card_id:
                    matched += 1
            logger.info("Dry run: %d/%d matched (%.0f%%)",
                        matched, len(listings),
                        100 * matched / len(listings) if listings else 0)
            return {"total": len(listings), "matched": matched, "inserted": 0,
                    "skipped": 0, "errors": 0, "query": query}
        finally:
            session.close()
    else:
        # Step 3: Ingest into fact_sales
        stats = ingest_ebay_results(listings)
        stats["query"] = query
        return stats


def run_batch(queries: list[str], max_pages: int = 3, dry_run: bool = False,
              resume: bool = False, delay_between: float = 5.0) -> list[dict]:
    """Run scrape_and_ingest for a batch of queries with rate limiting.

    Args:
        queries: List of search query strings.
        max_pages: Max eBay result pages per query.
        dry_run: If True, parse/match but don't insert.
        resume: If True, skip queries already in progress file.
        delay_between: Seconds to wait between queries.

    Returns:
        List of stats dicts, one per query.
    """
    completed = _load_completed() if resume else set()
    all_stats = []

    for i, query in enumerate(queries, 1):
        if resume and query in completed:
            logger.info("SKIP (already done): %s", query)
            continue

        logger.info("Progress: query %d/%d", i, len(queries))
        try:
            stats = scrape_and_ingest(query, max_pages=max_pages, dry_run=dry_run)
            all_stats.append(stats)

            if not dry_run:
                _mark_completed(query)

        except KeyboardInterrupt:
            logger.warning("Interrupted by user at query %d/%d", i, len(queries))
            break
        except Exception as e:
            logger.error("Failed on query %r: %s", query, e, exc_info=True)
            all_stats.append({"query": query, "error": str(e)})

        # Rate limiting between queries
        if i < len(queries):
            logger.debug("Sleeping %.1fs between queries", delay_between)
            time.sleep(delay_between)

    # Print summary
    total_listings = sum(s.get("total", 0) for s in all_stats)
    total_matched = sum(s.get("matched", 0) for s in all_stats)
    total_inserted = sum(s.get("inserted", 0) for s in all_stats)
    total_skipped = sum(s.get("skipped", 0) for s in all_stats)
    total_errors = sum(s.get("errors", 0) for s in all_stats)

    logger.info("")
    logger.info("=" * 60)
    logger.info("BATCH SUMMARY")
    logger.info("=" * 60)
    logger.info("  Queries run:  %d / %d", len(all_stats), len(queries))
    logger.info("  Listings:     %d total", total_listings)
    logger.info("  Matched:      %d (%.0f%%)", total_matched,
                100 * total_matched / total_listings if total_listings else 0)
    logger.info("  Inserted:     %d", total_inserted)
    logger.info("  Skipped(dup): %d", total_skipped)
    logger.info("  Errors:       %d", total_errors)

    return all_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape eBay sold listings and ingest into fact_sales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--card-name", type=str,
                       help="Search query for a specific card (e.g. 'Charizard Base Set')")
    group.add_argument("--set-name", type=str,
                       help="Set name to scrape top cards from (e.g. 'Base Set')")
    group.add_argument("--wotc", action="store_true",
                       help="Scrape all WotC era sets (Base through e-reader)")
    group.add_argument("--current", action="store_true",
                       help="Scrape current era sets (Scarlet & Violet block)")
    group.add_argument("--query", type=str,
                       help="Raw eBay search query (passed directly)")

    parser.add_argument("--max-pages", type=int, default=3,
                        help="Max result pages per query (default: 3, max: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and match but don't insert into DB")
    parser.add_argument("--resume", action="store_true",
                        help="Skip queries already completed (tracked in data/ebay_progress/)")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="Seconds between queries in batch mode (default: 5)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build query list
    if args.card_name:
        queries = [args.card_name]
    elif args.query:
        queries = [args.query]
    elif args.set_name:
        queries = _generate_set_queries(args.set_name)
        logger.info("Generated %d queries for set '%s'", len(queries), args.set_name)
    elif args.wotc:
        queries = []
        for s in WOTC_SETS:
            queries.extend(_generate_set_queries(s, TOP_POKEMON))
        logger.info("Generated %d queries for %d WotC sets", len(queries), len(WOTC_SETS))
    elif args.current:
        queries = []
        for s in CURRENT_SETS:
            queries.extend(_generate_set_queries(s, CURRENT_TOP_POKEMON))
        logger.info("Generated %d queries for %d current sets", len(queries), len(CURRENT_SETS))

    # Run
    if len(queries) == 1:
        stats = scrape_and_ingest(queries[0], max_pages=args.max_pages, dry_run=args.dry_run)
        logger.info("Result: %s", stats)
    else:
        run_batch(queries, max_pages=args.max_pages, dry_run=args.dry_run,
                  resume=args.resume, delay_between=args.delay)


if __name__ == "__main__":
    main()
