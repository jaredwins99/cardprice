#!/usr/bin/env python3
"""Download high-resolution card images from pokemontcg.io CDN.

Downloads `image_large` URLs for all cards in dim_cards to
data/card_images_hires/{set_id}/{card_id}_normal.png

Polite throttling: 2 concurrent downloads, 0.5s minimum between requests.
Resumes from where it left off (skips existing files).

Usage:
    python scripts/download_hires_images.py
    python scripts/download_hires_images.py --max-concurrent 4 --delay 0.25
    python scripts/download_hires_images.py --dry-run
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import aiohttp
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_URL = "postgresql+psycopg2://godli@/cardprice"
DEFAULT_OUTPUT = "data/card_images_hires"
DEFAULT_CONCURRENT = 2
DEFAULT_DELAY = 0.5


def fetch_cards_with_hires_urls(output_dir: Path):
    """Query dim_cards for all cards with image_large URLs.

    Returns list of (card_id, set_id, url, dest_file) tuples,
    filtering out cards whose destination file already exists.
    """
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT card_id, set_id, image_large
            FROM dim_cards
            WHERE image_large IS NOT NULL
            ORDER BY card_id
        """)).fetchall()

    total = len(rows)
    work = []
    skipped = 0

    for card_id, set_id, url in rows:
        safe_card_id = card_id.replace("/", "_")
        dest_dir = output_dir / set_id
        dest_file = dest_dir / f"{safe_card_id}.png"

        if dest_file.exists():
            skipped += 1
            continue
        work.append((card_id, set_id, url, dest_dir, dest_file))

    return total, skipped, work


async def download_all(work, max_concurrent: int, delay: float, dry_run: bool):
    """Download all images with rate limiting and concurrency control."""
    stats = {"downloaded": 0, "failed": 0}

    if dry_run:
        print(f"DRY RUN: would download {len(work)} images")
        return stats

    sem = asyncio.Semaphore(max_concurrent)
    lock = asyncio.Lock()
    last_request = [0.0]
    progress = [0]

    async def rate_limit():
        async with lock:
            now = asyncio.get_event_loop().time()
            wait = delay - (now - last_request[0])
            if wait > 0:
                await asyncio.sleep(wait)
            last_request[0] = asyncio.get_event_loop().time()

    async def download_one(session, card_id, url, dest_dir, dest_file):
        async with sem:
            await rate_limit()
            try:
                timeout = aiohttp.ClientTimeout(total=30)
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_file.write_bytes(data)
                        stats["downloaded"] += 1
                    else:
                        logger.warning("HTTP %d for %s", resp.status, card_id)
                        stats["failed"] += 1
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("Error downloading %s: %s", card_id, exc)
                stats["failed"] += 1

            progress[0] += 1
            if progress[0] % 100 == 0 or progress[0] == len(work):
                print(
                    f"  Progress: {progress[0]}/{len(work)} "
                    f"(downloaded: {stats['downloaded']}, "
                    f"failed: {stats['failed']})",
                    flush=True,
                )

    connector = aiohttp.TCPConnector(
        limit=max_concurrent, limit_per_host=max_concurrent
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            download_one(session, card_id, url, dest_dir, dest_file)
            for card_id, _, url, dest_dir, dest_file in work
        ]
        await asyncio.gather(*tasks)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download high-resolution Pokemon card images"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=DEFAULT_CONCURRENT,
        help=f"Max concurrent downloads (default: {DEFAULT_CONCURRENT})"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Min seconds between requests (default: {DEFAULT_DELAY})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without downloading"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir.resolve()}")
    print(f"Concurrency: {args.max_concurrent}, delay: {args.delay}s")
    print()

    # Query DB for work
    print("Querying database for image URLs...")
    total, skipped, work = fetch_cards_with_hires_urls(output_dir)
    print(f"Total cards with hires URLs: {total:,}")
    print(f"Already downloaded (skipped): {skipped:,}")
    print(f"To download: {len(work):,}")

    if not work:
        print("Nothing to download!")
        return

    # Estimate time
    est_seconds = len(work) * args.delay / args.max_concurrent
    est_minutes = est_seconds / 60
    print(f"Estimated time: ~{est_minutes:.0f} minutes")
    print()

    # Download
    t0 = time.time()
    stats = asyncio.run(
        download_all(work, args.max_concurrent, args.delay, args.dry_run)
    )
    elapsed = time.time() - t0

    print()
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Downloaded: {stats['downloaded']:,}")
    print(f"  Failed: {stats['failed']:,}")
    print(f"  Skipped (existing): {skipped:,}")


if __name__ == "__main__":
    main()
