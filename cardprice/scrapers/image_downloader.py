"""Download Pokemon card images from URLs stored in dim_cards.

Usage (programmatic):
    from cardprice.db.session import SessionLocal
    from cardprice.scrapers.image_downloader import download_card_images, build_image_index

    with SessionLocal() as session:
        result = download_card_images(session, size="small")
        print(result)  # {'downloaded': 100, 'skipped': 19978, 'failed': 0}

    # Async variant — ~4x faster (20 concurrent downloads)
    import asyncio
    with SessionLocal() as session:
        result = asyncio.run(download_card_images_async(session, size="small"))

    index = build_image_index()
    print(len(index))  # 20078

CLI:
    python -m cardprice.cli download-images --size small --output data/card_images
"""

import asyncio
import logging
import time
from pathlib import Path

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP session with retry
# ---------------------------------------------------------------------------

def _make_http_session(max_retries: int = 3) -> requests.Session:
    """Create a requests session with exponential-backoff retry."""
    s = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


# ---------------------------------------------------------------------------
# Core downloader
# ---------------------------------------------------------------------------

def download_card_images(
    session: Session,
    output_dir: str = "data/card_images",
    size: str = "small",
    batch_size: int = 50,
    max_concurrent: int = 5,  # unused for now; kept for future async upgrade
) -> dict:
    """Download card images from pokemontcg.io URLs stored in dim_cards.

    Parameters
    ----------
    session : SQLAlchemy session
    output_dir : directory to write images into (organized by set_id)
    size : "small" or "large"
    batch_size : number of rows to fetch per DB query batch
    max_concurrent : reserved for future concurrent downloads

    Returns
    -------
    dict with keys: downloaded, skipped, failed
    """
    if size not in ("small", "large"):
        raise ValueError(f"size must be 'small' or 'large', got '{size}'")

    col = {"small": "image_small", "large": "image_large"}[size]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    http = _make_http_session()

    # Rate limiting: 5 requests per second -> 0.2s between requests
    min_interval = 1.0 / 5

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    # Fetch all cards that have an image URL
    # col is from a fixed allowlist so safe to interpolate; we also select
    # set_id directly rather than re-deriving it from card_id.
    query = text(f"""
        SELECT card_id, set_id, {col}
        FROM dim_cards
        WHERE {col} IS NOT NULL
        ORDER BY card_id
    """)

    rows = session.execute(query).fetchall()
    total = len(rows)
    logger.info("Found %d cards with %s URLs", total, size)
    print(f"Found {total:,} cards with {size} image URLs", flush=True)

    last_request_time = 0.0
    created_dirs: set[Path] = set()

    for i, (card_id, set_id, url) in enumerate(rows):
        # Sanitize card_id for filesystem: replace "/" with "_"
        safe_card_id = card_id.replace("/", "_")
        dest_dir = output_path / set_id
        dest_file = dest_dir / f"{safe_card_id}.png"

        # Skip if already exists
        if dest_file.exists():
            stats["skipped"] += 1
            if (i + 1) % 100 == 0:
                _log_progress(i + 1, total, stats)
            continue

        # Rate limit (measured from start-to-start of requests)
        elapsed = time.monotonic() - last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        last_request_time = time.monotonic()

        try:
            resp = http.get(url, timeout=15)

            if resp.status_code == 404:
                logger.debug("404 for %s — skipping", card_id)
                stats["failed"] += 1
            elif resp.status_code != 200:
                logger.warning(
                    "HTTP %d for %s (%s)", resp.status_code, card_id, url
                )
                stats["failed"] += 1
            else:
                if dest_dir not in created_dirs:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    created_dirs.add(dest_dir)
                dest_file.write_bytes(resp.content)
                stats["downloaded"] += 1
        except requests.RequestException as exc:
            logger.warning("Download error for %s: %s", card_id, exc)
            stats["failed"] += 1

        if (i + 1) % 100 == 0:
            _log_progress(i + 1, total, stats)

    _log_progress(total, total, stats)
    return stats


# ---------------------------------------------------------------------------
# Async concurrent downloader
# ---------------------------------------------------------------------------

async def download_card_images_async(
    session: Session,
    output_dir: str = "data/card_images",
    size: str = "small",
    max_concurrent: int = 20,
    per_second: int = 20,
) -> dict:
    """Download card images concurrently using aiohttp.

    Uses a semaphore to cap concurrency and a token-bucket rate limiter
    to stay under *per_second* requests/sec against the CDN.

    The images.pokemontcg.io CDN is separate from the API rate limit
    (1000 req/day). CDN endpoints typically tolerate 20-50 req/sec
    without issue; we default to 20 for safety.

    Parameters
    ----------
    session : SQLAlchemy session (used to read dim_cards)
    output_dir : directory to write images into (organized by set_id)
    size : "small" or "large"
    max_concurrent : max simultaneous downloads (default 20)
    per_second : max new requests per second (default 20)

    Returns
    -------
    dict with keys: downloaded, skipped, failed
    """
    if size not in ("small", "large"):
        raise ValueError(f"size must be 'small' or 'large', got '{size}'")

    col = {"small": "image_small", "large": "image_large"}[size]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    # Fetch all cards that have an image URL (set_id from DB, not derived)
    query = text(f"""
        SELECT card_id, set_id, {col}
        FROM dim_cards
        WHERE {col} IS NOT NULL
        ORDER BY card_id
    """)
    rows = session.execute(query).fetchall()
    total = len(rows)
    logger.info("Found %d cards with %s URLs", total, size)
    print(f"Found {total:,} cards with {size} image URLs", flush=True)

    # Build work list, skipping already-downloaded files
    work = []
    for card_id, set_id, url in rows:
        safe_card_id = card_id.replace("/", "_")
        dest_dir = output_path / set_id
        dest_file = dest_dir / f"{safe_card_id}.png"

        if dest_file.exists():
            stats["skipped"] += 1
            continue
        work.append((card_id, url, dest_dir, dest_file))

    already_skipped = stats["skipped"]
    print(
        f"Skipped {already_skipped:,} already downloaded; "
        f"{len(work):,} to download",
        flush=True,
    )

    if not work:
        _log_progress(total, total, stats)
        return stats

    # Token-bucket rate limiter
    sem = asyncio.Semaphore(max_concurrent)
    interval = 1.0 / per_second
    _last_request = [0.0]  # mutable container for closure
    _lock = asyncio.Lock()

    async def _rate_limit():
        async with _lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = interval - (now - _last_request[0])
            if wait > 0:
                await asyncio.sleep(wait)
            _last_request[0] = asyncio.get_running_loop().time()

    progress_count = [0]  # mutable counter for progress logging

    async def _download_one(
        aio_session: aiohttp.ClientSession,
        card_id: str,
        url: str,
        dest_dir: Path,
        dest_file: Path,
    ):
        async with sem:
            await _rate_limit()
            try:
                async with aio_session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 404:
                        logger.debug("404 for %s — skipping", card_id)
                        stats["failed"] += 1
                    elif resp.status != 200:
                        logger.warning(
                            "HTTP %d for %s (%s)", resp.status, card_id, url
                        )
                        stats["failed"] += 1
                    else:
                        data = await resp.read()
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_file.write_bytes(data)
                        stats["downloaded"] += 1
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("Download error for %s: %s", card_id, exc)
                stats["failed"] += 1

            progress_count[0] += 1
            done = already_skipped + progress_count[0]
            if progress_count[0] % 100 == 0 or progress_count[0] == len(work):
                _log_progress(done, total, stats)

    # Retry-capable connector
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as aio_session:
        tasks = [
            _download_one(aio_session, card_id, url, dest_dir, dest_file)
            for card_id, url, dest_dir, dest_file in work
        ]
        await asyncio.gather(*tasks)

    _log_progress(total, total, stats)
    return stats


def _log_progress(current: int, total: int, stats: dict):
    msg = (
        f"Progress {current}/{total} — "
        f"downloaded: {stats['downloaded']}, "
        f"skipped: {stats['skipped']}, "
        f"failed: {stats['failed']}"
    )
    logger.info(msg)
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Image index builder
# ---------------------------------------------------------------------------

def build_image_index(image_dir: str = "data/card_images") -> dict:
    """Scan the image directory and return a {card_id: image_path} mapping.

    Reverses the filesystem-safe naming back to the canonical card_id format:
        "sv8/sv8-162_normal.png"  ->  card_id="sv8-162/normal"

    Returns
    -------
    dict mapping card_id (str) to absolute image path (str)
    """
    index = {}
    root = Path(image_dir)

    if not root.exists():
        logger.warning("Image directory %s does not exist", image_dir)
        return index

    for set_dir in sorted(root.iterdir()):
        if not set_dir.is_dir():
            continue
        for img_file in sorted(set_dir.iterdir()):
            if not img_file.suffix.lower() == ".png":
                continue
            # Reverse the safe name: "sv8-162_normal.png" -> "sv8-162/normal"
            stem = img_file.stem  # e.g. "sv8-162_normal"
            # Replace the LAST underscore with "/" to restore variant separator
            last_us = stem.rfind("_")
            if last_us != -1:
                card_id = stem[:last_us] + "/" + stem[last_us + 1:]
            else:
                card_id = stem
            index[card_id] = str(img_file.resolve())

    logger.info("Built image index with %d entries", len(index))
    return index
