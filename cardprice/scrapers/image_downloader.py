"""Download Pokemon card images from URLs stored in dim_cards.

Usage (programmatic):
    from cardprice.db.session import SessionLocal
    from cardprice.scrapers.image_downloader import download_card_images, build_image_index

    with SessionLocal() as session:
        result = download_card_images(session, size="small")
        print(result)  # {'downloaded': 100, 'skipped': 19978, 'failed': 0}

    index = build_image_index()
    print(len(index))  # 20078

CLI:
    python -m cardprice.cli download-images --size small --output data/card_images
"""

import logging
import os
import time
from pathlib import Path

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

    col = "image_small" if size == "small" else "image_large"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    http = _make_http_session()

    # Rate limiting: 5 requests per second -> 0.2s between requests
    min_interval = 1.0 / 5

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    # Fetch all cards that have an image URL
    query = text(f"""
        SELECT card_id, {col}
        FROM dim_cards
        WHERE {col} IS NOT NULL
        ORDER BY card_id
    """)

    rows = session.execute(query).fetchall()
    total = len(rows)
    logger.info("Found %d cards with %s URLs", total, size)

    last_request_time = 0.0

    for i, (card_id, url) in enumerate(rows):
        # Derive set_id from card_id: "sv8-162/normal" -> "sv8"
        set_id = card_id.split("-")[0]
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

        # Rate limit
        elapsed = time.monotonic() - last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        try:
            resp = http.get(url, timeout=15)
            last_request_time = time.monotonic()

            if resp.status_code == 404:
                logger.debug("404 for %s — skipping", card_id)
                stats["failed"] += 1
            elif resp.status_code != 200:
                logger.warning(
                    "HTTP %d for %s (%s)", resp.status_code, card_id, url
                )
                stats["failed"] += 1
            else:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file.write_bytes(resp.content)
                stats["downloaded"] += 1
        except requests.RequestException as exc:
            logger.warning("Download error for %s: %s", card_id, exc)
            stats["failed"] += 1

        if (i + 1) % 100 == 0:
            _log_progress(i + 1, total, stats)

    _log_progress(total, total, stats)
    return stats


def _log_progress(current: int, total: int, stats: dict):
    logger.info(
        "Progress %d/%d — downloaded: %d, skipped: %d, failed: %d",
        current,
        total,
        stats["downloaded"],
        stats["skipped"],
        stats["failed"],
    )


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
