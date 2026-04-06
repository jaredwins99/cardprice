"""TCGCSV scraper: live prices and archive ingestion for Pokemon (categoryId=3)."""

import json
import logging
import subprocess
import time
from datetime import date, datetime
from pathlib import Path

import requests
from sqlalchemy import text
from sqlalchemy.orm import Session

from cardprice.config import TCGCSV_BASE_URL, POKEMON_CATEGORY_ID

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ARCHIVES_DIR = DATA_DIR / "archives"
EXTRACTED_DIR = DATA_DIR / "extracted"

# Default category is English Pokemon (3); pass category_id=85 for Japanese.
def _base(category_id: int = POKEMON_CATEGORY_ID) -> str:
    return f"{TCGCSV_BASE_URL}/tcgplayer/{category_id}"

BASE = _base(POKEMON_CATEGORY_ID)
ARCHIVE_BASE = f"{TCGCSV_BASE_URL}/archive/tcgplayer"

# TCGPlayer marketplace_id — must match dim_marketplaces row.
TCGPLAYER_MARKETPLACE_ID = 1

# Respect the server: pause between group fetches.
REQUEST_DELAY = 0.3


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_json(url: str) -> dict:
    """GET a URL and return parsed JSON. Raises on HTTP errors."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Live API functions
# ---------------------------------------------------------------------------

def fetch_groups(category_id: int = POKEMON_CATEGORY_ID) -> list[dict]:
    """Return all Pokemon groups from TCGCSV for the given category."""
    data = _get_json(f"{_base(category_id)}/groups")
    if isinstance(data, list):
        return data
    return data.get("results", data)


def fetch_prices(group_id: int, category_id: int = POKEMON_CATEGORY_ID) -> list[dict]:
    """Fetch prices for one group. Returns list of price dicts."""
    data = _get_json(f"{_base(category_id)}/{group_id}/prices")
    return data.get("results", [])


def fetch_products(group_id: int, category_id: int = POKEMON_CATEGORY_ID) -> list[dict]:
    """Fetch products for one group. Returns list of product dicts."""
    data = _get_json(f"{_base(category_id)}/{group_id}/products")
    return data.get("results", [])


# ---------------------------------------------------------------------------
# DB insert helpers
# ---------------------------------------------------------------------------

_INSERT_SQL = text("""
    INSERT INTO fact_market_prices
        (tcg_product_id, marketplace_id, price_date, subtype_name,
         low_price, mid_price, high_price, market_price, direct_low)
    VALUES
        (:tcg_product_id, :marketplace_id, :price_date, :subtype_name,
         :low_price, :mid_price, :high_price, :market_price, :direct_low)
    ON CONFLICT (tcg_product_id, price_date, subtype_name) DO NOTHING
""")


def _insert_price_rows(session: Session, rows: list[dict], price_date: date) -> int:
    """Batch-insert price rows. Returns count of rows inserted."""
    if not rows:
        return 0

    params = []
    for r in rows:
        params.append({
            "tcg_product_id": r["productId"],
            "marketplace_id": TCGPLAYER_MARKETPLACE_ID,
            "price_date": price_date,
            "subtype_name": r.get("subTypeName"),
            "low_price": r.get("lowPrice"),
            "mid_price": r.get("midPrice"),
            "high_price": r.get("highPrice"),
            "market_price": r.get("marketPrice"),
            "direct_low": r.get("directLowPrice"),
        })

    result = session.execute(_INSERT_SQL, params)
    return result.rowcount


# ---------------------------------------------------------------------------
# Live ingestion
# ---------------------------------------------------------------------------

def ingest_live_prices(session: Session, target_date: date | None = None) -> dict:
    """Fetch today's prices for every Pokemon group and insert into DB.

    Returns summary dict with counts.
    """
    target_date = target_date or date.today()
    groups = fetch_groups()
    logger.info("Found %d Pokemon groups", len(groups))

    total_inserted = 0
    total_rows = 0
    errors = 0

    for i, g in enumerate(groups):
        gid = g["groupId"]
        gname = g.get("name", gid)
        try:
            prices = fetch_prices(gid)
            total_rows += len(prices)
            inserted = _insert_price_rows(session, prices, target_date)
            total_inserted += inserted
            if (i + 1) % 20 == 0:
                session.commit()
                logger.info("Progress: %d/%d groups  (%d rows inserted so far)",
                            i + 1, len(groups), total_inserted)
        except Exception:
            logger.exception("Failed to fetch prices for group %s (%s)", gid, gname)
            errors += 1
        time.sleep(REQUEST_DELAY)

    session.commit()
    summary = {
        "date": str(target_date),
        "groups": len(groups),
        "total_price_rows": total_rows,
        "inserted": total_inserted,
        "errors": errors,
    }
    logger.info("Live ingestion done: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Archive download + extraction
# ---------------------------------------------------------------------------

def download_archive(date_str: str) -> Path:
    """Download and extract a TCGCSV daily archive.

    Args:
        date_str: Date in YYYY-MM-DD format.

    Returns:
        Path to the extracted directory for this date.
    """
    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    archive_url = f"{ARCHIVE_BASE}/prices-{date_str}.ppmd.7z"
    local_7z = ARCHIVES_DIR / f"prices-{date_str}.ppmd.7z"
    extract_dir = EXTRACTED_DIR / date_str

    # Download if not already cached.
    if not local_7z.exists():
        logger.info("Downloading %s", archive_url)
        resp = requests.get(archive_url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(local_7z, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        logger.info("Saved %s (%.1f MB)", local_7z, local_7z.stat().st_size / 1e6)
    else:
        logger.info("Archive already cached: %s", local_7z)

    # Extract if not already done.
    if not extract_dir.exists():
        logger.info("Extracting to %s", extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["7z", "x", str(local_7z), f"-o{extract_dir}", "-y"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error("7z extraction failed: %s", result.stderr)
            raise RuntimeError(f"7z failed (rc={result.returncode}): {result.stderr}")
        logger.info("Extraction complete")
    else:
        logger.info("Already extracted: %s", extract_dir)

    return extract_dir


def _find_price_files(extract_dir: Path) -> list[Path]:
    """Find all price JSON files for Pokemon (categoryId=3) under an extracted archive."""
    cat_dir = extract_dir / str(POKEMON_CATEGORY_ID)
    if not cat_dir.exists():
        # Some archives put files at top level — try to find any prices file.
        candidates = list(extract_dir.rglob("*/prices"))
        if not candidates:
            candidates = list(extract_dir.rglob("*/prices.json"))
        # Filter to ones under a "3" directory if possible.
        pokemon_files = [p for p in candidates if f"/{POKEMON_CATEGORY_ID}/" in str(p)]
        return pokemon_files or candidates

    # Standard layout: {extract_dir}/3/{groupId}/prices
    price_files = sorted(cat_dir.rglob("prices"))
    if not price_files:
        price_files = sorted(cat_dir.rglob("prices.json"))
    return price_files


def _parse_price_file(path: Path) -> list[dict]:
    """Parse a TCGCSV price file (JSON). Returns list of price dicts."""
    raw = path.read_text()
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("results", [])
    return []


# ---------------------------------------------------------------------------
# Archive ingestion
# ---------------------------------------------------------------------------

def ingest_archive(session: Session, date_str: str) -> dict:
    """Download (if needed), extract, and load one day's archive into DB.

    Returns summary dict.
    """
    price_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    extract_dir = download_archive(date_str)
    price_files = _find_price_files(extract_dir)
    logger.info("Found %d price files for date %s", len(price_files), date_str)

    total_inserted = 0
    total_rows = 0

    for pf in price_files:
        prices = _parse_price_file(pf)
        total_rows += len(prices)
        inserted = _insert_price_rows(session, prices, price_date)
        total_inserted += inserted

    session.commit()

    summary = {
        "date": date_str,
        "price_files": len(price_files),
        "total_price_rows": total_rows,
        "inserted": total_inserted,
    }
    logger.info("Archive ingestion done: %s", summary)
    return summary
