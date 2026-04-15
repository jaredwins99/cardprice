"""Sales-velocity computation for scrape prioritization.

Each TCGPlayer Playwright scrape pulls the "latest 20 sales" per product.
Re-scraping a product that's had zero new sales since the last scrape is
a no-op on the UPSERT but still consumes the daily time budget (~3-5s per
product via Playwright).  Similarly, JustTCG calls are quota-capped at
1000/month.

This module computes per-product sales velocity (sales/day) from the
observed history in data/tcgplayer_sales.db.  Callers use it to:

  1. Skip products where too few new sales are expected since last scrape
     (threshold: EXPECTED_SALES_THRESHOLD new sales).
  2. Prioritize by expected-information-gain = (expected_new_sales) or
     (market_price × velocity) for value-weighted queues.

Products with no observed sales (or too little history) are treated as
low-velocity and rotated less frequently.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the SQLite DB where TCGPlayer Playwright sales are stored
_SALES_DB = Path(__file__).resolve().parents[2] / "data" / "tcgplayer_sales.db"

# Product-selection tuning knobs (override via env or call-site kwargs)
EXPECTED_SALES_THRESHOLD = 1.0  # skip if fewer than N new sales expected
MIN_SALES_FOR_VELOCITY = 3       # need at least this many sales to trust velocity
DEFAULT_UNKNOWN_VELOCITY = 1.0 / 30.0  # ~1 sale/month for products with no history


def compute_velocity(min_sales: int = MIN_SALES_FOR_VELOCITY) -> dict[int, float]:
    """Compute sales-per-day for every product in tcgplayer_sales.db.

    Uses the span between earliest and latest observed sale.  Filters out
    the chart-data pollution rows (sale_date LIKE 'N/N to N/N') that were
    scraped before the chart-data filter bug-fix.

    Returns dict {tcg_product_id: sales_per_day} for products with at
    least `min_sales` observed sales.  Products with fewer sales are
    omitted; callers should treat them as unknown velocity.
    """
    if not _SALES_DB.exists():
        return {}

    conn = sqlite3.connect(str(_SALES_DB))
    try:
        # Use ISO date prefix check to exclude pre-fix chart pollution rows
        # (e.g. "1/1 to 1/3"). Valid sales look like "2026-04-14T15:36:12..."
        rows = conn.execute(
            """
            SELECT
              tcg_product_id,
              COUNT(*) AS n,
              MIN(sale_date) AS first_sale,
              MAX(sale_date) AS last_sale
            FROM tcgplayer_sales
            WHERE sale_date LIKE '____-__-__%'
            GROUP BY tcg_product_id
            HAVING n >= ?
            """,
            (min_sales,),
        ).fetchall()
    finally:
        conn.close()

    velocity: dict[int, float] = {}
    for pid, n, first_sale, last_sale in rows:
        # Days between first and last observed sale.  Use julianday in
        # Python instead of SQL to avoid tz headaches with ISO-8601.
        from datetime import datetime
        try:
            d0 = datetime.fromisoformat(first_sale.replace("Z", "+00:00"))
            d1 = datetime.fromisoformat(last_sale.replace("Z", "+00:00"))
            span_days = max((d1 - d0).total_seconds() / 86400.0, 1.0)
        except Exception:
            span_days = 1.0
        velocity[pid] = float(n) / span_days

    logger.info(
        "Computed velocity for %d products (min %d sales, median=%.3f/day)",
        len(velocity), min_sales,
        sorted(velocity.values())[len(velocity) // 2] if velocity else 0.0,
    )
    return velocity


def get_last_scraped_at() -> dict[int, str]:
    """Return {tcg_product_id: ISO-8601 last_scraped} for every product
    recorded in the scrape_log.  Empty dict if the DB doesn't exist yet."""
    if not _SALES_DB.exists():
        return {}
    conn = sqlite3.connect(str(_SALES_DB))
    try:
        rows = conn.execute(
            "SELECT tcg_product_id, last_scraped FROM scrape_log"
        ).fetchall()
        return {pid: ts for pid, ts in rows}
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


def expected_new_sales(
    velocity_per_day: float,
    last_scraped_iso: str | None,
    now_iso: str,
) -> float:
    """How many new sales do we expect since the last scrape?

    velocity_per_day * days_elapsed.  Returns +inf for never-scraped so
    those are always candidates.
    """
    if last_scraped_iso is None:
        return float("inf")
    from datetime import datetime
    try:
        last = datetime.fromisoformat(last_scraped_iso.replace("Z", "+00:00"))
        now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        days = max((now - last).total_seconds() / 86400.0, 0.0)
    except Exception:
        return float("inf")  # if we can't parse, assume overdue
    return velocity_per_day * days


def select_by_velocity(
    candidates: list[tuple[int, float]],
    velocity: dict[int, float],
    last_scraped: dict[int, str],
    limit: int,
    *,
    threshold: float = EXPECTED_SALES_THRESHOLD,
    default_velocity: float = DEFAULT_UNKNOWN_VELOCITY,
) -> list[int]:
    """Select up to `limit` product IDs prioritized by expected information gain.

    Parameters
    ----------
    candidates : list[(tcg_product_id, market_price)]
        All products in the DB, ordered however the caller wants.  Market
        price is used only to break ties among never-scraped products.
    velocity : dict[int, float]
        Sales-per-day per product, from compute_velocity().
    last_scraped : dict[int, str]
        ISO timestamps of last successful scrape per product.
    limit : int
        Max products to return.
    threshold : float
        Skip products where fewer than this many new sales are expected.
        Products that have never been scraped bypass the threshold.
    default_velocity : float
        Velocity to assume for products with no observed sales.  Very low
        (1/30 day) so unknowns rotate slowly until we learn their pace.
    """
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()

    # Bucket 1: never-scraped.  Always candidates.  Sort by market price DESC.
    never_scraped: list[tuple[int, float]] = []
    # Bucket 2: scraped-before with expected_new_sales >= threshold.
    overdue: list[tuple[int, float, float]] = []  # (pid, expected, price)
    # Bucket 3: scraped-before but too soon to re-scrape (skipped).
    skipped = 0

    for pid, price in candidates:
        ts = last_scraped.get(pid)
        if ts is None:
            never_scraped.append((pid, price))
            continue
        vel = velocity.get(pid, default_velocity)
        expected = expected_new_sales(vel, ts, now_iso)
        if expected >= threshold:
            overdue.append((pid, expected, price))
        else:
            skipped += 1

    # Sort never-scraped by price DESC
    never_scraped.sort(key=lambda t: t[1], reverse=True)
    # Sort overdue by expected_new_sales DESC, tie-break on price DESC
    overdue.sort(key=lambda t: (t[1], t[2]), reverse=True)

    chosen: list[int] = [pid for pid, _ in never_scraped]
    chosen.extend(pid for pid, _, _ in overdue)
    chosen = chosen[:limit]

    logger.info(
        "select_by_velocity: never_scraped=%d, overdue=%d, skipped_too_soon=%d, picked=%d",
        len(never_scraped), len(overdue), skipped, len(chosen),
    )
    return chosen


def value_weighted_priority(
    candidates: list[tuple[int, float]],
    velocity: dict[int, float],
    *,
    default_velocity: float = DEFAULT_UNKNOWN_VELOCITY,
) -> list[tuple[int, float]]:
    """Re-rank (pid, price) pairs by price × velocity (expected USD-flow/day).

    Used for JustTCG where the quota is hard-capped and we want to spend
    each call on the products where price information changes fastest.
    Returns the list sorted by descending value-weighted score, paired
    with the score for logging.
    """
    scored: list[tuple[int, float]] = []
    for pid, price in candidates:
        vel = velocity.get(pid, default_velocity)
        scored.append((pid, float(price) * vel))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored
