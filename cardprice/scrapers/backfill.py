"""Batch historical price backfill from TCGCSV archives.

Downloads and ingests daily price archives from Feb 2024 to present.
Designed to run in chunks and be restartable (idempotent).
"""

import logging
import shutil
from datetime import date, timedelta
from pathlib import Path

from cardprice.db.session import SessionLocal
from cardprice.scrapers.tcgcsv import ingest_archive, EXTRACTED_DIR, ARCHIVES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# TCGCSV archives start from Feb 8, 2024
ARCHIVE_START = date(2024, 2, 8)


def generate_dates(start: date, end: date) -> list[str]:
    """Generate list of date strings from start to end inclusive."""
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def backfill(start: date | None = None, end: date | None = None, batch_size: int = 7, cleanup: bool = True) -> dict:
    """Run historical backfill in batches.

    Args:
        start: First date to backfill (default: ARCHIVE_START).
        end: Last date to backfill (default: yesterday).
        batch_size: Commit after this many days.

    Returns:
        Summary dict.
    """
    start = start or ARCHIVE_START
    end = end or (date.today() - timedelta(days=1))
    dates = generate_dates(start, end)

    logger.info("Backfill: %d days from %s to %s", len(dates), dates[0], dates[-1])

    session = SessionLocal()
    total_inserted = 0
    total_rows = 0
    days_done = 0
    errors = []

    try:
        for i, d in enumerate(dates):
            try:
                result = ingest_archive(session, d)
                total_rows += result["total_price_rows"]
                total_inserted += result["inserted"]
                days_done += 1
            except Exception as e:
                logger.error("Failed date %s: %s", d, e)
                errors.append(d)
                session.rollback()
                continue

            # Clean up extracted files to save disk (keep 7z for re-runs)
            if cleanup:
                extract_path = EXTRACTED_DIR / d
                if extract_path.exists():
                    shutil.rmtree(extract_path)

            if (i + 1) % batch_size == 0:
                logger.info("Progress: %d/%d days, %d rows inserted", i + 1, len(dates), total_inserted)
    finally:
        session.close()

    summary = {
        "days_attempted": len(dates),
        "days_done": days_done,
        "total_rows": total_rows,
        "total_inserted": total_inserted,
        "errors": errors,
    }
    logger.info("Backfill complete: %s", summary)
    return summary


if __name__ == "__main__":
    import sys
    start = None
    end = None
    if len(sys.argv) > 1:
        start = date.fromisoformat(sys.argv[1])
    if len(sys.argv) > 2:
        end = date.fromisoformat(sys.argv[2])
    backfill(start=start, end=end)
