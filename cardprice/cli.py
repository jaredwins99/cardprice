"""CLI entry point for the cardprice weekly refresh pipeline.

Usage:
    python -m cardprice.cli refresh       # weekly refresh (live prices + card catalog)
    python -m cardprice.cli backfill      # historical backfill
    python -m cardprice.cli migrate       # run DB migrations
    python -m cardprice.cli mapping       # TCGCSV product-to-card mapping
    python -m cardprice.cli status        # show DB row counts and date ranges
"""

import argparse
import logging
import sys
from datetime import date

from sqlalchemy import text

from cardprice.db.session import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_refresh(args):
    """Run weekly refresh: live prices + card catalog update."""
    from cardprice.scrapers.tcgcsv import ingest_live_prices
    from cardprice.scrapers.pokemontcg import ingest_all

    with SessionLocal() as session:
        print("=== Card catalog update (pokemontcg.io) ===")
        ingest_all(session)

        print("\n=== Live price ingestion (TCGCSV) ===")
        summary = ingest_live_prices(session)
        print(f"  Date:     {summary['date']}")
        print(f"  Groups:   {summary['groups']}")
        print(f"  Rows:     {summary['total_price_rows']}")
        print(f"  Inserted: {summary['inserted']}")
        print(f"  Errors:   {summary['errors']}")

    print("\nRefresh complete.")


def cmd_backfill(args):
    """Run historical price backfill from TCGCSV archives."""
    from cardprice.scrapers.backfill import backfill

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    summary = backfill(start=start, end=end)
    print(f"\nBackfill summary:")
    print(f"  Days attempted: {summary['days_attempted']}")
    print(f"  Days done:      {summary['days_done']}")
    print(f"  Total rows:     {summary['total_rows']}")
    print(f"  Inserted:       {summary['total_inserted']}")
    if summary["errors"]:
        print(f"  Failed dates:   {', '.join(summary['errors'])}")


def cmd_migrate(args):
    """Run DB migrations (create all tables)."""
    from cardprice.db.migrate import run

    run()


def cmd_mapping(args):
    """Run TCGCSV product-to-card mapping."""
    from cardprice.scrapers.mapping import run_mapping

    with SessionLocal() as session:
        run_mapping(session)


def cmd_status(args):
    """Show DB row counts and date ranges."""
    queries = {
        "dim_pokemon": "SELECT COUNT(*) FROM dim_pokemon",
        "dim_sets": "SELECT COUNT(*) FROM dim_sets",
        "dim_cards": "SELECT COUNT(*) FROM dim_cards",
        "dim_cards (mapped)": "SELECT COUNT(*) FROM dim_cards WHERE tcg_product_id IS NOT NULL",
        "dim_marketplaces": "SELECT COUNT(*) FROM dim_marketplaces",
        "fact_market_prices": "SELECT COUNT(*) FROM fact_market_prices",
        "fact_sales": "SELECT COUNT(*) FROM fact_sales",
    }

    date_queries = {
        "fact_market_prices": (
            "SELECT MIN(price_date), MAX(price_date) FROM fact_market_prices"
        ),
        "fact_sales": (
            "SELECT MIN(sale_date)::date, MAX(sale_date)::date FROM fact_sales"
        ),
    }

    with SessionLocal() as session:
        print("=== Row Counts ===")
        for label, sql in queries.items():
            try:
                count = session.execute(text(sql)).scalar()
                print(f"  {label:30s} {count:>12,}")
            except Exception as e:
                print(f"  {label:30s} ERROR: {e}")
                session.rollback()

        print("\n=== Date Ranges ===")
        for label, sql in date_queries.items():
            try:
                row = session.execute(text(sql)).fetchone()
                if row and row[0]:
                    print(f"  {label:30s} {row[0]}  to  {row[1]}")
                else:
                    print(f"  {label:30s} (no data)")
            except Exception as e:
                print(f"  {label:30s} ERROR: {e}")
                session.rollback()


def main():
    parser = argparse.ArgumentParser(
        prog="cardprice",
        description="Cardprice data pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # refresh
    sub.add_parser("refresh", help="Run weekly refresh (live prices + card catalog)")

    # backfill
    bf = sub.add_parser("backfill", help="Run historical price backfill")
    bf.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    bf.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")

    # migrate
    sub.add_parser("migrate", help="Run DB migrations (create tables)")

    # mapping
    sub.add_parser("mapping", help="Run TCGCSV product-to-card mapping")

    # status
    sub.add_parser("status", help="Show DB row counts and date ranges")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "refresh": cmd_refresh,
        "backfill": cmd_backfill,
        "migrate": cmd_migrate,
        "mapping": cmd_mapping,
        "status": cmd_status,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
