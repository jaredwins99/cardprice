#!/usr/bin/env python3
"""Map fact_market_prices rows to variant card_ids.

After migrate_variants.py creates variant rows in dim_cards, this script
updates fact_market_prices.card_id to point to the correct variant row
based on subtype_name.

The link goes through tcg_product_id:
  fact_market_prices.tcg_product_id -> dim_cards.tcg_product_id (the /normal row)
  fact_market_prices.subtype_name   -> maps to variant suffix

So for a row with tcg_product_id=12345 and subtype_name='Reverse Holofoil',
if the /normal card_id is 'sv8-42/normal', we update card_id to
'sv8-42/reverse_holofoil' (provided that card_id exists in dim_cards).

Usage:
    # Preview changes (default: dry run)
    python scripts/map_variant_prices.py

    # Actually update rows
    python scripts/map_variant_prices.py --execute

    # Only update one subtype
    python scripts/map_variant_prices.py --subtype "Reverse Holofoil" --execute
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import text

sys.path.insert(0, ".")
from cardprice.db.session import SessionLocal
from cardprice.ml.variant_detector import TCGCSV_SUBTYPE_TO_VARIANT


# Only Pokemon TCG subtypes (non-None values)
SUBTYPE_TO_VARIANT = {
    k: v for k, v in TCGCSV_SUBTYPE_TO_VARIANT.items() if v is not None
}


def main():
    parser = argparse.ArgumentParser(
        description="Map fact_market_prices.card_id to variant dim_cards rows."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually UPDATE rows. Without this flag, only previews changes.",
    )
    parser.add_argument(
        "--subtype",
        type=str,
        default=None,
        help="Only process this subtype_name (e.g. 'Reverse Holofoil').",
    )
    args = parser.parse_args()

    dry_run = not args.execute
    if dry_run:
        print("=== DRY RUN (pass --execute to apply) ===\n")

    session = SessionLocal()
    try:
        # ------------------------------------------------------------------
        # Step 1: Show current state
        # ------------------------------------------------------------------
        current = session.execute(text("""
            SELECT
                subtype_name,
                COUNT(*) AS total,
                COUNT(card_id) AS linked,
                COUNT(*) - COUNT(card_id) AS unlinked
            FROM fact_market_prices
            WHERE subtype_name IS NOT NULL
            GROUP BY subtype_name
            ORDER BY total DESC
        """)).fetchall()

        print("Current fact_market_prices state:")
        print(f"  {'Subtype':<30s} {'Total':>12s} {'Linked':>10s} {'Unlinked':>10s}")
        print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*10}")
        for row in current:
            print(f"  {row[0] or 'NULL':<30s} {row[1]:>12,} {row[2]:>10,} {row[3]:>10,}")
        print()

        # ------------------------------------------------------------------
        # Step 2: For each subtype, count how many rows would be updated
        # ------------------------------------------------------------------
        subtypes_to_process = (
            {args.subtype: SUBTYPE_TO_VARIANT[args.subtype]}
            if args.subtype
            else SUBTYPE_TO_VARIANT
        )

        total_updated = 0

        for subtype_name, variant_suffix in subtypes_to_process.items():
            # Count rows that would be updated:
            # - Have this subtype_name
            # - Have a tcg_product_id that matches a dim_cards row
            # - The target variant card_id exists in dim_cards
            # - card_id is currently wrong or NULL
            count_sql = text("""
                SELECT COUNT(*)
                FROM fact_market_prices fmp
                JOIN dim_cards dc ON dc.tcg_product_id = fmp.tcg_product_id
                WHERE fmp.subtype_name = :subtype
                  AND fmp.tcg_product_id IS NOT NULL
                  AND dc.variant = 'normal'
                  AND EXISTS (
                      SELECT 1 FROM dim_cards dc2
                      WHERE dc2.card_id = REGEXP_REPLACE(dc.card_id, '/[^/]+$', '')
                                          || '/' || :variant
                  )
                  AND (fmp.card_id IS NULL
                       OR fmp.card_id != REGEXP_REPLACE(dc.card_id, '/[^/]+$', '')
                                         || '/' || :variant)
            """)

            count = session.execute(
                count_sql, {"subtype": subtype_name, "variant": variant_suffix}
            ).scalar()

            if count > 0:
                print(f"  {subtype_name:<30s} -> /{variant_suffix:<25s} {count:>10,} rows")
                total_updated += count

        print(f"\nTotal rows to update: {total_updated:,}")

        if dry_run:
            print("\nNo changes made. Pass --execute to apply.")
            return

        if total_updated == 0:
            print("\nNothing to update.")
            return

        # ------------------------------------------------------------------
        # Step 3: Execute updates
        # ------------------------------------------------------------------
        print("\nExecuting updates...")

        update_sql = text("""
            UPDATE fact_market_prices fmp
            SET card_id = REGEXP_REPLACE(dc.card_id, '/[^/]+$', '')
                          || '/' || :variant
            FROM dim_cards dc
            WHERE dc.tcg_product_id = fmp.tcg_product_id
              AND fmp.subtype_name = :subtype
              AND fmp.tcg_product_id IS NOT NULL
              AND dc.variant = 'normal'
              AND EXISTS (
                  SELECT 1 FROM dim_cards dc2
                  WHERE dc2.card_id = REGEXP_REPLACE(dc.card_id, '/[^/]+$', '')
                                      || '/' || :variant
              )
              AND (fmp.card_id IS NULL
                   OR fmp.card_id != REGEXP_REPLACE(dc.card_id, '/[^/]+$', '')
                                     || '/' || :variant)
        """)

        for subtype_name, variant_suffix in subtypes_to_process.items():
            result = session.execute(
                update_sql,
                {"subtype": subtype_name, "variant": variant_suffix},
            )
            if result.rowcount > 0:
                print(f"  {subtype_name:<30s} -> /{variant_suffix:<25s} "
                      f"{result.rowcount:>10,} rows updated")

        session.commit()

        # ------------------------------------------------------------------
        # Step 4: Verify
        # ------------------------------------------------------------------
        print("\nPost-update state:")
        post = session.execute(text("""
            SELECT
                subtype_name,
                COUNT(*) AS total,
                COUNT(card_id) AS linked
            FROM fact_market_prices
            WHERE subtype_name IS NOT NULL
            GROUP BY subtype_name
            ORDER BY total DESC
        """)).fetchall()
        for row in post:
            pct = row[2] / row[1] * 100 if row[1] else 0
            print(f"  {row[0] or 'NULL':<30s} {row[2]:>10,}/{row[1]:>10,} "
                  f"linked ({pct:.1f}%)")

        # Check for unlinked rows (card_id pointing to non-existent dim_cards)
        orphans = session.execute(text("""
            SELECT COUNT(*)
            FROM fact_market_prices fmp
            WHERE fmp.card_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM dim_cards dc WHERE dc.card_id = fmp.card_id
              )
        """)).scalar()
        if orphans:
            print(f"\n  WARNING: {orphans:,} rows have card_id pointing to "
                  f"non-existent dim_cards rows!")

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
