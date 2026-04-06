#!/usr/bin/env python3
"""Migrate dim_cards: synthesize variant rows from ERA_VALID_VARIANTS.

For each existing /normal card, determines which variants it should have
based on its set/era (using ERA_VALID_VARIANTS and SET_SPECIAL_VARIANTS
from variant_detector.py), then INSERTs new dim_cards rows with the
appropriate card_id suffix (e.g. base1-4/holofoil, sv3-42/reverse_holofoil).

Usage:
    # Preview what would be inserted (default: dry run)
    python scripts/migrate/migrate_variants.py

    # Actually insert rows
    python scripts/migrate/migrate_variants.py --execute

    # Only migrate one set
    python scripts/migrate/migrate_variants.py --set sv8

    # Migrate with execute for one set
    python scripts/migrate/migrate_variants.py --set base1 --execute
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from sqlalchemy import text

# -- project imports --
sys.path.insert(0, ".")
from cardprice.db.session import SessionLocal
from cardprice.ml.era_detector import SET_TO_ERA
from cardprice.ml.variant_detector import (
    ERA_VALID_VARIANTS,
    SET_SPECIAL_VARIANTS,
    TCGCSV_SUBTYPE_TO_VARIANT,
    get_valid_variants,
)


# Variants that come from TCGCSV subtype data (have separate pricing).
# Visual-only variants (full_art, gold, rainbow_rare, shadowless*) are
# detected by CV and do NOT get separate dim_cards rows from this script
# because they share TCGCSV product/subtype with normal or holofoil.
PRICE_VARIANTS = set(TCGCSV_SUBTYPE_TO_VARIANT.values()) - {None}
# => {"normal", "holofoil", "reverse_holofoil",
#     "1st_edition", "1st_edition_holofoil",
#     "unlimited", "unlimited_holofoil"}


def get_variants_for_set(set_id: str) -> set[str]:
    """Return the set of price-relevant variants for a set.

    Intersects ERA_VALID_VARIANTS (which includes visual-only variants)
    with PRICE_VARIANTS (which only includes TCGCSV-tracked variants).
    """
    era = SET_TO_ERA.get(set_id, 0)
    all_valid = get_valid_variants(set_id, era)
    # Only keep variants that have separate TCGCSV pricing
    return all_valid & PRICE_VARIANTS


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize variant rows in dim_cards from era/set rules."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually INSERT rows. Without this flag, only previews changes.",
    )
    parser.add_argument(
        "--set",
        type=str,
        default=None,
        help="Only migrate cards from this set_id (e.g. 'sv8', 'base1').",
    )
    args = parser.parse_args()

    dry_run = not args.execute
    if dry_run:
        print("=== DRY RUN (pass --execute to apply) ===\n")

    session = SessionLocal()
    try:
        # ------------------------------------------------------------------
        # Step 1: Load all existing /normal cards
        # ------------------------------------------------------------------
        query = "SELECT card_id, name, set_id, card_number, rarity, supertype, subtypes, variant FROM dim_cards WHERE variant = 'normal'"
        params = {}
        if args.set:
            query += " AND set_id = :set_id"
            params["set_id"] = args.set

        rows = session.execute(text(query), params).fetchall()
        print(f"Found {len(rows)} existing /normal cards"
              + (f" in set {args.set}" if args.set else "") + ".\n")

        if not rows:
            print("Nothing to do.")
            return

        # ------------------------------------------------------------------
        # Step 2: For each card, determine which variants to create
        # ------------------------------------------------------------------
        to_insert: list[dict] = []
        stats: dict[str, int] = defaultdict(int)

        for row in rows:
            card_id = row[0]  # e.g. "sv8-162/normal"
            set_id = row[2]

            # Parse base_id: strip "/normal" suffix
            base_id = card_id.rsplit("/", 1)[0]  # e.g. "sv8-162"

            # Get valid price variants for this set
            valid_variants = get_variants_for_set(set_id)

            for variant in sorted(valid_variants):
                if variant == "normal":
                    continue  # Already exists

                new_card_id = f"{base_id}/{variant}"
                stats[variant] += 1
                to_insert.append({
                    "new_card_id": new_card_id,
                    "source_card_id": card_id,
                    "variant": variant,
                })

        # ------------------------------------------------------------------
        # Step 3: Filter out rows that already exist
        # ------------------------------------------------------------------
        # Check in batches to avoid huge IN clauses
        existing_ids: set[str] = set()
        new_ids = [r["new_card_id"] for r in to_insert]

        BATCH = 5000
        for i in range(0, len(new_ids), BATCH):
            batch = new_ids[i : i + BATCH]
            result = session.execute(
                text("SELECT card_id FROM dim_cards WHERE card_id = ANY(:ids)"),
                {"ids": batch},
            ).fetchall()
            existing_ids.update(r[0] for r in result)

        before_dedup = len(to_insert)
        to_insert = [r for r in to_insert if r["new_card_id"] not in existing_ids]
        skipped = before_dedup - len(to_insert)

        # Recount stats after dedup
        stats_final: dict[str, int] = defaultdict(int)
        for r in to_insert:
            stats_final[r["variant"]] += 1

        # ------------------------------------------------------------------
        # Step 4: Print summary
        # ------------------------------------------------------------------
        print(f"Variant rows to create: {len(to_insert)}")
        print(f"Already exist (skipped): {skipped}")
        print()

        if stats_final:
            print("Breakdown by variant:")
            for variant in sorted(stats_final):
                print(f"  {variant:<25s} {stats_final[variant]:>6,}")
            print()

        # Show sample of what would be inserted
        if to_insert and dry_run:
            print("Sample rows (first 20):")
            for r in to_insert[:20]:
                print(f"  {r['new_card_id']:<40s} (from {r['source_card_id']})")
            if len(to_insert) > 20:
                print(f"  ... and {len(to_insert) - 20} more")
            print()

        # ------------------------------------------------------------------
        # Step 5: Execute inserts (if not dry run)
        # ------------------------------------------------------------------
        if dry_run:
            print("No changes made. Pass --execute to apply.")
            return

        if not to_insert:
            print("Nothing to insert.")
            return

        INSERT_SQL = text("""
            INSERT INTO dim_cards (
                card_id, tcg_product_id, name, set_id, pokemon_id,
                card_number, rarity, supertype, subtypes, types,
                variant, hp, artist, image_small, image_large,
                tcgplayer_url
            )
            SELECT
                :new_card_id,
                NULL,
                name, set_id, pokemon_id,
                card_number, rarity, supertype, subtypes, types,
                :variant,
                hp, artist, image_small, image_large,
                tcgplayer_url
            FROM dim_cards
            WHERE card_id = :source_card_id
            ON CONFLICT (card_id) DO NOTHING
        """)

        inserted = 0
        errors = 0
        for i, r in enumerate(to_insert):
            try:
                result = session.execute(INSERT_SQL, r)
                if result.rowcount > 0:
                    inserted += 1
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  ERROR inserting {r['new_card_id']}: {e}")

            if (i + 1) % 10000 == 0:
                print(f"  Progress: {i + 1}/{len(to_insert)} "
                      f"({inserted} inserted, {errors} errors)")

        session.commit()
        print(f"\nDone. Inserted {inserted} variant rows, "
              f"{errors} errors.")

        # Verify
        total = session.execute(
            text("SELECT COUNT(*) FROM dim_cards")
        ).scalar()
        variant_dist = session.execute(text(
            "SELECT variant, COUNT(*) FROM dim_cards "
            "GROUP BY variant ORDER BY COUNT(*) DESC"
        )).fetchall()

        print(f"\nTotal dim_cards rows: {total:,}")
        print("Variant distribution:")
        for v, cnt in variant_dist:
            print(f"  {v or 'NULL':<25s} {cnt:>6,}")

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
