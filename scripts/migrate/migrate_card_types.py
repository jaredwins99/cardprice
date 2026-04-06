#!/usr/bin/env python3
"""Migration: Add card-level types to dim_cards.

Previously, types were only stored at the species level in dim_pokemon.
But Pokemon cards can have different types than their species (e.g., delta
species Flygon is Grass/Metal, not Dragon/Ground).

This script:
1. Adds a `types` TEXT[] column to dim_cards (if not exists)
2. Fetches card data from the GitHub mirror (PokemonTCG/pokemon-tcg-data)
3. Populates the types column for all Pokemon cards
"""

import logging
import sys
import time

import requests
from sqlalchemy import text

# Add parent dir to path so we can import cardprice
sys.path.insert(0, "/home/godli/cardprice")

from cardprice.db.session import engine, SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master"
MAX_RETRIES = 3


def _github_get(url):
    """GET from raw.githubusercontent.com with retry + backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** (attempt + 1)
            logger.warning("GitHub request failed (%s), retrying in %ds...", e, wait)
            time.sleep(wait)


def main():
    # Step 1: Add column
    logger.info("Step 1: Adding types column to dim_cards (if not exists)")
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE dim_cards ADD COLUMN IF NOT EXISTS types TEXT[]"))
        conn.commit()
    logger.info("Column added (or already existed)")

    # Step 2: Fetch sets list
    logger.info("Step 2: Fetching set list from GitHub mirror")
    sets_url = f"{GITHUB_RAW_BASE}/sets/en.json"
    sets_data = _github_get(sets_url)
    set_ids = [s["id"] for s in sets_data]
    logger.info("Found %d sets", len(set_ids))

    # Step 3: Fetch cards per set and build type map
    # Map: base_card_id (e.g. "base1-4") -> types list (e.g. ["Fire"])
    logger.info("Step 3: Fetching card types from GitHub mirror")
    type_map = {}  # base_card_id -> types
    failed_sets = []

    for i, set_id in enumerate(set_ids):
        url = f"{GITHUB_RAW_BASE}/cards/en/{set_id}.json"
        try:
            cards = _github_get(url)
        except Exception as e:
            logger.warning("Failed to fetch set %s: %s", set_id, e)
            failed_sets.append(set_id)
            continue

        for card in cards:
            card_id = card["id"]
            types = card.get("types")  # None for Trainer/Energy cards
            if types is not None:
                type_map[card_id] = types

        if (i + 1) % 20 == 0:
            logger.info("  Fetched %d/%d sets (%d cards with types so far)",
                        i + 1, len(set_ids), len(type_map))
        time.sleep(0.15)  # Be polite to GitHub

    logger.info("Fetched types for %d base card IDs from %d sets (%d failed)",
                len(type_map), len(set_ids), len(failed_sets))
    if failed_sets:
        logger.warning("Failed sets: %s", failed_sets)

    # Step 4: Update dim_cards
    logger.info("Step 4: Updating dim_cards with types")
    updated = 0
    skipped = 0

    with SessionLocal() as session:
        # Get all card_ids from dim_cards
        rows = session.execute(
            text("SELECT card_id FROM dim_cards WHERE supertype = 'Pokémon'")
        ).fetchall()
        logger.info("Found %d Pokemon card rows in dim_cards", len(rows))

        batch = []
        for row in rows:
            card_id = row[0]
            # card_id format: "base1-4/holofoil" -> base_id = "base1-4"
            base_id = card_id.rsplit("/", 1)[0]
            types = type_map.get(base_id)
            if types is not None:
                batch.append({"cid": card_id, "types": types})
            else:
                skipped += 1

            if len(batch) >= 500:
                for item in batch:
                    session.execute(
                        text("UPDATE dim_cards SET types = :types WHERE card_id = :cid"),
                        item,
                    )
                session.commit()
                updated += len(batch)
                batch = []

        # Final batch
        if batch:
            for item in batch:
                session.execute(
                    text("UPDATE dim_cards SET types = :types WHERE card_id = :cid"),
                    item,
                )
            session.commit()
            updated += len(batch)

    logger.info("Updated %d cards, skipped %d (no type data in mirror)", updated, skipped)

    # Step 5: Verification
    logger.info("Step 5: Verification")
    with SessionLocal() as session:
        # Total with types
        count = session.execute(
            text("SELECT count(*) FROM dim_cards WHERE types IS NOT NULL")
        ).scalar()
        logger.info("Cards with types: %d", count)

        # Pokemon cards without types (should be 0 or very few)
        missing = session.execute(
            text("SELECT count(*) FROM dim_cards WHERE supertype = 'Pokémon' AND types IS NULL")
        ).scalar()
        logger.info("Pokemon cards still missing types: %d", missing)

        # Delta species verification
        logger.info("\n=== Delta Species Verification ===")
        deltas = session.execute(
            text("""
                SELECT dc.card_id, dc.name, dc.set_id, dc.types,
                       dp.types AS species_types
                FROM dim_cards dc
                LEFT JOIN dim_pokemon dp ON dp.pokemon_id = dc.pokemon_id
                WHERE dc.name LIKE '%%δ%%'
                ORDER BY dc.set_id, dc.name
                LIMIT 20
            """)
        ).fetchall()

        for d in deltas:
            match = "SAME" if d.types == d.species_types else "DIFFERENT"
            logger.info("  %s | card_types=%s | species_types=%s | %s",
                        d.name.ljust(20), d.types, d.species_types, match)

    logger.info("\nMigration complete!")


if __name__ == "__main__":
    main()
