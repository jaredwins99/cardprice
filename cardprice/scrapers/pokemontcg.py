"""Ingest the full Pokemon card catalog from pokemontcg.io into dim_sets, dim_pokemon, dim_cards."""

import logging
import os
import time

import requests
from sqlalchemy import text

from cardprice.config import POKEMONTCG_API_KEY

logger = logging.getLogger(__name__)

BASE_URL = "https://api.pokemontcg.io/v2"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master"
PAGE_SIZE = 250
REQUEST_DELAY = 0.5
GITHUB_REQUEST_DELAY = 0.25
MAX_RETRIES = 3


def _headers():
    key = POKEMONTCG_API_KEY or os.environ.get("POKEMONTCG_API_KEY", "")
    h = {}
    if key:
        h["X-Api-Key"] = key
    return h


def _get(url, params=None):
    """GET with retry + exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=30)
            if resp.status_code == 504:
                raise requests.exceptions.HTTPError("504 Gateway Timeout")
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** (attempt + 1)
            logger.warning("Request failed (%s), retrying in %ds...", e, wait)
            time.sleep(wait)


def fetch_all_sets():
    """Fetch all sets from pokemontcg.io."""
    data = _get(f"{BASE_URL}/sets")
    return data["data"]


def fetch_all_cards():
    """Generator yielding all card dicts, paginated."""
    page = 1
    while True:
        logger.info("Fetching cards page %d", page)
        data = _get(f"{BASE_URL}/cards", params={"pageSize": PAGE_SIZE, "page": page})
        cards = data["data"]
        if not cards:
            break
        yield from cards
        # Check if we've fetched everything
        total = data.get("totalCount", 0)
        if page * PAGE_SIZE >= total:
            break
        page += 1
        time.sleep(REQUEST_DELAY)


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


def fetch_all_sets_github():
    """Fetch all sets from the GitHub mirror (PokemonTCG/pokemon-tcg-data).

    The GitHub data uses 'total' instead of 'totalCards'. We normalise to
    include 'totalCards' so downstream ingest_sets works without changes.
    """
    url = f"{GITHUB_RAW_BASE}/sets/en.json"
    logger.info("Fetching sets from GitHub mirror: %s", url)
    sets = _github_get(url)
    # Normalise: add 'totalCards' alias expected by ingest_sets
    for s in sets:
        if "totalCards" not in s and "total" in s:
            s["totalCards"] = s["total"]
    logger.info("Fetched %d sets from GitHub mirror", len(sets))
    return sets


def _extract_set_id_from_card_id(card_id):
    """Derive the set_id from a card id like 'base1-1' -> 'base1'."""
    # Card IDs are formatted as '{set_id}-{number}'
    parts = card_id.rsplit("-", 1)
    return parts[0] if len(parts) == 2 else None


def _normalise_github_card(card, set_id):
    """Normalise a GitHub mirror card dict so it looks like an API response.

    Key differences from the API:
    - No nested 'set' object -> we inject {"id": set_id}
    - No 'tcgplayer' object -> we leave it absent (variant defaults to 'normal')
    - 'evolvesTo' may be absent -> we leave it absent (handled via .get())
    """
    card["set"] = {"id": set_id}
    return card


def fetch_all_cards_github():
    """Generator yielding all card dicts from the GitHub mirror.

    Iterates over every set in sets/en.json, then fetches each
    cards/en/{set_id}.json and yields each card with a normalised 'set' field.
    """
    sets = fetch_all_sets_github()
    for i, s in enumerate(sets):
        set_id = s["id"]
        url = f"{GITHUB_RAW_BASE}/cards/en/{set_id}.json"
        logger.info("Fetching cards for set %s (%d/%d) from GitHub", set_id, i + 1, len(sets))
        try:
            cards = _github_get(url)
        except Exception:
            logger.warning("Failed to fetch cards for set %s from GitHub, skipping", set_id)
            continue
        for card in cards:
            # Derive set_id from card id as a sanity check, but prefer the known set_id
            yield _normalise_github_card(card, set_id)
        time.sleep(GITHUB_REQUEST_DELAY)


def ingest_sets(session):
    """Upsert all sets into dim_sets."""
    sets = fetch_all_sets()
    logger.info("Ingesting %d sets", len(sets))
    for s in sets:
        session.execute(
            text("""
                INSERT INTO dim_sets (set_id, name, series, total_cards, release_date)
                VALUES (:set_id, :name, :series, :total_cards, :release_date)
                ON CONFLICT (set_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    series = EXCLUDED.series,
                    total_cards = EXCLUDED.total_cards,
                    release_date = EXCLUDED.release_date
            """),
            {
                "set_id": s["id"],
                "name": s["name"],
                "series": s.get("series"),
                "total_cards": s.get("totalCards", s.get("total")),
                "release_date": s.get("releaseDate"),
            },
        )
    session.commit()
    logger.info("Sets ingestion complete: %d sets", len(sets))


def ingest_pokemon(session, cards):
    """Dedupe and upsert Pokemon species into dim_pokemon. Returns name->pokemon_id map."""
    seen = {}  # (name, pokedex_num) -> dict
    for c in cards:
        if c.get("supertype") != "Pokémon":
            continue
        name = c["name"]
        pokedex_nums = c.get("nationalPokedexNumbers", [])
        pokedex_num = pokedex_nums[0] if pokedex_nums else None
        key = (name, pokedex_num)
        if key in seen:
            continue
        seen[key] = {
            "name": name,
            "pokedex_num": pokedex_num,
            "types": c.get("types", []),
            "hp_base": int(c["hp"]) if c.get("hp", "").isdigit() else None,
            "evolves_from": c.get("evolvesFrom"),
            "evolves_to": c.get("evolvesTo", []),
        }

    logger.info("Ingesting %d unique Pokemon species", len(seen))
    for poke in seen.values():
        session.execute(
            text("""
                INSERT INTO dim_pokemon (name, pokedex_num, types, hp_base, evolves_from, evolves_to)
                VALUES (:name, :pokedex_num, :types, :hp_base, :evolves_from, :evolves_to)
                ON CONFLICT (name, COALESCE(pokedex_num, -1)) DO UPDATE SET
                    types = EXCLUDED.types,
                    hp_base = EXCLUDED.hp_base,
                    evolves_from = EXCLUDED.evolves_from,
                    evolves_to = EXCLUDED.evolves_to
            """),
            {
                "name": poke["name"],
                "pokedex_num": poke["pokedex_num"],
                "types": poke["types"],
                "hp_base": poke["hp_base"],
                "evolves_from": poke["evolves_from"],
                "evolves_to": poke["evolves_to"],
            },
        )
    session.commit()

    # Build lookup map: (name, pokedex_num) -> pokemon_id
    rows = session.execute(text("SELECT pokemon_id, name, pokedex_num FROM dim_pokemon")).fetchall()
    return {(r[1], r[2]): r[0] for r in rows}


def _get_variants(card):
    """Extract variant keys from tcgplayer.prices, defaulting to ['normal']."""
    prices = (card.get("tcgplayer") or {}).get("prices") or {}
    variants = list(prices.keys())
    return variants if variants else ["normal"]


def ingest_cards(session, cards, pokemon_map):
    """Explode variants and upsert into dim_cards."""
    count = 0
    for c in cards:
        base_id = c["id"]
        set_id = c.get("set", {}).get("id")
        name = c["name"]

        # Resolve pokemon_id
        pokemon_id = None
        if c.get("supertype") == "Pokémon":
            pokedex_nums = c.get("nationalPokedexNumbers", [])
            pokedex_num = pokedex_nums[0] if pokedex_nums else None
            pokemon_id = pokemon_map.get((name, pokedex_num))

        hp = int(c["hp"]) if c.get("hp", "").isdigit() else None
        images = c.get("images", {})
        tcgplayer_url = (c.get("tcgplayer") or {}).get("url")
        subtypes = c.get("subtypes", [])
        types = c.get("types")  # Card-level types (may differ from species)

        for variant in _get_variants(c):
            card_id = f"{base_id}/{variant}"
            session.execute(
                text("""
                    INSERT INTO dim_cards (
                        card_id, name, set_id, pokemon_id, card_number,
                        rarity, supertype, subtypes, types, variant, hp,
                        artist, image_small, image_large, tcgplayer_url
                    ) VALUES (
                        :card_id, :name, :set_id, :pokemon_id, :card_number,
                        :rarity, :supertype, :subtypes, :types, :variant, :hp,
                        :artist, :image_small, :image_large, :tcgplayer_url
                    )
                    ON CONFLICT (card_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        set_id = EXCLUDED.set_id,
                        pokemon_id = EXCLUDED.pokemon_id,
                        rarity = EXCLUDED.rarity,
                        supertype = EXCLUDED.supertype,
                        subtypes = EXCLUDED.subtypes,
                        types = EXCLUDED.types,
                        hp = EXCLUDED.hp,
                        artist = EXCLUDED.artist,
                        image_small = EXCLUDED.image_small,
                        image_large = EXCLUDED.image_large,
                        tcgplayer_url = EXCLUDED.tcgplayer_url
                """),
                {
                    "card_id": card_id,
                    "name": name,
                    "set_id": set_id,
                    "pokemon_id": pokemon_id,
                    "card_number": c.get("number"),
                    "rarity": c.get("rarity"),
                    "supertype": c.get("supertype"),
                    "subtypes": subtypes,
                    "types": types,
                    "variant": variant,
                    "hp": hp,
                    "artist": c.get("artist"),
                    "image_small": images.get("small"),
                    "image_large": images.get("large"),
                    "tcgplayer_url": tcgplayer_url,
                },
            )
            count += 1
            if count % 5000 == 0:
                session.commit()
                logger.info("Inserted %d card rows so far", count)

    session.commit()
    logger.info("Cards ingestion complete: %d total rows", count)


def _try_api_ingestion(session):
    """Attempt ingestion from the pokemontcg.io API. Raises on failure."""
    # 1. Sets
    ingest_sets(session)

    # 2. Fetch all cards into memory (needed for two passes: pokemon + cards)
    logger.info("Fetching all cards from API...")
    all_cards = list(fetch_all_cards())
    logger.info("Fetched %d cards total", len(all_cards))

    return all_cards


def _github_fallback_ingestion(session):
    """Ingest from the GitHub mirror as a fallback."""
    logger.info("Using GitHub mirror fallback for ingestion")

    # 1. Sets from GitHub
    sets = fetch_all_sets_github()
    logger.info("Ingesting %d sets from GitHub mirror", len(sets))
    for s in sets:
        session.execute(
            text("""
                INSERT INTO dim_sets (set_id, name, series, total_cards, release_date)
                VALUES (:set_id, :name, :series, :total_cards, :release_date)
                ON CONFLICT (set_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    series = EXCLUDED.series,
                    total_cards = EXCLUDED.total_cards,
                    release_date = EXCLUDED.release_date
            """),
            {
                "set_id": s["id"],
                "name": s["name"],
                "series": s.get("series"),
                "total_cards": s.get("totalCards", s.get("total")),
                "release_date": s.get("releaseDate"),
            },
        )
    session.commit()
    logger.info("Sets ingestion from GitHub complete: %d sets", len(sets))

    # 2. Fetch all cards from GitHub into memory
    logger.info("Fetching all cards from GitHub mirror...")
    all_cards = list(fetch_all_cards_github())
    logger.info("Fetched %d cards from GitHub mirror", len(all_cards))

    return all_cards


def ingest_all(session):
    """Orchestrate full ingestion: sets, then cards (populates dim_pokemon + dim_cards).

    Tries the pokemontcg.io API first. If it fails after retries (e.g. 504 errors),
    falls back to the GitHub mirror at PokemonTCG/pokemon-tcg-data.
    """
    logger.info("Starting full pokemontcg.io ingestion")

    try:
        all_cards = _try_api_ingestion(session)
        logger.info("API ingestion succeeded")
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.warning(
            "API ingestion failed after retries (%s), falling back to GitHub mirror", e
        )
        all_cards = _github_fallback_ingestion(session)

    # 3. Pokemon species
    pokemon_map = ingest_pokemon(session, all_cards)

    # 4. Cards with variant explosion
    ingest_cards(session, all_cards, pokemon_map)

    logger.info("Full ingestion complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from cardprice.db.session import SessionLocal

    with SessionLocal() as session:
        ingest_all(session)
