"""Fetch Pokemon species metadata from PokeAPI and upsert into dim_pokemon_features."""

import logging
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sqlalchemy import text

from cardprice.db.session import SessionLocal

logger = logging.getLogger(__name__)

POKEAPI_BASE = "https://pokeapi.co/api/v2"
DELAY = 0.5


def _http_session() -> requests.Session:
    """Create a requests session with retry/backoff."""
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def _fetch_species(http: requests.Session, n: int) -> dict | None:
    """Fetch /pokemon-species/{n} and /pokemon/{n}, return merged dict or None."""
    try:
        sp_resp = http.get(f"{POKEAPI_BASE}/pokemon-species/{n}", timeout=30)
        sp_resp.raise_for_status()
        species = sp_resp.json()

        pk_resp = http.get(f"{POKEAPI_BASE}/pokemon/{n}", timeout=30)
        pk_resp.raise_for_status()
        pokemon = pk_resp.json()
    except requests.RequestException as e:
        logger.warning("PokeAPI error for pokedex_num=%d: %s", n, e)
        return None

    # Extract egg groups
    egg_groups = [eg["name"] for eg in species.get("egg_groups", [])]

    # Extract types
    types = [
        t["type"]["name"]
        for t in sorted(pokemon.get("types", []), key=lambda t: t["slot"])
    ]

    # Extract base stats
    stat_map = {}
    for stat in pokemon.get("stats", []):
        stat_map[stat["stat"]["name"]] = stat["base_stat"]

    hp = stat_map.get("hp")
    attack = stat_map.get("attack")
    defense = stat_map.get("defense")
    sp_attack = stat_map.get("special-attack")
    sp_defense = stat_map.get("special-defense")
    speed = stat_map.get("speed")
    bst = sum(v for v in [hp, attack, defense, sp_attack, sp_defense, speed] if v is not None)

    return {
        "pokedex_num": n,
        "name": species.get("name", ""),
        "generation": species.get("generation", {}).get("name"),
        "is_legendary": species.get("is_legendary", False),
        "is_mythical": species.get("is_mythical", False),
        "capture_rate": species.get("capture_rate"),
        "base_happiness": species.get("base_happiness"),
        "egg_groups": egg_groups,
        "color": species.get("color", {}).get("name") if species.get("color") else None,
        "shape": species.get("shape", {}).get("name") if species.get("shape") else None,
        "habitat": species.get("habitat", {}).get("name") if species.get("habitat") else None,
        "hp": hp,
        "attack": attack,
        "defense": defense,
        "sp_attack": sp_attack,
        "sp_defense": sp_defense,
        "speed": speed,
        "bst": bst,
        "height": pokemon.get("height"),
        "weight": pokemon.get("weight"),
        "types": types,
    }


UPSERT_SQL = text("""
    INSERT INTO dim_pokemon_features (
        pokedex_num, name, generation, is_legendary, is_mythical,
        capture_rate, base_happiness, egg_groups, color, shape, habitat,
        hp, attack, defense, sp_attack, sp_defense, speed, bst,
        height, weight, types
    ) VALUES (
        :pokedex_num, :name, :generation, :is_legendary, :is_mythical,
        :capture_rate, :base_happiness, :egg_groups, :color, :shape, :habitat,
        :hp, :attack, :defense, :sp_attack, :sp_defense, :speed, :bst,
        :height, :weight, :types
    )
    ON CONFLICT (pokedex_num) DO UPDATE SET
        name = EXCLUDED.name,
        generation = EXCLUDED.generation,
        is_legendary = EXCLUDED.is_legendary,
        is_mythical = EXCLUDED.is_mythical,
        capture_rate = EXCLUDED.capture_rate,
        base_happiness = EXCLUDED.base_happiness,
        egg_groups = EXCLUDED.egg_groups,
        color = EXCLUDED.color,
        shape = EXCLUDED.shape,
        habitat = EXCLUDED.habitat,
        hp = EXCLUDED.hp,
        attack = EXCLUDED.attack,
        defense = EXCLUDED.defense,
        sp_attack = EXCLUDED.sp_attack,
        sp_defense = EXCLUDED.sp_defense,
        speed = EXCLUDED.speed,
        bst = EXCLUDED.bst,
        height = EXCLUDED.height,
        weight = EXCLUDED.weight,
        types = EXCLUDED.types
""")


def fetch_all_species(session):
    """Fetch PokeAPI metadata for every pokedex_num in dim_pokemon and upsert."""
    rows = session.execute(
        text("SELECT DISTINCT pokedex_num FROM dim_pokemon WHERE pokedex_num IS NOT NULL ORDER BY pokedex_num")
    ).fetchall()
    pokedex_nums = [r[0] for r in rows]
    total = len(pokedex_nums)
    logger.info("Found %d unique pokedex_num values to fetch from PokeAPI", total)

    http = _http_session()
    inserted = 0
    errors = 0

    for i, n in enumerate(pokedex_nums, 1):
        data = _fetch_species(http, n)
        if data is None:
            errors += 1
            continue

        try:
            session.execute(UPSERT_SQL, data)
            session.commit()
            inserted += 1
        except Exception as e:
            logger.warning("DB upsert error for pokedex_num=%d: %s", n, e)
            session.rollback()
            errors += 1

        if i % 50 == 0:
            logger.info("PokeAPI progress: %d/%d fetched (%d inserted, %d errors)", i, total, inserted, errors)

        time.sleep(DELAY)

    logger.info(
        "PokeAPI complete: %d/%d inserted, %d errors", inserted, total, errors
    )
    return {"total": total, "inserted": inserted, "errors": errors}
