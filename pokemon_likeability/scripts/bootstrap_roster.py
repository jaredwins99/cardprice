#!/usr/bin/env python3
"""Bootstrap the Pokémon roster + seed the ratings file.

Idempotent:
  * Re-running refreshes ``data/roster.json`` only.
  * ``data/ratings.json`` is preserved — existing ratings/comparison counts are
    kept verbatim; any new species in the roster are inserted at 1000 (or at
    their seeded value if they appear in SEED_BOOST).
  * ``data/votes.jsonl`` is never touched.

The roster is fetched from the PokéAPI species index, which currently lists
all ~1025 species through Gen 9 plus the Hisuian/Paldean forms that are
considered separate species. Sprite URLs are built against the PokeAPI/sprites
GitHub raw CDN — that repo mirrors every numeric ID under
``sprites/pokemon/<id>.png`` and ``sprites/pokemon/other/official-artwork/<id>.png``.
We use the official-artwork URL as the primary image (high-res) and the small
sprite as a thumbnail fallback.

If the PokéAPI list call fails entirely we fall back to a hard-coded
``MAX_NATIONAL_ID`` numeric sweep so the UI is at least functional offline.
Partial fetches are saved atomically so a crash mid-way still leaves a valid
JSON behind.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
ROSTER_PATH = DATA / "roster.json"
RATINGS_PATH = DATA / "ratings.json"

SPECIES_INDEX_URL = "https://pokeapi.co/api/v2/pokemon-species?limit=2000"

# Through Gen 9 / Paldea. PokéAPI currently exposes 1025 species.
MAX_NATIONAL_ID = 1025

SPRITE_BASE = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"
ART_BASE = f"{SPRITE_BASE}/other/official-artwork"


# ELO seed boosts. Match by name, case-insensitive.
SEED_BOOST: dict[str, float] = {
    # Tier 1 -> 1500
    "Charizard": 1500, "Gengar": 1500, "Umbreon": 1500,
    "Pikachu":   1500, "Mewtwo": 1500,
    # Tier 2 -> 1350
    "Espeon":    1350, "Dragonite": 1350, "Bulbasaur": 1350, "Mew": 1350,
    # Tier 3 -> 1200
    "Snorlax":   1200, "Eevee":   1200, "Lugia":    1200, "Giratina": 1200,
    # Tier 4 -> 1100
    "Arcanine":  1100,
}


def _http_get_json(url: str, timeout: int = 30) -> dict | None:
    req = Request(url, headers={"User-Agent": "pokemon-likeability-bootstrap/0.1"})
    try:
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, ConnectionError, OSError) as e:
        print(f"  ! {url} -> {e}", file=sys.stderr)
        return None


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    tmp.replace(path)


def _capitalize_name(slug: str) -> str:
    # PokéAPI slugs are lowercase, hyphenated for multi-word names.
    # ``mr-mime`` -> ``Mr. Mime``, ``ho-oh`` -> ``Ho-Oh``, ``nidoran-f`` -> ``Nidoran-F``.
    parts = slug.split("-")
    return "-".join(p.capitalize() for p in parts)


def _build_entry(pid: int, slug: str) -> dict:
    return {
        "id": pid,
        "slug": slug,
        "name": _capitalize_name(slug),
        "sprite_url": f"{SPRITE_BASE}/{pid}.png",
        "art_url":    f"{ART_BASE}/{pid}.png",
    }


def _fetch_via_index(limit: int | None) -> list[dict]:
    """Preferred path: single PokéAPI call returns name+url for every species."""
    index = _http_get_json(SPECIES_INDEX_URL)
    if not index or "results" not in index:
        return []
    out: list[dict] = []
    for r in index["results"]:
        slug = r.get("name") or ""
        url = r.get("url") or ""
        # url looks like https://pokeapi.co/api/v2/pokemon-species/25/
        pid_str = url.rstrip("/").rsplit("/", 1)[-1]
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid > MAX_NATIONAL_ID:
            # PokéAPI exposes some experimental/regional-variant species at
            # IDs > 10000; skip those, they're not in the National Pokédex
            # and won't have sprites under the numeric path.
            continue
        out.append(_build_entry(pid, slug))
        if limit and len(out) >= limit:
            break
    out.sort(key=lambda e: e["id"])
    return out


def _fetch_fallback(limit: int | None, sleep_s: float) -> list[dict]:
    """Fallback: walk numeric IDs and resolve each species individually.

    Slow (~1 req / species) but robust to a broken index endpoint.
    """
    out: list[dict] = []
    upper = min(limit, MAX_NATIONAL_ID) if limit else MAX_NATIONAL_ID
    for pid in range(1, upper + 1):
        sp = _http_get_json(f"https://pokeapi.co/api/v2/pokemon-species/{pid}/")
        slug = (sp or {}).get("name") or f"pokemon-{pid}"
        out.append(_build_entry(pid, slug))
        if pid % 50 == 0:
            print(f"  fetched {pid}/{upper}", flush=True)
            _atomic_write(ROSTER_PATH, {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "pokeapi-fallback-partial",
                "pokemon": out,
            })
        time.sleep(sleep_s)
    return out


def fetch_roster(limit: int | None, sleep_s: float) -> list[dict]:
    print(f"-> fetching species index: {SPECIES_INDEX_URL}")
    roster = _fetch_via_index(limit)
    if roster:
        print(f"   got {len(roster)} species from index")
        return roster
    print("   index failed; falling back to per-ID walk (slow)")
    return _fetch_fallback(limit, sleep_s)


def _load_existing_ratings() -> dict:
    if not RATINGS_PATH.is_file():
        return {}
    try:
        with open(RATINGS_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"  ! could not parse existing ratings.json: {e}", file=sys.stderr)
        return {}


def seed_ratings(roster: list[dict]) -> tuple[int, int]:
    """Merge roster into ratings.json. Returns (new_entries, seeded_total)."""
    seed_lookup = {k.lower(): v for k, v in SEED_BOOST.items()}
    existing = _load_existing_ratings()
    ratings: dict[str, dict] = dict(existing.get("ratings") or {})

    added = 0
    seeded_total = 0
    for sp in roster:
        sid = str(sp["id"])
        if sid in ratings:
            # Preserve existing rating + n
            if ratings[sid].get("seeded"):
                seeded_total += 1
            continue
        boost = seed_lookup.get(sp["name"].lower())
        rating = float(boost) if boost is not None else 1000.0
        ratings[sid] = {
            "name": sp["name"],
            "rating": rating,
            "n": 0,
            "seeded": boost is not None,
        }
        added += 1
        if boost is not None:
            seeded_total += 1

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "ratings": ratings,
    }
    _atomic_write(RATINGS_PATH, payload)
    return added, seeded_total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap roster size (for testing)")
    ap.add_argument("--sleep", type=float, default=0.05,
                    help="sleep between fallback per-ID requests (seconds)")
    args = ap.parse_args()

    roster = fetch_roster(args.limit, args.sleep)
    if not roster:
        print("FATAL: failed to fetch any species", file=sys.stderr)
        return 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "pokeapi",
        "pokemon": roster,
    }
    _atomic_write(ROSTER_PATH, payload)
    print(f"-> wrote {ROSTER_PATH}  ({len(roster)} species)")

    added, seeded = seed_ratings(roster)
    print(f"-> wrote {RATINGS_PATH}  (+{added} new entries, {seeded} seeded > 1000)")

    # Sanity-check the seed
    rats = _load_existing_ratings().get("ratings") or {}
    by_name = {v.get("name", "").lower(): v for v in rats.values()}
    print("   seed check:")
    for nm, want in SEED_BOOST.items():
        got = by_name.get(nm.lower())
        ok = got and abs(got.get("rating", 0) - want) < 0.01
        mark = "OK" if ok else "MISSING"
        rating = got.get("rating") if got else "?"
        print(f"     [{mark}] {nm:10s} want={want} got={rating}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
