"""Fetch competitive usage stats from Smogon (via pkmn.github.io) and upsert into dim_smogon_usage."""

import logging
import re
import time

import requests
from sqlalchemy import text

logger = logging.getLogger(__name__)

BASE_URL = "https://pkmn.github.io/smogon/data/stats"
DEFAULT_FORMATS = ["gen9ou", "gen9uu", "gen9ubers", "gen9vgc2025"]
REQUEST_DELAY = 0.5
MAX_RETRIES = 3


def _get_json(url: str) -> dict:
    """GET JSON with retry + exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** (attempt + 1)
            logger.warning("Request failed (%s), retrying in %ds...", e, wait)
            time.sleep(wait)


def _normalize_name(smogon_name: str) -> str:
    """Normalize a Smogon Pokemon name to match dim_pokemon.name.

    Smogon uses forms like 'Landorus-Therian', 'Urshifu-Rapid-Strike'.
    dim_pokemon stores base names like 'Landorus', 'Urshifu'.
    We strip the form suffix for the mapping lookup.
    """
    # Keep the full smogon name in dim_smogon_usage.pokemon_name, but for
    # mapping to dim_pokemon we return the base name (before first hyphen that
    # indicates a form).
    #
    # Exceptions: some Pokemon have hyphens in their base name.
    HYPHEN_BASE_NAMES = {
        "Ho-Oh", "Porygon-Z", "Jangmo-o", "Hakamo-o", "Kommo-o",
        "Wo-Chien", "Chien-Pao", "Ting-Lu", "Chi-Yu",
        "Type: Null",  # colon, not hyphen, but included for completeness
    }
    if smogon_name in HYPHEN_BASE_NAMES:
        return smogon_name

    # For form variants like "Landorus-Therian", extract "Landorus"
    # Common suffixes: -Therian, -Mega, -Alola, -Galar, -Hisui, -Paldea,
    # -Rapid-Strike, -Single-Strike, -Origin, -Crowned, etc.
    form_pattern = re.compile(
        r"^(.+?)-(Mega|Therian|Incarnate|Origin|Altered|Crowned|Unbound|"
        r"Alola|Galar|Hisui|Paldea|Rapid-Strike|Single-Strike|"
        r"Black|White|Dusk-Mane|Dawn-Wings|Ultra|Ice|Shadow|"
        r"Wellspring|Hearthflame|Cornerstone|Bloodmoon|Teal-Mask|"
        r"Small|Large|Super|Average|10%|50%|Complete|"
        r"Attack|Defense|Speed|Heat|Wash|Frost|Fan|Mow|"
        r"Midnight|Dusk|School|Blade|Busted|Hangry|Noice|"
        r"Low-Key|Amped|Eternamax).*$",
        re.IGNORECASE,
    )
    m = form_pattern.match(smogon_name)
    if m:
        return m.group(1)
    return smogon_name


def _parse_stats(data: dict, format_name: str) -> list[dict]:
    """Parse the Smogon JSON into a list of row dicts for upsert.

    The JSON structure from pkmn.github.io is:
    {
      "info": {...},
      "data": {
        "PokemonName": {
          "usage": <float>,            # weighted usage %
          "raw": { "count": <int>, "weight": <float>, ... },
          "real": { "count": <int>, "weight": <float>, ... },
          "lead": { "usage": <float>, "raw": {...}, "real": {...} },
          "gxe": { ... },
          ...
        },
        ...
      }
    }
    """
    pokemon_data = data.get("data", data)
    rows = []

    for name, stats in pokemon_data.items():
        if not isinstance(stats, dict):
            continue

        # Extract usage - could be a float directly or nested
        usage_weighted = None
        usage_raw = None
        usage_real = None
        viability_gxe = None
        count = None
        lead_weighted = None

        # Usage percentage (weighted)
        if isinstance(stats.get("usage"), (int, float)):
            usage_weighted = float(stats["usage"])
        elif isinstance(stats.get("usage"), dict):
            usage_weighted = stats["usage"].get("weighted")
            usage_raw = stats["usage"].get("raw")
            usage_real = stats["usage"].get("real")

        # Raw count
        raw_data = stats.get("raw", {})
        if isinstance(raw_data, dict):
            count = raw_data.get("count")
            if usage_raw is None and "weight" in raw_data:
                usage_raw = raw_data.get("weight")

        # Real usage
        real_data = stats.get("real", {})
        if isinstance(real_data, dict):
            if usage_real is None and "weight" in real_data:
                usage_real = real_data.get("weight")

        # GXE / Viability Ceiling
        gxe_data = stats.get("gxe")
        if isinstance(gxe_data, (int, float)):
            viability_gxe = float(gxe_data)
        elif isinstance(gxe_data, dict):
            # Take the max GXE if it's a distribution
            vals = [v for v in gxe_data.values() if isinstance(v, (int, float))]
            if vals:
                viability_gxe = max(vals)

        # Lead usage
        lead_data = stats.get("lead", {})
        if isinstance(lead_data, dict):
            if isinstance(lead_data.get("usage"), (int, float)):
                lead_weighted = float(lead_data["usage"])
            elif isinstance(lead_data.get("usage"), dict):
                lead_weighted = lead_data["usage"].get("weighted")

        rows.append({
            "pokemon_name": name,
            "format": format_name,
            "usage_weighted": usage_weighted,
            "usage_raw": usage_raw,
            "usage_real": usage_real,
            "viability_gxe": viability_gxe,
            "count": count,
            "lead_weighted": lead_weighted,
        })

    return rows


UPSERT_SQL = text("""
    INSERT INTO dim_smogon_usage
        (pokemon_name, format, usage_weighted, usage_raw, usage_real,
         viability_gxe, count, lead_weighted, fetched_at)
    VALUES
        (:pokemon_name, :format, :usage_weighted, :usage_raw, :usage_real,
         :viability_gxe, :count, :lead_weighted, now())
    ON CONFLICT (pokemon_name, format) DO UPDATE SET
        usage_weighted = EXCLUDED.usage_weighted,
        usage_raw      = EXCLUDED.usage_raw,
        usage_real     = EXCLUDED.usage_real,
        viability_gxe  = EXCLUDED.viability_gxe,
        count          = EXCLUDED.count,
        lead_weighted  = EXCLUDED.lead_weighted,
        fetched_at     = now()
""")

POKEMON_LINK_SQL = text("""
    UPDATE dim_smogon_usage s
    SET pokemon_name = s.pokemon_name  -- no-op, just for the WHERE clause
    FROM dim_pokemon p
    WHERE LOWER(p.name) = LOWER(:base_name)
      AND s.pokemon_name = :smogon_name
      AND s.format = :format
""")


def fetch_smogon_usage(session, formats=None):
    """Fetch Smogon usage stats and upsert into dim_smogon_usage.

    Args:
        session: SQLAlchemy session (from SessionLocal()).
        formats: list of format slugs, e.g. ["gen9ou", "gen9uu"].
                 Defaults to DEFAULT_FORMATS.

    Returns:
        dict with summary: {formats_fetched, total_rows, errors}.
    """
    if formats is None:
        formats = DEFAULT_FORMATS

    total_rows = 0
    errors = []

    for fmt in formats:
        url = f"{BASE_URL}/{fmt}.json"
        logger.info("Fetching Smogon stats: %s", url)

        try:
            data = _get_json(url)
        except Exception as e:
            logger.error("Failed to fetch %s: %s", fmt, e)
            errors.append(fmt)
            continue

        rows = _parse_stats(data, fmt)
        logger.info("  Parsed %d Pokemon for %s", len(rows), fmt)

        for row in rows:
            session.execute(UPSERT_SQL, row)

        session.commit()
        total_rows += len(rows)

        # Build name mapping to dim_pokemon
        mapped = 0
        for row in rows:
            base_name = _normalize_name(row["pokemon_name"])
            result = session.execute(
                text("SELECT 1 FROM dim_pokemon WHERE LOWER(name) = LOWER(:name)"),
                {"name": base_name},
            ).fetchone()
            if result:
                mapped += 1

        logger.info("  Name mapping: %d/%d matched in dim_pokemon", mapped, len(rows))

        if fmt != formats[-1]:
            time.sleep(REQUEST_DELAY)

    summary = {
        "formats_fetched": len(formats) - len(errors),
        "total_rows": total_rows,
        "errors": errors,
    }
    logger.info(
        "Smogon fetch complete: %d formats, %d rows, %d errors",
        summary["formats_fetched"],
        summary["total_rows"],
        len(errors),
    )
    return summary
