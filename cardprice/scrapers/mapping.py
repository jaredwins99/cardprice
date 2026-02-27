"""Cross-source ID mapping: TCGCSV productIds -> dim_cards entries.

Matches TCGCSV groups to dim_sets (by name) and TCGCSV products to
dim_cards (by set + card name + card number). Updates tcg_group_id
and tcg_product_id columns respectively.
"""

import logging
import re
import time
from difflib import SequenceMatcher

import requests
from sqlalchemy import text

from cardprice.config import TCGCSV_BASE_URL, POKEMON_CATEGORY_ID

logger = logging.getLogger(__name__)

GROUPS_URL = f"{TCGCSV_BASE_URL}/tcgplayer/{POKEMON_CATEGORY_ID}/groups"
PRODUCTS_URL = f"{TCGCSV_BASE_URL}/tcgplayer/{POKEMON_CATEGORY_ID}/{{group_id}}/products"

FUZZY_THRESHOLD = 0.80

# Manual alias table for known name mismatches between TCGCSV and pokemontcg.io.
# Maps TCGCSV group name (lowercase) -> pokemontcg.io set_id.
MANUAL_SET_ALIASES = {
    "base set": "base1",
    "base set (shadowless)": "base1",  # same cards, different print run
    "expedition": "ecard1",
    "rumble": "ru1",
    "wotc promo": "basep",
    "best of promos": "bp",
    "nintendo promos": "np",
    "black and white promos": "bwp",
    "diamond and pearl promos": "dpp",
    "hgss promos": "hsp",
    "sm promos": "smp",
    "xy promos": "xyp",
    "sm base set": "sm1",
    "xy base set": "xy1",
    "sv: scarlet & violet 151": "sv3pt5",
    "swsh: sword & shield promo cards": "swshp",
    "sv: scarlet & violet promo cards": "svp",
    "sv01: scarlet & violet base set": "sv1",
    "swsh01: sword & shield base set": "swsh1",
    "swsh02: rebel clash": "swsh2",
    "swsh04: vivid voltage": "swsh4",
    "swsh05: battle styles": "swsh5",
    "swsh08: fusion strike": "swsh8",
    "swsh11: lost origin": "swsh11",
    # McDonald's: TCGCSV uses "Promos", pokemontcg.io uses "Collection"
    "mcdonald's promos 2011": "mcd11",
    "mcdonald's promos 2012": "mcd12",
    "mcdonald's promos 2014": "mcd14",
    "mcdonald's promos 2015": "mcd15",
    "mcdonald's promos 2016": "mcd16",
    "mcdonald's promos 2017": "mcd17",
    "mcdonald's promos 2018": "mcd18",
    "mcdonald's promos 2019": "mcd19",
    "mcdonald's 25th anniversary promos": "mcd21",
    "mcdonald's promos 2022": "mcd22",
}

# Gender symbols that TCGCSV spells out but pokemontcg.io uses unicode
GENDER_MAP = {"♂": " m", "♀": " f", "♂": " m", "♀": " f"}

# TCGCSV appends " - NNN/NNN" or " - PREFNNN" suffixes to many product names.
# This regex strips the trailing card-number portion so the name can match dim_cards.
_PRODUCT_NAME_NUM_SUFFIX_RE = re.compile(
    r"\s*-\s*[A-Za-z]*\d+(?:/\d+)?\s*$"
)

# Parenthetical qualifiers TCGCSV adds that pokemontcg.io does not include in card names.
# e.g. "(Full Art)", "(Alternate Full Art)", "(Delta Species)", "(Secret)", etc.
_PAREN_QUALIFIER_RE = re.compile(
    r"\s*\("
    r"(?:Full Art|Alternate Full Art|Alternate Art|Secret|Cracked Ice Holo"
    r"|Cosmos Holo|Delta Species|Holo|Reverse Holo|Stamped|Shiny"
    r"|Special Art Rare|Special Art|Illustration Rare|Ultra Rare"
    r"|Hyper Rare|Art Rare|Trainer Gallery|Pokemon Day Stamped"
    r"|Staff|Pre-Release Promo|Box Topper|Jumbo|Gold Secret Rare"
    r"|Gold|Rainbow Rare|Rainbow|Character Rare|Character Secret Rare"
    r"|Galarian Gallery|Special Illustration Rare"
    r"|Immersive Rare|Immersive Art Rare|Crown Rare"
    r"|Team Plasma|Team Flare|Team Aqua|Team Magma"
    r"|Prerelease|Pre-Release|Build and Battle|Non-Holo)"
    r"\)\s*",
    re.IGNORECASE,
)

# TCGCSV sometimes disambiguates duplicate names with "(NNN)" parenthetical card numbers.
# e.g. "Roselia (002)" for card #002 - strip these.
_PAREN_NUMBER_RE = re.compile(r"\s*\(\d+\)\s*")

# pokemontcg.io uses "★" and "δ" symbols; TCGCSV spells them out as "Star" / "(Delta Species)".
# After stripping the parenthetical, "Star" may remain as a trailing word in TCGCSV names.
_STAR_SUFFIX_RE = re.compile(r"\bStar\b", re.IGNORECASE)

# Series prefixes TCGCSV prepends to set names (pokemontcg.io does not)
_SET_PREFIX_RE = re.compile(
    r"^(?:SM\s*[-:]?\s*|XY\s*[-:]?\s*|SWSH\d*\s*[-:]?\s*|SV\d*\s*[-:]?\s*"
    r"|BW\s*[-:]?\s*|DP\s*[-:]?\s*|EX\s*[-:]?\s*|HS\s*[-:]?\s*)",
    re.IGNORECASE,
)


def _clean_tcgcsv_product_name(name: str) -> str:
    """Strip TCGCSV-specific name decorations before normalizing.

    Removes:
    - Trailing " - NNN/NNN" or " - PREFIX_NNN (qualifier) [tag]" suffixes
    - Parenthetical art/print qualifiers like "(Full Art)", "(Delta Species)"
    - Square-bracket tags like "[Staff]"
    - "Star" -> "★" alignment (pokemontcg.io uses the symbol, which _normalize strips)
    """
    # Strip everything after " - <card-number-like>" including trailing qualifiers.
    # Handles: "Rillaboom - SWSH006 (Prerelease) [Staff]" -> "Rillaboom"
    #          "Sprigatito - 012/193" -> "Sprigatito"
    name = re.sub(
        r"\s*-\s*[A-Za-z]*\d+(?:/\d+)?(?:\s*[\(\[].*)?$",
        "",
        name,
    )
    # Strip parenthetical qualifiers: "Leafeon V (Full Art)" -> "Leafeon V"
    name = _PAREN_QUALIFIER_RE.sub(" ", name)
    # Strip parenthetical card numbers: "Roselia (002)" -> "Roselia"
    name = _PAREN_NUMBER_RE.sub(" ", name)
    # Strip square-bracket tags: "Some Card [Staff]" -> "Some Card"
    name = re.sub(r"\s*\[.*?\]\s*", " ", name)
    # "Charizard Star" -> "Charizard" (pokemontcg.io uses ★ which gets stripped)
    name = _STAR_SUFFIX_RE.sub("", name)
    return name.strip()


def _normalize(s: str) -> str:
    """Lowercase, strip, remove special chars for comparison."""
    # Replace gender symbols before stripping
    for sym, repl in GENDER_MAP.items():
        s = s.replace(sym, repl)
    s = s.lower().strip()
    # Normalize & to and
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_set_name(s: str) -> str:
    """Normalize a set name: strip series prefixes, then standard normalize."""
    s = _SET_PREFIX_RE.sub("", s.strip())
    return _normalize(s)


def _fuzzy_match(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _fetch_json(url: str) -> dict:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------- SET MAPPING ----------


def map_sets(session) -> dict:
    """Match TCGCSV groups to dim_sets rows by name.

    Updates dim_sets.tcg_group_id for matched sets.
    Returns dict with keys: matched, unmatched_tcg, unmatched_db, total_tcg, total_db.
    """
    # Fetch TCGCSV groups
    data = _fetch_json(GROUPS_URL)
    groups = data.get("results", data) if isinstance(data, dict) else data
    logger.info("Fetched %d TCGCSV groups", len(groups))

    # Load all dim_sets from DB
    rows = session.execute(text("SELECT set_id, name FROM dim_sets")).fetchall()
    db_sets = {row.set_id: row.name for row in rows}
    logger.info("Found %d dim_sets rows in DB", len(db_sets))

    # Build normalized lookup: norm_name -> (set_id, original_name)
    # Use both plain and prefix-stripped normalization for broader matching
    db_lookup = {}
    for set_id, name in db_sets.items():
        db_lookup[_normalize(name)] = (set_id, name)
        db_lookup[_normalize_set_name(name)] = (set_id, name)

    matched = 0
    matched_set_ids = set()
    unmatched_tcg = []

    for group in groups:
        group_id = group.get("groupId")
        group_name = group.get("name", "")

        # 0. Manual alias check first (highest priority)
        alias_set_id = MANUAL_SET_ALIASES.get(group_name.lower().strip())
        if alias_set_id and alias_set_id not in matched_set_ids:
            # Verify alias target exists in DB
            if alias_set_id in db_sets:
                session.execute(
                    text("UPDATE dim_sets SET tcg_group_id = :gid WHERE set_id = :sid"),
                    {"gid": group_id, "sid": alias_set_id},
                )
                matched += 1
                matched_set_ids.add(alias_set_id)
                continue

        # Try both plain and prefix-stripped normalization
        norm_group = None
        for candidate_norm in [_normalize(group_name), _normalize_set_name(group_name)]:
            if candidate_norm in db_lookup:
                candidate_set_id = db_lookup[candidate_norm][0]
                if candidate_set_id not in matched_set_ids:
                    norm_group = candidate_norm
                    break
        if norm_group is None:
            norm_group = _normalize_set_name(group_name)

        # Exact normalized match
        if norm_group in db_lookup and db_lookup[norm_group][0] not in matched_set_ids:
            set_id = db_lookup[norm_group][0]
            session.execute(
                text("UPDATE dim_sets SET tcg_group_id = :gid WHERE set_id = :sid"),
                {"gid": group_id, "sid": set_id},
            )
            matched += 1
            matched_set_ids.add(set_id)
            continue

        # Fuzzy match using prefix-stripped name
        norm_group_stripped = _normalize_set_name(group_name)
        best_score = 0.0
        best_set_id = None
        for norm_name, (set_id, _) in db_lookup.items():
            if set_id in matched_set_ids:
                continue
            score = _fuzzy_match(norm_group_stripped, norm_name)
            if score > best_score:
                best_score = score
                best_set_id = set_id

        if best_score >= FUZZY_THRESHOLD and best_set_id:
            session.execute(
                text("UPDATE dim_sets SET tcg_group_id = :gid WHERE set_id = :sid"),
                {"gid": group_id, "sid": best_set_id},
            )
            matched += 1
            matched_set_ids.add(best_set_id)
            logger.debug(
                "Fuzzy matched TCGCSV '%s' -> dim_sets '%s' (%.2f)",
                group_name, db_sets[best_set_id], best_score,
            )
        else:
            unmatched_tcg.append({"groupId": group_id, "name": group_name})

    session.commit()

    # Check which DB sets still have no tcg_group_id
    unmapped_db = session.execute(
        text("SELECT set_id, name FROM dim_sets WHERE tcg_group_id IS NULL")
    ).fetchall()

    for item in unmatched_tcg:
        logger.warning("Unmatched TCGCSV group: %s (id=%s)", item["name"], item["groupId"])
    for row in unmapped_db:
        logger.warning("Unmapped dim_sets row: %s (%s)", row.name, row.set_id)

    stats = {
        "matched": matched,
        "unmatched_tcg": len(unmatched_tcg),
        "unmatched_db": len(unmapped_db),
        "total_tcg": len(groups),
        "total_db": len(db_sets),
    }
    return stats


# ---------- CARD MAPPING ----------


def _parse_card_number(num_str: str) -> str:
    """Normalize card number: extract the card number portion, strip leading zeros.

    Handles formats like "001/102", "SV001", "TG05/TG30", "1", "SWSH001", etc.
    We extract just the collector number (before slash) and strip leading zeros.
    """
    if not num_str:
        return ""
    num_str = num_str.strip()

    # Take portion before "/" if present (e.g. "001/102" -> "001")
    if "/" in num_str:
        num_str = num_str.split("/")[0].strip()

    # Strip leading zeros from numeric portion
    # Match optional alpha prefix + digits + optional alpha suffix
    # e.g. "SV001" -> "SV1", "001" -> "1", "SM103a" -> "sm103a"
    m = re.match(r"^([A-Za-z]*)0*(\d+)([A-Za-z]?)$", num_str)
    if m:
        prefix, digits, suffix = m.groups()
        result = f"{prefix.lower()}{digits}" if prefix else digits
        if suffix:
            result += suffix.lower()
        return result

    return num_str.lower()


def map_cards(session) -> dict:
    """Match TCGCSV products to dim_cards for all mapped sets.

    Updates dim_cards.tcg_product_id for matched cards.
    Returns dict with keys: matched, unmatched, total_products.
    """
    # Get all sets that have a tcg_group_id
    mapped_sets = session.execute(
        text("SELECT set_id, tcg_group_id FROM dim_sets WHERE tcg_group_id IS NOT NULL")
    ).fetchall()
    logger.info("Processing %d mapped sets", len(mapped_sets))

    total_matched = 0
    total_unmatched = 0
    total_products = 0

    for set_row in mapped_sets:
        set_id = set_row.set_id
        group_id = set_row.tcg_group_id

        # Fetch products for this group
        url = PRODUCTS_URL.format(group_id=group_id)
        try:
            data = _fetch_json(url)
        except requests.RequestException as e:
            logger.error("Failed to fetch products for group %s: %s", group_id, e)
            continue

        products = data.get("results", data) if isinstance(data, dict) else data
        total_products += len(products)

        # Load dim_cards for this set
        cards = session.execute(
            text(
                "SELECT card_id, name, card_number, variant "
                "FROM dim_cards WHERE set_id = :sid"
            ),
            {"sid": set_id},
        ).fetchall()

        # Build lookup: (norm_name, norm_number) -> list of card rows
        card_lookup = {}
        for card in cards:
            norm_name = _normalize(card.name)
            norm_num = _parse_card_number(card.card_number or "")
            key = (norm_name, norm_num)
            card_lookup.setdefault(key, []).append(card)

        set_matched = 0
        set_unmatched = []

        for product in products:
            product_id = product.get("productId")
            product_name = product.get("name", "")
            # TCGCSV extendedData may contain number; also check 'number' field
            ext_data = {
                d["name"]: d["value"]
                for d in product.get("extendedData", [])
            }
            product_number = ext_data.get("Number", product.get("number", ""))

            # Skip sealed products (booster packs, boxes, etc.) — no card equivalent
            if product_number in ("N/A", "", None):
                continue

            # Clean TCGCSV product name: strip number suffixes, art qualifiers, etc.
            clean_product_name = _clean_tcgcsv_product_name(product_name)
            norm_pname = _normalize(clean_product_name)
            norm_pnum = _parse_card_number(str(product_number) if product_number else "")

            # Exact match on (name, number)
            candidates = card_lookup.get((norm_pname, norm_pnum))
            if candidates:
                # If multiple variants, match the first unmatched one
                for card in candidates:
                    session.execute(
                        text(
                            "UPDATE dim_cards SET tcg_product_id = :pid "
                            "WHERE card_id = :cid AND tcg_product_id IS NULL"
                        ),
                        {"pid": product_id, "cid": card.card_id},
                    )
                set_matched += 1
                continue

            # Substring / starts-with match on name, exact on number.
            # Handles cases where one source has a subtitle the other doesn't,
            # e.g. TCGCSV "Professor's Research" vs DB "Professor's Research (Professor Magnolia)".
            substr_card = None
            for (norm_name, norm_num), card_list in card_lookup.items():
                if norm_num != norm_pnum:
                    continue
                if (norm_name.startswith(norm_pname)
                        or norm_pname.startswith(norm_name)):
                    substr_card = card_list[0]
                    break

            if substr_card:
                session.execute(
                    text(
                        "UPDATE dim_cards SET tcg_product_id = :pid "
                        "WHERE card_id = :cid AND tcg_product_id IS NULL"
                    ),
                    {"pid": product_id, "cid": substr_card.card_id},
                )
                set_matched += 1
                continue

            # Fuzzy match on name, exact on number
            best_score = 0.0
            best_card = None
            for (norm_name, norm_num), card_list in card_lookup.items():
                if norm_num != norm_pnum:
                    continue
                score = _fuzzy_match(norm_pname, norm_name)
                if score > best_score:
                    best_score = score
                    best_card = card_list[0]

            if best_score >= FUZZY_THRESHOLD and best_card:
                session.execute(
                    text(
                        "UPDATE dim_cards SET tcg_product_id = :pid "
                        "WHERE card_id = :cid AND tcg_product_id IS NULL"
                    ),
                    {"pid": product_id, "cid": best_card.card_id},
                )
                set_matched += 1
            else:
                set_unmatched.append({
                    "productId": product_id,
                    "name": product_name,
                    "number": product_number,
                })

        total_matched += set_matched
        total_unmatched += len(set_unmatched)

        if set_unmatched:
            logger.warning(
                "Set %s: %d matched, %d unmatched",
                set_id, set_matched, len(set_unmatched),
            )
            for item in set_unmatched[:5]:  # Log first 5 per set
                logger.warning(
                    "  Unmatched product: %s #%s (id=%s)",
                    item["name"], item["number"], item["productId"],
                )
            if len(set_unmatched) > 5:
                logger.warning("  ... and %d more", len(set_unmatched) - 5)

        session.commit()

        # Be polite to the API
        time.sleep(0.3)

    stats = {
        "matched": total_matched,
        "unmatched": total_unmatched,
        "total_products": total_products,
    }
    return stats


# ---------- ORCHESTRATOR ----------


def run_mapping(session) -> None:
    """Run full set + card mapping pipeline. Prints summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("=== Set Mapping ===")
    set_stats = map_sets(session)
    print(f"  TCGCSV groups:  {set_stats['total_tcg']}")
    print(f"  DB sets:        {set_stats['total_db']}")
    print(f"  Matched:        {set_stats['matched']}")
    print(f"  Unmatched TCG:  {set_stats['unmatched_tcg']}")
    print(f"  Unmatched DB:   {set_stats['unmatched_db']}")

    print("\n=== Card Mapping ===")
    card_stats = map_cards(session)
    print(f"  Total products: {card_stats['total_products']}")
    print(f"  Matched:        {card_stats['matched']}")
    print(f"  Unmatched:      {card_stats['unmatched']}")

    if card_stats["total_products"] > 0:
        rate = card_stats["matched"] / card_stats["total_products"] * 100
        print(f"  Match rate:     {rate:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    from cardprice.db.session import SessionLocal

    with SessionLocal() as session:
        run_mapping(session)
