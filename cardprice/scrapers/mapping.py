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

FUZZY_THRESHOLD = 0.85


def _normalize(s: str) -> str:
    """Lowercase, strip, remove special chars for comparison."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


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
    db_lookup = {}
    for set_id, name in db_sets.items():
        norm = _normalize(name)
        db_lookup[norm] = (set_id, name)

    matched = 0
    unmatched_tcg = []

    for group in groups:
        group_id = group.get("groupId")
        group_name = group.get("name", "")
        norm_group = _normalize(group_name)

        # Exact normalized match first
        if norm_group in db_lookup:
            set_id = db_lookup[norm_group][0]
            session.execute(
                text("UPDATE dim_sets SET tcg_group_id = :gid WHERE set_id = :sid"),
                {"gid": group_id, "sid": set_id},
            )
            matched += 1
            continue

        # Fuzzy match
        best_score = 0.0
        best_set_id = None
        for norm_name, (set_id, _) in db_lookup.items():
            score = _fuzzy_match(norm_group, norm_name)
            if score > best_score:
                best_score = score
                best_set_id = set_id

        if best_score >= FUZZY_THRESHOLD and best_set_id:
            session.execute(
                text("UPDATE dim_sets SET tcg_group_id = :gid WHERE set_id = :sid"),
                {"gid": group_id, "sid": best_set_id},
            )
            matched += 1
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
    """Normalize card number: strip leading zeros, lowercase."""
    if not num_str:
        return ""
    num_str = num_str.strip().lower()
    # Strip leading zeros from purely numeric numbers
    if num_str.isdigit():
        return str(int(num_str))
    return num_str


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

            norm_pname = _normalize(product_name)
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
