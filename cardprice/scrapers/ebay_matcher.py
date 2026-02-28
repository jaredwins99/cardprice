"""Match parsed eBay listings to dim_cards and ingest into fact_sales.

Uses fuzzy string matching on card name + set name to find the best
matching card_id in the database.
"""

import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher

from sqlalchemy import text
from sqlalchemy.orm import Session

from cardprice.db.session import SessionLocal

logger = logging.getLogger(__name__)

# Minimum similarity score to accept a match
CONFIDENCE_THRESHOLD = 0.55

# eBay marketplace ID — must match a row in dim_marketplaces
EBAY_MARKETPLACE_ID = 2   # Insert into dim_marketplaces if not present


def _normalize(s: str) -> str:
    """Lowercase and strip extra whitespace for comparison."""
    if not s:
        return ""
    return " ".join(s.lower().split())


def _similarity(a: str, b: str) -> float:
    """Return SequenceMatcher ratio between two strings."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _load_candidates(session: Session) -> list[dict]:
    """Load all cards from dim_cards with their set names for matching."""
    rows = session.execute(text("""
        SELECT c.card_id, c.name, c.card_number, c.set_id, c.variant,
               s.name AS set_name
        FROM dim_cards c
        LEFT JOIN dim_sets s ON c.set_id = s.set_id
    """)).fetchall()
    return [
        {
            "card_id": r.card_id,
            "name": r.name,
            "card_number": r.card_number,
            "set_id": r.set_id,
            "variant": r.variant,
            "set_name": r.set_name,
        }
        for r in rows
    ]


def match_listing(parsed_title: dict, session: Session,
                   candidates: list[dict] | None = None) -> tuple[str | None, float]:
    """Match a parsed eBay title to a card_id in dim_cards.

    Args:
        parsed_title: Dict from ebay_title_parser.parse_title() with keys:
            card_name, set_name, card_number, grading_authority, grade, etc.
        session: SQLAlchemy session.
        candidates: Pre-loaded candidate list (optional, avoids repeated queries).

    Returns:
        Tuple of (card_id, confidence). card_id is None if no match
        exceeds the confidence threshold.
    """
    if candidates is None:
        candidates = _load_candidates(session)

    card_name = parsed_title.get("card_name") or ""
    set_name = parsed_title.get("set_name") or ""
    card_number = parsed_title.get("card_number") or ""

    if not card_name:
        return None, 0.0

    best_id = None
    best_score = 0.0

    for cand in candidates:
        score = 0.0

        # Name similarity (weighted most heavily)
        name_sim = _similarity(card_name, cand["name"])
        score += name_sim * 0.50

        # Set name similarity
        if set_name and cand["set_name"]:
            set_sim = _similarity(set_name, cand["set_name"])
            score += set_sim * 0.30
        elif not set_name and not cand["set_name"]:
            score += 0.10  # neutral — neither has set info

        # Card number exact match bonus
        if card_number and cand["card_number"]:
            # Normalize: strip leading zeros for comparison
            def _norm_num(n):
                parts = n.split("/")
                return "/".join(p.lstrip("0") or "0" for p in parts)

            if _norm_num(card_number) == _norm_num(cand["card_number"]):
                score += 0.20  # strong signal
            else:
                # Partial: numerator matches
                try:
                    if _norm_num(card_number).split("/")[0] == _norm_num(cand["card_number"]).split("/")[0]:
                        score += 0.05
                except (IndexError, ValueError):
                    pass

        if score > best_score:
            best_score = score
            best_id = cand["card_id"]

    if best_score >= CONFIDENCE_THRESHOLD:
        return best_id, round(best_score, 4)
    return None, round(best_score, 4)


def _ensure_ebay_marketplace(session: Session) -> int:
    """Ensure eBay exists in dim_marketplaces and return its ID."""
    row = session.execute(
        text("SELECT marketplace_id FROM dim_marketplaces WHERE name = 'eBay'")
    ).fetchone()
    if row:
        return row.marketplace_id
    session.execute(text(
        "INSERT INTO dim_marketplaces (name, source_system) VALUES ('eBay', 'ebay_scraper')"
    ))
    session.flush()
    row = session.execute(
        text("SELECT marketplace_id FROM dim_marketplaces WHERE name = 'eBay'")
    ).fetchone()
    return row.marketplace_id


def ingest_ebay_results(listings: list[dict], session: Session | None = None) -> dict:
    """Match and insert eBay sold listings into fact_sales.

    Args:
        listings: List of dicts from ebay.scrape_sold_listings(), each
            containing at minimum: title, sold_price, sold_date, item_id.
            Titles are parsed via ebay_title_parser.parse_title().
        session: Optional SQLAlchemy session. A new one is created if None.

    Returns:
        Summary dict with counts: total, matched, inserted, skipped, errors.
    """
    from cardprice.scrapers.ebay_title_parser import parse_title

    own_session = session is None
    if own_session:
        session = SessionLocal()

    stats = {"total": len(listings), "matched": 0, "inserted": 0,
             "skipped": 0, "errors": 0}

    try:
        marketplace_id = _ensure_ebay_marketplace(session)

        # Pre-load all candidates once for the batch
        candidates = _load_candidates(session)
        logger.info("Loaded %d card candidates for matching", len(candidates))

        for listing in listings:
            try:
                title = listing.get("title", "")
                parsed = parse_title(title)
                card_id, confidence = match_listing(parsed, session, candidates)

                if card_id:
                    stats["matched"] += 1

                # Build the sold_date
                sold_date = listing.get("sold_date")
                if sold_date is None:
                    sold_date = datetime.now(timezone.utc)

                source_item_id = listing.get("item_id")

                # Skip duplicates (source_item_id unique index)
                if source_item_id:
                    existing = session.execute(
                        text("SELECT 1 FROM fact_sales WHERE source_item_id = :sid"),
                        {"sid": source_item_id},
                    ).fetchone()
                    if existing:
                        stats["skipped"] += 1
                        continue

                # Determine condition from grading or parsed condition
                condition = parsed.get("condition")
                if parsed.get("is_graded") and parsed.get("grade"):
                    # Map numeric grade to condition equivalent
                    try:
                        g = float(parsed["grade"])
                        if g >= 9:
                            condition = "NM"
                        elif g >= 7:
                            condition = "LP"
                        elif g >= 5:
                            condition = "MP"
                        else:
                            condition = "HP"
                    except ValueError:
                        pass

                session.execute(text("""
                    INSERT INTO fact_sales (
                        card_id, marketplace_id, sale_date, sale_price,
                        condition, listing_url, source_item_id,
                        sale_type, shipping_price, grading_authority,
                        grade, raw_title, image_urls, match_confidence
                    ) VALUES (
                        :card_id, :marketplace_id, :sale_date, :sale_price,
                        :condition, :listing_url, :source_item_id,
                        :sale_type, :shipping_price, :grading_authority,
                        :grade, :raw_title, :image_urls, :match_confidence
                    )
                """), {
                    "card_id": card_id,
                    "marketplace_id": marketplace_id,
                    "sale_date": sold_date,
                    "sale_price": listing.get("sold_price"),
                    "condition": condition,
                    "listing_url": listing.get("listing_url"),
                    "source_item_id": source_item_id,
                    "sale_type": listing.get("sale_type"),
                    "shipping_price": listing.get("shipping_price"),
                    "grading_authority": parsed.get("grading_authority"),
                    "grade": parsed.get("grade"),
                    "raw_title": title,
                    "image_urls": [listing["image_url"]] if listing.get("image_url") else None,
                    "match_confidence": confidence if card_id else None,
                })
                stats["inserted"] += 1

            except Exception as e:
                logger.error("Error processing listing %r: %s",
                             listing.get("item_id", "?"), e)
                stats["errors"] += 1

        session.commit()
        logger.info(
            "eBay ingest complete: %d total, %d matched, %d inserted, "
            "%d skipped (dup), %d errors",
            stats["total"], stats["matched"], stats["inserted"],
            stats["skipped"], stats["errors"],
        )

    except Exception as e:
        session.rollback()
        logger.error("eBay ingest batch failed: %s", e)
        raise
    finally:
        if own_session:
            session.close()

    return stats
