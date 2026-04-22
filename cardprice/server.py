"""Minimal HTTP server for phone-based card scanning.

Start: python -m cardprice.server [--port 8888]
Then open http://<your-wsl-ip>:8888 on your phone.

Endpoints:
    GET  /           -> Mobile-friendly upload page (with QR code)
    GET  /qr         -> QR code PNG image of the server URL
    POST /scan       -> Upload image, identify card, return JSON
    POST /scan-url   -> Download image from URL, identify card, return JSON
    POST /scan-page  -> Upload binder page photo, segment & identify cards
    GET  /pending    -> List pending scans awaiting identification
    GET  /history    -> Last 50 scans (resolved + pending) sorted by timestamp desc
    GET  /stats      -> Scanning statistics (counts, methods, confidence, index sizes)
    GET  /price-history/<card_id> -> Last 30 days of market prices as JSON array
    GET  /events/<scan_id> -> SSE stream for scan result updates
    POST /resolve    -> Resolve a pending scan with correct card_id
    POST /resolve-batch -> Resolve multiple pending scans at once
    POST /inventory/add -> Add card to inventory (upsert)
    POST /inventory/remove -> Remove card from inventory (decrement)
    GET  /inventory  -> Current inventory as JSON
    GET  /export     -> Export inventory as CSV attachment
    POST /cart/add   -> Add card to shopping cart (in-memory)
    POST /cart/remove -> Remove card from shopping cart
    GET  /cart       -> Current cart contents as JSON
    GET  /cart/clear -> Clear the entire shopping cart
    GET  /card-image/<card_id> -> Serve local card reference image (PNG)
    GET  /card-image-variant/<card_id>?variants=... -> Card image with variant overlays
    GET  /slide-scan-v7 -> Auto-snap slide scanner UI
    POST /slide-scan/identify -> Identify individually captured card images
    GET  /condition  -> Condition assessment capture UI (4-angle wizard)
    GET  /condition/camera -> Live camera overlay UI for condition capture
    GET  /condition/camera/<card_id> -> Per-card camera UI with card identity pre-filled
    POST /condition/camera/assess -> Receive camera captures, run condition pipeline
    GET  /condition/capture/<card_id> -> Per-card capture UI with card identity pre-filled
    POST /condition/photo/<card_id>/<step> -> Upload one photo, get immediate quality feedback
    GET  /condition/report/<card_id> -> Get combined condition report for a card
    POST /condition/assess -> Receive 4 photos, run condition assessment pipeline
"""

import argparse
import csv
import io
import json
import logging
import os
import re
import socket
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from cardprice.scan_logger import log_scan_result, get_accuracy_summary, get_card_ledger

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/inbox")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PENDING_DIR = Path("data/pending_scans")
PENDING_DIR.mkdir(parents=True, exist_ok=True)

CARD_IMAGES_DIR = Path("data/card_images")
JP_CARD_IMAGES_DIR = Path("data/card_images_jp")
CARD_IMAGES_VARIANTS_DIR = Path("data/card_images_variants")

# Reverse mapping: english card_id -> japanese image path (relative)
_jp_image_index = {}  # type: dict[str, str]

def _load_jp_image_index():
    """Load JP->EN card mapping and build reverse index (EN card_id -> JP image path)."""
    mapping_path = Path("data/jp_en_card_mapping.json")
    if not mapping_path.is_file():
        return
    try:
        mapping = json.loads(mapping_path.read_text())
        for jp_path, en_card_id in mapping.items():
            _jp_image_index[en_card_id] = jp_path
        logger.info("Loaded %d JP card image mappings", len(_jp_image_index))
    except Exception as e:
        logger.warning("Failed to load JP card mapping: %s", e)

_load_jp_image_index()

# Variant image index: card_id -> {variant_type: image_path}
# Maps detected variants to actual downloaded variant reference images.
_variant_image_index = {}  # type: dict[str, dict[str, str]]

# Map detected_variant values to variant image index keys
_DETECTED_VARIANT_TO_IMAGE_KEY = {
    "ex_set_stamp": "stamped",
    "stamped": "stamped",
    "holofoil": "holofoil",
    "reverse_holofoil": "reverse_holo",
}

def _load_variant_image_index():
    """Load variant image index from data/variant_image_index.json."""
    index_path = Path("data/variant_image_index.json")
    if not index_path.is_file():
        return
    try:
        data = json.loads(index_path.read_text())
        for card_id, variants in data.items():
            _variant_image_index[card_id] = variants
        logger.info("Loaded variant image index: %d cards", len(_variant_image_index))
    except Exception as e:
        logger.warning("Failed to load variant image index: %s", e)

_load_variant_image_index()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# ---------------------------------------------------------------------------
# JustTCG real condition prices (cached in-memory)
# ---------------------------------------------------------------------------
_JUSTTCG_DB_PATH = Path("data/justtcg_prices.db")

# Condition name mapping: JustTCG long names -> our short codes
_JUSTTCG_COND_MAP = {
    "Near Mint": "NM",
    "Lightly Played": "LP",
    "Moderately Played": "MP",
    "Heavily Played": "HP",
    "Damaged": "DMG",
}

# Variant -> JustTCG printing name mapping (same as TCGCSV subtypes)
_VARIANT_TO_JUSTTCG_PRINTING = {
    "normal":           "Normal",
    "holofoil":         "Holofoil",
    "reverse_holofoil": "Reverse Holofoil",
    "stamped":          "Reverse Holofoil",
    "promo":            "Normal",
    "shadowless":       "Normal",
    "full_art":         "Holofoil",
    "gold":             "Holofoil",
    "rainbow_rare":     "Holofoil",
    "cosmos":           "Holofoil",
    "cracked_ice":      "Holofoil",
    "1st_edition":      "1st Edition",
}

# Cache: {(tcg_product_id, printing) -> {"NM": price, "LP": price, ...}}
_justtcg_cache: dict | None = None


def _load_justtcg_cache():
    """Load all JustTCG prices into memory, keyed by (tcg_product_id, printing)."""
    global _justtcg_cache
    if _justtcg_cache is not None:
        return
    _justtcg_cache = {}
    if not _JUSTTCG_DB_PATH.exists():
        logger.warning("JustTCG DB not found at %s — using estimated prices only", _JUSTTCG_DB_PATH)
        return
    try:
        import sqlite3
        conn = sqlite3.connect(str(_JUSTTCG_DB_PATH))
        rows = conn.execute(
            "SELECT tcg_product_id, printing, condition, price FROM justtcg_prices"
        ).fetchall()
        conn.close()
        for product_id, printing, condition, price in rows:
            short_cond = _JUSTTCG_COND_MAP.get(condition)
            if not short_cond:
                continue
            key = (product_id, printing)
            if key not in _justtcg_cache:
                _justtcg_cache[key] = {}
            _justtcg_cache[key][short_cond] = float(price)
        logger.info("Loaded %d JustTCG price entries (%d card/printing combos)",
                     len(rows), len(_justtcg_cache))
    except Exception as e:
        logger.error("Failed to load JustTCG prices: %s", e)
        _justtcg_cache = {}


def _get_justtcg_prices(tcg_product_id, printing="Normal"):
    """Look up real condition prices from JustTCG cache.

    Args:
        tcg_product_id: TCGPlayer product ID (int)
        printing: JustTCG printing name (e.g. "Normal", "Holofoil")

    Returns dict with condition short codes as keys and price as values,
    or None if not found.
    """
    _load_justtcg_cache()
    if not tcg_product_id or not _justtcg_cache:
        return None
    # Try exact printing first, then fall back to any printing for this product
    prices = _justtcg_cache.get((tcg_product_id, printing))
    if prices and len(prices) >= 3:  # need at least 3 conditions to be useful
        return prices
    # Fallback: try other common printings
    for fallback_printing in ("Holofoil", "Normal", "Reverse Holofoil"):
        if fallback_printing == printing:
            continue
        prices = _justtcg_cache.get((tcg_product_id, fallback_printing))
        if prices and len(prices) >= 3:
            return prices
    return None

# In-memory shopping cart: card_id -> {quantity, card_name, set_name, market_price,
#                                       condition_prices, image_url, tcgplayer_url}
CART = {}

# Duplicate detection: max Hamming distance to consider a duplicate
_DEDUP_HASH_THRESHOLD = 3

# Server port stored at startup so QR code can encode the full URL
_server_port = 8888


def _compute_phash(image_path):
    """Compute perceptual hash of an image, returning hex string or None."""
    try:
        import imagehash
        from PIL import Image
        img = Image.open(image_path)
        return str(imagehash.phash(img))
    except Exception as e:
        logger.warning("Failed to compute phash for %s: %s", image_path, e)
        return None


def _find_duplicate_scan(phash_hex):
    """Check all scans in pending_scans/ for a matching phash.

    Returns the cached scan dict if a duplicate is found (Hamming distance
    < _DEDUP_HASH_THRESHOLD), otherwise None.
    """
    if not phash_hex:
        return None
    try:
        import imagehash
        query_hash = imagehash.hex_to_hash(phash_hex)
    except Exception:
        return None

    for f in PENDING_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        stored_hex = data.get("phash")
        if not stored_hex:
            continue
        try:
            stored_hash = imagehash.hex_to_hash(stored_hex)
            distance = query_hash - stored_hash
            if distance < _DEDUP_HASH_THRESHOLD:
                logger.info(
                    "Duplicate scan detected (distance=%d): %s matches %s",
                    distance, phash_hex, f.name,
                )
                return data
        except Exception:
            continue
    return None



# ---------------------------------------------------------------------------
# Shared price-lookup SQL + condition-price builder
# ---------------------------------------------------------------------------
# Prefers Normal subtype, falls back to Holofoil, then any subtype.
# This ensures holo-only cards (e.g. Absol) still get a price.
# Unified price lookup: dispatches between English and Japanese tables
# based on the card_id prefix.  Japanese cards have card_id like "jp_<id>"
# from dim_cards_jp.  This UNION ALL means the same SQL works for both
# without per-call-site dispatch.
_PRICE_LOOKUP_SQL = """
    SELECT c.name, s.name as set_name, c.image_small,
           c.tcg_product_id, p.market_price, NULL::text as name_jp
    FROM dim_cards c
    JOIN dim_sets s ON s.set_id = c.set_id
    LEFT JOIN LATERAL (
        SELECT market_price FROM fact_market_prices
        WHERE card_id = c.card_id
        ORDER BY
            CASE subtype_name WHEN 'Normal' THEN 0 WHEN 'Holofoil' THEN 1 ELSE 2 END,
            price_date DESC
        LIMIT 1
    ) p ON true
    WHERE c.card_id = :cid AND :cid NOT LIKE 'jp\\_%' ESCAPE '\\'
    UNION ALL
    SELECT c.name, s.name as set_name, c.image_url as image_small,
           c.tcg_product_id, p.market_price, c.name_jp
    FROM dim_cards_jp c
    JOIN dim_sets_jp s ON s.set_id = c.set_id
    LEFT JOIN LATERAL (
        SELECT market_price FROM fact_market_prices_jp
        WHERE tcg_product_id = c.tcg_product_id
        ORDER BY
            CASE subtype_name WHEN 'Normal' THEN 0 WHEN 'Holofoil' THEN 1 ELSE 2 END,
            price_date DESC
        LIMIT 1
    ) p ON true
    WHERE c.card_id = :cid AND :cid LIKE 'jp\\_%' ESCAPE '\\'
"""


def _is_jp_card_id(card_id: str | None) -> bool:
    """Detect Japanese card_id format (jp_<product_id> from dim_cards_jp)."""
    return bool(card_id and card_id.startswith("jp_"))


# -----------------------------------------------------------------------
# Manual-search indexes (name autocomplete + move fuzzy match)
# -----------------------------------------------------------------------

_NAME_AUTOCOMPLETE_INDEX: list[tuple[str, int]] | None = None
_NAME_AUTOCOMPLETE_LOCK = __import__("threading").Lock()

# Japanese-catalog counterparts (dim_cards_jp).  Separate indexes because
# JP product names carry card-number + variant suffixes we need to strip
# (e.g. "Pineco - 001/098 (Master Ball Pattern)" -> "Pineco" base name).
_NAME_AUTOCOMPLETE_INDEX_JP: list[tuple[str, int]] | None = None
_NAME_AUTOCOMPLETE_LOCK_JP = __import__("threading").Lock()
_SEARCH_BY_NAME_INDEX_JP: dict[str, list[tuple]] | None = None
_SEARCH_INDEX_LOCK_JP = __import__("threading").Lock()

# Authoritative per-card variant catalog, built by
# scripts/scrapers/build_variant_catalog.py from TCGCSV product metadata.
# Format: {card_id: {variant_key: tcg_product_id}}. Enumerates every
# real TCGPlayer SKU per card — covers prime, cracked_ice_holofoil,
# cosmos_holofoil, delta_species, full_art, team_plasma, prerelease,
# staff, winner, JP master/poke ball patterns, etc.
_CARD_VARIANT_CATALOG: dict[str, dict[str, int]] | None = None
_CARD_VARIANT_CATALOG_LOCK = __import__("threading").Lock()
_CARD_VARIANT_CATALOG_PATH = Path(__file__).resolve().parent.parent / \
    "data" / "dim_card_variants.json"


def _get_card_variant_catalog() -> dict[str, dict[str, int]]:
    """Lazy-load the TCGCSV-derived per-card variant catalog."""
    global _CARD_VARIANT_CATALOG
    if _CARD_VARIANT_CATALOG is not None:
        return _CARD_VARIANT_CATALOG
    with _CARD_VARIANT_CATALOG_LOCK:
        if _CARD_VARIANT_CATALOG is not None:
            return _CARD_VARIANT_CATALOG
        try:
            if not _CARD_VARIANT_CATALOG_PATH.is_file():
                logger.warning(
                    "dim_card_variants.json not found at %s — variant picker "
                    "will fall back to variant_tree.json eras only. Run "
                    "scripts/scrapers/build_variant_catalog.py to generate.",
                    _CARD_VARIANT_CATALOG_PATH,
                )
                _CARD_VARIANT_CATALOG = {}
                return _CARD_VARIANT_CATALOG
            with open(_CARD_VARIANT_CATALOG_PATH) as f:
                blob = json.load(f)
            _CARD_VARIANT_CATALOG = blob.get("by_card", {})
            logger.info("Loaded variant catalog: %d cards, %d SKUs",
                        blob.get("card_count", len(_CARD_VARIANT_CATALOG)),
                        blob.get("total_variants", 0))
        except Exception as _e:
            logger.warning("variant catalog load failed: %s", _e)
            _CARD_VARIANT_CATALOG = {}
        return _CARD_VARIANT_CATALOG

# Lowercased-name → list of search-card rows (base_id, name, set_id,
# set_name, rarity, release_date, tcg_product_id). Built once at first
# use from a single DB scan and reused for every /search-cards request.
# Total footprint is ~5MB for 20k cards — cheap in exchange for turning
# Postgres hits into Python dict lookups.
_SEARCH_BY_NAME_INDEX: dict[str, list[tuple]] | None = None
_SEARCH_INDEX_LOCK = __import__("threading").Lock()

# Pokemon-TCG name suffixes that render as glyphs most mobile fonts don't
# ship (delta species δ, prism star ◇, gold star ★). We keep the raw DB
# name for SQL lookups but send a phone-safe display label in the API
# response so "Latias δ" renders as "Latias (Delta Species)" on the client.
_NAME_GLYPH_MAP = {
    "δ": " (Delta Species)",   # δ
    "◇": " (Prism Star)",       # ◇
    "★": " (Gold Star)",        # ★
    "♀": " F",                  # ♀ (Nidoran F)
    "♂": " M",                  # ♂ (Nidoran M)
}


def _display_name(name: str) -> str:
    """Return a phone-safe rendering of a Pokemon name.

    Replaces rare-glyph suffixes (δ/◇/★/♀/♂) with their English
    equivalents. Internal card_ids and DB queries still use the raw
    name — this is display-only.
    """
    if not name:
        return name
    out = name
    for src, dst in _NAME_GLYPH_MAP.items():
        out = out.replace(src, dst)
    # Collapse doubled spaces created by suffix substitutions.
    while "  " in out:
        out = out.replace("  ", " ")
    return out.strip()


# Regex to strip card-number suffix "- 123/456" and variant suffix "(...)"
# from dim_cards_jp.name, leaving just the base Pokemon/Trainer name.
# Example: "Pikachu - 025/165 (Master Ball Pattern)" -> "Pikachu"
_JP_NAME_STRIP_RE = re.compile(
    r"\s*-\s*\S+?/\S+?\s*(?:\([^)]*\))?\s*$"
    r"|\s*\([^)]*\)\s*$"
)


def _jp_base_name(full_name: str) -> str:
    """Extract the base name from a dim_cards_jp.name value.

    TCGPlayer JP naming format: "<Name> - <CardNum>/<SetSize> (<Variant>)".
    We strip the numeric suffix and parenthesized variant to match how
    English search keys on plain Pokemon names.
    """
    if not full_name:
        return ""
    stripped = _JP_NAME_STRIP_RE.sub("", full_name).strip()
    return stripped or full_name.strip()


def _get_name_autocomplete_index_jp() -> list[tuple[str, int]]:
    """JP counterpart of _get_name_autocomplete_index.

    Queries dim_cards_jp, strips card-number + variant suffixes, groups
    by base name.  Returns (base_name, distinct_card_count) sorted by
    count desc.
    """
    global _NAME_AUTOCOMPLETE_INDEX_JP
    if _NAME_AUTOCOMPLETE_INDEX_JP is not None:
        return _NAME_AUTOCOMPLETE_INDEX_JP
    with _NAME_AUTOCOMPLETE_LOCK_JP:
        if _NAME_AUTOCOMPLETE_INDEX_JP is not None:
            return _NAME_AUTOCOMPLETE_INDEX_JP
        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text
            counts: dict[str, int] = {}
            with SessionLocal() as session:
                rows = session.execute(sql_text(
                    "SELECT name, card_id FROM dim_cards_jp WHERE name IS NOT NULL"
                )).fetchall()
            for full_name, _cid in rows:
                base = _jp_base_name(str(full_name))
                if not base:
                    continue
                counts[base] = counts.get(base, 0) + 1
            _NAME_AUTOCOMPLETE_INDEX_JP = sorted(
                counts.items(), key=lambda kv: (-kv[1], kv[0])
            )
            logger.info(
                "name-autocomplete (JP): %d distinct base names",
                len(_NAME_AUTOCOMPLETE_INDEX_JP),
            )
        except Exception as _e:
            logger.warning("name-autocomplete JP DB load failed: %s", _e)
            _NAME_AUTOCOMPLETE_INDEX_JP = []
        return _NAME_AUTOCOMPLETE_INDEX_JP


def _get_search_by_name_index_jp() -> dict[str, list[tuple]]:
    """JP counterpart of _get_search_by_name_index.

    Keys on the stripped base name (so "pikachu" matches all Pikachu JP
    products).  Entries carry the JP card_id (jp_<id>), the full
    TCGPlayer name (keeps the variant suffix for display), set info,
    image URL, and tcg_product_id.
    """
    global _SEARCH_BY_NAME_INDEX_JP
    if _SEARCH_BY_NAME_INDEX_JP is not None:
        return _SEARCH_BY_NAME_INDEX_JP
    with _SEARCH_INDEX_LOCK_JP:
        if _SEARCH_BY_NAME_INDEX_JP is not None:
            return _SEARCH_BY_NAME_INDEX_JP
        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text
            out: dict[str, list[tuple]] = {}
            with SessionLocal() as session:
                rows = session.execute(sql_text(
                    "SELECT c.card_id, c.name, c.name_jp, c.set_id, "
                    "       s.name AS set_name, c.rarity, s.published_on, "
                    "       c.tcg_product_id, c.image_url "
                    "FROM dim_cards_jp c "
                    "LEFT JOIN dim_sets_jp s ON c.set_id = s.set_id "
                    "WHERE c.name IS NOT NULL"
                )).fetchall()
            for r in rows:
                card_id = r[0]
                full_name = r[1] or ""
                base = _jp_base_name(full_name)
                if not base:
                    continue
                entry = (
                    card_id,        # 0 jp_<id>
                    full_name,      # 1 full TCGPlayer name w/ variant suffix
                    r[2],           # 2 name_jp (kanji, may be None)
                    r[3],           # 3 set_id
                    r[4],           # 4 set_name
                    r[5],           # 5 rarity
                    r[6],           # 6 published_on (date)
                    r[7],           # 7 tcg_product_id
                    r[8],           # 8 image_url
                )
                out.setdefault(base.lower(), []).append(entry)
            # Sort each bucket by published_on desc (None last).
            for lst in out.values():
                lst.sort(key=lambda e: (e[6] or ""), reverse=True)
            _SEARCH_BY_NAME_INDEX_JP = out
            logger.info(
                "search-name index (JP): %d distinct names, %d rows",
                len(out), sum(len(v) for v in out.values()),
            )
        except Exception as _e:
            logger.warning("search-name JP index build failed: %s", _e)
            _SEARCH_BY_NAME_INDEX_JP = {}
        return _SEARCH_BY_NAME_INDEX_JP


def _get_name_autocomplete_index() -> list[tuple[str, int]]:
    """Return list of (name, printing_count) sorted by count desc.

    Uses dim_cards as the source of truth so it always reflects the
    current catalog. Cached on first call; cache rebuilds on server
    restart.
    """
    global _NAME_AUTOCOMPLETE_INDEX
    if _NAME_AUTOCOMPLETE_INDEX is not None:
        return _NAME_AUTOCOMPLETE_INDEX
    with _NAME_AUTOCOMPLETE_LOCK:
        if _NAME_AUTOCOMPLETE_INDEX is not None:
            return _NAME_AUTOCOMPLETE_INDEX
        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text
            counts: dict[str, int] = {}
            with SessionLocal() as session:
                rows = session.execute(sql_text(
                    "SELECT name, COUNT(DISTINCT card_id) "
                    "FROM dim_cards WHERE name IS NOT NULL GROUP BY name"
                )).fetchall()
            for name, count in rows:
                if name:
                    counts[str(name)] = int(count)
            _NAME_AUTOCOMPLETE_INDEX = sorted(
                counts.items(), key=lambda kv: (-kv[1], kv[0])
            )
        except Exception as _e:
            logger.warning("name-autocomplete DB load failed: %s", _e)
            _NAME_AUTOCOMPLETE_INDEX = []
        return _NAME_AUTOCOMPLETE_INDEX


def _get_search_by_name_index() -> dict[str, list[tuple]]:
    """Return {lowered_name: [(base_id, name, set_id, set_name, rarity,
    release_date, tcg_product_id), ...]}. Each list is sorted by
    release_date desc.

    Built from a single dim_cards + dim_sets join on first access and
    cached. Invalidated only by server restart — dim_cards updates are
    rare (batch ingests), so a stale index is acceptable until next boot.
    """
    global _SEARCH_BY_NAME_INDEX
    if _SEARCH_BY_NAME_INDEX is not None:
        return _SEARCH_BY_NAME_INDEX
    with _SEARCH_INDEX_LOCK:
        if _SEARCH_BY_NAME_INDEX is not None:
            return _SEARCH_BY_NAME_INDEX
        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text
            out: dict[str, list[tuple]] = {}
            with SessionLocal() as session:
                rows = session.execute(sql_text(
                    "SELECT c.card_id, c.name, c.set_id, s.name AS set_name, "
                    "       c.rarity, s.release_date, c.tcg_product_id "
                    "FROM dim_cards c "
                    "LEFT JOIN dim_sets s ON c.set_id = s.set_id "
                    "WHERE c.name IS NOT NULL"
                )).fetchall()
            for r in rows:
                base_id = (r[0] or "").split("/", 1)[0]
                if not base_id:
                    continue
                raw_name = r[1] or ""
                entry = (base_id, r[1], r[2], r[3], r[4], r[5], r[6])
                # Index under the raw name *and* the display-safe name
                # (e.g. "Nidoran ♀" and "Nidoran F") so a client-typed
                # ASCII query matches the same row as its raw form.
                keys = {raw_name.lower()}
                display = _display_name(raw_name)
                if display:
                    keys.add(display.lower())
                for key in keys:
                    out.setdefault(key, []).append(entry)
            # Sort each bucket by release_date desc (None last), stable on id.
            for k, lst in out.items():
                lst.sort(key=lambda e: (e[5] is None, e[5] or "", e[0]),
                         reverse=False)
                # flip: reverse=False with tuple would put None last, but
                # we want newest first -> resort:
                lst.sort(key=lambda e: (e[5] or ""), reverse=True)
            _SEARCH_BY_NAME_INDEX = out
            logger.info("search-name index: %d distinct names, %d rows",
                        len(out), sum(len(v) for v in out.values()))
        except Exception as _e:
            logger.warning("search-name index build failed: %s", _e)
            _SEARCH_BY_NAME_INDEX = {}
        return _SEARCH_BY_NAME_INDEX


def _rarity_canonical_printing(rarity: str | None) -> str:
    """Map a card's rarity to its canonical JustTCG printing value.

    "Rare Holo", "Trainer Gallery Rare Holo", "Illustration Rare", "Full
    Art", "Hyper/Rainbow/Gold/Secret", V/VMAX/VSTAR/GX/EX/ex etc. are
    printed on holofoil stock — use the Holofoil JustTCG row for them.
    Everything else (Common, Uncommon, non-holo Rare, Promo) defaults to
    Normal.
    """
    if not rarity:
        return "Normal"
    rlow = rarity.lower()
    _HOLO_KEYWORDS = (
        "holo", "illustration", "full art", "hyper", "rainbow",
        "gold", "secret", "amazing", "radiant", "prism star",
        "shining", "shiny", "ultra", "special illustration",
        "break", "lv.x", "lvx", "lv x",
    )
    for kw in _HOLO_KEYWORDS:
        if kw in rlow:
            return "Holofoil"
    # Short token-match for suffix markers like "GX", "V", "VMAX", "EX".
    toks = rlow.replace("-", " ").split()
    for t in ("gx", "ex", "v", "vmax", "vstar"):
        if t in toks:
            return "Holofoil"
    return "Normal"


def _batch_mp_prices(entries: list[tuple]) -> dict[str, tuple[float | None, bool]]:
    """Build (mp_price, estimated) for each base_id in one pass.

    Uses each card's rarity (entries[4]) to pick the JustTCG *printing*
    (Normal vs Holofoil) so holo cards don't fall through to an
    unrelated-Normal-row + bad estimate. Falls back to NM market price
    (matched by printing) * 0.62 when JustTCG has no MP row.
    """
    if not entries:
        return {}
    base_ids = [e[0] for e in entries]
    tcg_pids = [e[6] for e in entries if e[6] is not None]

    # Bulk NM market prices from fact_market_prices. Query by the indexed
    # card_id column directly (no SPLIT_PART wrapper) — we know every
    # dim_cards row stores card_id as "<base>/normal", so mapping
    # base_id → full_id is a string append, not a substring call.
    nm_by_base: dict[str, float | None] = {}
    full_to_base = {b + "/normal": b for b in base_ids}
    full_ids = list(full_to_base.keys())
    try:
        from cardprice.db.session import SessionLocal
        from sqlalchemy import text as sql_text
        with SessionLocal() as session:
            rows = session.execute(sql_text(
                "SELECT card_id, "
                "       MIN(market_price) FILTER (WHERE subtype_name='Normal') AS nm_normal, "
                "       MIN(market_price) FILTER (WHERE subtype_name='Holofoil') AS nm_holo, "
                "       MIN(market_price) AS nm_any "
                "FROM fact_market_prices "
                "WHERE card_id = ANY(:ids) "
                "GROUP BY card_id"
            ), {"ids": full_ids}).fetchall()
        for full_id, p_norm, p_holo, p_any in rows:
            base_id = full_to_base.get(full_id, full_id.split("/", 1)[0])
            nm_by_base[base_id] = float(p_norm or p_holo or p_any) if (p_norm or p_holo or p_any) else None
    except Exception as _e:
        logger.debug("bulk market_price lookup failed: %s", _e)

    # Bulk JustTCG MP rows. Pull BOTH printings so we can pick per-card.
    jtcg_mp: dict[tuple[int, str], float] = {}
    if tcg_pids:
        try:
            import sqlite3
            db_path = Path(__file__).resolve().parent.parent / "data" / "justtcg_prices.db"
            if db_path.is_file():
                con = sqlite3.connect(str(db_path))
                try:
                    placeholders = ",".join("?" * len(tcg_pids))
                    for row in con.execute(
                        "SELECT tcg_product_id, price, printing "
                        "FROM justtcg_prices "
                        f"WHERE tcg_product_id IN ({placeholders}) "
                        "AND condition = 'Moderately Played' "
                        "AND printing IN ('Normal','Holofoil')",
                        tcg_pids,
                    ):
                        jtcg_mp[(row[0], row[2])] = float(row[1])
                finally:
                    con.close()
        except Exception as _e:
            logger.debug("bulk JustTCG MP lookup failed: %s", _e)

    # We also pulled holo NM alongside normal NM earlier — reshape the
    # existing nm_by_base lookup so we can pick the printing-appropriate
    # fallback. Refetch per-printing breakdown.
    nm_by_base_printing: dict[tuple[str, str], float] = {}
    try:
        from cardprice.db.session import SessionLocal
        from sqlalchemy import text as sql_text
        with SessionLocal() as session:
            rows = session.execute(sql_text(
                "SELECT card_id, subtype_name, MIN(market_price) "
                "FROM fact_market_prices "
                "WHERE card_id = ANY(:ids) AND subtype_name IN ('Normal','Holofoil') "
                "GROUP BY card_id, subtype_name"
            ), {"ids": full_ids}).fetchall()
        for full_id, subtype, price in rows:
            base_id = full_to_base.get(full_id, full_id.split("/", 1)[0])
            if price is not None:
                nm_by_base_printing[(base_id, subtype)] = float(price)
    except Exception as _e:
        logger.debug("per-printing NM lookup failed: %s", _e)

    out: dict[str, tuple[float | None, bool]] = {}
    for e in entries:
        base_id, tcg_pid = e[0], e[6]
        rarity = e[4] if len(e) > 4 else None
        printing = _rarity_canonical_printing(rarity)
        # Prefer the canonical printing; fall back to the other.
        mp_price = None
        if tcg_pid is not None:
            mp_price = jtcg_mp.get((tcg_pid, printing))
            if mp_price is None:
                other = "Normal" if printing == "Holofoil" else "Holofoil"
                mp_price = jtcg_mp.get((tcg_pid, other))
        if mp_price is not None:
            out[base_id] = (mp_price, False)
            continue
        # Estimate from the printing-appropriate NM
        nm = (nm_by_base_printing.get((base_id, printing))
              or nm_by_base_printing.get((base_id, "Holofoil" if printing == "Normal" else "Normal"))
              or nm_by_base.get(base_id))
        if nm is not None:
            out[base_id] = (round(nm * 0.62, 2), True)
        else:
            out[base_id] = (None, False)
    return out


def _search_cards_by_name_and_move(name: str, move: str, limit: int) -> list[dict]:
    """Return card printings matching an exact Pokemon name + optional move.

    Name match: case-insensitive equality (user picked from autocomplete).
    Move match: fuzzy substring against structured_attacks.json attack/ability
    names for that card. When both filters are applied and exactly one card
    matches, the frontend can treat it as a resolved identification.

    Each result carries the base card_id (no variant suffix); the client
    fetches variants separately to populate the variant picker.
    """
    try:
        from cardprice.ml import _load_structured_attacks
    except Exception as _e:
        logger.warning("search-cards import failed: %s", _e)
        return []

    structured = _load_structured_attacks()
    name_q = name.strip().lower()
    index = _get_search_by_name_index()

    # Prefix match over the name index. Exact matches always included.
    # For broader prefix hits, skip keys whose suffix after the query is
    # a mechanic token — "Pikachu ex", "Pikachu V", "Pikachu VMAX",
    # "Pikachu-GX", tag-team "Pikachu & Zekrom-GX" are different cards
    # with different moves. Keep subseries markers like δ / ◇ / ★ and
    # "(Delta Species)" / "(Prism Star)" / "(Gold Star)".
    _MECHANIC_RE = re.compile(
        r"(?:\bex\b|\bgx\b|\bv\b|\bvmax\b|\bvstar\b|\bv-union\b|"
        r"\bbreak\b|\blv\.?\s*x\b|&)",
        re.IGNORECASE,
    )
    raw_entries: list[tuple] = []
    exact = index.get(name_q, [])
    if exact:
        raw_entries.extend(exact)
    for key, bucket in index.items():
        if key == name_q:
            continue
        if not key.startswith(name_q):
            continue
        suffix = key[len(name_q):]
        # Normalize hyphen to space so "-ex" is tokenized as "ex".
        if _MECHANIC_RE.search(suffix.replace("-", " ")):
            continue
        raw_entries.extend(bucket)

    # First pass: collect matching base_ids (dedupe, apply move filter).
    seen_base: set[str] = set()
    matched_entries: list[tuple] = []
    matched_moves_by_base: dict[str, list[str]] = {}
    for entry in raw_entries:
        base_id = entry[0]
        if base_id in seen_base:
            continue
        seen_base.add(base_id)

        if move:
            data = structured.get(base_id, {}) or {}
            moves_list = [a.get("name", "") for a in data.get("attacks", [])]
            moves_list += [a.get("name", "") for a in data.get("abilities", [])]
            matched = [m for m in moves_list if m and move in m.lower()]
            if not matched:
                continue
            matched_moves_by_base[base_id] = matched
        matched_entries.append(entry)
        if len(matched_entries) >= limit:
            break

    # Second pass: batched MP price lookup for all surviving candidates.
    mp_lookup = _batch_mp_prices(matched_entries)

    results = []
    for entry in matched_entries:
        base_id, card_name, set_id, set_name, rarity, _rel, tcg_pid = entry
        image_url = None
        if set_id and "-" in base_id:
            num = base_id.rsplit("-", 1)[-1]
            image_url = f"https://images.pokemontcg.io/{set_id}/{num}.png"

        mp_price, mp_est = mp_lookup.get(base_id, (None, False))

        results.append({
            "card_id": base_id,
            "name": _display_name(card_name),
            "raw_name": card_name,
            "set_id": set_id,
            "set_name": set_name,
            "rarity": rarity,
            "image_url": image_url,
            "matched_moves": matched_moves_by_base.get(base_id, []),
            "mp_price": mp_price,
            "mp_estimated": mp_est,
        })

    return results


def _search_cards_by_name_jp(name: str, limit: int) -> list[dict]:
    """Search JP catalog (dim_cards_jp) for a Pokemon name.

    User types an English name (the Pokemon name TCGPlayer uses for JP
    products).  Returns JP product rows with JP images + JP market prices.
    JP has no structured_attacks data, so no move filtering.
    """
    name_q = name.strip().lower()
    index = _get_search_by_name_index_jp()

    # Prefix match over the base-name keys, with exact matches first.
    raw_entries: list[tuple] = []
    exact = index.get(name_q, [])
    if exact:
        raw_entries.extend(exact)
    _MECHANIC_RE = re.compile(
        r"(?:\bex\b|\bgx\b|\bv\b|\bvmax\b|\bvstar\b|\bv-union\b|"
        r"\bbreak\b|\blv\.?\s*x\b|&)",
        re.IGNORECASE,
    )
    for key, bucket in index.items():
        if key == name_q:
            continue
        if not key.startswith(name_q):
            continue
        suffix = key[len(name_q):]
        if _MECHANIC_RE.search(suffix.replace("-", " ")):
            continue
        raw_entries.extend(bucket)

    # Dedupe by card_id (same JP product could be indexed under both
    # name variants if we ever alias).
    seen: set[str] = set()
    matched: list[tuple] = []
    for entry in raw_entries:
        card_id = entry[0]
        if card_id in seen:
            continue
        seen.add(card_id)
        matched.append(entry)
        if len(matched) >= limit:
            break

    # Batched price lookup from fact_market_prices_jp.
    pids = [e[7] for e in matched if e[7]]
    price_by_pid: dict[int, float] = {}
    if pids:
        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text
            with SessionLocal() as session:
                # Prefer Normal subtype, fall back to Holofoil, then any.
                rows = session.execute(sql_text("""
                    SELECT DISTINCT ON (tcg_product_id)
                           tcg_product_id, market_price
                    FROM fact_market_prices_jp
                    WHERE tcg_product_id = ANY(:pids)
                    ORDER BY tcg_product_id,
                             CASE subtype_name WHEN 'Normal' THEN 0
                                               WHEN 'Holofoil' THEN 1
                                               ELSE 2 END,
                             price_date DESC
                """), {"pids": pids}).fetchall()
            for pid, mp in rows:
                if mp is not None:
                    price_by_pid[int(pid)] = float(mp)
        except Exception as _e:
            logger.warning("JP batch price lookup failed: %s", _e)

    results = []
    for e in matched:
        (card_id, full_name, name_jp, set_id, set_name, rarity,
         _rel, tcg_pid, image_url) = e
        mp_price = price_by_pid.get(int(tcg_pid)) if tcg_pid else None
        results.append({
            "card_id": card_id,
            "name": full_name,
            "raw_name": full_name,
            "name_jp": name_jp,
            "set_id": set_id,
            "set_name": set_name,
            "rarity": rarity,
            "image_url": image_url,
            "matched_moves": [],  # not available for JP
            "mp_price": mp_price,
            "mp_estimated": False,
        })
    return results


# -----------------------------------------------------------------------
# Card detail page (big image + variant picker + pricing)
# -----------------------------------------------------------------------

def _decompose_variants(raw: list[str], rarity: str | None = None) -> dict:
    """Split a card's possible variant strings into orthogonal axes.

    Each raw variant is parsed into (edition, finish, modifier):
      - edition: normal | 1st_edition | shadowless | unlimited
      - finish: non_holo | holofoil | reverse_holofoil | full_art
                | gold | rainbow_rare
      - modifier: None | ex_set_stamp | prerelease | staff | ...

    Returned dict has axis-specific ordered lists plus the raw set for
    the client to validate its composition against.

    If rarity indicates the card is intrinsically holo (Trainer Gallery
    Rare Holo, Illustration Rare, Full Art, Hyper/Rainbow/Gold/Secret,
    Rare Holo V/VMAX/VSTAR/EX, etc.), "non_holo" is dropped from the
    finish options. These cards are never printed flat.
    """
    # Rarities that imply holofoil/foil-only printings. Match case-insensitive
    # substring so "Trainer Gallery Rare Holo" / "Rare Holo V" / "Rare Holo
    # VMAX" all map to holo-only without listing each explicitly.
    _HOLO_ONLY_KEYWORDS = (
        "holo", "illustration", "full art", "hyper",
        "rainbow", "gold", "secret", "amazing", "radiant",
        "prism star", "shining", "shiny",
        "lv.x", "lvx", "lv x",
        "break", "gx", "ex", "v", "vmax", "vstar",
        "special illustration", "ultra",
    )
    holo_only = False
    if rarity:
        rlow = rarity.lower().strip()
        # Single-letter matches like "V" need word-boundary to avoid
        # "Common"/"Uncommon"/etc false positives. Use exact match for
        # short tokens and substring for long ones.
        for kw in _HOLO_ONLY_KEYWORDS:
            if len(kw) <= 3:
                # token-match
                tokens = rlow.replace("-", " ").split()
                if kw in tokens:
                    holo_only = True
                    break
            else:
                if kw in rlow:
                    holo_only = True
                    break

    editions: list[str] = []
    finishes: list[str] = []
    modifiers: list[str] = []

    def _add(lst, v):
        if v not in lst:
            lst.append(v)

    # Variant keys that are a visible finish/foil rather than a modifier
    _FINISH_KEYS = {
        "holofoil", "reverse_holofoil", "full_art", "gold", "rainbow_rare",
        "secret", "secret_rare", "hyper_rare", "illustration_rare",
        "special_illustration_rare", "alternate_full_art",
        "alternate_art_secret", "cosmos_holofoil", "cracked_ice_holofoil",
        "shiny", "non_holo", "foil",
        # JP finish patterns
        "poke_ball_pattern", "master_ball_pattern", "energy_symbol_pattern",
        "dusk_ball", "love_ball", "friend_ball", "quick_ball", "poke_ball",
    }
    # Variant keys that are a stamp/modifier layered on a printing
    _MODIFIER_KEYS = {
        "ex_set_stamp", "prerelease", "staff", "pc_exclusive",
        "pokemon_center_exclusive", "winner", "crosshatch", "wc", "ditto",
        "tru", "black_star_promo", "promo", "ghost_stamp", "grey_stamp",
        "movie_exclusive", "mcdonalds", "pokemon_day", "jumbo",
        "black_dot_error", "team_rocket", "omega",
    }

    for v in raw:
        vl = v.lower()
        edition = "normal"
        finish = "non_holo"
        modifier = None

        if vl.startswith("1st_edition"):
            edition = "1st_edition"
            rest = vl[len("1st_edition"):].lstrip("_")
        elif vl.startswith("shadowless"):
            edition = "shadowless"
            rest = vl[len("shadowless"):].lstrip("_")
        elif vl.startswith("unlimited"):
            edition = "unlimited"
            rest = vl[len("unlimited"):].lstrip("_")
        else:
            rest = vl

        if rest in ("holofoil", "holo"):
            finish = "holofoil"
        elif rest in _FINISH_KEYS:
            finish = rest
        elif rest in ("", "normal"):
            # For holo-only rarities, TCGCSV lists the canonical product
            # as just the card name ("Charizard") — that product is the
            # Holofoil printing, not a non-holo one. Map accordingly.
            finish = "holofoil" if holo_only else "non_holo"
        elif rest in _MODIFIER_KEYS:
            modifier = rest
        else:
            # Unknown variant key — treat as modifier so it's still selectable.
            modifier = rest

        _add(editions, edition)
        _add(finishes, finish)
        if modifier:
            _add(modifiers, modifier)

    # Canonicalize ordering for nicer UX
    EDITION_ORDER = ["normal", "unlimited", "1st_edition", "shadowless"]
    FINISH_ORDER = [
        "non_holo", "holofoil", "reverse_holofoil",
        "cosmos_holofoil", "cracked_ice_holofoil",
        "full_art", "alternate_full_art", "illustration_rare",
        "special_illustration_rare",
        "secret_rare", "alternate_art_secret", "secret",
        "gold", "rainbow_rare", "hyper_rare", "shiny",
        # JP finish patterns
        "poke_ball_pattern", "master_ball_pattern", "energy_symbol_pattern",
        "poke_ball", "dusk_ball", "love_ball", "friend_ball", "quick_ball",
        "foil",
    ]
    editions = sorted(editions, key=lambda e: (EDITION_ORDER.index(e)
                                               if e in EDITION_ORDER
                                               else len(EDITION_ORDER)))
    finishes = sorted(finishes, key=lambda f: (FINISH_ORDER.index(f)
                                               if f in FINISH_ORDER
                                               else len(FINISH_ORDER)))
    # Drop non_holo from the finish set when the card's rarity implies
    # a foil-only printing. Keeps the dropdown honest.
    if holo_only and "non_holo" in finishes:
        finishes = [f for f in finishes if f != "non_holo"]
    # Guarantee at least one finish — if modifier-only variants like
    # delta_species leave the finish axis empty, the card is still a
    # physical object the user needs to pick a finish for. Fall back to
    # holofoil for holo-only rarities, non_holo otherwise.
    if not finishes:
        finishes = ["holofoil" if holo_only else "non_holo"]
    return {
        "raw": list(raw),
        "edition": editions,
        "finish": finishes,
        "modifier": modifiers,
    }


def _card_detail_payload(card_id: str) -> dict | None:
    """Return the full detail payload for a card_id (bare or with variant).

    Shape mirrors the scan-result fields (name, set_name, image_url,
    market_price, condition_prices, tcgplayer_url) plus possible variants.
    """
    base_id = card_id.split("/", 1)[0]
    full_id = card_id if "/" in card_id else f"{base_id}/normal"

    name = None
    set_name = None
    market_price = None
    image_url = None
    condition_prices = None
    tcg_product_id = None
    rarity = None

    try:
        from cardprice.db.session import SessionLocal
        from sqlalchemy import text as sql_text
        with SessionLocal() as session:
            row = session.execute(
                sql_text(_PRICE_LOOKUP_SQL), {"cid": full_id}
            ).fetchone()
            if not row:
                # Fall back to bare id
                row = session.execute(
                    sql_text(_PRICE_LOOKUP_SQL), {"cid": base_id}
                ).fetchone()
            if row:
                name = row.name
                set_name = row.set_name
                image_url = row.image_small
                tcg_product_id = row.tcg_product_id
                market_price = float(row.market_price) if row.market_price else None
                variant_suffix = full_id.rsplit("/", 1)[-1] if "/" in full_id else "normal"
                condition_prices = _build_condition_prices(
                    row.market_price,
                    tcg_product_id=tcg_product_id,
                    variant=variant_suffix,
                )
            # Separate query for rarity (not in the price SQL SELECT list).
            rarity_row = session.execute(sql_text(
                "SELECT rarity FROM dim_cards WHERE card_id = :cid LIMIT 1"
            ), {"cid": full_id}).fetchone()
            if not rarity_row:
                rarity_row = session.execute(sql_text(
                    "SELECT rarity FROM dim_cards WHERE card_id LIKE :p LIMIT 1"
                ), {"p": base_id + "/%"}).fetchone()
            if rarity_row:
                rarity = rarity_row[0]
    except Exception as _e:
        logger.warning("card-info DB lookup failed for %s: %s", card_id, _e)
        return None

    if name is None:
        return None

    # Per-card variant catalog (TCGCSV-derived) is the authoritative list:
    # it enumerates every TCGPlayer product SKU for this card, which means
    # prime / cracked_ice_holofoil / delta_species / cosmos_holofoil /
    # prerelease / staff / alternate_full_art / master_ball_pattern / etc.
    # all surface correctly when the data has them.
    catalog = _get_card_variant_catalog()
    possible: list[str] = []
    if base_id in catalog:
        possible = list(catalog[base_id].keys())

    # Fallback to variant_tree.json-derived possible_variants when the
    # catalog doesn't have this card (promo/freshly-added sets). This
    # keeps the picker functional for cards the TCGCSV scrape missed.
    if not possible:
        try:
            from cardprice.ml.card_attributes import get_card_attrs
            attrs = get_card_attrs(full_id) or get_card_attrs(f"{base_id}/normal")
            if attrs:
                possible = list(attrs.possible_variants or [])
        except Exception as _e:
            logger.debug("possible_variants lookup failed for %s: %s", base_id, _e)

    variants = _decompose_variants(possible, rarity=rarity)

    tcgplayer_url = None
    if tcg_product_id:
        tcgplayer_url = f"https://www.tcgplayer.com/product/{tcg_product_id}"

    # Inline sales — cheap SQLite query, saves a second network round-trip
    # from the client. Capped at 100 rows to keep the payload reasonable.
    sales = []
    if tcg_product_id:
        try:
            import sqlite3
            db_path = Path(__file__).resolve().parent.parent / "data" / "tcgplayer_sales.db"
            if db_path.is_file():
                con = sqlite3.connect(str(db_path))
                try:
                    for sale_date, price, cond, qty, printing in con.execute(
                        "SELECT sale_date, sale_price, condition, quantity, printing "
                        "FROM tcgplayer_sales WHERE tcg_product_id=? "
                        "ORDER BY sale_date DESC LIMIT 100", (tcg_product_id,)
                    ):
                        sales.append({
                            "date": (sale_date or "")[:10],
                            "datetime": sale_date,
                            "price": float(price) if price is not None else None,
                            "condition": cond,
                            "quantity": qty,
                            "printing": printing,
                        })
                finally:
                    con.close()
        except Exception as _e:
            logger.debug("inline sales lookup failed for %s: %s", card_id, _e)

    return {
        "card_id": base_id,
        "name": _display_name(name),
        "raw_name": name,
        "set_name": set_name,
        "image_url": image_url,
        "market_price": market_price,
        "condition_prices": condition_prices,
        "tcgplayer_url": tcgplayer_url,
        "tcg_product_id": tcg_product_id,
        "rarity": rarity,
        "variants": variants,
        "sales": sales,
    }


CARD_DETAIL_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Card Detail</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 500px; margin: 20px auto; padding: 0 15px; background: #1a1a2e; color: #eee; }
a { color: #f39c12; text-decoration: none; }
.topbar { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.backbtn { background: #16213e; color: #f39c12; border: 1px solid #333; padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 14px; }
.card-img { display: block; width: 100%; max-width: 320px; margin: 0 auto 15px; border-radius: 10px; box-shadow: 0 4px 18px rgba(0,0,0,0.6); }
.card-title { text-align: center; font-size: 22px; font-weight: 700; margin: 4px 0; color: #fff; }
.card-meta { text-align: center; font-size: 13px; color: #888; margin-bottom: 10px; }
.price-big { text-align: center; font-size: 32px; font-weight: 700; color: #4ecca3; margin: 6px 0; }
.panel { background: #16213e; border-radius: 10px; padding: 14px; margin: 10px 0; }
.panel h4 { margin: 0 0 8px; color: #f39c12; font-size: 15px; }
.v-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
.v-row label { min-width: 80px; font-size: 13px; color: #bbb; }
.v-row select { flex: 1; padding: 8px; border-radius: 6px; background: #0f3460; color: #eee; border: 1px solid #333; font-size: 14px; }
.cond-row { display: flex; justify-content: center; gap: 6px; flex-wrap: wrap; margin: 6px 0 4px; font-size: 13px; font-weight: 600; }
.cond-pill { padding: 4px 10px; border-radius: 14px; background: #0f3460; border: 1px solid transparent; cursor: pointer; user-select: none; }
.cond-pill:hover { background: #14447a; }
.cond-pill.selected { outline: 2px solid currentColor; outline-offset: 1px; filter: brightness(1.15); font-weight: 800; }
.cond-hint { font-size: 11px; color: #666; text-align: center; margin-top: 4px; }
.sales-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.sales-table th, .sales-table td { text-align: left; padding: 4px 6px; border-bottom: 1px solid #222; }
.sales-table th { color: #888; font-weight: 600; font-size: 11px; }
.sales-cond { font-weight: 700; }
.addbtn { width: 100%; padding: 14px; background: #4ecca3; color: #1a1a2e; border: none; border-radius: 8px; font-size: 16px; font-weight: 700; cursor: pointer; margin-top: 8px; }
.addbtn:disabled { opacity: 0.5; cursor: default; }
.msg { text-align: center; font-size: 13px; margin-top: 6px; min-height: 16px; }
.selected-variant { color: #f39c12; font-family: monospace; font-size: 13px; }
.invalid { opacity: 0.4; }
</style>
</head>
<body>
<div class="topbar">
  <button class="backbtn" onclick="history.back()">&larr; Back</button>
  <span style="color:#888;font-size:12px;">Card detail</span>
</div>
<div id="err" style="display:none;color:#e94560;text-align:center;padding:20px;"></div>
<div id="loading" style="text-align:center;padding:40px;color:#888;">Loading…</div>
<div id="view" style="display:none;">
  <img id="img" class="card-img" alt="">
  <div id="title" class="card-title"></div>
  <div id="meta" class="card-meta"></div>
  <div class="price-big" id="priceBig">—</div>
  <div id="priceBigLabel" style="text-align:center;font-size:11px;color:#888;margin-top:-6px;margin-bottom:8px;letter-spacing:1px;">MODERATELY PLAYED</div>

  <div class="panel">
    <h4>Variant</h4>
    <div class="v-row"><label>Edition</label><select id="selEdition"></select></div>
    <div class="v-row"><label>Finish</label><select id="selFinish"></select></div>
    <div class="v-row" id="modRow" style="display:none;"><label>Modifier</label><select id="selMod"></select></div>
    <div style="font-size:12px;color:#888;margin-top:6px;">
      Selected: <span id="variantStr" class="selected-variant">normal</span>
      <span id="variantWarn" style="color:#e94560;margin-left:8px;display:none;">not a known variant for this card</span>
    </div>
  </div>

  <div class="panel" id="condPanel" style="display:none;">
    <h4>Condition Prices</h4>
    <div class="cond-row" id="condRow"></div>
    <div class="cond-hint">Tap a condition to select it for your inventory</div>
    <div id="condSource" style="font-size:11px;color:#666;text-align:center;margin-top:4px;"></div>
  </div>

  <button id="addBtn" class="addbtn">Add MP to Inventory</button>
  <div id="addMsg" class="msg"></div>

  <div class="panel" id="salesPanel" style="display:none;">
    <h4>Recent Sales</h4>
    <div id="salesFilter" style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px;"></div>
    <div id="salesSummary" style="font-size:12px;color:#bbb;margin-bottom:6px;"></div>
    <table class="sales-table" id="salesTable"></table>
  </div>

  <div style="text-align:center;margin-top:14px;">
    <a id="tcgLink" href="#" target="_blank">View on TCGplayer</a>
  </div>
</div>

<script>
const CARD_ID = "__CARD_ID__";
let SELECTED_COND = 'MP';
const COND_LONG = {NM:'Near Mint', LP:'Lightly Played', MP:'Moderately Played', HP:'Heavily Played', DMG:'Damaged'};
function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[c])); }

async function load() {
    try {
        // Fire /card-info and /card-sales in parallel — sales doesn't
        // depend on info, so serializing them is wasted wall time.
        // Sales needs tcg_product_id which only comes from /card-info,
        // but we can guess from base_id isn't viable — so fire info first,
        // then sales as soon as we see tcg_product_id. Actually simpler:
        // one info request, then paint immediately and fire sales in
        // the background render() path. Keep that flow.
        const r = await fetch('/card-info?card_id=' + encodeURIComponent(CARD_ID));
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const data = await r.json();
        if (data.error) throw new Error(data.error);
        render(data);
    } catch (e) {
        document.getElementById('loading').style.display = 'none';
        const err = document.getElementById('err');
        err.style.display = 'block';
        err.textContent = 'Could not load card: ' + e.message;
    }
}

function composeVariant() {
    const ed = document.getElementById('selEdition').value;
    const fi = document.getElementById('selFinish').value;
    const mod = document.getElementById('selMod').value;
    if (mod) return mod;
    let base;
    if (ed === 'normal') {
        base = (fi === 'non_holo') ? 'normal' : fi;
    } else {
        base = (fi === 'non_holo') ? ed : (ed + '_' + fi);
    }
    return base;
}

function labelize(v) {
    return v.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

let RAW_VARIANTS = [];

function refreshVariantStr() {
    const v = composeVariant();
    document.getElementById('variantStr').textContent = v;
    const warn = document.getElementById('variantWarn');
    if (RAW_VARIANTS.length && !RAW_VARIANTS.includes(v)) {
        warn.style.display = 'inline';
    } else {
        warn.style.display = 'none';
    }
}

const SALES_CONDCOLOR = {'Near Mint':'#4ecca3','Lightly Played':'#a8d8a8','Moderately Played':'#f0c040','Heavily Played':'#e08040','Damaged':'#e94560'};
const SALES_CONDSHORT = {'Near Mint':'NM','Lightly Played':'LP','Moderately Played':'MP','Heavily Played':'HP','Damaged':'DMG'};
const SALES_CONDS_ORDER = ['Near Mint','Lightly Played','Moderately Played','Heavily Played','Damaged'];
let ALL_SALES = [];
let SALES_FILTER = 'all';

function renderSales(sales) {
    ALL_SALES = sales;
    // Build filter pill row
    const filterDiv = document.getElementById('salesFilter');
    const counts = {all: sales.length};
    for (const s of sales) counts[s.condition] = (counts[s.condition]||0) + 1;
    const buttons = [['all','All',counts.all,'#eee']];
    for (const c of SALES_CONDS_ORDER) {
        if (counts[c]) buttons.push([c, SALES_CONDSHORT[c], counts[c], SALES_CONDCOLOR[c]]);
    }
    filterDiv.innerHTML = '';
    for (const [key,label,count,color] of buttons) {
        const b = document.createElement('button');
        b.className = 'sf-btn';
        const active = (SALES_FILTER === key);
        b.style.cssText = 'padding:4px 9px;border-radius:12px;border:1px solid ' + color
            + ';background:' + (active ? color : 'transparent')
            + ';color:' + (active ? '#1a1a2e' : color)
            + ';font-size:12px;font-weight:700;cursor:pointer;';
        b.innerHTML = esc(label) + ' <span style="opacity:0.75;font-weight:500;">' + count + '</span>';
        b.addEventListener('click', () => { SALES_FILTER = key; renderSales(ALL_SALES); });
        filterDiv.appendChild(b);
    }
    const filtered = (SALES_FILTER === 'all') ? sales : sales.filter(s => s.condition === SALES_FILTER);
    document.getElementById('salesSummary').textContent =
        filtered.length + ' sale' + (filtered.length===1?'':'s')
        + (SALES_FILTER === 'all' ? '' : (' in ' + SALES_CONDSHORT[SALES_FILTER]));
    const tbl = document.getElementById('salesTable');
    tbl.innerHTML = '<tr><th>Date</th><th>Cond</th><th>Printing</th><th style="text-align:right;">Price</th></tr>';
    for (const s of filtered) {
        const short = SALES_CONDSHORT[s.condition] || s.condition || '';
        const color = SALES_CONDCOLOR[s.condition] || '#eee';
        const tr = document.createElement('tr');
        tr.innerHTML =
            '<td>' + esc(s.date) + '</td>'
          + '<td class="sales-cond" style="color:' + color + ';">' + esc(short) + '</td>'
          + '<td>' + esc(s.printing || '') + '</td>'
          + '<td style="text-align:right;">' + (s.price!=null ? '$' + Number(s.price).toFixed(2) : '—') + '</td>';
        tbl.appendChild(tr);
    }
}

function render(data) {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('view').style.display = 'block';
    if (data.image_url) document.getElementById('img').src = data.image_url;
    else document.getElementById('img').style.display = 'none';
    document.getElementById('title').textContent = data.name || data.card_id;
    // Meta line: "Set Name - #<collector number>". We strip the set prefix
    // from card_id because "swsh12pt5gg-GG20" is an internal identifier
    // that reads as gibberish; the number ("GG20") is the useful bit.
    const cidParts = (data.card_id || '').split('-');
    const cardNum = cidParts.length > 1 ? cidParts.slice(1).join('-') : data.card_id;
    const metaStr = [data.set_name, cardNum ? ('#' + cardNum) : null].filter(Boolean).join(' - ');
    document.getElementById('meta').textContent = metaStr;
    // Headline price: Moderately Played. MP is the typical "played card"
    // anchor — we always want to see it up top. Fall back to market price
    // if MP isn't in the condition grid.
    const priceEl = document.getElementById('priceBig');
    const priceLabel = document.getElementById('priceBigLabel');
    const cpForBig = data.condition_prices || {};
    const mp = cpForBig.MP || cpForBig.mp;
    if (mp && mp.price != null) {
        priceEl.textContent = '$' + Number(mp.price).toFixed(2);
        priceEl.style.color = '#f0c040';
        priceLabel.textContent = 'MODERATELY PLAYED';
    } else if (data.market_price) {
        priceEl.textContent = '$' + Number(data.market_price).toFixed(2);
        priceEl.style.color = '#4ecca3';
        priceLabel.textContent = 'MARKET (NM)';
    } else {
        priceEl.textContent = '—';
        priceLabel.textContent = '';
    }

    const variants = data.variants || {};
    RAW_VARIANTS = variants.raw || [];

    function fill(sel, opts, fallback) {
        sel.innerHTML = '';
        const list = (opts && opts.length) ? opts : [fallback];
        for (const v of list) {
            const o = document.createElement('option');
            o.value = v;
            o.textContent = labelize(v);
            sel.appendChild(o);
        }
    }
    // Edition row: only show when there's a real choice (more than just
    // "normal"). Hides the row for cards with no 1st-edition / shadowless /
    // unlimited distinction.
    const edRow = document.getElementById('selEdition').closest('.v-row');
    const editions = (variants.edition || []).filter(e => e && e !== 'normal');
    if (editions.length > 0) {
        fill(document.getElementById('selEdition'), ['normal'].concat(editions), 'normal');
        edRow.style.display = 'flex';
    } else {
        fill(document.getElementById('selEdition'), ['normal'], 'normal');
        edRow.style.display = 'none';
    }
    fill(document.getElementById('selFinish'), variants.finish, 'non_holo');

    const modRow = document.getElementById('modRow');
    const selMod = document.getElementById('selMod');
    if (variants.modifier && variants.modifier.length) {
        modRow.style.display = 'flex';
        selMod.innerHTML = '<option value="">None</option>';
        for (const m of variants.modifier) {
            const o = document.createElement('option');
            o.value = m;
            o.textContent = labelize(m);
            selMod.appendChild(o);
        }
    }

    ['selEdition', 'selFinish', 'selMod'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', refreshVariantStr);
    });
    refreshVariantStr();

    // Condition prices — 5 pills, matches the scan result. MP is emphasized
    // (bold border + glow) since that's the typical visual anchor when
    // pricing a played card. Each pill is clickable to pick the condition
    // that the Add-to-Inventory button will use.
    const cp = data.condition_prices;
    const COLORS = {NM:'#4ecca3', LP:'#a8d8a8', MP:'#f0c040', HP:'#e08040', DMG:'#e94560'};
    const COND_LONG = {NM:'Near Mint', LP:'Lightly Played', MP:'Moderately Played', HP:'Heavily Played', DMG:'Damaged'};
    SELECTED_COND = 'MP';
    function refreshCondSelection() {
        document.querySelectorAll('#condRow .cond-pill').forEach(el => {
            if (el.getAttribute('data-cond') === SELECTED_COND) el.classList.add('selected');
            else el.classList.remove('selected');
        });
        const btn = document.getElementById('addBtn');
        btn.textContent = 'Add ' + SELECTED_COND + ' to Inventory';
    }
    if (cp) {
        const row = document.getElementById('condRow');
        row.innerHTML = '';
        const CONDS = ['NM','LP','MP','HP','DMG'];
        let allJtcg = true, anyJtcg = false;
        for (const key of CONDS) {
            const info = cp[key] || cp[key.toLowerCase()];
            if (!info) continue;
            if (info.source === 'justtcg') anyJtcg = true; else allJtcg = false;
            const prefix = '$';
            const estStyle = info.estimated ? 'font-style:italic;opacity:0.7;' : '';
            const pill = document.createElement('span');
            pill.className = 'cond-pill';
            pill.setAttribute('data-cond', key);
            pill.style.color = COLORS[key];
            pill.style.cssText += estStyle;
            pill.innerHTML = esc(key) + ' ' + prefix + Number(info.price).toFixed(2);
            pill.addEventListener('click', () => {
                SELECTED_COND = key;
                refreshCondSelection();
            });
            row.appendChild(pill);
        }
        if (anyJtcg) {
            document.getElementById('condSource').textContent = allJtcg ? 'market' : 'mixed';
        }
        document.getElementById('condPanel').style.display = 'block';
        refreshCondSelection();
    }

    const link = document.getElementById('tcgLink');
    if (data.tcgplayer_url) link.href = data.tcgplayer_url;
    else link.style.display = 'none';

    // Sales are inlined in /card-info to save a round-trip.
    const sales = Array.isArray(data.sales) ? data.sales : [];
    if (sales.length > 0) {
        renderSales(sales);
        document.getElementById('salesPanel').style.display = 'block';
    }

    document.getElementById('addBtn').addEventListener('click', async () => {
        const v = composeVariant();
        const fullId = CARD_ID + '/' + v;
        const cond = COND_LONG[SELECTED_COND] || null;
        const btn = document.getElementById('addBtn');
        const msg = document.getElementById('addMsg');
        btn.disabled = true; btn.textContent = 'Adding…'; msg.textContent = '';
        try {
            const r = await fetch('/inventory/add', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({card_id: fullId, quantity: 1, condition: cond}),
            });
            const j = await r.json();
            if (r.ok) {
                msg.style.color = '#4ecca3';
                msg.textContent = 'Added ' + fullId + (cond ? ' (' + SELECTED_COND + ')' : '');
            } else {
                msg.style.color = '#e94560';
                msg.textContent = j.error || 'Failed';
            }
        } catch (e) {
            msg.style.color = '#e94560'; msg.textContent = 'Network error';
        } finally {
            btn.disabled = false;
            btn.textContent = 'Add ' + SELECTED_COND + ' to Inventory';
        }
    });
}

load();
</script>
</body>
</html>
"""


_PRICE_LOOKUP_BULK_SQL = """
    SELECT c.card_id, c.name, p.market_price
    FROM dim_cards c
    LEFT JOIN LATERAL (
        SELECT market_price FROM fact_market_prices
        WHERE card_id = c.card_id
        ORDER BY
            CASE subtype_name WHEN 'Normal' THEN 0 WHEN 'Holofoil' THEN 1 ELSE 2 END,
            price_date DESC
        LIMIT 1
    ) p ON true
    WHERE c.card_id = ANY(:ids)
"""


def _build_condition_prices(nm_price, tcg_product_id=None, variant="normal"):
    """Build condition_prices dict for all 5 raw conditions.

    First tries real per-condition prices from JustTCG (keyed by tcg_product_id
    and printing/variant). Falls back to fixed multipliers from NM price.

    Each condition entry has: price, source ("justtcg" or "estimated").
    Estimated entries also include multiplier, range_low, range_high.

    Returns None if nm_price is falsy and no JustTCG data found.
    """
    # Try JustTCG real prices first
    if tcg_product_id:
        printing = _VARIANT_TO_JUSTTCG_PRINTING.get(variant or "normal", "Normal")
        jtcg = _get_justtcg_prices(tcg_product_id, printing)
        if jtcg:
            cond_prices = {}
            for cond in ("NM", "LP", "MP", "HP", "DMG"):
                if cond in jtcg:
                    cond_prices[cond] = {
                        "price": round(jtcg[cond], 2),
                        "source": "justtcg",
                        "estimated": False,
                    }
            if len(cond_prices) >= 3:
                # Fill any missing conditions with multipliers from NM
                base = jtcg.get("NM") or nm_price
                if base:
                    from cardprice.models.condition_pricing import CONDITION_MULTIPLIERS_WITH_CI
                    base = float(base)
                    for cond in ("NM", "LP", "MP", "HP", "DMG"):
                        if cond not in cond_prices:
                            mult, ci_lo, ci_hi = CONDITION_MULTIPLIERS_WITH_CI[cond]
                            cond_prices[cond] = {
                                "price": round(base * mult, 2),
                                "multiplier": mult,
                                "range_low": round(base * ci_lo, 2),
                                "range_high": round(base * ci_hi, 2),
                                "source": "estimated",
                                "estimated": True,
                            }
                return cond_prices

    # Fallback: fixed multipliers from NM price
    if not nm_price:
        return None
    from cardprice.models.condition_pricing import CONDITION_MULTIPLIERS_WITH_CI
    nm = float(nm_price)
    cond_prices = {}
    for cond in ("NM", "LP", "MP", "HP", "DMG"):
        mult, ci_lo, ci_hi = CONDITION_MULTIPLIERS_WITH_CI[cond]
        cond_prices[cond] = {
            "price": round(nm * mult, 2),
            "multiplier": mult,
            "range_low": round(nm * ci_lo, 2),
            "range_high": round(nm * ci_hi, 2),
            "source": "estimated",
            "estimated": True,
        }
    return cond_prices


# ---------------------------------------------------------------------------
# Variant → TCGCSV subtype mapping + variant price lookup
# ---------------------------------------------------------------------------
# Maps ML-detected variant names to fact_market_prices.subtype_name values.
# The ML pipeline returns: normal, holofoil, reverse_holofoil, 1st_edition,
# promo, full_art, shadowless, gold, rainbow_rare, stamped.
_VARIANT_TO_SUBTYPE = {
    "normal":           "Normal",
    "holofoil":         "Holofoil",
    "reverse_holofoil": "Reverse Holofoil",
    "stamped":          "Reverse Holofoil",   # EX stamped = reverse holo pricing
    "promo":            "Normal",             # promos have their own card_id
    "shadowless":       "Normal",             # no separate TCGCSV subtype
    "full_art":         "Holofoil",           # full arts priced as holofoil
    "gold":             "Holofoil",
    "rainbow_rare":     "Holofoil",
    "cosmos":           "Holofoil",
    "cracked_ice":      "Holofoil",
    # --- Japanese variant printings ---
    # Each JP variant printing is its own tcg_product_id in dim_cards_jp,
    # and TCGCSV cat 85 prices them under the "Holofoil" subtype_name.
    # The card_id itself encodes the variant; the subtype map just routes
    # the price query to the right fact_market_prices_jp row.
    "master_ball_pattern":   "Holofoil",
    "poke_ball_pattern":     "Holofoil",
    "mirror_holofoil":       "Holofoil",
    "mirror_foil":           "Holofoil",
    "energy_symbol_pattern": "Holofoil",
    "dusk_ball_pattern":     "Holofoil",
    "love_ball_pattern":     "Holofoil",
    "friend_ball_pattern":   "Holofoil",
    "quick_ball_pattern":    "Holofoil",
    "team_rocket_pattern":   "Holofoil",
    "terastal_pattern":      "Holofoil",
}

# 1st Edition has multiple possible subtypes in TCGCSV; try them in priority order.
_1ST_EDITION_SUBTYPES = ["1st Edition Holofoil", "1st Edition", "1st Edition Normal"]


def _lookup_variant_price(session, card_id, detected_variant):
    """Look up variant-specific price from fact_market_prices.

    For Japanese cards (card_id starts with "jp_"), looks up
    fact_market_prices_jp instead.

    For 1st Edition, tries multiple subtype names since TCGCSV uses different
    ones depending on era (Holofoil vs Normal vs bare "1st Edition").

    Returns the market_price as float, or None if not found.
    """
    from sqlalchemy import text as sql_text

    # Japanese cards: route to fact_market_prices_jp.
    # JP table joins on tcg_product_id, not card_id — bridge via dim_cards_jp.
    if _is_jp_card_id(card_id):
        subtype = _VARIANT_TO_SUBTYPE.get(detected_variant or "normal", "Normal")
        vrow = session.execute(
            sql_text("""
                SELECT p.market_price
                FROM dim_cards_jp c
                JOIN fact_market_prices_jp p ON p.tcg_product_id = c.tcg_product_id
                WHERE c.card_id = :cid AND p.subtype_name = :subtype
                ORDER BY p.price_date DESC LIMIT 1
            """),
            {"cid": card_id, "subtype": subtype},
        ).fetchone()
        if vrow and vrow.market_price:
            return float(vrow.market_price)
        # Fall back to any subtype if specific one missing
        vrow = session.execute(
            sql_text("""
                SELECT p.market_price
                FROM dim_cards_jp c
                JOIN fact_market_prices_jp p ON p.tcg_product_id = c.tcg_product_id
                WHERE c.card_id = :cid
                ORDER BY p.price_date DESC LIMIT 1
            """),
            {"cid": card_id},
        ).fetchone()
        if vrow and vrow.market_price:
            return float(vrow.market_price)
        return None

    if detected_variant == "1st_edition":
        # Try each 1st Edition subtype in priority order
        for subtype in _1ST_EDITION_SUBTYPES:
            vrow = session.execute(
                sql_text("""
                    SELECT market_price FROM fact_market_prices
                    WHERE card_id = :cid AND subtype_name = :subtype
                    ORDER BY price_date DESC LIMIT 1
                """),
                {"cid": card_id, "subtype": subtype},
            ).fetchone()
            if vrow and vrow.market_price:
                return float(vrow.market_price)
        return None

    subtype = _VARIANT_TO_SUBTYPE.get(detected_variant)
    if not subtype:
        return None

    vrow = session.execute(
        sql_text("""
            SELECT market_price FROM fact_market_prices
            WHERE card_id = :cid AND subtype_name = :subtype
            ORDER BY price_date DESC LIMIT 1
        """),
        {"cid": card_id, "subtype": subtype},
    ).fetchone()
    if vrow and vrow.market_price:
        return float(vrow.market_price)
    return None


def _get_lan_ip():
    """Return the LAN IP address of this machine."""
    try:
        # Connect to an external address to determine which interface is used
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _generate_qr_png(url):
    """Generate a QR code PNG as bytes using the qrcode library.

    Returns PNG bytes, or None if the library is not installed.
    """
    try:
        import qrcode
        import io
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="white", back_color="#1a1a2e")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return None

HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Card Scanner</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 500px; margin: 20px auto; padding: 0 15px; background: #1a1a2e; color: #eee; }
h1 { text-align: center; color: #e94560; }
.upload-btn { display: block; width: 100%; padding: 20px; font-size: 18px; background: #e94560; color: white; border: none; border-radius: 12px; cursor: pointer; margin: 10px 0; }
.upload-btn:active { background: #c23152; }
input[type=file] { display: none; }
.toggle-row { display: flex; align-items: center; justify-content: space-between; background: #16213e; padding: 12px 15px; border-radius: 8px; margin: 10px 0; }
.toggle-row label { color: #ccc; font-size: 15px; }
.toggle-switch { position: relative; width: 50px; height: 28px; }
.toggle-switch input { opacity: 0; width: 0; height: 0; }
.toggle-slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background: #333; border-radius: 28px; transition: 0.3s; }
.toggle-slider:before { content: ""; position: absolute; height: 22px; width: 22px; left: 3px; bottom: 3px; background: #eee; border-radius: 50%; transition: 0.3s; }
.toggle-switch input:checked + .toggle-slider { background: #4ecca3; }
.toggle-switch input:checked + .toggle-slider:before { transform: translateX(22px); }
.result { background: #16213e; padding: 15px; border-radius: 8px; margin: 15px 0; display: none; }
.result.show { display: block; }
.result h3 { color: #e94560; margin: 0 0 10px; }
.price { font-size: 24px; color: #4ecca3; font-weight: bold; }
.cond-row { display: flex; justify-content: center; gap: 6px; flex-wrap: wrap; margin: 6px 0 4px; font-size: 12px; font-weight: 600; }
.cond-pill { padding: 2px 7px; border-radius: 10px; background: #1a1a2e; }
.variant-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; background: #f0c040; color: #1a1a2e; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-right: 4px; margin-bottom: 2px; }
.variant-badge.first-edition { background: #f1c40f; color: #333; }
.variant-badge.shadowless { background: #bdc3c7; color: #333; }
.variant-badge.ghost { background: #95a5a6; color: #fff; }
.variant-badge.no-symbol { background: #e67e22; color: #fff; }
.variant-badge.promo { background: #2c3e50; color: #fff; }
.variant-badge.prerelease { background: #3498db; color: #fff; }
.variant-badge.staff { background: linear-gradient(135deg, #f1c40f, #3498db); color: #fff; }
.variant-badge.stamped { background: #9b59b6; color: #fff; }
.variant-badge.pc-exclusive { background: #e74c3c; color: #fff; }
.variant-badge.bb-promo { background: #e74c3c; color: #fff; }
.variant-badge.winner { background: #f39c12; color: #333; }
.variant-badge.crosshatch { background: #1abc9c; color: #fff; }
.variant-badge.wc { background: #7f8c8d; color: #fff; }
.variant-badge.ditto { background: #9b59b6; color: #fff; }
.variant-badge.tru { background: #2980b9; color: #fff; }
.variant-badge.reverse-holo { background: #95a5a6; color: #fff; }
.variant-badge.holo { background: linear-gradient(135deg, #e74c3c, #f1c40f, #2ecc71, #3498db); color: #fff; }
.btn-row { display: flex; gap: 8px; justify-content: center; margin: 12px 0 4px; }
.btn-row button { flex: 1; max-width: 180px; padding: 10px 16px; border: none; border-radius: 8px; font-size: 15px; font-weight: bold; cursor: pointer; }
.btn-inv { background: #4ecca3; color: #1a1a2e; }
.btn-cart { background: #e94560; color: #fff; }
.btn-row button:disabled { opacity: 0.5; }
.confidence { color: #888; }
.section-preview { max-width: 100%; border-radius: 8px; margin: 10px 0; display: none; }
.spinner { display: none; flex-direction: column; align-items: center; justify-content: center; padding: 32px 24px; gap: 12px; }
.spinner.show { display: flex; }
.spinner .scan-spinner-wrap { position: relative; width: 96px; height: 96px; }
.spinner .scan-spinner-ring { width: 96px; height: 96px; border: 4px solid #333; border-top-color: #e94560; border-radius: 50%; animation: spinAnim 0.8s linear infinite; }
.spinner .scan-timer { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 24px; font-weight: 700; color: #fff; font-variant-numeric: tabular-nums; }
.spinner .scan-label .dots::after { content: ''; animation: dotAnim 1.5s steps(4, end) infinite; }
@keyframes spinAnim { to { transform: rotate(360deg); } }
@keyframes dotAnim { 0% { content: ''; } 25% { content: '.'; } 50% { content: '..'; } 75% { content: '...'; } 100% { content: ''; } }
.qr-section { text-align: center; background: #16213e; border-radius: 12px; padding: 15px; margin: 0 0 15px; }
.qr-section p { margin: 5px 0 10px; color: #888; font-size: 14px; }
.qr-section .url { font-family: monospace; color: #4ecca3; font-size: 13px; word-break: break-all; }
#qrCanvas { image-rendering: pixelated; border-radius: 4px; }
</style>
</head>
<body>
<h1>Pokemon Card Scanner</h1>
<div style="display:flex;gap:6px;margin:0 0 10px;flex-wrap:wrap;">
    <a href="/inventory/view" style="flex:1;display:block;background:#4ecca3;color:#1a1a2e;padding:12px 8px;border-radius:8px;text-decoration:none;text-align:center;font-size:13px;font-weight:700;">Inventory</a>
</div>
<style>
.cond-grid{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr auto 1fr;gap:1px;font-size:8px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:2px;}
.cond-cell{text-align:center;background:rgba(255,255,255,0.04);border-radius:3px;padding:2px 0;}
.cond-cell span{opacity:0.5;font-size:7px;display:block;}
.cond-main{grid-column:1/3;text-align:center;font-size:14px;font-weight:700;padding:3px 0;}
</style>
<div class="qr-section" id="qrSection">
    <p>Scan QR code to open on your phone</p>
    <canvas id="qrCanvas"></canvas>
    <br>
    <span class="url" id="serverUrl"></span>
</div>

<!-- ===== SECTION 0: MANUAL SEARCH (orange #f39c12) ===== -->
<div style="background:#16213e;border-radius:12px;padding:15px;margin:10px 0;border-left:4px solid #f39c12;">
<h3 style="color:#f39c12;margin:0 0 10px;font-size:18px;">Search by Name</h3>
<div style="display:grid;grid-template-columns:110px 1fr 1fr;gap:8px;">
    <select id="searchLang"
            style="padding:10px;border-radius:6px;border:1px solid #333;background:#0f3460;color:#eee;font-size:15px;box-sizing:border-box;">
        <option value="en" selected>English</option>
        <option value="ja">Japanese</option>
    </select>
    <div style="position:relative;">
        <input id="searchName" type="text" autocomplete="off" placeholder="Pokemon name"
               style="width:100%;padding:10px;border-radius:6px;border:1px solid #333;background:#0f3460;color:#eee;font-size:15px;box-sizing:border-box;">
        <div id="searchNameDropdown"
             style="display:none;position:absolute;top:100%;left:0;right:0;background:#0f3460;border:1px solid #f39c12;border-radius:0 0 6px 6px;max-height:220px;overflow-y:auto;z-index:20;"></div>
    </div>
    <input id="searchMove" type="text" placeholder="Move text (optional)"
           style="width:100%;padding:10px;border-radius:6px;border:1px solid #333;background:#0f3460;color:#eee;font-size:15px;box-sizing:border-box;">
</div>
<div id="searchResults" style="margin-top:10px;"></div>
</div>

<!-- ===== SECTION 1: MY INVENTORY (green #4ecca3) ===== -->
<div style="background:#16213e;border-radius:12px;padding:15px;margin:10px 0;border-left:4px solid #4ecca3;">
<h3 style="color:#4ecca3;margin:0 0 10px;font-size:18px;">My Inventory</h3>
<div style="display:flex;gap:8px;">
    <label class="upload-btn" for="invCamera" style="flex:1;margin:0;padding:14px 10px;font-size:16px;background:#4ecca3;color:#1a1a2e;">Scan Single</label>
    <label class="upload-btn" for="invGallery" style="flex:1;margin:0;padding:14px 10px;font-size:16px;background:#0f3460;border:2px solid #4ecca3;color:#4ecca3;">Gallery Single</label>
</div>
<input type="file" id="invCamera" accept="image/*" capture="environment">
<input type="file" id="invGallery" accept="image/*">
<div style="display:flex;gap:8px;margin:8px 0 0;">
    <label class="upload-btn" for="invPageCamera" style="flex:1;margin:0;padding:14px 10px;font-size:16px;background:#3a9d7e;color:#fff;">Scan Page</label>
    <label class="upload-btn" for="invPageGallery" style="flex:1;margin:0;padding:14px 10px;font-size:16px;background:#0f3460;border:2px solid #3a9d7e;color:#3a9d7e;">Gallery Page</label>
</div>
<input type="file" id="invPageCamera" accept="image/*" capture="environment">
<input type="file" id="invPageGallery" accept="image/*">
<img id="invPreview" class="section-preview">
<div class="spinner" id="invSpinner">
    <div class="scan-spinner-wrap">
        <div class="scan-spinner-ring"></div>
        <div class="scan-timer" id="invScanTimer">0s</div>
    </div>
    <span class="scan-label">Scanning<span class="dots"></span></span>
</div>
<div class="result" id="invResult">
    <h3 id="invCardName" style="color:#4ecca3;"></h3>
    <div id="invVariantBadges" style="display:none;margin:4px 0;"></div>
    <div class="price" id="invCardPrice"></div>
    <div id="invConditionPrices" class="cond-row" style="display:none;"></div>
    <div class="confidence" id="invCardConf"></div>
    <div id="invCardMeta" style="font-size:12px;color:#666;margin:2px 0;"></div>
    <img id="invRefImage" style="display:none;max-width:200px;margin:12px auto;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,0.5)" />
    <canvas id="invSparkline" width="150" height="40" style="display:none;margin:10px 0;"></canvas>
    <div id="invSparkLabel" style="display:none;font-size:11px;color:#888;"></div>
    <div class="btn-row">
        <button id="invAddBtn" class="btn-inv" style="display:none;" onclick="addToSection('inv')">Add to Inventory</button>
    </div>
    <div id="invMsg" style="display:none;font-size:13px;margin-top:6px;text-align:center;"></div>
</div>
<div class="result" id="invPageResult">
    <h3 style="color:#4ecca3;">Binder Page Results</h3>
    <div id="invPageTotal" class="price"></div>
    <div id="invPageCards"></div>
    <div class="btn-row" id="invPageBtnRow" style="display:none;">
        <button class="btn-inv" onclick="addAllPage('inv')">Add All to Inventory</button>
    </div>
</div>
</div>

<!-- ===== DIVIDER ===== -->
<hr style="border:none;height:2px;background:linear-gradient(to right,transparent,#333,transparent);margin:20px 0;">

<!-- ===== SECTION 2: SHOPPING CART (blue #3498db) ===== -->
<div style="background:#16213e;border-radius:12px;padding:15px;margin:10px 0;border-left:4px solid #3498db;">
<h3 style="color:#3498db;margin:0 0 10px;font-size:18px;">Shopping Cart</h3>
<div style="display:flex;gap:8px;">
    <label class="upload-btn" for="cartCamera" style="flex:1;margin:0;padding:14px 10px;font-size:16px;background:#3498db;color:#fff;">Scan Single</label>
    <label class="upload-btn" for="cartGallery" style="flex:1;margin:0;padding:14px 10px;font-size:16px;background:#0f3460;border:2px solid #3498db;color:#3498db;">Gallery Single</label>
</div>
<input type="file" id="cartCamera" accept="image/*" capture="environment">
<input type="file" id="cartGallery" accept="image/*">
<div style="display:flex;gap:8px;margin:8px 0 0;">
    <label class="upload-btn" for="cartPageCamera" style="flex:1;margin:0;padding:14px 10px;font-size:16px;background:#2980b9;color:#fff;">Scan Page</label>
    <label class="upload-btn" for="cartPageGallery" style="flex:1;margin:0;padding:14px 10px;font-size:16px;background:#0f3460;border:2px solid #2980b9;color:#2980b9;">Gallery Page</label>
</div>
<input type="file" id="cartPageCamera" accept="image/*" capture="environment">
<input type="file" id="cartPageGallery" accept="image/*">
<img id="cartPreview" class="section-preview">
<div class="spinner" id="cartSpinner">
    <div class="scan-spinner-wrap">
        <div class="scan-spinner-ring"></div>
        <div class="scan-timer" id="cartScanTimer">0s</div>
    </div>
    <span class="scan-label">Scanning<span class="dots"></span></span>
</div>
<div class="result" id="cartResult">
    <h3 id="cartCardName" style="color:#3498db;"></h3>
    <div id="cartVariantBadges" style="display:none;margin:4px 0;"></div>
    <div class="price" id="cartCardPrice" style="color:#3498db;"></div>
    <div id="cartConditionPrices" class="cond-row" style="display:none;"></div>
    <div class="confidence" id="cartCardConf"></div>
    <div id="cartCardMeta" style="font-size:12px;color:#666;margin:2px 0;"></div>
    <img id="cartRefImage" style="display:none;max-width:200px;margin:12px auto;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,0.5)" />
    <canvas id="cartSparkline" width="150" height="40" style="display:none;margin:10px 0;"></canvas>
    <div id="cartSparkLabel" style="display:none;font-size:11px;color:#888;"></div>
    <div class="btn-row">
        <button id="cartAddBtn" class="btn-cart" style="display:none;" onclick="addToSection('cart')">Add to Cart</button>
    </div>
    <div id="cartMsg" style="display:none;font-size:13px;margin-top:6px;text-align:center;"></div>
</div>
<div class="result" id="cartPageResult">
    <h3 style="color:#3498db;">Binder Page Results</h3>
    <div id="cartPageTotal" class="price" style="color:#3498db;"></div>
    <div id="cartPageCards"></div>
    <div class="btn-row" id="cartPageBtnRow" style="display:none;">
        <button class="btn-cart" onclick="addAllPage('cart')">Add All to Cart</button>
    </div>
</div>
<div style="text-align:center;margin:10px 0 0;">
    <a href="/cart/view" style="color:#3498db;font-size:13px;text-decoration:none;">View Cart</a>
</div>
</div>
<div id="cardDetailOverlay" style="display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.85);z-index:1000;align-items:center;justify-content:center;padding:16px;" onclick="if(event.target===this)this.style.display='none';">
    <div style="background:#16213e;border-radius:12px;padding:20px;max-width:360px;width:100%;max-height:90vh;overflow-y:auto;position:relative;text-align:center;">
        <button onclick="document.getElementById('cardDetailOverlay').style.display='none';" style="position:absolute;top:8px;right:12px;background:none;border:none;color:#888;font-size:24px;cursor:pointer;line-height:1;">&times;</button>
        <div id="cardDetailBody"></div>
    </div>
</div>
<script>
// Minimal QR Code generator in JS — zero external dependencies.
// Supports byte-mode encoding up to version 6 (EC level M).
var QRGen=(function(){
"use strict";
var EXP=new Array(256),LOG=new Array(256);
(function(){var v=1;for(var i=0;i<255;i++){EXP[i]=v;LOG[v]=i;v<<=1;if(v>=256)v^=0x11d;}EXP[255]=EXP[0];})();
function gfMul(a,b){return a===0||b===0?0:EXP[(LOG[a]+LOG[b])%255];}
function polyMul(a,b){var r=new Array(a.length+b.length-1).fill(0);for(var i=0;i<a.length;i++)for(var j=0;j<b.length;j++)r[i+j]^=gfMul(a[i],b[j]);return r;}
function ecBytes(data,ecLen){
    var gen=[1];for(var i=0;i<ecLen;i++)gen=polyMul(gen,[1,EXP[i]]);
    var msg=new Array(data.length+ecLen).fill(0);for(var i=0;i<data.length;i++)msg[i]=data[i];
    for(var i=0;i<data.length;i++){var c=msg[i];if(c!==0)for(var j=0;j<gen.length;j++)msg[i+j]^=gfMul(gen[j],c);}
    return msg.slice(data.length);
}
var VERSIONS=[
    null,
    {total:26,ec:10,cap:16},{total:44,ec:16,cap:28},{total:70,ec:26,cap:44},
    {total:100,ec:18,cap:82},{total:134,ec:26,cap:108},{total:172,ec:18,cap:154}
];
var ALIGN=[null,null,[6,18],[6,22],[6,26],[6,30],[6,34]];
function chooseVersion(len){for(var v=1;v<=6;v++){if(len<=VERSIONS[v].cap)return v;}return 6;}
function makeMatrix(sz){var m=[];for(var i=0;i<sz;i++){var r=[];for(var j=0;j<sz;j++)r.push(null);m.push(r);}return m;}
function addFinder(m,row,col){
    for(var r=-1;r<=7;r++)for(var c=-1;c<=7;c++){
        var rr=row+r,cc=col+c;if(rr<0||rr>=m.length||cc<0||cc>=m.length)continue;
        m[rr][cc]=((r>=0&&r<=6&&(c===0||c===6))||(c>=0&&c<=6&&(r===0||r===6))||(r>=2&&r<=4&&c>=2&&c<=4))?1:0;
    }
}
function addAlignment(m,row,col){
    for(var r=-2;r<=2;r++)for(var c=-2;c<=2;c++)
        m[row+r][col+c]=(Math.abs(r)===2||Math.abs(c)===2||(r===0&&c===0))?1:0;
}
function addTimingPatterns(m){var sz=m.length;for(var i=8;i<sz-8;i++){if(m[6][i]===null)m[6][i]=(i%2===0)?1:0;if(m[i][6]===null)m[i][6]=(i%2===0)?1:0;}}
function reserveFormatInfo(m){
    var sz=m.length;
    for(var i=0;i<8;i++){if(m[8][i]===null)m[8][i]=0;if(m[i][8]===null)m[i][8]=0;if(m[8][sz-1-i]===null)m[8][sz-1-i]=0;if(m[sz-1-i][8]===null)m[sz-1-i][8]=0;}
    if(m[8][8]===null)m[8][8]=0;m[sz-8][8]=1;
}
function placeData(m,bits){
    var sz=m.length,idx=0;
    for(var col=sz-1;col>=1;col-=2){
        if(col===6)col=5;
        for(var row=0;row<sz;row++){for(var c=0;c<2;c++){
            var cc=col-c,goUp=((Math.floor((sz-1-col)/2))%2===0),rr=goUp?(sz-1-row):row;
            if(m[rr][cc]===null){m[rr][cc]=(idx<bits.length)?bits[idx]:0;idx++;}
        }}
    }
}
function isReserved(m,r,c,sz){
    if(r<9&&c<9)return true;if(r<9&&c>=sz-8)return true;if(r>=sz-8&&c<9)return true;
    if(r===6||c===6)return true;return false;
}
function applyMask0(m,sz){for(var r=0;r<sz;r++)for(var c=0;c<sz;c++){if(!isReserved(m,r,c,sz)&&(r+c)%2===0)m[r][c]^=1;}}
function writeFormatInfo(m){
    var sz=m.length;var bits=[1,0,1,0,1,0,0,0,0,0,1,0,0,1,0]; // ECM mask0
    var hpos=[0,1,2,3,4,5,7,8];
    for(var i=0;i<8;i++)m[8][hpos[i]]=bits[i];
    for(var i=0;i<7;i++)m[8][sz-7+i]=bits[8+i];
    for(var i=0;i<8;i++)m[hpos[7-i]][8]=bits[i];
    for(var i=0;i<7;i++)m[sz-1-i][8]=bits[8+i];
}
function pushBitsTo(arr,val,len){for(var i=len-1;i>=0;i--)arr.push((val>>i)&1);}
function encode(text){
    var bytes=[];for(var i=0;i<text.length;i++){var cp=text.charCodeAt(i);if(cp<128)bytes.push(cp);else if(cp<0x800){bytes.push(0xc0|(cp>>6));bytes.push(0x80|(cp&0x3f));}else{bytes.push(0xe0|(cp>>12));bytes.push(0x80|((cp>>6)&0x3f));bytes.push(0x80|(cp&0x3f));}}
    var version=chooseVersion(bytes.length);var vi=VERSIONS[version];var sz=17+version*4;
    var dataBits=[];
    pushBitsTo(dataBits,4,4);pushBitsTo(dataBits,bytes.length,version<=9?8:16);
    for(var i=0;i<bytes.length;i++)pushBitsTo(dataBits,bytes[i],8);
    var totalBits=vi.cap*8;var tl=Math.min(4,totalBits-dataBits.length);pushBitsTo(dataBits,0,tl);
    while(dataBits.length%8!==0)dataBits.push(0);
    var pad=[0xEC,0x11],pi=0;while(dataBits.length<totalBits){pushBitsTo(dataBits,pad[pi],8);pi^=1;}
    var dataBytes=[];for(var i=0;i<dataBits.length;i+=8){var b=0;for(var j=0;j<8;j++)b=(b<<1)|dataBits[i+j];dataBytes.push(b);}
    var ecCW=ecBytes(dataBytes,vi.ec);
    var allBits=[];for(var i=0;i<dataBytes.length;i++)pushBitsTo(allBits,dataBytes[i],8);
    for(var i=0;i<ecCW.length;i++)pushBitsTo(allBits,ecCW[i],8);
    var m=makeMatrix(sz);addFinder(m,0,0);addFinder(m,0,sz-7);addFinder(m,sz-7,0);
    if(ALIGN[version]){var ap=ALIGN[version];for(var i=0;i<ap.length;i++)for(var j=0;j<ap.length;j++){if(i===0&&j===0)continue;if(i===0&&j===ap.length-1)continue;if(i===ap.length-1&&j===0)continue;addAlignment(m,ap[i],ap[j]);}}
    addTimingPatterns(m);reserveFormatInfo(m);placeData(m,allBits);applyMask0(m,sz);writeFormatInfo(m);
    return m;
}
return{encode:encode};
})();

function drawQR(canvasId,text,cellSize){
    cellSize=cellSize||6;var matrix=QRGen.encode(text);var sz=matrix.length;
    var canvas=document.getElementById(canvasId);canvas.width=sz*cellSize;canvas.height=sz*cellSize;
    var ctx=canvas.getContext('2d');ctx.fillStyle='#1a1a2e';ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='#ffffff';
    for(var r=0;r<sz;r++)for(var c=0;c<sz;c++)if(matrix[r][c])ctx.fillRect(c*cellSize,r*cellSize,cellSize,cellSize);
}
// Render QR code with the server URL; hide on mobile (already on the phone)
(function(){
    var url=location.protocol+'//'+location.host;
    document.getElementById('serverUrl').textContent=url;
    drawQR('qrCanvas',url,6);
    if(/Mobi|Android|iPhone|iPad/i.test(navigator.userAgent))
        document.getElementById('qrSection').style.display='none';
    // Check for HTTPS tunnel URL (better for camera features)
    fetch('/tunnel-url').then(r=>r.json()).then(d=>{
        if(d.url){
            document.getElementById('serverUrl').textContent=d.url;
            drawQR('qrCanvas',d.url,6);
            var note=document.createElement('p');
            note.style.cssText='color:#4ecca3;font-size:11px;margin-top:6px;';
            note.textContent='HTTPS tunnel (camera works)';
            document.getElementById('qrSection').appendChild(note);
        }
    }).catch(function(){});
})();
</script>
<script>
// ===== Section-parameterized scan timer =====
var _scanTimers = {};
function startScanTimer(sec) {
    var start = Date.now();
    var timerEl = document.getElementById(sec + 'ScanTimer');
    timerEl.textContent = '0.0s';
    _scanTimers[sec] = setInterval(function() {
        timerEl.textContent = ((Date.now() - start) / 1000).toFixed(1) + 's';
    }, 100);
}
function stopScanTimer(sec) {
    if (_scanTimers[sec]) { clearInterval(_scanTimers[sec]); _scanTimers[sec] = null; }
}

// ===== Single card scan — parameterized by section =====
function handleFile(file, sec) {
    if (!file) return;
    var preview = document.getElementById(sec + 'Preview');
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
    var spinner = document.getElementById(sec + 'Spinner');
    var result = document.getElementById(sec + 'Result');
    spinner.classList.add('show');
    startScanTimer(sec);
    result.classList.remove('show');
    var fd = new FormData();
    fd.append('image', file);
    fetch('/scan', {method: 'POST', body: fd})
        .then(function(r) { return r.json(); })
        .then(function(data) {
            stopScanTimer(sec);
            spinner.classList.remove('show');
            result.classList.add('show');
            if (data.status === 'pending') {
                document.getElementById(sec + 'CardName').textContent = 'Queued for identification...';
                document.getElementById(sec + 'CardPrice').textContent = 'Checking every 3s';
                document.getElementById(sec + 'CardConf').textContent = '';
                document.getElementById(sec + 'CardMeta').textContent = '';
                pollResult(data.scan_id, sec);
            } else {
                showResult(data, sec);
            }
        })
        .catch(function(e) {
            stopScanTimer(sec);
            spinner.classList.remove('show');
            result.classList.add('show');
            document.getElementById(sec + 'CardName').textContent = 'Error: ' + e;
        });
}

// Global stamp badge map for variant display
var _STAMP_BADGE_MAP = {
    '1st_edition':        {label: '1st Ed',       cssClass: 'first-edition'},
    'shadowless':         {label: 'Shadowless',    cssClass: 'shadowless'},
    'ghost_stamp':        {label: 'Ghost',         cssClass: 'ghost'},
    'no_symbol':          {label: 'No Symbol',     cssClass: 'no-symbol'},
    'reverse_holo':       {label: 'Rev Holo',      cssClass: 'reverse-holo'},
    'ex_set_stamp':       {label: 'Stamped',       cssClass: 'stamped'},
    'black_star_promo':   {label: 'Promo',         cssClass: 'promo'},
    'modern_promo':       {label: 'Promo',         cssClass: 'promo'},
    'promo_stamp':        {label: 'Promo',         cssClass: 'promo'},
    'promo':              {label: 'Promo',         cssClass: 'promo'},
    'prerelease':         {label: 'Prerelease',    cssClass: 'prerelease'},
    'staff':              {label: 'Staff',          cssClass: 'staff'},
    'pokemon_center':     {label: 'PC',            cssClass: 'pc-exclusive'},
    'build_battle':       {label: 'B&B',           cssClass: 'bb-promo'},
    'winner':             {label: 'Winner',        cssClass: 'winner'},
    'crosshatch':         {label: 'Crosshatch',    cssClass: 'crosshatch'},
    'world_championship': {label: 'WC Deck',       cssClass: 'wc'},
    'ditto':              {label: 'Ditto!',        cssClass: 'ditto'},
    'toys_r_us':          {label: 'TRU',           cssClass: 'tru'},
    'stamped':            {label: 'Stamped',        cssClass: 'stamped'},
};

function showResult(data, sec) {
    var nameEl = document.getElementById(sec + 'CardName');
    var cardName = data.card_name || 'Unknown Card';
    var accentColor = sec === 'inv' ? '#4ecca3' : '#3498db';
    // Append Japanese name in parens when present (e.g., "Pineco (クヌギダマ)")
    var displayName = cardName;
    if (data.card_name_jp) {
        displayName = cardName + ' <span style="color:#888;font-size:0.85em;font-weight:normal;">(' + data.card_name_jp + ')</span>';
    }
    if (data.tcgplayer_url) {
        nameEl.innerHTML = '<a href="' + data.tcgplayer_url + '" target="_blank" rel="noopener" style="color:' + accentColor + ';text-decoration:underline;">' + displayName + '</a>';
    } else {
        nameEl.innerHTML = displayName;
    }
    var badgeContainer = document.getElementById(sec + 'VariantBadges');
    badgeContainer.innerHTML = '';
    var stampBadgesSeen = {};
    var stampSources = [];
    // Primary: stamp_details (multiple stamps)
    if (data.stamp_details && typeof data.stamp_details === 'object') {
        var skeys = Object.keys(data.stamp_details);
        for (var si = 0; si < skeys.length; si++) { stampSources.push(skeys[si]); }
    }
    // Fallback: stamps_detected array
    if (stampSources.length === 0 && data.stamps_detected && data.stamps_detected.length > 0) {
        for (var si2 = 0; si2 < data.stamps_detected.length; si2++) { stampSources.push(data.stamps_detected[si2]); }
    }
    // Fallback: detected_variant scalar
    if (stampSources.length === 0 && data.detected_variant && data.detected_variant !== 'normal') {
        stampSources.push(data.detected_variant);
    }
    for (var si3 = 0; si3 < stampSources.length; si3++) {
        var stype = stampSources[si3];
        var sinfo = _STAMP_BADGE_MAP[stype];
        var slabel = sinfo ? sinfo.label : stype.replace(/_/g, ' ');
        var scls = sinfo ? sinfo.cssClass : '';
        if (stampBadgesSeen[slabel]) continue;
        stampBadgesSeen[slabel] = true;
        var sb = document.createElement('span');
        sb.className = 'variant-badge' + (scls ? ' ' + scls : '');
        sb.textContent = slabel;
        badgeContainer.appendChild(sb);
    }
    if (stampSources.length > 0) {
        badgeContainer.style.display = 'block';
    } else {
        badgeContainer.style.display = 'none';
    }
    var displayPrice = data.variant_price || data.market_price;
    document.getElementById(sec + 'CardPrice').textContent = displayPrice ? '$' + parseFloat(displayPrice).toFixed(2) : 'No price data';
    var cpDiv = document.getElementById(sec + 'ConditionPrices');
    var conditions = ['NM','LP','MP','HP','DMG'];
    var colors = {'NM':'#4ecca3','LP':'#a8d8a8','MP':'#f0c040','HP':'#e08040','DMG':'#e94560'};
    if (data.condition_prices) {
        var cp = data.condition_prices;
        var html = '';
        for (var ci = 0; ci < conditions.length; ci++) {
            var cond = conditions[ci];
            var info = cp[cond];
            if (!info) continue;
            var prefix = '$';
            var estStyle = info.estimated ? 'font-style:italic;opacity:0.7;' : '';
            html += '<span class="cond-pill" style="color:' + colors[cond] + ';' + estStyle + '">' + cond + ' ' + prefix + info.price.toFixed(2) + '</span>';
        }
        // Show source indicator
        var allJtcg = Object.keys(cp).every(function(k){ return cp[k].source === 'justtcg'; });
        var anyJtcg = Object.keys(cp).some(function(k){ return cp[k].source === 'justtcg'; });
        if (anyJtcg) {
            html += '<span style="font-size:10px;color:#666;margin-left:4px;">' + (allJtcg ? 'market' : 'mixed') + '</span>';
        }
        cpDiv.innerHTML = html;
        cpDiv.style.display = 'flex';
    } else {
        cpDiv.style.display = 'none';
    }
    document.getElementById(sec + 'CardConf').innerHTML =
        (data.elapsed_seconds ? '<span style="color:#3498db;">' + data.elapsed_seconds + 's</span> — ' : '') +
        (data.confidence ? (data.confidence * 100).toFixed(0) + '% confidence' : '') +
        (data.method ? ' via ' + data.method : '');
    document.getElementById(sec + 'CardMeta').textContent = data.card_id || '';
    var refImg = document.getElementById(sec + 'RefImage');
    if (data.image_url) { refImg.src = data.image_url; refImg.style.display = 'block'; }
    else { refImg.style.display = 'none'; }
    // Sparkline
    var spark = document.getElementById(sec + 'Sparkline');
    var sparkLabel = document.getElementById(sec + 'SparkLabel');
    spark.style.display = 'none';
    sparkLabel.style.display = 'none';
    if (data.card_id) {
        fetch('/price-history/' + encodeURIComponent(data.card_id))
            .then(function(r) { return r.json(); })
            .then(function(pts) {
                if (!pts || pts.length < 2) return;
                pts = pts.slice().reverse();
                var prices = pts.map(function(p) { return p.price; });
                var minP = Math.min.apply(null, prices), maxP = Math.max.apply(null, prices);
                var range = maxP - minP || 1;
                var W = 150, H = 40, pad = 2;
                spark.width = W; spark.height = H;
                spark.style.display = 'block';
                var ctx = spark.getContext('2d');
                ctx.clearRect(0, 0, W, H);
                var up = prices[prices.length - 1] >= prices[0];
                ctx.strokeStyle = up ? '#4ecca3' : '#e94560';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                for (var i = 0; i < prices.length; i++) {
                    var x = pad + (i / (prices.length - 1)) * (W - 2 * pad);
                    var y = H - pad - ((prices[i] - minP) / range) * (H - 2 * pad);
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.stroke();
                sparkLabel.style.display = 'block';
                var diff = prices[prices.length - 1] - prices[0];
                var sign = diff >= 0 ? '+' : '';
                sparkLabel.textContent = '30d: ' + sign + diff.toFixed(2) + ' (' + pts[0].date + ' \u2192 ' + pts[pts.length-1].date + ')';
                sparkLabel.style.color = up ? '#4ecca3' : '#e94560';
            })
            .catch(function() {});
    }
    // Show add button
    var addBtn = document.getElementById(sec + 'AddBtn');
    var msg = document.getElementById(sec + 'Msg');
    msg.style.display = 'none'; msg.textContent = '';
    if (data.card_id) {
        addBtn.style.display = 'block';
        addBtn.dataset.cardId = data.card_id;
        addBtn.dataset.cardName = cardName;
        addBtn.dataset.price = displayPrice || '';
    } else {
        addBtn.style.display = 'none';
    }
    // Store last scan data for section
    _lastScanData[sec] = data;
    reopenCamera(sec);
}

// ===== Add to Inventory / Cart — unified =====
var _lastScanData = {inv: null, cart: null};
function addToSection(sec) {
    var btn = document.getElementById(sec + 'AddBtn');
    var msg = document.getElementById(sec + 'Msg');
    var cardId = btn.dataset.cardId;
    if (!cardId) return;
    btn.disabled = true;
    btn.textContent = 'Adding...';
    var url, body, label;
    if (sec === 'inv') {
        url = '/inventory/add';
        body = JSON.stringify({card_id: cardId, quantity: 1});
        label = 'Add to Inventory';
    } else {
        url = '/cart/add';
        var cardName = btn.dataset.cardName || cardId;
        var price = parseFloat(btn.dataset.price) || 0;
        body = JSON.stringify({card_id: cardId, card_name: cardName, market_price: price, quantity: 1});
        label = 'Add to Cart';
    }
    fetch(url, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: body })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        btn.disabled = false; btn.textContent = label;
        msg.style.display = 'block';
        if (data.error) {
            msg.style.color = '#e94560'; msg.textContent = data.error;
        } else if (sec === 'inv') {
            msg.style.color = '#4ecca3'; msg.textContent = 'Added! Total in inventory: ' + data.quantity;
        } else {
            msg.style.color = '#3498db'; msg.textContent = 'In cart! Qty: ' + data.quantity + ' | Cart total: $' + (data.cart_total || 0).toFixed(2);
        }
    })
    .catch(function(e) {
        btn.disabled = false; btn.textContent = label;
        msg.style.display = 'block'; msg.style.color = '#e94560'; msg.textContent = 'Error: ' + e;
    });
}

// ===== Polling — parameterized =====
function pollResult(scanId, sec) {
    if (typeof EventSource !== 'undefined') {
        var es = new EventSource('/events/' + scanId);
        es.addEventListener('resolved', function(e) { es.close(); showResult(JSON.parse(e.data), sec); });
        es.addEventListener('timeout', function() {
            es.close();
            document.getElementById(sec + 'CardName').textContent = 'Identification timed out';
            document.getElementById(sec + 'CardPrice').textContent = '';
        });
        es.onerror = function() { es.close(); _pollFallback(scanId, sec); };
    } else {
        _pollFallback(scanId, sec);
    }
}
function _pollFallback(scanId, sec) {
    var poll = setInterval(function() {
        fetch('/result/' + scanId).then(function(r) { return r.json(); }).then(function(data) {
            if (data.status === 'resolved') { clearInterval(poll); showResult(data, sec); }
        });
    }, 3000);
}

// ===== Continuous scanning — parameterized =====
var continuousState = {inv: {active: false, count: 0}, cart: {active: false, count: 0}};
function toggleContinuous(sec) {
    var other = sec === 'inv' ? 'cart' : 'inv';
    var checkbox = document.getElementById(sec + 'Continuous');
    var otherCheckbox = document.getElementById(other + 'Continuous');
    if (checkbox.checked) {
        otherCheckbox.checked = false;
        continuousState[other].active = false;
        document.getElementById(other + 'ContinuousInfo').style.display = 'none';
        continuousState[sec].active = true;
        continuousState[sec].count = 0;
        document.getElementById(sec + 'ContinuousInfo').style.display = 'block';
        document.getElementById(sec + 'ContinuousCount').textContent = '0';
    } else {
        continuousState[sec].active = false;
        document.getElementById(sec + 'ContinuousInfo').style.display = 'none';
    }
}
function stopContinuous(sec) {
    document.getElementById(sec + 'Continuous').checked = false;
    continuousState[sec].active = false;
    document.getElementById(sec + 'ContinuousInfo').style.display = 'none';
}
function reopenCamera(sec) {
    if (continuousState[sec] && continuousState[sec].active) {
        continuousState[sec].count++;
        document.getElementById(sec + 'ContinuousCount').textContent = continuousState[sec].count;
        setTimeout(function() { document.getElementById(sec + 'Camera').click(); }, 1200);
    }
}
function reopenPageCamera(sec) {
    if (continuousState[sec] && continuousState[sec].active) {
        continuousState[sec].count++;
        document.getElementById(sec + 'ContinuousCount').textContent = continuousState[sec].count;
        setTimeout(function() { document.getElementById(sec + 'PageCamera').click(); }, 1200);
    }
}

// ===== Wire up file inputs for both sections =====
['inv', 'cart'].forEach(function(sec) {
    document.getElementById(sec + 'Camera').onchange = function() { handleFile(this.files[0], sec); this.value=''; };
    document.getElementById(sec + 'Gallery').onchange = function() { handleFile(this.files[0], sec); this.value=''; };
    document.getElementById(sec + 'PageCamera').onchange = function() { handlePageFile(this.files[0], sec); this.value=''; };
    document.getElementById(sec + 'PageGallery').onchange = function() { handlePageFile(this.files[0], sec); this.value=''; };
});

// ===== Page scan — parameterized by section =====
var _pageCardsData = {inv: [], cart: []};
function _showCardDetail(sec, idx) {
    var c = _pageCardsData[sec][idx];
    if (!c) return;
    var overlay = document.getElementById('cardDetailOverlay');
    var body = document.getElementById('cardDetailBody');
    var accentColor = sec === 'inv' ? '#4ecca3' : '#3498db';
    var displayPrice = c.variant_price || c.market_price;
    var h = '';
    var imgSrc = c.local_image_url || c.image_url || '';
    if (imgSrc) {
        h += '<img src="' + imgSrc + '" style="width:100%;max-width:280px;display:block;margin:0 auto 12px;border-radius:8px;" />';
    }
    var nameText = c.card_name || 'Unknown';
    if (c.card_name_jp) {
        nameText = nameText + ' <span style="color:#888;font-size:0.85em;font-weight:normal;">(' + c.card_name_jp + ')</span>';
    }
    if (c.tcgplayer_url) {
        h += '<div style="font-size:18px;font-weight:bold;margin-bottom:4px;"><a href="' + c.tcgplayer_url + '" target="_blank" rel="noopener" style="color:#e0e0e0;text-decoration:underline;text-decoration-color:' + accentColor + ';">' + nameText + '</a></div>';
    } else {
        h += '<div style="font-size:18px;font-weight:bold;color:#e0e0e0;margin-bottom:4px;">' + nameText + '</div>';
    }
    if (c.set_name) h += '<div style="font-size:13px;color:#888;margin-bottom:8px;">' + c.set_name + '</div>';
    // Render all detected stamp badges for inventory/cart list items
    var listStamps = [];
    if (c.stamp_details && typeof c.stamp_details === 'object') {
        var lsKeys = Object.keys(c.stamp_details);
        for (var lsi = 0; lsi < lsKeys.length; lsi++) listStamps.push(lsKeys[lsi]);
    } else if (c.stamps_detected && c.stamps_detected.length > 0) {
        for (var lsi2 = 0; lsi2 < c.stamps_detected.length; lsi2++) listStamps.push(c.stamps_detected[lsi2]);
    } else if (c.detected_variant && c.detected_variant !== 'normal') {
        listStamps.push(c.detected_variant);
    }
    if (listStamps.length > 0) {
        var lsSeen = {};
        h += '<div style="margin-bottom:8px;">';
        for (var lsi3 = 0; lsi3 < listStamps.length; lsi3++) {
            var lst = listStamps[lsi3];
            var lsInfo = _STAMP_BADGE_MAP[lst];
            var lsLabel = lsInfo ? lsInfo.label : lst.replace(/_/g, ' ');
            if (lsSeen[lsLabel]) continue;
            lsSeen[lsLabel] = true;
            var lsCls = lsInfo ? lsInfo.cssClass : '';
            h += '<span class="variant-badge' + (lsCls ? ' ' + lsCls : '') + '">' + lsLabel + '</span>';
        }
        h += '</div>';
    }
    if (displayPrice) {
        h += '<div style="font-size:24px;font-weight:bold;color:' + accentColor + ';margin:8px 0;">$' + parseFloat(displayPrice).toFixed(2) + '</div>';
    } else {
        h += '<div style="font-size:16px;color:#666;margin:8px 0;">No price data</div>';
    }
    if (c.condition_prices) {
        var cp = c.condition_prices;
        var conds = ['NM','LP','MP','HP','DMG'];
        var clrs = {'NM':'#4ecca3','LP':'#a8d8a8','MP':'#f0c040','HP':'#e08040','DMG':'#e94560'};
        var allJtcg2 = Object.keys(cp).every(function(k){ return cp[k].source === 'justtcg'; });
        h += '<table style="width:100%;border-collapse:collapse;font-size:13px;margin:8px 0;">';
        h += '<tr style="border-bottom:1px solid #333;"><th style="text-align:left;padding:4px 8px;color:#888;">Cond</th><th style="text-align:right;padding:4px 8px;color:#888;">Price</th><th style="text-align:right;padding:4px 8px;color:#888;">' + (allJtcg2 ? 'Source' : 'Range') + '</th></tr>';
        for (var ci = 0; ci < conds.length; ci++) {
            var cond = conds[ci];
            var info = cp[cond];
            if (!info) continue;
            h += '<tr style="border-bottom:1px solid #222;">';
            h += '<td style="padding:4px 8px;color:' + clrs[cond] + ';font-weight:bold;">' + cond + '</td>';
            var pricePrefix = '$';
            h += '<td style="padding:4px 8px;text-align:right;color:#e0e0e0;">' + pricePrefix + info.price.toFixed(2) + '</td>';
            if (info.source === 'justtcg') {
                h += '<td style="padding:4px 8px;text-align:right;color:#4ecca3;font-size:11px;">market</td>';
            } else if (info.range_low != null) {
                h += '<td style="padding:4px 8px;text-align:right;color:#888;font-size:11px;">$' + info.range_low.toFixed(2) + ' - $' + info.range_high.toFixed(2) + '</td>';
            } else {
                h += '<td></td>';
            }
            h += '</tr>';
        }
        h += '</table>';
    }
    if (c.confidence) {
        h += '<div style="font-size:12px;color:#888;margin-top:6px;">' + (c.confidence * 100).toFixed(0) + '% confidence' + (c.method ? ' via ' + c.method : '') + '</div>';
    }
    if (c.card_id) {
        h += '<div style="font-size:11px;color:#555;margin-top:4px;">' + c.card_id + '</div>';
    }
    body.innerHTML = h;
    overlay.style.display = 'flex';
}

function handlePageFile(file, sec) {
    if (!file) return;
    var preview = document.getElementById(sec + 'Preview');
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
    var spinner = document.getElementById(sec + 'Spinner');
    var pageResult = document.getElementById(sec + 'PageResult');
    var singleResult = document.getElementById(sec + 'Result');
    spinner.classList.add('show');
    startScanTimer(sec);
    pageResult.classList.remove('show');
    singleResult.classList.remove('show');
    var fd = new FormData();
    fd.append('image', file);
    fetch('/scan-page?variants=true', {method: 'POST', body: fd})
        .then(function(r) { return r.json(); })
        .then(function(data) {
            stopScanTimer(sec);
            var elapsedSec = document.getElementById(sec + 'ScanTimer').textContent;
            spinner.classList.remove('show');
            pageResult.classList.add('show');
            if (data.error) {
                document.getElementById(sec + 'PageTotal').textContent = 'Error';
                document.getElementById(sec + 'PageCards').innerHTML = '<div style="color:#e94560;padding:16px;">' + data.error + '</div>';
                return;
            }
            var cards = data.cards || [];
            _pageCardsData[sec] = cards;
            var total = 0;
            if (data.status === 'pending') {
                document.getElementById(sec + 'PageTotal').textContent = 'Page queued (' + (data.scan_id || '') + ')';
                document.getElementById(sec + 'PageCards').innerHTML = '<div style="color:#888;">Segmentation unavailable. Full page image saved for later processing.</div>';
                return;
            }
            var accentColor = sec === 'inv' ? '#4ecca3' : '#3498db';
            var condColors = {'NM':'#4ecca3','LP':'#a8d8a8','MP':'#f1c40f','HP':'#e67e22','DMG':'#e74c3c'};
            var condKeys = ['NM','LP','MP','HP','DMG'];
            var numCols = 3;
            var html = '<div style="display:grid;grid-template-columns:repeat(' + numCols + ',1fr);gap:8px;max-width:600px;margin:12px auto;">';
            for (var i = 0; i < cards.length; i++) {
                var c = cards[i];
                var displayPrice = c.variant_price || c.market_price;
                var price = displayPrice ? parseFloat(displayPrice) : 0;
                total += price;
                var imgSrc = c.local_image_url || c.image_url || '';
                html += '<div onclick="_showCardDetail(\'' + sec + '\',' + i + ')" style="background:#0f3460;border-radius:8px;overflow:hidden;text-align:center;position:relative;cursor:pointer;-webkit-tap-highlight-color:rgba(78,204,163,0.2);">';
                if (imgSrc) {
                    html += '<img src="' + imgSrc + '" style="width:100%;display:block;border-radius:8px 8px 0 0;" />';
                } else {
                    html += '<div style="width:100%;aspect-ratio:5/7;background:#16213e;display:flex;align-items:center;justify-content:center;color:#666;font-size:12px;border-radius:8px 8px 0 0;">No image</div>';
                }
                html += '<div style="padding:6px 4px;">';
                var nameText = c.card_name || 'Unknown';
                if (c.tcgplayer_url) {
                    html += '<div style="font-size:12px;font-weight:bold;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"><a href="' + c.tcgplayer_url + '" target="_blank" rel="noopener" onclick="event.stopPropagation();" style="color:#e0e0e0;text-decoration:underline;text-decoration-color:' + accentColor + '55;">' + nameText + '</a></div>';
                } else {
                    html += '<div style="font-size:12px;font-weight:bold;color:#e0e0e0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + nameText + '</div>';
                }
                if (c.condition_prices || displayPrice) {
                    var cp = c.condition_prices || {};
                    var mpInfo = cp['MP'];
                    var mpPrice = mpInfo && mpInfo.price != null ? mpInfo.price : (displayPrice ? parseFloat(displayPrice) * 0.8 : null);
                    if (mpPrice != null) {
                        html += '<div style="font-size:16px;font-weight:bold;color:' + condColors['MP'] + ';">$' + mpPrice.toFixed(2) + '</div>';
                    } else if (displayPrice) {
                        html += '<div style="font-size:16px;font-weight:bold;color:' + condColors['MP'] + ';">$' + parseFloat(displayPrice).toFixed(2) + '</div>';
                    } else {
                        html += '<div style="font-size:13px;color:#666;">No price</div>';
                    }
                    // 2x2 grid: LP top-left, HP top-right, NM bottom-left, DMG bottom-right
                    var corners = [['LP','HP'],['NM','DMG']];
                    for (var cri = 0; cri < corners.length; cri++) {
                    html += '<div style="display:flex;gap:1px;font-size:8px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:' + (cri === 0 ? '2' : '1') + 'px;">';
                    for (var cci = 0; cci < corners[cri].length; cci++) {
                        var ccond = corners[cri][cci];
                        var cinfo = cp[ccond];
                        var cclr = cinfo && cinfo.price != null ? condColors[ccond] : '#555';
                        var cval = cinfo && cinfo.price != null ? '$' + cinfo.price.toFixed(cinfo.price >= 10 ? 0 : 2) : '\u2014';
                        var cestStyle = (cinfo && cinfo.estimated) ? 'font-style:italic;opacity:0.7;' : '';
                        html += '<div style="flex:1;text-align:center;color:' + cclr + ';background:rgba(255,255,255,0.04);border-radius:3px;padding:2px 0;' + cestStyle + '"><span style="opacity:0.5;font-size:7px;display:block;">' + ccond + '</span>' + cval + '</div>';
                    }
                    html += '</div>';
                    }
                } else if (displayPrice) {
                    html += '<div style="font-size:16px;font-weight:bold;color:' + condColors['MP'] + ';">$' + parseFloat(displayPrice).toFixed(2) + '</div>';
                } else {
                    html += '<div style="font-size:13px;color:#666;">No price</div>';
                }
                html += '<div style="font-size:10px;color:#888;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + (c.set_name || '') + '</div>';
                html += '</div></div>';
            }
            html += '</div>';
            var condTotals = {};
            for (var ci = 0; ci < condKeys.length; ci++) condTotals[condKeys[ci]] = 0;
            for (var ti = 0; ti < cards.length; ti++) {
                if (cards[ti].condition_prices) {
                    for (var ci = 0; ci < condKeys.length; ci++) {
                        var ck = condKeys[ci];
                        if (cards[ti].condition_prices[ck]) condTotals[ck] += cards[ti].condition_prices[ck].price;
                    }
                }
            }
            var identifiedCount = cards.filter(function(c) { return c.card_id && c.confidence >= 0.5; }).length;
            var totalEl = document.getElementById(sec + 'PageTotal');
            totalEl.innerHTML = '<span style="color:#3498db;">' + identifiedCount + '</span>' +
                '<span style="color:#888;"> cards in </span>' +
                '<span style="color:#3498db;">' + elapsedSec + '</span>' +
                '<span style="color:#888;"> \u2014 </span>' +
                '<span style="color:' + condColors['MP'] + ';">MP: $' + (condTotals.MP > 0 ? condTotals.MP.toFixed(2) : (total * 0.8).toFixed(2)) + '</span>' +
                '<span style="color:#888;"> \u00b7 </span>' +
                '<span style="color:' + condColors['NM'] + ';">NM: $' + total.toFixed(2) + '</span>';
            document.getElementById(sec + 'PageCards').innerHTML = html || '<div style="color:#888">No cards identified</div>';
            // Show "Add All" button if cards have IDs
            var hasCards = cards.some(function(c) { return c.card_id; });
            document.getElementById(sec + 'PageBtnRow').style.display = hasCards ? 'flex' : 'none';
            reopenPageCamera(sec);
        })
        .catch(function(e) {
            stopScanTimer(sec);
            spinner.classList.remove('show');
            pageResult.classList.add('show');
            document.getElementById(sec + 'PageTotal').textContent = 'Error';
            document.getElementById(sec + 'PageCards').textContent = '' + e;
            reopenPageCamera(sec);
        });
}

// ===== Add All from page =====
function addAllPage(sec) {
    var cards = _pageCardsData[sec] || [];
    var url = sec === 'inv' ? '/inventory/add' : '/cart/add';
    var promises = [];
    for (var i = 0; i < cards.length; i++) {
        var c = cards[i];
        if (!c.card_id) continue;
        var body;
        if (sec === 'inv') {
            body = JSON.stringify({card_id: c.card_id, quantity: 1});
        } else {
            body = JSON.stringify({card_id: c.card_id, card_name: c.card_name || c.card_id, market_price: parseFloat(c.variant_price || c.market_price) || 0, quantity: 1});
        }
        promises.push(fetch(url, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: body }));
    }
    var btn = document.getElementById(sec + 'PageBtnRow').querySelector('button');
    btn.disabled = true; btn.textContent = 'Adding...';
    Promise.all(promises).then(function() {
        btn.disabled = false;
        btn.textContent = sec === 'inv' ? 'Added All!' : 'Added All!';
        setTimeout(function() { btn.textContent = sec === 'inv' ? 'Add All to Inventory' : 'Add All to Cart'; }, 2000);
    }).catch(function() {
        btn.disabled = false;
        btn.textContent = sec === 'inv' ? 'Add All to Inventory' : 'Add All to Cart';
    });
}

// ===== Manual search (name + move → results → card detail page) =====
(function() {
    const nameInput = document.getElementById('searchName');
    const moveInput = document.getElementById('searchMove');
    const langSelect = document.getElementById('searchLang');
    const dropdown = document.getElementById('searchNameDropdown');
    const resultsDiv = document.getElementById('searchResults');
    let debounceTimer = null;
    let LAST_RESULTS = [];
    // Sort modes: 'release' (default, newest first), 'price_desc', 'price_asc'
    let SORT_MODE = 'release';

    function esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g,
            c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[c]));
    }

    function currentLang() {
        return (langSelect && langSelect.value) || 'en';
    }

    // Per-language autocomplete caches — switching EN/JP loads a different
    // name index, so we key the cache by language and fetch lazily.
    const NAMES_CACHE = {};      // lang -> items[]
    const NAMES_PROMISES = {};   // lang -> inflight promise
    function ensureAllNames() {
        const lang = currentLang();
        if (NAMES_CACHE[lang]) return Promise.resolve(NAMES_CACHE[lang]);
        if (NAMES_PROMISES[lang]) return NAMES_PROMISES[lang];
        NAMES_PROMISES[lang] = fetch('/names/all?lang=' + encodeURIComponent(lang))
            .then(r => r.json())
            .then(items => {
                NAMES_CACHE[lang] = Array.isArray(items) ? items : [];
                return NAMES_CACHE[lang];
            })
            .catch(() => { NAMES_CACHE[lang] = []; return NAMES_CACHE[lang]; });
        return NAMES_PROMISES[lang];
    }

    function renderAutocomplete(items) {
        if (!items.length) {
            dropdown.style.display='none'; dropdown.innerHTML=''; return;
        }
        dropdown.innerHTML = items.map(it =>
            '<div class="ac-item" data-name="' + esc(it.name) + '" '
            + 'style="padding:14px 12px;cursor:pointer;border-bottom:1px solid #222;display:flex;justify-content:space-between;align-items:center;">'
            + '<span>' + esc(it.name) + '</span>'
            + '<span style="color:#888;font-size:12px;">' + it.count + ' prints</span>'
            + '</div>'
        ).join('');
        dropdown.style.display = 'block';
        dropdown.querySelectorAll('.ac-item').forEach(el => {
            el.addEventListener('click', () => {
                nameInput.value = el.getAttribute('data-name');
                dropdown.style.display = 'none';
                runSearch();
            });
        });
    }

    function filterAutocomplete(q) {
        const names = NAMES_CACHE[currentLang()];
        if (!q || !names) return [];
        const ql = q.toLowerCase();
        const prefix = [], sub = [];
        for (const it of names) {
            const n = (it.name || '').toLowerCase();
            const rn = (it.raw_name || '').toLowerCase();
            if (n.startsWith(ql) || rn.startsWith(ql)) prefix.push(it);
            else if (n.includes(ql) || rn.includes(ql)) sub.push(it);
            if (prefix.length >= 40) break;
        }
        const hits = prefix.length ? prefix : sub;
        return hits.slice(0, 10);
    }

    function updateAutocomplete(q) {
        if (!q) { dropdown.style.display='none'; dropdown.innerHTML=''; return; }
        if (NAMES_CACHE[currentLang()]) {
            renderAutocomplete(filterAutocomplete(q));
        } else {
            ensureAllNames().then(() => {
                if (nameInput.value.trim().toLowerCase() === q.toLowerCase()) {
                    renderAutocomplete(filterAutocomplete(q));
                }
            });
        }
    }

    // Language switch — hide move input (no move data for JP) and
    // re-run the current query under the new catalog.
    if (langSelect) {
        langSelect.addEventListener('change', () => {
            const jp = currentLang() === 'ja';
            moveInput.style.display = jp ? 'none' : '';
            moveInput.value = '';
            dropdown.style.display = 'none';
            resultsDiv.innerHTML = '';
            LAST_RESULTS = [];
            // Preload the JP name index so autocomplete works on first keystroke.
            ensureAllNames();
            const q = nameInput.value.trim();
            if (q) updateAutocomplete(q);
        });
    }

    function openCardDetail(cardId) {
        window.location.href = '/card/' + encodeURIComponent(cardId);
    }

    function sortItems(items, mode) {
        const copy = items.slice();
        if (mode === 'price_desc') {
            copy.sort((a, b) => (b.mp_price || 0) - (a.mp_price || 0));
        } else if (mode === 'price_asc') {
            copy.sort((a, b) => (a.mp_price || 999999) - (b.mp_price || 999999));
        }
        // 'release' = server order (release_date DESC), no client-side reorder
        return copy;
    }

    function renderResults() {
        if (!LAST_RESULTS.length) { resultsDiv.innerHTML=''; return; }
        const sorted = sortItems(LAST_RESULTS, SORT_MODE);
        // Sort toolbar
        function btn(val, label) {
            const active = (SORT_MODE === val);
            return '<button data-sort="' + val + '" '
                + 'style="padding:4px 10px;border-radius:12px;border:1px solid #333;'
                + 'background:' + (active ? '#f39c12' : 'transparent') + ';'
                + 'color:' + (active ? '#1a1a2e' : '#f39c12') + ';'
                + 'font-size:11px;font-weight:700;cursor:pointer;">' + esc(label) + '</button>';
        }
        const toolbar = '<div style="display:flex;gap:4px;margin-bottom:6px;">'
            + btn('release', 'Newest')
            + btn('price_desc', 'Price high')
            + btn('price_asc', 'Price low')
            + '</div>';
        const cards = sorted.map((c, i) => {
            const mm = (c.matched_moves && c.matched_moves.length) ? ' — ' + c.matched_moves.map(esc).join(', ') : '';
            let mpHtml = '';
            if (c.mp_price != null) {
                const prefix = '$';
                mpHtml = '<div style="text-align:right;min-width:60px;">'
                       +   '<div style="color:#f0c040;font-size:14px;font-weight:800;">' + prefix + Number(c.mp_price).toFixed(2) + '</div>'
                       +   '<div style="color:#888;font-size:9px;letter-spacing:1px;">MP</div>'
                       + '</div>';
            }
            return '<div class="sr-card" data-idx="' + i + '" '
                + 'style="display:flex;gap:8px;background:#0f3460;margin:4px 0;padding:6px;border-radius:6px;cursor:pointer;align-items:center;">'
                + (c.image_url ? '<img src="' + esc(c.image_url) + '" style="width:44px;height:61px;object-fit:cover;border-radius:4px;">' : '<div style="width:44px;height:61px;background:#222;border-radius:4px;"></div>')
                + '<div style="flex:1;font-size:13px;min-width:0;">'
                +   '<div style="color:#fff;font-weight:600;">' + esc(c.name) + '</div>'
                +   '<div style="color:#888;font-size:11px;">' + esc(c.set_name || c.set_id || '') + ' - ' + esc(c.rarity || '-') + '</div>'
                +   '<div style="color:#f39c12;font-size:11px;">#' + esc((c.card_id.split("-").slice(1).join("-")) || c.card_id) + mm + '</div>'
                + '</div>'
                + mpHtml
                + '</div>';
        }).join('');
        resultsDiv.innerHTML = toolbar + cards;
        resultsDiv.querySelectorAll('button[data-sort]').forEach(b => {
            b.addEventListener('click', () => { SORT_MODE = b.getAttribute('data-sort'); renderResults(); });
        });
        resultsDiv.querySelectorAll('.sr-card').forEach(el => {
            // Prefetch /card-info on first touch/hover so the browser has
            // the payload cached before navigation kicks off the real fetch.
            let prefetched = false;
            const prefetch = () => {
                if (prefetched) return;
                prefetched = true;
                const idx = parseInt(el.getAttribute('data-idx'));
                const cid = sorted[idx].card_id;
                fetch('/card-info?card_id=' + encodeURIComponent(cid)).catch(()=>{});
            };
            el.addEventListener('touchstart', prefetch, {passive: true});
            el.addEventListener('mouseenter', prefetch);
            el.addEventListener('click', () => {
                const idx = parseInt(el.getAttribute('data-idx'));
                openCardDetail(sorted[idx].card_id);
            });
        });
    }

    async function runSearch() {
        const name = nameInput.value.trim();
        const move = moveInput.value.trim();
        if (!name) { resultsDiv.innerHTML=''; return; }
        resultsDiv.innerHTML = '<div style="color:#888;font-size:13px;padding:6px;">Searching…</div>';
        try {
            const u = '/search-cards?name=' + encodeURIComponent(name)
                    + (move ? '&move=' + encodeURIComponent(move) : '')
                    + '&lang=' + encodeURIComponent(currentLang())
                    + '&limit=300';
            const r = await fetch(u);
            const items = await r.json();
            if (!Array.isArray(items)) {
                resultsDiv.innerHTML = '<div style="color:#e94560;font-size:13px;">' + esc(items.error || 'Error') + '</div>';
                return;
            }
            if (items.length === 0) {
                resultsDiv.innerHTML = '<div style="color:#888;font-size:13px;padding:6px;">No matches</div>';
                return;
            }
            // Single-result shortcut: jump straight to the detail page.
            if (items.length === 1) {
                openCardDetail(items[0].card_id);
                return;
            }
            LAST_RESULTS = items;
            renderResults();
        } catch (e) {
            resultsDiv.innerHTML = '<div style="color:#e94560;font-size:13px;">Search error</div>';
        }
    }

    nameInput.addEventListener('input', () => {
        // Local filter is instant — no debounce needed.
        updateAutocomplete(nameInput.value.trim());
    });
    // Pre-warm the index so the first keystroke is already instant.
    ensureAllNames();
    nameInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); dropdown.style.display='none'; runSearch(); }
        else if (e.key === 'Escape') { dropdown.style.display='none'; }
    });
    moveInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => { if (nameInput.value.trim()) runSearch(); }, 200);
    });
    document.addEventListener('click', (e) => {
        if (!nameInput.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.style.display = 'none';
        }
    });
})();
</script>
</body>
</html>
"""


def _parse_multipart(body, content_type):
    """Extract first file from multipart/form-data body.

    Returns (filename, file_bytes) or (None, None).
    """
    m = re.search(r'boundary="?([^\s";]+)"?', content_type)
    if not m:
        return None, None
    boundary = m.group(1).encode()
    parts = body.split(b"--" + boundary)
    for part in parts:
        if b"Content-Disposition" not in part:
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end < 0:
            continue
        header_block = part[:header_end].decode(errors="replace")
        file_data = part[header_end + 4:]
        # Strip trailing CRLF and closing boundary marker (--)
        if file_data.endswith(b"--\r\n"):
            file_data = file_data[:-4]
        if file_data.endswith(b"\r\n"):
            file_data = file_data[:-2]
        fn_match = re.search(r'filename="([^"]*)"', header_block)
        if fn_match and fn_match.group(1):
            return fn_match.group(1), file_data
    return None, None


def _parse_multipart_named(body, content_type):
    """Extract all named files from multipart/form-data body.

    Returns dict of {field_name: (filename, file_bytes)}.
    """
    m = re.search(r'boundary="?([^\s";]+)"?', content_type)
    if not m:
        return {}
    boundary = m.group(1).encode()
    parts = body.split(b"--" + boundary)
    result = {}
    for part in parts:
        if b"Content-Disposition" not in part:
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end < 0:
            continue
        header_block = part[:header_end].decode(errors="replace")
        file_data = part[header_end + 4:]
        # Strip trailing CRLF and closing boundary marker (--)
        if file_data.endswith(b"--\r\n"):
            file_data = file_data[:-4]
        if file_data.endswith(b"\r\n"):
            file_data = file_data[:-2]
        name_match = re.search(r'name="([^"]*)"', header_block)
        fn_match = re.search(r'filename="([^"]*)"', header_block)
        if name_match and fn_match and fn_match.group(1):
            result[name_match.group(1)] = (fn_match.group(1), file_data)
        elif name_match and (not fn_match or not fn_match.group(1)):
            # Plain text field (no filename) — store as (None, raw_bytes)
            result[name_match.group(1)] = (None, file_data)
    return result


def _local_image_url(card_id, ocr_raw=None, variant=None):
    """Return local /card-image/ URL for a card_id if the image file exists.

    card_id format: "ex14-94/normal" or "bw5-107" (with or without variant).
    Checks for normal variant PNG on disk.  Returns None if not found.

    If ocr_raw contains "[JP]" and a Japanese reference image exists for this
    card_id, returns a /jp-card-image/ URL instead.

    If variant is set to a non-normal value, returns a /card-image-variant/
    URL so the overlay system renders stamp/badge indicators on the image.
    """
    if not card_id:
        return None

    # Check for Japanese image preference
    is_jp = ocr_raw and isinstance(ocr_raw, str) and "[JP]" in ocr_raw
    if is_jp and card_id in _jp_image_index:
        jp_path = _jp_image_index[card_id]
        if Path(jp_path).is_file():
            return f"/jp-card-image/{jp_path}"

    # Strip variant suffix if present (e.g. "ex14-94/normal" -> "ex14-94")
    base_id = card_id.split("/")[0] if "/" in card_id else card_id
    # Extract set_id: everything before the last '-' (e.g. "ecard3-H32" -> "ecard3")
    last_dash = base_id.rfind("-")
    if last_dash <= 0:
        return None
    set_id = base_id[:last_dash]
    image_path = CARD_IMAGES_DIR / set_id / f"{base_id}_normal.png"
    if image_path.is_file():
        if variant and variant != "normal":
            from urllib.parse import quote
            return f"/card-image-variant/{quote(base_id)}?variants={quote(variant)}"
        return f"/card-image/{base_id}/normal"
    return None


def _ref_image_path(card_id, variant=None):
    """Return the Path to the local reference image for a card_id, or None.

    If variant is specified and a real variant image exists in the variant
    image index (data/card_images_variants/), returns that path.
    Otherwise falls back to the normal-variant PNG in data/card_images/<set_id>/.
    """
    if not card_id:
        return None
    base_id = card_id.split("/")[0] if "/" in card_id else card_id

    # Check for variant image if a variant type is specified
    if variant and variant != "normal":
        image_key = _DETECTED_VARIANT_TO_IMAGE_KEY.get(variant)
        if image_key and base_id in _variant_image_index:
            vpath_str = _variant_image_index[base_id].get(image_key)
            if vpath_str:
                vpath = Path(vpath_str)
                if vpath.is_file():
                    return vpath

    last_dash = base_id.rfind("-")
    if last_dash <= 0:
        return None
    set_id = base_id[:last_dash]
    image_path = CARD_IMAGES_DIR / set_id / f"{base_id}_normal.png"
    return image_path if image_path.is_file() else None


def _corner_condition(ratio):
    """Map a corner whitening proxy ratio to a TCG condition abbreviation.

    Uses the same thresholds as edge whitening but slightly more lenient
    since corner wear is derived indirectly from edge measurements.
    """
    if ratio <= 0.0:
        return "NM", "Gem Mint"
    if ratio < 0.008:
        return "NM", "Near Mint"
    if ratio < 0.03:
        return "LP", "Lightly Played"
    if ratio < 0.07:
        return "MP", "Moderately Played"
    return "HP", "Heavily Played"


# ---------------------------------------------------------------------------
# Auto-crop: extract a single card from a messy slide-scan frame
# ---------------------------------------------------------------------------

def _autocrop_card(img_path: str) -> str:
    """Try to find and crop the most prominent Pokemon card from a frame.

    Uses multiple detection strategies (Canny edges at various thresholds,
    HSV-based blue binder exclusion) to find card-shaped quadrilaterals.
    Picks the candidate closest to the image centre with the best aspect
    ratio match (Pokemon cards are 63x88mm, ratio ~0.716).

    Falls back to the original image if no suitable card region is found.
    """
    import cv2
    import numpy as np

    CARD_RATIO = 0.716          # 63/88
    OUT_W, OUT_H = 420, 586     # standard output portrait size

    img = cv2.imread(str(img_path))
    if img is None:
        return str(img_path)

    h, w = img.shape[:2]
    min_area = 0.05 * w * h
    max_area = 0.85 * w * h
    center = np.array([w / 2.0, h / 2.0])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates = []  # list of (4,2) float32 arrays

    # --- Strategy 1: Canny edges at multiple thresholds ---
    for lo, hi in [(20, 80), (30, 100), (50, 150)]:
        edges = cv2.Canny(blur, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.02, 0.03, 0.04, 0.05, 0.06):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4:
                    candidates.append(approx.reshape(4, 2).astype(np.float32))
                    break

    # --- Strategy 2: minAreaRect from merged/dilated edge contours ---
    for lo, hi in [(20, 80), (30, 100)]:
        edges = cv2.Canny(blur, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=3)
        edges = cv2.erode(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            rect = cv2.minAreaRect(cnt)
            rw, rh = sorted(rect[1])
            if rh == 0:
                continue
            ratio = rw / rh
            if 0.55 < ratio < 0.90:
                box = cv2.boxPoints(rect).astype(np.float32)
                candidates.append(box)

    # --- Strategy 3: HSV blue-binder exclusion ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, (90, 40, 40), (130, 255, 255))
    mask_card = cv2.bitwise_not(mask_blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask_card = cv2.morphologyEx(mask_card, cv2.MORPH_OPEN, kernel)
    mask_card = cv2.morphologyEx(mask_card, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_card, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = sorted(rect[1])
        if rh == 0:
            continue
        ratio = rw / rh
        if 0.55 < ratio < 0.90:
            box = cv2.boxPoints(rect).astype(np.float32)
            candidates.append(box)

    # --- Score candidates: prefer central, card-shaped, large ---
    best = None
    best_score = float('inf')

    for pts in candidates:
        area = cv2.contourArea(pts)
        if area < min_area or area > max_area:
            continue

        rect = cv2.minAreaRect(pts)
        rw, rh = sorted(rect[1])
        if rh == 0:
            continue
        ratio = rw / rh
        if not (0.60 < ratio < 0.85):
            continue

        M = cv2.moments(pts)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        dist = np.linalg.norm(np.array([cx, cy]) - center)

        ratio_penalty = abs(ratio - CARD_RATIO) * 2.0
        size_bonus = -0.3 * (area / (w * h))
        score = dist / max(w, h) + ratio_penalty + size_bonus

        if score < best_score:
            best_score = score
            best = pts

    if best is None:
        return str(img_path)

    # Validate: reject crops that are too small or uniform (e.g. carpet)
    best_area = cv2.contourArea(best)
    area_frac = best_area / (w * h)

    # If the detected region is already most of the frame, skip cropping
    if area_frac > 0.70:
        return str(img_path)

    # Check the detected region for card-like content (not uniform texture)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, best.astype(np.int32), 255)
    roi_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    roi_pixels = roi_hsv[mask == 255]
    if len(roi_pixels) > 0:
        hue_std = roi_pixels[:, 0].astype(np.float64).std()
        sat_mean = roi_pixels[:, 1].astype(np.float64).mean()
        val_std = roi_pixels[:, 2].astype(np.float64).std()
        # Carpet/uniform backgrounds have very low hue AND value variance.
        # Cards always have printed text/art with meaningful contrast.
        if hue_std < 10 and val_std < 25:
            logger.debug("Auto-crop rejected (uniform region): hue_std=%.1f val_std=%.1f",
                         hue_std, val_std)
            return str(img_path)

    # Order points: TL, TR, BR, BL
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = best.sum(axis=1)
    diff = np.diff(best, axis=1).ravel()
    ordered[0] = best[np.argmin(s)]       # top-left
    ordered[2] = best[np.argmax(s)]       # bottom-right
    ordered[1] = best[np.argmin(diff)]    # top-right
    ordered[3] = best[np.argmax(diff)]    # bottom-left

    # Determine if the detected quad is landscape
    width_avg = (np.linalg.norm(ordered[1] - ordered[0])
                 + np.linalg.norm(ordered[2] - ordered[3])) / 2.0
    height_avg = (np.linalg.norm(ordered[3] - ordered[0])
                  + np.linalg.norm(ordered[2] - ordered[1])) / 2.0

    out_w, out_h = OUT_W, OUT_H
    if width_avg > height_avg:
        out_w, out_h = OUT_H, OUT_W

    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]],
                   dtype=np.float32)
    M_warp = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, M_warp, (out_w, out_h))

    # Rotate landscape to portrait
    if out_w > out_h:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    path_str = str(img_path)
    base, ext = os.path.splitext(path_str)
    crop_path = f"{base}_cropped{ext or '.jpg'}"
    cv2.imwrite(crop_path, warped)
    logger.info("Auto-cropped card: %s -> %s", img_path, crop_path)
    return crop_path


class ScanHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._send_html(HTML_PAGE)
        elif self.path == "/multi":
            from cardprice.scanner_ui import MULTI_CARD_HTML
            self._send_html(MULTI_CARD_HTML)
        elif self.path == "/qr":
            self._send_qr()
        elif self.path == "/inventory/view":
            from cardprice.inventory_ui import INVENTORY_HTML
            self._send_html(INVENTORY_HTML)
        elif self.path == "/cart/view":
            from cardprice.cart_ui import CART_HTML
            self._send_html(CART_HTML)
        elif self.path == "/cart":
            self._send_cart()
        elif self.path == "/cart/clear":
            self._handle_cart_clear()
        elif self.path == "/inventory":
            self._send_inventory()
        elif self.path == "/export":
            self._send_csv_export()
        elif self.path == "/pending":
            self._send_pending()
        elif self.path == "/history":
            self._send_history()
        elif self.path == "/stats":
            self._send_stats()
        elif self.path == "/scan-stats":
            self._send_scan_stats()
        elif self.path == "/card-ledger":
            self._send_card_ledger()
        elif self.path.startswith("/result/"):
            self._send_result(self.path.split("/result/", 1)[1])
        elif self.path.startswith("/events/"):
            self._stream_sse(self.path.split("/events/", 1)[1])
        elif self.path.startswith("/price-history/"):
            from urllib.parse import unquote
            card_id = unquote(self.path.split("/price-history/", 1)[1])
            self._send_price_history(card_id)
        elif self.path.startswith("/card-image-variant/"):
            from urllib.parse import unquote, urlparse, parse_qs
            parsed = urlparse(self.path)
            card_id = unquote(parsed.path.split("/card-image-variant/", 1)[1])
            variants = parse_qs(parsed.query).get("variants", [""])[0]
            self._send_card_image_variant(card_id, variants)
        elif self.path.startswith("/jp-card-image/"):
            from urllib.parse import unquote
            jp_path = unquote(self.path.split("/jp-card-image/", 1)[1])
            self._send_jp_card_image(jp_path)
        elif self.path.startswith("/card-image/"):
            from urllib.parse import unquote
            card_path = unquote(self.path.split("/card-image/", 1)[1])
            self._send_card_image(card_path)
        elif self.path.startswith("/segment-image/"):
            from urllib.parse import unquote
            seg_path = unquote(self.path.split("/segment-image/", 1)[1])
            self._send_segment_image(seg_path)
        elif self.path == "/condition/training/stats":
            self._handle_training_stats()
        elif self.path == "/condition/camera":
            from cardprice.condition_camera_ui import CAMERA_HTML
            self._send_html(CAMERA_HTML)
        elif self.path.startswith("/condition/camera/"):
            from urllib.parse import unquote
            from cardprice.condition_camera_ui import render_camera_html
            card_id = unquote(self.path.split("/condition/camera/", 1)[1])
            # Look up card name for display
            card_name = None
            try:
                from cardprice.db.session import SessionLocal
                from sqlalchemy import text as sql_text
                with SessionLocal() as session:
                    row = session.execute(
                        sql_text("SELECT name FROM dim_cards WHERE card_id = :cid"),
                        {"cid": card_id},
                    ).fetchone()
                    if row:
                        card_name = row[0]
            except Exception:
                pass
            self._send_html(render_camera_html(card_id, card_name))
        elif self.path == "/condition":
            from cardprice.condition_ui import CONDITION_HTML
            self._send_html(CONDITION_HTML)
        elif self.path.startswith("/condition/capture/"):
            from urllib.parse import unquote
            card_id = unquote(self.path.split("/condition/capture/", 1)[1])
            self._send_condition_capture(card_id)
        elif self.path.startswith("/condition/report/"):
            from urllib.parse import unquote
            card_id = unquote(self.path.split("/condition/report/", 1)[1])
            self._send_condition_report(card_id)
        elif self.path.startswith("/condition/heatmap/"):
            self._send_condition_heatmap(self.path.split("/condition/heatmap/", 1)[1])
        elif self.path == "/camera-test":
            from cardprice.camera_test import CAMERA_TEST_HTML
            self._send_html(CAMERA_TEST_HTML)
        elif self.path == "/camera-diag":
            from cardprice.camera_diag import CAMERA_DIAG_HTML
            self._send_html(CAMERA_DIAG_HTML)
        elif self.path == "/page-scanner":
            from cardprice.page_scanner import PAGE_SCANNER_HTML
            self._send_html(PAGE_SCANNER_HTML)
        elif self.path == "/slide-scan-v7":
            from cardprice.slide_scan_v7 import SLIDE_SCAN_V7_HTML
            self._send_html(SLIDE_SCAN_V7_HTML)
        elif self.path.startswith("/names/autocomplete"):
            self._handle_names_autocomplete()
        elif self.path == "/names/all":
            self._handle_names_all()
        elif self.path.startswith("/search-cards"):
            self._handle_search_cards()
        elif self.path.startswith("/card-info"):
            self._handle_card_info()
        elif self.path.startswith("/card-sales"):
            self._handle_card_sales()
        elif self.path.startswith("/card/"):
            self._handle_card_detail_page()
        elif self.path == "/tunnel-url":
            tunnel_file = Path(__file__).resolve().parent.parent / "data" / "tunnel_url.txt"
            url = tunnel_file.read_text().strip() if tunnel_file.is_file() else ""
            self._send_json({"url": url})
        elif self.path == "/install-cert":
            cert_path = Path(__file__).resolve().parent.parent / "data" / "server.crt"
            if cert_path.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "application/x-x509-ca-cert")
                self.send_header("Content-Disposition", "attachment; filename=cardprice-ca.crt")
                self.end_headers()
                self.wfile.write(cert_path.read_bytes())
            else:
                self.send_error(404, "No cert file found")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/scan":
            self._handle_scan()
        elif self.path == "/scan-url":
            self._handle_scan_url()
        elif self.path == "/scan-page" or self.path.startswith("/scan-page?"):
            self._handle_scan_page()
        elif self.path == "/detect-variants" or self.path.startswith("/detect-variants?"):
            self._handle_detect_variants()
        elif self.path == "/resolve":
            self._handle_resolve()
        elif self.path == "/resolve-batch":
            self._handle_resolve_batch()
        elif self.path == "/cart/add":
            self._handle_cart_add()
        elif self.path == "/cart/remove":
            self._handle_cart_remove()
        elif self.path == "/inventory/add":
            self._handle_inventory_add()
        elif self.path == "/inventory/remove":
            self._handle_inventory_remove()
        elif self.path == "/condition/camera/assess":
            self._handle_camera_condition_assess()
        elif self.path == "/condition/training/save":
            self._handle_training_save()
        elif self.path == "/condition/assess":
            self._handle_condition_assess()
        elif self.path == "/slide-scan/identify" or self.path.startswith("/slide-scan/identify?"):
            self._handle_slide_scan_identify()
        elif self.path.startswith("/condition/photo/"):
            self._handle_condition_photo()
        else:
            self.send_error(404)

    def _send_html(self, html):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_condition_heatmap(self, tmpdir_name):
        """Serve the defect heatmap PNG from a condition assessment temp dir.

        URL: /condition/heatmap/<tmpdir_name>
        The heatmap is rendered during /condition/assess and saved as heatmap.png.
        """
        import tempfile as _tf

        tmpdir_name = tmpdir_name.strip("/")
        # Security: only allow simple directory names (no traversal)
        if ".." in tmpdir_name or "/" in tmpdir_name or not tmpdir_name.startswith("condition_"):
            self.send_error(400, "Invalid heatmap path")
            return

        heatmap_path = Path(_tf.gettempdir()) / tmpdir_name / "heatmap.png"
        if not heatmap_path.is_file():
            self.send_error(404, "Heatmap not found (run /condition/assess first)")
            return

        png_data = heatmap_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png_data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(png_data)

    def _send_qr(self):
        """Serve QR code PNG for the server URL (requires qrcode library)."""
        url = f"http://{_get_lan_ip()}:{_server_port}"
        png_data = _generate_qr_png(url)
        if png_data:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png_data)))
            self.end_headers()
            self.wfile.write(png_data)
        else:
            # qrcode library not installed — return a helpful message
            self._send_json(
                {"error": "qrcode library not installed; QR is rendered client-side via JS"},
                status=501,
            )

    def _handle_scan(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.send_error(411, "Content-Length required")
            return
        try:
            length = int(raw_length)
        except (ValueError, TypeError):
            self.send_error(400, "Invalid Content-Length")
            return
        if length <= 0:
            self.send_error(400, "Empty request body")
            return
        if length > MAX_UPLOAD_BYTES:
            self.send_error(413, "Upload too large (max 20 MB)")
            return

        body = self.rfile.read(length)

        filename, file_data = _parse_multipart(body, content_type)
        if not filename or not file_data:
            self.send_error(400, "No image uploaded")
            return

        # Save uploaded image
        ext = Path(filename).suffix or ".jpg"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = UPLOAD_DIR / f"scan_{timestamp}{ext}"
        save_path.write_bytes(file_data)

        # Duplicate detection: compute phash and check for previous scan
        phash_hex = _compute_phash(str(save_path))
        if phash_hex:
            cached = _find_duplicate_scan(phash_hex)
            if cached:
                logger.info("Returning cached result for duplicate scan")
                cached_response = dict(cached)
                cached_response["duplicate"] = True
                self._send_json(cached_response)
                return

        # Run identification (v2 pipeline — same as scan-page)
        try:
            import time as _time
            _t_start = _time.time()
            from cardprice.ml import identify_card_v2
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                result = identify_card_v2(str(save_path), session=session, detect_variants=True)

                detected_variant = result.get("detected_variant", "normal")

                # Stamp text OCR for EX-era cards (same as page pipeline)
                _STAMPED_EX_SETS = {"ex7","ex8","ex9","ex10","ex11",
                                    "ex12","ex13","ex14","ex15","ex16"}
                if result.get("card_id") and detected_variant in (None, "normal"):
                    _set_id = result["card_id"].split("-")[0]
                    if _set_id in _STAMPED_EX_SETS:
                        try:
                            import cv2
                            from cardprice.ml.ocr_matcher import get_rapid_engine
                            _img = cv2.imread(str(save_path))
                            if _img is not None:
                                _h, _w = _img.shape[:2]
                                _crop = _img[int(_h*0.35):int(_h*0.55), int(_w*0.55):int(_w*0.90)]
                                _up = cv2.resize(_crop, (_crop.shape[1]*3, _crop.shape[0]*3))
                                _sr, _ = get_rapid_engine()(_up)
                                if _sr:
                                    _NON_STAMP = {"illus","arita","sugimori","nishida","imakuni",
                                                  "komiya","tokiya","mitsuhiro","atsuko","ken",
                                                  "kagemaru","himeno","masakazu","ryo","kouki",
                                                  "saya","planeta","cr.","5ban","graphics"}
                                    _texts = []
                                    for _, _t, _c in _sr:
                                        if float(_c) < 0.4 or len(_t.strip()) < 3:
                                            continue
                                        if any(w.strip(".,").lower() in _NON_STAMP for w in _t.split()):
                                            continue
                                        if "illus" in _t.lower() or "ilus" in _t.lower():
                                            continue
                                        _texts.append(_t)
                                    if _texts:
                                        detected_variant = "ex_set_stamp"
                                        result["detected_variant"] = "ex_set_stamp"
                        except Exception:
                            pass

                _elapsed = round(_time.time() - _t_start, 1)

                response = {
                    "card_id": result["card_id"],
                    "confidence": result["confidence"],
                    "method": result["method"],
                    "elapsed_seconds": _elapsed,
                    "card_name": None,
                    "market_price": None,
                    "variant_price": None,
                    "set_name": None,
                    "image_url": None,
                    "local_image_url": _local_image_url(result["card_id"], ocr_raw=result.get("raw_response", {}).get("ocr_raw"), variant=detected_variant),
                    "phash": phash_hex,
                    "detected_variant": detected_variant,
                    "variant_confidence": result.get("variant_confidence"),
                    "stamps_detected": result.get("stamps_detected", []),
                    "stamp_details": result.get("stamp_details", {}),
                    "tcgplayer_url": None,
                }

                if result["card_id"]:
                    row = session.execute(
                        sql_text(_PRICE_LOOKUP_SQL),
                        {"cid": result["card_id"]},
                    ).fetchone()
                    if row:
                        response["card_name"] = row.name
                        if getattr(row, "name_jp", None):
                            response["card_name_jp"] = row.name_jp
                        response["set_name"] = row.set_name
                        response["market_price"] = (
                            float(row.market_price) if row.market_price else None
                        )
                        response["image_url"] = row.image_small
                        if row.tcg_product_id:
                            response["tcgplayer_url"] = f"https://www.tcgplayer.com/product/{row.tcg_product_id}"

                        # Look up variant-specific price
                        if detected_variant != "normal":
                            vprice = _lookup_variant_price(session, result["card_id"], detected_variant)
                            if vprice:
                                response["variant_price"] = vprice

                        # Condition-adjusted prices for all raw conditions
                        price_for_conditions = response.get("variant_price") or response.get("market_price")
                        cond_prices = _build_condition_prices(
                            price_for_conditions,
                            tcg_product_id=row.tcg_product_id,
                            variant=detected_variant,
                        )
                        if cond_prices:
                            response["condition_prices"] = cond_prices
                else:
                    # No confident ML match — queue for Claude Code identification
                    scan_id = timestamp
                    pending_meta = PENDING_DIR / f"{scan_id}.json"
                    pending_meta.write_text(json.dumps({
                        "scan_id": scan_id,
                        "image_path": str(save_path),
                        "status": "pending",
                        "phash": phash_hex,
                        "ml_response": result.get("raw_response", {}),
                    }))
                    response["status"] = "pending"
                    response["scan_id"] = scan_id
                    response["message"] = "Card queued for identification"

            # Save resolved scan metadata so future dedup checks can find it
            if response.get("card_id") and phash_hex:
                resolved_meta = PENDING_DIR / f"{timestamp}.json"
                if not resolved_meta.exists():
                    resolved_meta.write_text(json.dumps({
                        "scan_id": timestamp,
                        "image_path": str(save_path),
                        "status": "resolved",
                        "phash": phash_hex,
                        "card_id": response["card_id"],
                        "confidence": response["confidence"],
                        "method": response["method"],
                        "card_name": response.get("card_name"),
                        "market_price": response.get("market_price"),
                        "set_name": response.get("set_name"),
                        "image_url": response.get("image_url"),
                    }))

            self._send_json(response)

        except Exception as e:
            logger.error("Scan error: %s", e)
            self._send_json({"error": str(e)}, status=500)


    def _read_json_body(self, max_bytes=1024 * 1024):
        """Read and parse a JSON request body.  Returns dict or None (sends error)."""
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            self.send_error(400, "Expected application/json")
            return None
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.send_error(411, "Content-Length required")
            return None
        try:
            length = int(raw_length)
        except (ValueError, TypeError):
            self.send_error(400, "Invalid Content-Length")
            return None
        if length <= 0:
            self.send_error(400, "Empty request body")
            return None
        if length > max_bytes:
            self.send_error(413, "Request body too large")
            return None
        body = self.rfile.read(length)
        try:
            return json.loads(body.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_error(400, "Invalid JSON")
            return None

    def _resolve_single(self, scan_id, card_id, confidence=0.95):
        """Resolve a single pending scan.  Returns response dict or error dict."""
        pending_file = PENDING_DIR / f"{scan_id}.json"
        if not pending_file.exists():
            return {"error": f"Scan {scan_id} not found", "_status": 404}

        try:
            scan_data = json.loads(pending_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            return {"error": f"Failed to read scan data: {e}", "_status": 500}

        # Look up card info from database
        card_name = None
        set_name = None
        market_price = None
        image_url = None
        condition_prices = None
        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                row = session.execute(
                    sql_text(_PRICE_LOOKUP_SQL),
                    {"cid": card_id},
                ).fetchone()
                if row:
                    card_name = row.name
                    set_name = row.set_name
                    market_price = float(row.market_price) if row.market_price else None
                    image_url = row.image_small
                    # Extract variant from card_id (format: "set-num/variant")
                    _variant = card_id.rsplit("/", 1)[-1] if "/" in card_id else "normal"
                    condition_prices = _build_condition_prices(
                        row.market_price,
                        tcg_product_id=row.tcg_product_id,
                        variant=_variant,
                    )
        except Exception as e:
            logger.warning("DB lookup failed during resolve for %s: %s", card_id, e)

        # Update the pending scan file
        scan_data["status"] = "resolved"
        scan_data["card_id"] = card_id
        scan_data["confidence"] = confidence
        scan_data["method"] = "manual"
        scan_data["card_name"] = card_name
        scan_data["set_name"] = set_name
        scan_data["market_price"] = market_price
        scan_data["image_url"] = image_url
        pending_file.write_text(json.dumps(scan_data))

        result = {
            "scan_id": scan_id,
            "status": "resolved",
            "card_id": card_id,
            "confidence": confidence,
            "method": "manual",
            "card_name": card_name,
            "set_name": set_name,
            "market_price": market_price,
            "image_url": image_url,
            "local_image_url": _local_image_url(card_id),
        }
        if condition_prices:
            result["condition_prices"] = condition_prices
        return result

    def _handle_resolve(self):
        """Resolve a single pending/unknown scan by providing the correct card_id.

        Accepts JSON: {"scan_id": "...", "card_id": "...", "confidence": 0.95}
        Updates the pending scan JSON and returns card info.
        """
        data = self._read_json_body()
        if data is None:
            return

        scan_id = data.get("scan_id")
        card_id = data.get("card_id")
        if not scan_id or not card_id:
            self._send_json({"error": "Missing required fields: scan_id, card_id"}, status=400)
            return

        confidence = data.get("confidence", 0.95)
        result = self._resolve_single(scan_id, card_id, confidence)
        status = result.pop("_status", 200)
        self._send_json(result, status=status)

    def _handle_resolve_batch(self):
        """Resolve multiple pending scans at once.

        Accepts JSON: {"resolutions": [{"scan_id": "...", "card_id": "...", "confidence": 0.95}, ...]}
        Returns results for each resolution.
        """
        data = self._read_json_body()
        if data is None:
            return

        resolutions = data.get("resolutions")
        if not resolutions or not isinstance(resolutions, list):
            self._send_json({"error": "Missing or invalid 'resolutions' array"}, status=400)
            return

        results = []
        for item in resolutions:
            scan_id = item.get("scan_id")
            card_id = item.get("card_id")
            if not scan_id or not card_id:
                results.append({"error": "Missing scan_id or card_id", "scan_id": scan_id})
                continue
            confidence = item.get("confidence", 0.95)
            result = self._resolve_single(scan_id, card_id, confidence)
            result.pop("_status", None)
            results.append(result)

        self._send_json({"results": results, "count": len(results)})

    def _handle_scan_url(self):
        """Handle image download from URL: fetch the image, identify card, return JSON.

        Accepts JSON: {"url": "https://example.com/card.jpg"}
        Downloads with 10-second timeout and 20MB size limit.
        Returns same response format as /scan.
        """
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            self.send_error(400, "Expected application/json")
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.send_error(411, "Content-Length required")
            return
        try:
            length = int(raw_length)
        except (ValueError, TypeError):
            self.send_error(400, "Invalid Content-Length")
            return
        if length <= 0:
            self.send_error(400, "Empty request body")
            return
        if length > 1024 * 1024:  # 1 MB max for JSON request itself
            self.send_error(413, "Request body too large")
            return

        body = self.rfile.read(length)
        try:
            request_data = json.loads(body.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_error(400, "Invalid JSON")
            return

        url = request_data.get("url")
        if not url:
            self.send_error(400, "Missing 'url' field in JSON body")
            return

        # Validate URL (basic check)
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            self.send_error(400, "Invalid URL: must start with http:// or https://")
            return

        # Download the image with timeout and size limit
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                file_data = b""
                # Read in chunks to enforce size limit
                while True:
                    chunk = response.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    file_data += chunk
                    if len(file_data) > MAX_UPLOAD_BYTES:
                        self.send_error(413, f"Downloaded image too large (max {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB)")
                        return

            if not file_data:
                self.send_error(400, "Downloaded image is empty")
                return

        except urllib.error.URLError as e:
            logger.warning("URL fetch failed: %s", e)
            self.send_error(400, f"Failed to download image: {str(e)[:100]}")
            return
        except socket.timeout:
            self.send_error(408, "Download timeout (10s)")
            return
        except Exception as e:
            logger.warning("URL download error: %s", e)
            self.send_error(400, f"Download error: {str(e)[:100]}")
            return

        # Infer file extension from URL or default to .jpg
        ext = ".jpg"
        if "?" in url:
            path_part = url.split("?")[0]
        else:
            path_part = url
        if "." in path_part:
            ext_candidate = "." + path_part.split(".")[-1].lower()
            if ext_candidate in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
                ext = ext_candidate

        # Save downloaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = UPLOAD_DIR / f"scan_{timestamp}{ext}"
        save_path.write_bytes(file_data)
        logger.info("Downloaded image saved: %s (%d bytes from %s)", save_path, len(file_data), url)

        # Duplicate detection: compute phash and check for previous scan
        phash_hex = _compute_phash(str(save_path))
        if phash_hex:
            cached = _find_duplicate_scan(phash_hex)
            if cached:
                logger.info("Returning cached result for duplicate scan from URL")
                cached_response = dict(cached)
                cached_response["duplicate"] = True
                self._send_json(cached_response)
                return

        # Run identification
        try:
            from cardprice.ml import identify_card_v2
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                result = identify_card_v2(str(save_path), session=session)

                response = {
                    "card_id": result["card_id"],
                    "confidence": result["confidence"],
                    "method": result["method"],
                    "card_name": None,
                    "market_price": None,
                    "set_name": None,
                    "image_url": None,
                    "local_image_url": _local_image_url(result["card_id"], ocr_raw=result.get("raw_response", {}).get("ocr_raw")),
                    "phash": phash_hex,
                    "source_url": url,
                }

                if result["card_id"]:
                    row = session.execute(
                        sql_text(_PRICE_LOOKUP_SQL),
                        {"cid": result["card_id"]},
                    ).fetchone()
                    if row:
                        response["card_name"] = row.name
                        if getattr(row, "name_jp", None):
                            response["card_name_jp"] = row.name_jp
                        response["set_name"] = row.set_name
                        response["market_price"] = (
                            float(row.market_price) if row.market_price else None
                        )
                        response["image_url"] = row.image_small

                        # Condition-adjusted prices for all raw conditions
                        _variant = result["card_id"].rsplit("/", 1)[-1] if "/" in result["card_id"] else "normal"
                        cond_prices = _build_condition_prices(
                            row.market_price,
                            tcg_product_id=row.tcg_product_id,
                            variant=_variant,
                        )
                        if cond_prices:
                            response["condition_prices"] = cond_prices
                else:
                    # No confident ML match — queue for Claude Code identification
                    scan_id = timestamp
                    pending_meta = PENDING_DIR / f"{scan_id}.json"
                    pending_meta.write_text(json.dumps({
                        "scan_id": scan_id,
                        "image_path": str(save_path),
                        "status": "pending",
                        "phash": phash_hex,
                        "source_url": url,
                        "ml_response": result.get("raw_response", {}),
                    }))
                    response["status"] = "pending"
                    response["scan_id"] = scan_id
                    response["message"] = "Card queued for identification"

            # Save resolved scan metadata so future dedup checks can find it
            if response.get("card_id") and phash_hex:
                resolved_meta = PENDING_DIR / f"{timestamp}.json"
                if not resolved_meta.exists():
                    resolved_meta.write_text(json.dumps({
                        "scan_id": timestamp,
                        "image_path": str(save_path),
                        "status": "resolved",
                        "phash": phash_hex,
                        "card_id": response["card_id"],
                        "confidence": response["confidence"],
                        "method": response["method"],
                        "card_name": response.get("card_name"),
                        "market_price": response.get("market_price"),
                        "set_name": response.get("set_name"),
                        "image_url": response.get("image_url"),
                        "source_url": url,
                    }))

            self._send_json(response)

        except Exception as e:
            logger.error("Scan-URL error: %s", e)
            self._send_json({"error": str(e)}, status=500)

    def _handle_scan_page(self):
        """Handle binder page upload: segment into individual cards, identify each.

        Query parameters:
            variants=true  — run variant detection (holo/reverse/stamp checks)
            variants=false — skip variant detection (default, faster)
            correct=1      — apply perspective correction to each card before
                             identification (default off)
        """
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        detect_variants = qs.get("variants", ["false"])[0].lower() in ("true", "1", "yes")
        correct_perspective = qs.get("correct", ["false"])[0].lower() in ("true", "1", "yes")

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.send_error(411, "Content-Length required")
            return
        try:
            length = int(raw_length)
        except (ValueError, TypeError):
            self.send_error(400, "Invalid Content-Length")
            return
        if length <= 0:
            self.send_error(400, "Empty request body")
            return
        if length > MAX_UPLOAD_BYTES:
            self.send_error(413, "Upload too large (max 20 MB)")
            return

        body = self.rfile.read(length)
        filename, file_data = _parse_multipart(body, content_type)
        if not filename or not file_data:
            self.send_error(400, "No image uploaded")
            return

        # Save uploaded page image
        ext = Path(filename).suffix or ".jpg"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = UPLOAD_DIR / f"page_{timestamp}{ext}"
        save_path.write_bytes(file_data)
        logger.info("Binder page saved: %s (%d bytes)", save_path, len(file_data))

        # Try to segment cards from the page
        card_images = []
        segmentation_ok = False
        try:
            from cardprice.ml.card_segmenter import segment_cards
            card_images = segment_cards(str(save_path))
            segmentation_ok = True
            logger.info("Segmented %d cards from binder page", len(card_images))
        except ImportError:
            logger.info("card_segmenter not available, queuing whole page")
        except Exception as e:
            logger.warning("Segmentation failed: %s, queuing whole page", e)

        if not segmentation_ok or not card_images:
            # Queue the whole page image for later processing
            scan_id = f"page_{timestamp}"
            pending_meta = PENDING_DIR / f"{scan_id}.json"
            pending_meta.write_text(json.dumps({
                "scan_id": scan_id,
                "image_path": str(save_path),
                "status": "pending",
                "type": "binder_page",
            }))
            self._send_json({
                "status": "pending",
                "scan_id": scan_id,
                "message": "Binder page queued for processing",
                "cards": [],
            })
            return

        # Identify each segmented card
        cards = []
        try:
            from cardprice.ml import identify_page_v2 as identify_page
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                page_results = identify_page(card_images, session=session,
                                             detect_variants=detect_variants,
                                             correct_perspective=correct_perspective)
                for idx, (card_img_path, result) in enumerate(zip(card_images, page_results)):
                    # Compute grid position (assume 3 columns for binder pages)
                    num_cols = 3
                    row = idx // num_cols
                    col = idx % num_cols
                    # Build URL for the segmented card image
                    seg_rel = str(Path(card_img_path).relative_to(UPLOAD_DIR))
                    detected_variant = result.get("detected_variant", "normal")

                    card_data = {
                        "position": idx,
                        "row": row,
                        "col": col,
                        "card_id": result["card_id"],
                        "confidence": result["confidence"],
                        "method": result["method"],
                        "detected_variant": detected_variant,
                        "variant_confidence": result.get("variant_confidence"),
                        "stamps_detected": result.get("stamps_detected", []),
                        "stamp_details": result.get("stamp_details", {}),
                        "ocr_name": result.get("raw_response", {}).get("ocr_name") if isinstance(result.get("raw_response"), dict) else None,
                        "ocr_conf": result.get("raw_response", {}).get("ocr_confidence") if isinstance(result.get("raw_response"), dict) else None,
                        "card_name": None,
                        "market_price": None,
                        "variant_price": None,
                        "set_name": None,
                        "image_url": None,
                        "tcgplayer_url": None,
                        "local_image_url": _local_image_url(result["card_id"], ocr_raw=result.get("raw_response", {}).get("ocr_raw"), variant=detected_variant),
                        "segment_image_url": f"/segment-image/{seg_rel}",
                    }

                    if result["card_id"]:
                        row = session.execute(
                            sql_text(_PRICE_LOOKUP_SQL),
                            {"cid": result["card_id"]},
                        ).fetchone()
                        if row:
                            card_data["card_name"] = row.name
                            if getattr(row, "name_jp", None):
                                card_data["card_name_jp"] = row.name_jp
                            card_data["set_name"] = row.set_name
                            card_data["market_price"] = (
                                float(row.market_price) if row.market_price else None
                            )
                            card_data["image_url"] = row.image_small
                            if row.tcg_product_id:
                                card_data["tcgplayer_url"] = f"https://www.tcgplayer.com/product/{row.tcg_product_id}"

                            # Look up variant-specific price
                            if detected_variant != "normal":
                                vprice = _lookup_variant_price(session, result["card_id"], detected_variant)
                                if vprice:
                                    card_data["variant_price"] = vprice

                            # Use variant_price for condition pricing if available, else market_price
                            price_for_conditions = card_data["variant_price"] or card_data["market_price"]
                            if price_for_conditions:
                                card_data["condition_prices"] = _build_condition_prices(
                                    price_for_conditions,
                                    tcg_product_id=row.tcg_product_id,
                                    variant=detected_variant,
                                )

                    cards.append(card_data)

                # Log scan results for accuracy tracking
                try:
                    page_id = f"page_{timestamp}_cards"
                    for card_data in cards:
                        log_scan_result(page_id, card_data.get("position", 0), {
                            "card_id": card_data.get("card_id"),
                            "card_name": card_data.get("card_name"),
                            "confidence": card_data.get("confidence"),
                            "method": card_data.get("method"),
                            "ocr_name": card_data.get("raw_response", {}).get("ocr_name") if isinstance(card_data.get("raw_response"), dict) else None,
                            "ocr_conf": card_data.get("raw_response", {}).get("ocr_confidence") if isinstance(card_data.get("raw_response"), dict) else None,
                            "variant_detected": card_data.get("detected_variant"),
                        })
                except Exception as e:
                    logger.warning(f"Failed to log scan results: {e}")

        except Exception as e:
            logger.error("Page scan identification error: %s", e)
            # Reclaim memory even on error path
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            self._send_json({"error": str(e), "cards": []}, status=500)
            return

        total_value = sum(
            (c["variant_price"] or c["market_price"])
            for c in cards
            if (c["variant_price"] or c["market_price"])
        )
        total_mp = sum(
            (c.get("condition_prices", {}) or {}).get("MP", {}).get("price", 0) or 0
            for c in cards
        )
        self._send_json({
            "status": "ok",
            "cards": cards,
            "total_cards": len(cards),
            "total_value": round(total_value, 2),
            "total_mp": round(total_mp, 2),
        })

        # Release temporary objects and reclaim memory after page scan.
        # Page scans allocate many large image arrays and intermediate tensors
        # that Python's refcount GC may not collect promptly.
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _handle_slide_scan_identify(self):
        """Handle slide-scan: receive individual card images, identify each.

        POST /slide-scan/identify
        Multipart form data with fields card_0 through card_8 (each a JPEG).
        Skips segmentation entirely — images are already individual cards.

        Query parameters:
            variants=true  — run variant detection (default true for slide-scan)
            variants=false — skip variant detection

        Returns same JSON format as /scan-page for UI compatibility.
        """
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        detect_variants = qs.get("variants", ["true"])[0].lower() in ("true", "1", "yes")

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.send_error(411, "Content-Length required")
            return
        try:
            length = int(raw_length)
        except (ValueError, TypeError):
            self.send_error(400, "Invalid Content-Length")
            return
        if length <= 0:
            self.send_error(400, "Empty request body")
            return
        if length > MAX_UPLOAD_BYTES:
            self.send_error(413, "Upload too large (max 20 MB)")
            return

        body = self.rfile.read(length)
        fields = _parse_multipart_named(body, content_type)
        if not fields:
            self.send_error(400, "No card images uploaded")
            return

        # Extract card images from card_0 through card_8
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cards_dir = UPLOAD_DIR / f"slide_{timestamp}_cards"
        cards_dir.mkdir(parents=True, exist_ok=True)

        card_images = {}  # position -> path
        num_cols = 3  # default binder page columns
        for field_name, (filename, file_data) in fields.items():
            if not field_name.startswith("card_"):
                continue
            # Support both "card_0" and "card_r0_c1" field name formats
            suffix = field_name.split("card_", 1)[1]
            try:
                if suffix.startswith("r") and "_c" in suffix:
                    # Format: card_r0_c1 -> position = row * cols + col
                    parts = suffix.split("_c")
                    r = int(parts[0][1:])  # strip leading 'r'
                    c = int(parts[1])
                    pos = r * num_cols + c
                else:
                    pos = int(suffix)
            except (ValueError, IndexError):
                continue
            if not file_data or len(file_data) < 100:
                continue
            save_path = cards_dir / f"card_{pos:02d}.jpg"
            save_path.write_bytes(file_data)
            card_images[pos] = str(save_path)

        if not card_images:
            self.send_error(400, "No valid card images found")
            return

        logger.info("Slide-scan: received %d card images in %s",
                     len(card_images), cards_dir)

        # Auto-crop: try to extract a clean card from each frame
        for pos, path in list(card_images.items()):
            try:
                cropped = _autocrop_card(path)
                if cropped != path:
                    card_images[pos] = cropped
            except Exception as e:
                logger.warning("Auto-crop failed for %s: %s", path, e)

        # Sort by position for consistent ordering
        sorted_positions = sorted(card_images.keys())
        card_paths = [card_images[p] for p in sorted_positions]

        # Identify cards using identify_page_v2 (handles parallel OCR + DINOv2)
        cards = []
        try:
            from cardprice.ml import identify_page_v2 as identify_page
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                page_results = identify_page(card_paths, session=session,
                                             detect_variants=detect_variants)

                for idx, (pos, result) in enumerate(zip(sorted_positions, page_results)):
                    num_cols = 3
                    row = pos // num_cols
                    col = pos % num_cols

                    seg_rel = str(Path(card_images[pos]).relative_to(UPLOAD_DIR))
                    detected_variant = result.get("detected_variant", "normal")

                    card_data = {
                        "position": pos,
                        "row": row,
                        "col": col,
                        "card_id": result["card_id"],
                        "confidence": result["confidence"],
                        "method": result["method"],
                        "detected_variant": detected_variant,
                        "variant_confidence": result.get("variant_confidence"),
                        "stamps_detected": result.get("stamps_detected", []),
                        "stamp_details": result.get("stamp_details", {}),
                        "ocr_name": result.get("raw_response", {}).get("ocr_name") if isinstance(result.get("raw_response"), dict) else None,
                        "ocr_conf": result.get("raw_response", {}).get("ocr_confidence") if isinstance(result.get("raw_response"), dict) else None,
                        "card_name": None,
                        "market_price": None,
                        "variant_price": None,
                        "set_name": None,
                        "image_url": None,
                        "tcgplayer_url": None,
                        "local_image_url": _local_image_url(result["card_id"], ocr_raw=result.get("raw_response", {}).get("ocr_raw"), variant=detected_variant),
                        "segment_image_url": f"/segment-image/{seg_rel}",
                    }

                    if result["card_id"]:
                        row_db = session.execute(
                            sql_text(_PRICE_LOOKUP_SQL),
                            {"cid": result["card_id"]},
                        ).fetchone()
                        if row_db:
                            card_data["card_name"] = row_db.name
                            if getattr(row_db, "name_jp", None):
                                card_data["card_name_jp"] = row_db.name_jp
                            card_data["set_name"] = row_db.set_name
                            card_data["market_price"] = (
                                float(row_db.market_price) if row_db.market_price else None
                            )
                            card_data["image_url"] = row_db.image_small
                            if row_db.tcg_product_id:
                                card_data["tcgplayer_url"] = f"https://www.tcgplayer.com/product/{row_db.tcg_product_id}"

                            if detected_variant != "normal":
                                vprice = _lookup_variant_price(session, result["card_id"], detected_variant)
                                if vprice:
                                    card_data["variant_price"] = vprice

                            price_for_conditions = card_data["variant_price"] or card_data["market_price"]
                            if price_for_conditions:
                                card_data["condition_prices"] = _build_condition_prices(
                                    price_for_conditions,
                                    tcg_product_id=row_db.tcg_product_id,
                                    variant=detected_variant,
                                )

                    cards.append(card_data)

        except Exception as e:
            logger.error("Slide-scan identification error: %s", e, exc_info=True)
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            self._send_json({"error": str(e), "cards": []}, status=500)
            return

        total_value = sum(
            (c["variant_price"] or c["market_price"])
            for c in cards
            if (c["variant_price"] or c["market_price"])
        )
        total_mp = sum(
            (c.get("condition_prices", {}) or {}).get("MP", {}).get("price", 0) or 0
            for c in cards
        )
        self._send_json({
            "status": "ok",
            "scan_type": "slide_scan",
            "cards": cards,
            "total_cards": len(cards),
            "total_value": round(total_value, 2),
            "total_mp": round(total_mp, 2),
        })

        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _handle_detect_variants(self):
        """Run variant detection on already-identified cards.

        POST /detect-variants
        Body: JSON with card_ids and their image paths.
        Example: {"cards": [{"card_id": "base1-4/normal", "image_path": "/path/to/img.jpg"}, ...]}

        Returns variant info for each card (detected_variant, variant_confidence,
        stamps_detected, variant_price).
        """
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            self.send_error(400, "Expected application/json")
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.send_error(411, "Content-Length required")
            return
        try:
            length = int(raw_length)
        except (ValueError, TypeError):
            self.send_error(400, "Invalid Content-Length")
            return

        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        card_list = payload.get("cards", [])
        if not card_list:
            self._send_json({"status": "ok", "cards": []})
            return

        try:
            from cardprice.ml import _apply_variant_detection
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            results = []
            with SessionLocal() as session:
                for card_info in card_list:
                    card_id = card_info.get("card_id")
                    image_path = card_info.get("image_path")
                    if not card_id or not image_path:
                        results.append({"card_id": card_id, "error": "missing card_id or image_path"})
                        continue

                    result = {"card_id": card_id}
                    _apply_variant_detection(result, image_path, detect_variants=True)

                    detected_variant = result.get("detected_variant", "normal")
                    variant_price = None
                    if detected_variant != "normal":
                        variant_price = _lookup_variant_price(session, card_id, detected_variant)

                    results.append({
                        "card_id": card_id,
                        "detected_variant": detected_variant,
                        "variant_confidence": result.get("variant_confidence"),
                        "stamps_detected": result.get("stamps_detected", []),
                        "stamp_details": result.get("stamp_details", {}),
                        "variant_checks_run": result.get("variant_checks_run", []),
                        "variant_price": variant_price,
                    })

            self._send_json({"status": "ok", "cards": results})
        except Exception as e:
            logger.error("Variant detection error: %s", e)
            self._send_json({"error": str(e), "cards": []}, status=500)

    def _send_price_history(self, card_id):
        """Return last 30 days of market prices for a card as JSON array."""
        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                # Pick best available subtype: Normal > Holofoil > any
                best_sub_row = session.execute(
                    sql_text("""
                        SELECT subtype_name FROM fact_market_prices
                        WHERE card_id = :cid
                        ORDER BY
                            CASE subtype_name WHEN 'Normal' THEN 0 WHEN 'Holofoil' THEN 1 ELSE 2 END,
                            price_date DESC
                        LIMIT 1
                    """),
                    {"cid": card_id},
                ).fetchone()
                best_subtype = best_sub_row.subtype_name if best_sub_row else 'Normal'

                rows = session.execute(
                    sql_text("""
                        SELECT price_date, market_price
                        FROM fact_market_prices
                        WHERE card_id = :cid AND subtype_name = :subtype
                        ORDER BY price_date DESC
                        LIMIT 30
                    """),
                    {"cid": card_id, "subtype": best_subtype},
                ).fetchall()
                result = [
                    {
                        "date": str(r.price_date),
                        "price": float(r.market_price) if r.market_price else 0,
                    }
                    for r in rows
                ]
                self._send_json(result)
        except Exception as e:
            logger.error("Price history error: %s", e)
            self._send_json([])

    # ---- Shopping Cart endpoints ----

    def _handle_cart_add(self):
        """Add a card to the in-memory shopping cart (upsert).

        Accepts JSON: {"card_id": "ex13-18/normal", "card_name": "Absol",
                        "market_price": 4.79, "set_name": "...", ...}
        Increments quantity if already in cart, otherwise adds with quantity 1.
        """
        data = self._read_json_body()
        if data is None:
            return

        card_id = data.get("card_id")
        if not card_id:
            self._send_json({"error": "Missing required field: card_id"}, status=400)
            return

        quantity = data.get("quantity", 1)
        try:
            quantity = int(quantity)
        except (ValueError, TypeError):
            self._send_json({"error": "quantity must be an integer"}, status=400)
            return
        if quantity < 1:
            self._send_json({"error": "quantity must be >= 1"}, status=400)
            return

        if card_id in CART:
            CART[card_id]["quantity"] += quantity
        else:
            CART[card_id] = {
                "quantity": quantity,
                "card_name": data.get("card_name"),
                "set_name": data.get("set_name"),
                "market_price": data.get("market_price"),
                "condition_prices": data.get("condition_prices"),
                "image_url": data.get("image_url"),
                "tcgplayer_url": data.get("tcgplayer_url"),
            }

        cart_total = sum(
            (item.get("market_price") or 0) * item.get("quantity", 1)
            for item in CART.values()
        )
        self._send_json({
            "card_id": card_id,
            "quantity": CART[card_id]["quantity"],
            "action": "added",
            "cart_size": len(CART),
            "cart_total": round(cart_total, 2),
        })

    def _handle_cart_remove(self):
        """Remove a card from the shopping cart (decrement or delete).

        Accepts JSON: {"card_id": "ex13-18/normal", "quantity": 1}
        Decrements quantity; removes entry if result <= 0.
        """
        data = self._read_json_body()
        if data is None:
            return

        card_id = data.get("card_id")
        if not card_id:
            self._send_json({"error": "Missing required field: card_id"}, status=400)
            return

        quantity = data.get("quantity", 1)
        try:
            quantity = int(quantity)
        except (ValueError, TypeError):
            self._send_json({"error": "quantity must be an integer"}, status=400)
            return
        if quantity < 1:
            self._send_json({"error": "quantity must be >= 1"}, status=400)
            return

        if card_id not in CART:
            self._send_json({"error": f"Card not in cart: {card_id}"}, status=404)
            return

        new_qty = CART[card_id]["quantity"] - quantity
        if new_qty <= 0:
            del CART[card_id]
            new_qty = 0
        else:
            CART[card_id]["quantity"] = new_qty

        self._send_json({
            "card_id": card_id,
            "quantity": new_qty,
            "action": "removed",
            "cart_size": len(CART),
        })

    def _send_cart(self):
        """Return current cart contents as JSON."""
        items = []
        total_value = 0.0
        for card_id, entry in CART.items():
            price = entry.get("market_price")
            qty = entry.get("quantity", 1)
            line_total = float(price) * qty if price is not None else None
            if line_total is not None:
                total_value += line_total
            items.append({
                "card_id": card_id,
                "card_name": entry.get("card_name"),
                "set_name": entry.get("set_name"),
                "quantity": qty,
                "market_price": price,
                "condition_prices": entry.get("condition_prices"),
                "image_url": entry.get("image_url"),
                "tcgplayer_url": entry.get("tcgplayer_url"),
                "line_total": line_total,
            })
        self._send_json({
            "items": items,
            "count": len(items),
            "total_value": round(total_value, 2),
        })

    def _handle_cart_clear(self):
        """Clear all items from the shopping cart."""
        count = len(CART)
        CART.clear()
        self._send_json({
            "action": "cleared",
            "items_removed": count,
        })

    # ------------------------------------------------------------------
    # Manual search (text-based lookup alternative to image scan)
    # ------------------------------------------------------------------

    def _handle_names_autocomplete(self):
        """Return Pokemon names matching a prefix, ranked by printing count.

        GET /names/autocomplete?q=la&limit=10&lang=en
        -> [{"name": "Lapras", "count": 21}, {"name": "Latias", "count": 14}, ...]

        lang=en (default) hits dim_cards; lang=ja hits dim_cards_jp.
        Popularity proxy = number of distinct card_ids with that English name.
        """
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        q = (qs.get("q", [""])[0] or "").strip().lower()
        lang = (qs.get("lang", ["en"])[0] or "en").strip().lower()
        try:
            limit = int(qs.get("limit", ["10"])[0])
        except ValueError:
            limit = 10
        limit = max(1, min(limit, 50))

        if not q:
            self._send_json([])
            return

        index = (
            _get_name_autocomplete_index_jp() if lang == "ja"
            else _get_name_autocomplete_index()
        )
        # Prefix match; substring match as a weaker fallback when no prefix hits.
        prefix_hits = [(n, c) for n, c in index if n.lower().startswith(q)]
        if prefix_hits:
            hits = prefix_hits
        else:
            hits = [(n, c) for n, c in index if q in n.lower()]

        exact = [(n, c) for n, c in hits if n.lower() == q]
        others = [(n, c) for n, c in hits if n.lower() != q]
        ranked = (exact + others)[:limit]
        self._send_json([
            {"name": _display_name(n), "raw_name": n, "count": c}
            for n, c in ranked
        ])

    def _handle_names_all(self):
        """Return the full autocomplete index in one shot.

        GET /names/all?lang=en   (default)
        GET /names/all?lang=ja   (Japanese catalog)

        Used by the client to filter locally.  ~7k entries (EN) / ~3k
        (JP); small enough for one request per language.
        """
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        lang = (qs.get("lang", ["en"])[0] or "en").strip().lower()
        index = (
            _get_name_autocomplete_index_jp() if lang == "ja"
            else _get_name_autocomplete_index()
        )
        self._send_json([
            {"name": _display_name(n), "raw_name": n, "count": c}
            for n, c in index
        ])

    def _handle_search_cards(self):
        """Search for cards matching a Pokemon name and optional move text.

        GET /search-cards?name=Latias&move=psychic+drain&limit=20
        -> [{"card_id":"ex6-22","name":"Latias","set_id":"ex6",
             "set_name":"Legend Maker","rarity":"Rare Holo",
             "image_url":"https://images.pokemontcg.io/ex6/22.png",
             "matched_moves":["psychic drain"]}, ...]

        Returns a single result when the combination narrows to one card,
        otherwise the full matching set (sorted by set age descending).
        """
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        name = (qs.get("name", [""])[0] or "").strip()
        move = (qs.get("move", [""])[0] or "").strip().lower()
        lang = (qs.get("lang", ["en"])[0] or "en").strip().lower()
        try:
            limit = int(qs.get("limit", ["200"])[0])
        except ValueError:
            limit = 200
        limit = max(1, min(limit, 500))

        if not name:
            self._send_json({"error": "name required"}, status=400)
            return

        if lang == "ja":
            results = _search_cards_by_name_jp(name, limit)
        else:
            results = _search_cards_by_name_and_move(name, move, limit)
        self._send_json(results)

    def _handle_card_info(self):
        """Return detail view data for one card.

        GET /card-info?card_id=base1-4
        -> {"card_id": "base1-4", "name": "Charizard", "set_name": "Base Set",
            "image_url": "...", "market_price": 345.0,
            "condition_prices": {...},
            "variants": {"raw":[...], "edition":[...], "finish":[...]},
            "tcgplayer_url": "..."}

        card_id may be bare ("base1-4") or include a variant
        ("base1-4/1st_edition_holofoil"). Pricing prefers the normal
        printing for breadth; the client chooses the variant for the
        final add-to-inventory call.
        """
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        card_id = (qs.get("card_id", [""])[0] or "").strip()
        if not card_id:
            self._send_json({"error": "card_id required"}, status=400)
            return
        info = _card_detail_payload(card_id)
        if info is None:
            self._send_json({"error": f"Card not found: {card_id}"}, status=404)
            return
        self._send_json(info)

    def _handle_card_sales(self):
        """Return recent TCGplayer sales for a card.

        GET /card-sales?tcg_product_id=42382&limit=50
        -> [{"date":"2026-04-16","price":499.0,"condition":"Lightly Played",
             "quantity":1,"printing":"Holofoil"}, ...]
        Reads from data/tcgplayer_sales.db; returns [] if no sales recorded.
        """
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        pid_s = (qs.get("tcg_product_id", [""])[0] or "").strip()
        try:
            pid = int(pid_s) if pid_s else 0
        except ValueError:
            pid = 0
        if pid <= 0:
            self._send_json([])
            return
        try:
            limit = int(qs.get("limit", ["50"])[0])
        except ValueError:
            limit = 50
        limit = max(1, min(limit, 200))

        try:
            import sqlite3
            db_path = Path(__file__).resolve().parent.parent / "data" / "tcgplayer_sales.db"
            if not db_path.is_file():
                self._send_json([])
                return
            con = sqlite3.connect(str(db_path))
            try:
                rows = con.execute(
                    "SELECT sale_date, sale_price, condition, quantity, printing "
                    "FROM tcgplayer_sales WHERE tcg_product_id=? "
                    "ORDER BY sale_date DESC LIMIT ?", (pid, limit)
                ).fetchall()
            finally:
                con.close()
        except Exception as _e:
            logger.warning("card-sales failed for pid=%s: %s", pid, _e)
            self._send_json([])
            return

        sales = []
        for sale_date, price, cond, qty, printing in rows:
            sales.append({
                "date": (sale_date or "")[:10],
                "datetime": sale_date,
                "price": float(price) if price is not None else None,
                "condition": cond,
                "quantity": qty,
                "printing": printing,
            })
        self._send_json(sales)

    def _handle_card_detail_page(self):
        """Serve the card detail HTML page.

        GET /card/<card_id>
        Client-side fetches /card-info and /inventory/add.
        """
        from urllib.parse import unquote
        # /card/<id> -> id (id may include slash if user added variant)
        raw = self.path[len("/card/"):]
        # Strip any query string before decoding
        if "?" in raw:
            raw = raw.split("?", 1)[0]
        card_id = unquote(raw)
        # Guard against path traversal / injection
        if not card_id or any(ch in card_id for ch in ("<", ">", "\"", "'", "\n", "\r")):
            self.send_error(400, "Invalid card_id")
            return
        html = CARD_DETAIL_HTML.replace("__CARD_ID__", card_id)
        self._send_html(html)

    # ---- Inventory endpoints ----

    def _handle_inventory_add(self):
        """Add a card to user inventory (upsert, keyed on card_id + condition).

        Accepts JSON: {"card_id": "base1-4/holofoil", "quantity": 1,
                       "condition": "Moderately Played"}
        Validates card exists in dim_cards, then upserts into user_inventory.
        When condition is given, rows are keyed by (card_id, condition) so a
        Charizard NM and a Charizard MP live as separate rows. When no
        condition is supplied, behaves like the old un-conditioned upsert
        (legacy callers).
        """
        data = self._read_json_body()
        if data is None:
            return

        card_id = data.get("card_id")
        if not card_id:
            self._send_json({"error": "Missing required field: card_id"}, status=400)
            return

        quantity = data.get("quantity", 1)
        try:
            quantity = int(quantity)
        except (ValueError, TypeError):
            self._send_json({"error": "quantity must be an integer"}, status=400)
            return
        if quantity < 1:
            self._send_json({"error": "quantity must be >= 1"}, status=400)
            return

        # user_inventory.condition is constrained to {NM,LP,MP,HP,DMG}. Accept
        # either the 2-letter code or the full name from the client and
        # normalize before the upsert.
        condition = data.get("condition")
        if condition is not None:
            condition = str(condition).strip() or None
        _COND_FULL_TO_CODE = {
            "Near Mint": "NM", "Lightly Played": "LP",
            "Moderately Played": "MP", "Heavily Played": "HP",
            "Damaged": "DMG",
        }
        _COND_CODES = {"NM", "LP", "MP", "HP", "DMG"}
        if condition:
            if condition in _COND_FULL_TO_CODE:
                condition = _COND_FULL_TO_CODE[condition]
            elif condition.upper() in _COND_CODES:
                condition = condition.upper()
            else:
                self._send_json({"error": f"Unknown condition: {condition}"}, status=400)
                return

        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                card = session.execute(
                    sql_text("SELECT card_id FROM dim_cards WHERE card_id = :cid"),
                    {"cid": card_id},
                ).fetchone()
                if not card:
                    self._send_json({"error": f"Card not found: {card_id}"}, status=404)
                    return

                # Upsert keyed on (card_id, condition). NULL-safe equality so
                # pre-existing rows with no condition still match legacy adds.
                if condition:
                    existing = session.execute(
                        sql_text("""
                            SELECT id, quantity FROM user_inventory
                            WHERE card_id = :cid AND condition = :cond
                            ORDER BY id LIMIT 1
                        """),
                        {"cid": card_id, "cond": condition},
                    ).fetchone()
                else:
                    existing = session.execute(
                        sql_text("""
                            SELECT id, quantity FROM user_inventory
                            WHERE card_id = :cid AND condition IS NULL
                            ORDER BY id LIMIT 1
                        """),
                        {"cid": card_id},
                    ).fetchone()

                if existing:
                    new_qty = existing.quantity + quantity
                    session.execute(
                        sql_text("""
                            UPDATE user_inventory
                            SET quantity = :qty, updated_at = NOW()
                            WHERE id = :rid
                        """),
                        {"qty": new_qty, "rid": existing.id},
                    )
                else:
                    new_qty = quantity
                    session.execute(
                        sql_text("""
                            INSERT INTO user_inventory
                                (card_id, quantity, condition, created_at, updated_at)
                            VALUES (:cid, :qty, :cond, NOW(), NOW())
                        """),
                        {"cid": card_id, "qty": new_qty, "cond": condition},
                    )
                session.commit()

                self._send_json({
                    "card_id": card_id,
                    "condition": condition,
                    "quantity": new_qty,
                    "action": "added",
                })
        except Exception as e:
            logger.error("Inventory add error: %s", e)
            self._send_json({"error": str(e)}, status=500)

    def _handle_inventory_remove(self):
        """Remove a card from user inventory (decrement or delete).

        Accepts JSON: {"card_id": "base1-4/holofoil", "quantity": 1}
        Decrements quantity; deletes row if result <= 0.
        """
        data = self._read_json_body()
        if data is None:
            return

        card_id = data.get("card_id")
        if not card_id:
            self._send_json({"error": "Missing required field: card_id"}, status=400)
            return

        quantity = data.get("quantity", 1)
        try:
            quantity = int(quantity)
        except (ValueError, TypeError):
            self._send_json({"error": "quantity must be an integer"}, status=400)
            return
        if quantity < 1:
            self._send_json({"error": "quantity must be >= 1"}, status=400)
            return

        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                row = session.execute(
                    sql_text("SELECT quantity FROM user_inventory WHERE card_id = :cid"),
                    {"cid": card_id},
                ).fetchone()
                if not row:
                    self._send_json({"error": f"Card not in inventory: {card_id}"}, status=404)
                    return

                new_qty = row.quantity - quantity
                if new_qty <= 0:
                    session.execute(
                        sql_text("DELETE FROM user_inventory WHERE card_id = :cid"),
                        {"cid": card_id},
                    )
                    session.commit()
                    self._send_json({
                        "card_id": card_id,
                        "quantity": 0,
                        "action": "removed",
                    })
                else:
                    session.execute(
                        sql_text("""
                            UPDATE user_inventory
                            SET quantity = :qty, updated_at = NOW()
                            WHERE card_id = :cid
                        """),
                        {"cid": card_id, "qty": new_qty},
                    )
                    session.commit()
                    self._send_json({
                        "card_id": card_id,
                        "quantity": new_qty,
                        "action": "decremented",
                    })
        except Exception as e:
            logger.error("Inventory remove error: %s", e)
            self._send_json({"error": str(e)}, status=500)

    def _handle_camera_condition_assess(self):
        """Receive photos from the camera overlay UI, run condition assessment.

        The camera UI sends fields named by step ID (front_straight,
        front_tilt_left, etc.) plus an optional card_id text field.
        We map these to the standard angle names used by the pipeline.
        """
        import tempfile

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.send_error(411, "Content-Length required")
            return
        try:
            length = int(raw_length)
        except (ValueError, TypeError):
            self.send_error(400, "Invalid Content-Length")
            return
        if length <= 0:
            self.send_error(400, "Empty request body")
            return
        max_bytes = 80 * 1024 * 1024
        if length > max_bytes:
            self.send_error(413, f"Upload too large (max {max_bytes // (1024*1024)} MB)")
            return

        body = self.rfile.read(length)
        files = _parse_multipart_named(body, content_type)

        # Map camera step IDs to pipeline angle names
        step_to_angle = {
            "front_straight": "front",
            "front_tilt_left": "oblique",
            "front_tilt_right": "oblique_right",
            "back_straight": "back",
        }

        # Extract optional card_id text field
        supplied_card_id = None
        if "card_id" in files:
            _, raw = files["card_id"]
            supplied_card_id = raw.decode("utf-8", errors="replace").strip() or None

        # Save images to a temp directory
        tmpdir = tempfile.mkdtemp(prefix="condition_cam_")
        saved_paths = {}
        for step_id, (filename, file_data) in files.items():
            if step_id == "card_id":
                continue
            angle = step_to_angle.get(step_id, step_id)
            if filename is None:
                continue
            ext = Path(filename).suffix or ".jpg"
            save_path = Path(tmpdir) / f"{angle}{ext}"
            save_path.write_bytes(file_data)
            saved_paths[angle] = str(save_path)

        if "front" not in saved_paths:
            self._send_json(
                {"error": "At least a front_straight image is required"}, status=400
            )
            return

        logger.info(
            "Camera condition assess: received %d images (card_id=%s), saved to %s",
            len(saved_paths), supplied_card_id, tmpdir,
        )

        # Build response using the same pipeline as _handle_condition_assess
        front_path = saved_paths.get("front")
        tmpdir_name = Path(tmpdir).name
        response = {
            "overall_grade": None,
            "overall_confidence": 0.0,
            "condition": None,
            "sub_grades": {
                "centering": 0.0,
                "surface": 0.0,
                "edges": 0.0,
                "corners": 0.0,
            },
            "defects": [],
            "card_id": supplied_card_id,
            "card_name": None,
            "angles_received": list(saved_paths.keys()),
            "price_multiplier": None,
            "heatmap_url": None,
        }

        # --- Centering ---
        try:
            from cardprice.ml.centering_detector import measure_centering
            centering = measure_centering(image_path=front_path)
            response["sub_grades"]["centering"] = round(
                centering.get("centering_score", 0.0), 1
            )
        except Exception as e:
            logger.warning("Camera assess centering failed: %s", e)

        # --- Edge whitening ---
        try:
            from cardprice.ml.edge_whitening import measure_edge_whitening
            whitening = measure_edge_whitening(front_path)
            tcg_cond = whitening.get("tcg_condition", "NM")
            edge_grade_map = {"NM": 9.5, "LP": 7.0, "MP": 4.5, "HP": 2.0}
            response["sub_grades"]["edges"] = edge_grade_map.get(tcg_cond, 5.0)

            # Corner wear from edge data
            try:
                edge_ratios = sorted(
                    [whitening["edges"][s]["whitening_ratio"]
                     for s in ("top", "bottom", "left", "right")],
                    reverse=True,
                )
                corner_ratio = (edge_ratios[0] + edge_ratios[1]) / 2
                corner_label, _ = _corner_condition(corner_ratio)
                corner_grade_map = {
                    "NM": 9.5, "LP": 7.0, "MP": 4.5, "HP": 2.0, "DMG": 1.0,
                }
                response["sub_grades"]["corners"] = corner_grade_map.get(
                    corner_label, 5.0
                )
            except Exception as ce:
                logger.warning("Camera assess corner derivation failed: %s", ce)
        except Exception as e:
            logger.warning("Camera assess edge whitening failed: %s", e)

        # --- Surface defect detection (if we have a card_id) ---
        card_id = supplied_card_id
        if not card_id:
            try:
                from cardprice.ml import identify_card_v2
                from cardprice.db.session import SessionLocal
                with SessionLocal() as session:
                    id_result = identify_card_v2(front_path, session=session, detect_variants=False)
                    card_id = id_result.get("card_id")
                    if card_id:
                        response["card_id"] = card_id
            except Exception as e:
                logger.warning("Camera assess: card identification failed: %s", e)

        if card_id:
            try:
                from cardprice.db.session import SessionLocal
                from sqlalchemy import text as sql_text
                with SessionLocal() as session:
                    row = session.execute(
                        sql_text("SELECT name FROM dim_cards WHERE card_id = :cid"),
                        {"cid": card_id},
                    ).fetchone()
                    if row:
                        response["card_name"] = row[0]
            except Exception:
                pass

            # Surface detection
            ref_image_path = None
            ref_dir = Path("data/ref_images")
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                candidate = ref_dir / f"{card_id}{ext}"
                if candidate.is_file():
                    ref_image_path = str(candidate)
                    break

            if ref_image_path:
                try:
                    from cardprice.ml.surface_detector import detect_surface_defects
                    surface = detect_surface_defects(
                        front_path, ref_image_path,
                        output_dir=tmpdir,
                    )
                    surface_score = surface.get("surface_score", 5.0)
                    response["sub_grades"]["surface"] = round(surface_score, 1)
                    response["defects"] = surface.get("defects", [])
                    heatmap_file = Path(tmpdir) / "heatmap.png"
                    if heatmap_file.is_file():
                        response["heatmap_url"] = f"/condition/heatmap/{tmpdir_name}"
                except Exception as e:
                    logger.warning("Camera assess surface detection failed: %s", e)

        # --- Compute overall grade ---
        grades = response["sub_grades"]
        non_zero = [v for v in grades.values() if v > 0]
        if non_zero:
            avg = sum(non_zero) / len(non_zero)
            # Map numeric avg to TCG condition
            if avg >= 9.0:
                response["overall_grade"] = "NM"
                response["price_multiplier"] = 1.0
            elif avg >= 7.0:
                response["overall_grade"] = "LP"
                response["price_multiplier"] = 0.80
            elif avg >= 4.5:
                response["overall_grade"] = "MP"
                response["price_multiplier"] = 0.60
            elif avg >= 2.0:
                response["overall_grade"] = "HP"
                response["price_multiplier"] = 0.40
            else:
                response["overall_grade"] = "DMG"
                response["price_multiplier"] = 0.20
            response["overall_confidence"] = min(len(non_zero) / 4.0, 1.0)
            response["condition"] = response["overall_grade"]

        self._send_json(response)

    def _handle_training_save(self):
        """Save captured images as labeled training data for condition grading.

        Receives multipart/form-data with:
          - condition (required): NM/LP/MP/HP/DMG
          - card_id (optional): card identifier
          - front_straight, front_tilt_left, etc.: image files
        """
        import json as _json
        from datetime import datetime, timezone

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.send_error(411, "Content-Length required")
            return
        try:
            length = int(raw_length)
        except (ValueError, TypeError):
            self.send_error(400, "Invalid Content-Length")
            return
        if length <= 0 or length > 80 * 1024 * 1024:
            self.send_error(400, "Invalid request size")
            return

        body = self.rfile.read(length)
        files = _parse_multipart_named(body, content_type)

        # Extract condition label
        valid_conditions = {"NM", "LP", "MP", "HP", "DMG"}
        condition = None
        if "condition" in files:
            _, raw = files["condition"]
            condition = raw.decode("utf-8", errors="replace").strip().upper()
        if condition not in valid_conditions:
            self._send_json(
                {"error": f"condition must be one of {sorted(valid_conditions)}"},
                status=400,
            )
            return

        # Extract optional card_id
        card_id = None
        if "card_id" in files:
            _, raw = files["card_id"]
            card_id = raw.decode("utf-8", errors="replace").strip() or None

        # Setup directories
        base_dir = Path("data/condition_training/camera")
        cond_dir = base_dir / condition
        cond_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        card_id_safe = (card_id or "unknown").replace("/", "_").replace("\\", "_")

        # Save image files
        saved_images = []
        for field_name, (filename, file_data) in files.items():
            if field_name in ("condition", "card_id"):
                continue
            if filename is None:
                continue
            ext = Path(filename).suffix or ".jpg"
            save_name = f"{ts}_{card_id_safe}_{field_name}{ext}"
            save_path = cond_dir / save_name
            save_path.write_bytes(file_data)
            saved_images.append(str(save_path))

        if not saved_images:
            self._send_json({"error": "No image files received"}, status=400)
            return

        # Append to labels.jsonl
        labels_path = base_dir / "labels.jsonl"
        label_entry = {
            "images": saved_images,
            "condition": condition,
            "card_id": card_id or None,
            "timestamp": ts,
            "source": "camera_ui",
        }
        with open(labels_path, "a") as f:
            f.write(_json.dumps(label_entry) + "\n")

        # Count total samples
        total = 0
        if labels_path.is_file():
            with open(labels_path) as f:
                total = sum(1 for _ in f)

        logger.info(
            "Training data saved: condition=%s card_id=%s images=%d total=%d",
            condition, card_id, len(saved_images), total,
        )

        self._send_json({
            "saved": len(saved_images),
            "condition": condition,
            "total_count": total,
        })

    def _handle_training_stats(self):
        """Return counts of training samples per condition."""
        import json as _json

        labels_path = Path("data/condition_training/camera/labels.jsonl")
        counts = {"NM": 0, "LP": 0, "MP": 0, "HP": 0, "DMG": 0}
        total = 0

        if labels_path.is_file():
            with open(labels_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = _json.loads(line)
                        cond = entry.get("condition", "")
                        if cond in counts:
                            counts[cond] += 1
                        total += 1
                    except _json.JSONDecodeError:
                        continue

        self._send_json({"counts": counts, "total": total})

    def _handle_condition_assess(self):
        """Receive up to 4 card photos, run condition assessment pipeline.

        Accepts multipart/form-data with fields:
          - front (required): JPEG image of card front
          - back, oblique, edge (optional): additional angle images
          - card_id (optional): text field with known card_id to skip identification
            and enable surface defect comparison against the reference image

        Returns JSON with overall grade, sub-grades, defect annotations,
        and condition-adjusted pricing.
        """
        import tempfile

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.send_error(411, "Content-Length required")
            return
        try:
            length = int(raw_length)
        except (ValueError, TypeError):
            self.send_error(400, "Invalid Content-Length")
            return
        if length <= 0:
            self.send_error(400, "Empty request body")
            return
        # Condition assess can receive 4 images — allow up to 80 MB
        max_bytes = 80 * 1024 * 1024
        if length > max_bytes:
            self.send_error(413, f"Upload too large (max {max_bytes // (1024*1024)} MB)")
            return

        body = self.rfile.read(length)
        files = _parse_multipart_named(body, content_type)

        if "front" not in files:
            self._send_json(
                {"error": "At least a front image is required"}, status=400
            )
            return

        # Extract optional card_id text field
        supplied_card_id = None
        if "card_id" in files:
            _, raw = files["card_id"]
            supplied_card_id = raw.decode("utf-8", errors="replace").strip() or None

        # Save images to a temp directory
        tmpdir = tempfile.mkdtemp(prefix="condition_")
        saved_paths = {}
        angle_names = ["front", "back", "oblique", "edge"]
        for name in angle_names:
            if name not in files:
                continue
            filename, file_data = files[name]
            if filename is None:
                continue  # skip text fields
            ext = Path(filename).suffix or ".jpg"
            save_path = Path(tmpdir) / f"{name}{ext}"
            save_path.write_bytes(file_data)
            saved_paths[name] = str(save_path)

        logger.info(
            "Condition assess: received %d images (card_id=%s), saved to %s",
            len(saved_paths), supplied_card_id, tmpdir,
        )

        # --- Run assessment pipeline on the front image ---
        front_path = saved_paths.get("front")
        tmpdir_name = Path(tmpdir).name
        response = {
            "overall_grade": 0.0,
            "condition": "NM",
            "sub_grades": {
                "centering": 0.0,
                "surface": 0.0,
                "edges": 0.0,
                "corners": 0.0,
            },
            "defects": [],
            "card_id": supplied_card_id,
            "card_name": None,
            "set_name": None,
            "image_url": None,
            "local_image_url": None,
            "nm_price": None,
            "assessed_price": None,
            "angles_received": list(saved_paths.keys()),
            "temp_dir": tmpdir,
            "heatmap_url": None,
        }

        # Centering detector (HSV-based border measurement)
        try:
            from cardprice.ml.centering_detector import measure_centering
            centering = measure_centering(image_path=front_path)
            centering_score = centering.get("centering_score", 0.0)
            response["sub_grades"]["centering"] = round(centering_score, 1)
            response["centering_detail"] = {
                "lr": centering.get("front_lr", ""),
                "tb": centering.get("front_tb", ""),
                "confidence": centering.get("confidence", 0.0),
            }
        except Exception as e:
            logger.warning("Centering detector failed: %s", e)
            response["centering_detail"] = {"error": str(e)}

        # Edge whitening detector (LAB+HSV border wear)
        try:
            from cardprice.ml.edge_whitening import measure_edge_whitening
            whitening = measure_edge_whitening(front_path)
            tcg_cond = whitening.get("tcg_condition", "NM")
            # Map whitening result to an edge sub-grade (10 = NM, 7 = LP, 4 = MP, 2 = HP)
            edge_grade_map = {"NM": 9.5, "LP": 7.0, "MP": 4.5, "HP": 2.0}
            response["sub_grades"]["edges"] = edge_grade_map.get(tcg_cond, 5.0)
            response["whitening_detail"] = {
                "overall_ratio": whitening.get("overall_ratio", 0.0),
                "worst_edge": whitening.get("worst_edge", ""),
                "worst_ratio": whitening.get("worst_ratio", 0.0),
                "condition_label": whitening.get("condition_label", ""),
                "tcg_condition": tcg_cond,
            }

            # Corner wear: derive from the 4 corner regions of the edge data
            # Use the average of the two worst per-edge whitening ratios as a
            # proxy for corner condition (corners sit at edge intersections).
            try:
                edge_ratios = sorted(
                    [whitening["edges"][s]["whitening_ratio"]
                     for s in ("top", "bottom", "left", "right")],
                    reverse=True,
                )
                # Two worst edges contribute most to corner wear
                corner_ratio = (edge_ratios[0] + edge_ratios[1]) / 2
                corner_label, _ = _corner_condition(corner_ratio)
                corner_grade_map = {
                    "NM": 9.5, "LP": 7.0, "MP": 4.5, "HP": 2.0, "DMG": 1.0,
                }
                response["sub_grades"]["corners"] = corner_grade_map.get(
                    corner_label, 5.0
                )
                response["corner_detail"] = {
                    "proxy_ratio": round(corner_ratio, 6),
                    "condition": corner_label,
                }
            except Exception as ce:
                logger.warning("Corner grade derivation failed: %s", ce)
        except Exception as e:
            logger.warning("Edge whitening detector failed: %s", e)
            response["whitening_detail"] = {"error": str(e)}

        # --- Card identification (use supplied card_id or auto-detect) ---
        card_id = supplied_card_id
        try:
            from cardprice.db.session import SessionLocal
            from cardprice.models.condition_pricing import get_conditioned_price
            from sqlalchemy import text as sql_text

            if not card_id:
                # Auto-identify from the front image
                from cardprice.ml import identify_card_v2
                with SessionLocal() as session:
                    id_result = identify_card_v2(front_path, session=session, detect_variants=False)
                    card_id = id_result.get("card_id")
                    if card_id:
                        response["card_id"] = card_id
                        response["identification_confidence"] = id_result.get("confidence")
                        response["identification_method"] = id_result.get("method")
        except Exception as e:
            logger.warning("Condition assess: card identification failed: %s", e)

        # --- Surface defect detection (DINOv2 patch comparison) ---
        # Requires a reference image, so we need a known card_id
        ref_image_path = None
        if card_id:
            ref_image_path = _ref_image_path(card_id)

        if ref_image_path and ref_image_path.is_file():
            try:
                from cardprice.ml.surface_detector import (
                    detect_surface_defects,
                    estimate_condition,
                    render_heatmap,
                )

                surface_result = detect_surface_defects(
                    front_path, str(ref_image_path)
                )
                surface_cond = estimate_condition(surface_result)

                # Map surface grade: NM=9.5, LP=7, MP=4.5, HP=2, DMG=1
                surface_grade_map = {
                    "NM": 9.5, "LP": 7.0, "MP": 4.5, "HP": 2.0, "DMG": 1.0,
                }
                response["sub_grades"]["surface"] = surface_grade_map.get(
                    surface_cond["grade_abbrev"], 5.0
                )
                response["surface_detail"] = {
                    "defect_score": round(surface_result["defect_score"], 4),
                    "defect_count": surface_result["defect_count"],
                    "defect_ratio": round(surface_result["defect_ratio"], 4),
                    "mean_similarity": round(surface_result["mean_similarity"], 4),
                    "min_similarity": round(surface_result["min_similarity"], 4),
                    "grade": surface_cond["grade"],
                    "grade_abbrev": surface_cond["grade_abbrev"],
                    "confidence": surface_cond["confidence"],
                }

                # Serialize defect patch locations for the client
                response["defects"] = [
                    {"row": r, "col": c, "similarity": round(s, 4)}
                    for r, c, s in surface_result["defect_patches"][:20]  # top-20 worst
                ]

                # Render and save heatmap overlay for the visualization endpoint
                try:
                    heatmap_path = Path(tmpdir) / "heatmap.png"
                    render_heatmap(
                        surface_result["anomaly_map"],
                        output_path=str(heatmap_path),
                        title=f"Surface Defects — score={surface_result['defect_score']:.3f}",
                    )
                    if heatmap_path.is_file():
                        response["heatmap_url"] = f"/condition/heatmap/{tmpdir_name}"
                except Exception as he:
                    logger.warning("Heatmap render failed: %s", he)

            except Exception as e:
                logger.warning("Surface defect detector failed: %s", e)
                response["surface_detail"] = {"error": str(e)}
        else:
            msg = "no reference image" if card_id else "card not identified"
            response["surface_detail"] = {"skipped": msg}

        # --- Compute overall grade from available sub-grades ---
        # Only average sub-grades that have been populated (> 0)
        populated = [
            v for v in response["sub_grades"].values() if v > 0
        ]
        if populated:
            overall = sum(populated) / len(populated)
            response["overall_grade"] = round(overall, 1)
            # Map overall score to TCG condition
            if overall >= 8.5:
                response["condition"] = "NM"
            elif overall >= 6.5:
                response["condition"] = "LP"
            elif overall >= 4.0:
                response["condition"] = "MP"
            elif overall >= 2.0:
                response["condition"] = "HP"
            else:
                response["condition"] = "DMG"

        # --- Look up card metadata and apply condition-adjusted pricing ---
        if card_id:
            try:
                from cardprice.db.session import SessionLocal
                from cardprice.models.condition_pricing import get_conditioned_price
                from sqlalchemy import text as sql_text

                with SessionLocal() as session:
                    row = session.execute(
                        sql_text("""
                            SELECT c.name, s.name as set_name, c.image_small
                            FROM dim_cards c
                            JOIN dim_sets s ON s.set_id = c.set_id
                            WHERE c.card_id = :cid
                        """),
                        {"cid": card_id},
                    ).fetchone()
                    if row:
                        response["card_name"] = row.name
                        response["set_name"] = row.set_name
                        response["image_url"] = row.image_small
                        response["local_image_url"] = _local_image_url(card_id)

                    # Apply condition-adjusted pricing
                    pricing = get_conditioned_price(
                        card_id, response["condition"], session=session
                    )
                    response["nm_price"] = pricing["nm_price"]
                    response["assessed_price"] = pricing["assessed_price"]
                    response["multiplier"] = pricing["multiplier"]
                    response["price_range_low"] = pricing["price_range_low"]
                    response["price_range_high"] = pricing["price_range_high"]
                    response["price_date"] = pricing["price_date"]
            except Exception as e:
                logger.warning("Condition assess: pricing lookup failed: %s", e)

        self._send_json(response)

    def _send_condition_capture(self, card_id):
        """Serve the per-card 4-step condition capture UI.

        GET /condition/capture/<card_id>

        Looks up card metadata (name, set, image) and renders the capture
        wizard pre-filled with that card's identity.
        """
        card_name = None
        set_name = None
        image_url = None

        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                row = session.execute(
                    sql_text("""
                        SELECT c.name, s.name as set_name, c.image_small
                        FROM dim_cards c
                        JOIN dim_sets s ON s.set_id = c.set_id
                        WHERE c.card_id = :cid
                    """),
                    {"cid": card_id},
                ).fetchone()
                if row:
                    card_name = row.name
                    set_name = row.set_name
                    image_url = row.image_small
        except Exception as e:
            logger.warning("Condition capture: card lookup failed: %s", e)

        # Use local image URL if available
        local_url = _local_image_url(card_id)
        if local_url:
            image_url = local_url

        from cardprice.condition_ui import render_capture_html
        html = render_capture_html(card_id, card_name, set_name, image_url)
        self._send_html(html)

    def _handle_condition_photo(self):
        """Receive a single photo for one step, return immediate quality feedback.

        POST /condition/photo/<card_id>/<step>

        Accepts multipart/form-data with a single field 'photo'.
        Returns JSON with quality assessment:
          - quality: "good" | "acceptable" | "poor"
          - message: human-readable feedback
          - blur_score: Laplacian variance (higher = sharper)
          - brightness: mean pixel brightness (0-255)
        """
        # Parse card_id and step from path
        path_parts = self.path.split("/condition/photo/", 1)
        if len(path_parts) < 2:
            self._send_json({"error": "Invalid path"}, status=400)
            return

        from urllib.parse import unquote
        remainder = unquote(path_parts[1]).rstrip("/")
        # remainder is "<card_id>/<step>" where step is 0-3
        last_slash = remainder.rfind("/")
        if last_slash < 0:
            self._send_json({"error": "Missing step index in path"}, status=400)
            return

        card_id = remainder[:last_slash]
        step_str = remainder[last_slash + 1:]
        try:
            step_idx = int(step_str)
        except ValueError:
            self._send_json({"error": "Invalid step index"}, status=400)
            return

        step_names = ["front", "back", "oblique", "edge"]
        if step_idx < 0 or step_idx >= len(step_names):
            self._send_json({"error": "Step must be 0-3"}, status=400)
            return

        step_name = step_names[step_idx]

        # Read and parse the upload
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.send_error(411, "Content-Length required")
            return
        try:
            length = int(raw_length)
        except (ValueError, TypeError):
            self.send_error(400, "Invalid Content-Length")
            return
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self.send_error(400, "Invalid upload size")
            return

        body = self.rfile.read(length)
        files = _parse_multipart_named(body, content_type)

        if "photo" not in files:
            self._send_json({"error": "No 'photo' field in upload"}, status=400)
            return

        filename, file_data = files["photo"]

        # Save to temp file for analysis
        import tempfile
        tmpdir = Path(tempfile.mkdtemp(prefix="condphoto_"))
        ext = Path(filename).suffix if filename else ".jpg"
        photo_path = tmpdir / f"{step_name}{ext}"
        photo_path.write_bytes(file_data)

        # Run quality checks
        quality = "good"
        message = ""
        blur_score = 0.0
        brightness = 128.0

        try:
            import cv2
            import numpy as np

            img = cv2.imread(str(photo_path))
            if img is None:
                self._send_json({
                    "quality": "poor",
                    "message": "Could not decode image",
                    "blur_score": 0,
                    "brightness": 0,
                    "step": step_name,
                })
                return

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Blur detection: Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = float(laplacian.var())

            # Brightness: mean of grayscale
            brightness = float(gray.mean())

            # Resolution check
            h, w = img.shape[:2]
            resolution_ok = min(h, w) >= 480

            # Quality thresholds
            issues = []

            if blur_score < 50:
                issues.append("Image is too blurry")
                quality = "poor"
            elif blur_score < 150:
                issues.append("Image is slightly soft")
                if quality == "good":
                    quality = "acceptable"

            if brightness < 40:
                issues.append("Image is too dark")
                quality = "poor"
            elif brightness < 70:
                issues.append("Image is a bit dark")
                if quality == "good":
                    quality = "acceptable"
            elif brightness > 230:
                issues.append("Image is overexposed")
                quality = "poor"
            elif brightness > 200:
                issues.append("Image is a bit bright")
                if quality == "good":
                    quality = "acceptable"

            if not resolution_ok:
                issues.append("Resolution is low")
                if quality == "good":
                    quality = "acceptable"

            # Step-specific checks
            if step_name == "oblique":
                # For oblique, we actually expect some glare/highlights
                # Check if there are bright spots (potential reflections)
                bright_pixels = np.sum(gray > 220) / gray.size
                if bright_pixels < 0.01 and quality == "good":
                    issues.append("No reflections visible - try angling toward light")
                    quality = "acceptable"

            if quality == "good":
                good_messages = {
                    "front": "Sharp and well-lit - good for centering analysis",
                    "back": "Clear back image - good for whitening detection",
                    "oblique": "Good angle capture - reflections visible for scratch detection",
                    "edge": "Clear edge view - good for corner and edge wear analysis",
                }
                message = good_messages.get(step_name, "Good quality capture")
            elif quality == "acceptable":
                message = ". ".join(issues) + " - usable but consider retaking"
            else:
                message = ". ".join(issues) + " - please retake"

        except ImportError:
            # cv2 not available -- skip quality checks, accept the photo
            quality = "good"
            message = "Photo received (quality check unavailable)"
        except Exception as e:
            logger.warning("Photo quality check failed: %s", e)
            quality = "acceptable"
            message = "Could not fully assess quality"

        # Clean up temp file
        try:
            photo_path.unlink(missing_ok=True)
            tmpdir.rmdir()
        except Exception:
            pass

        logger.info(
            "Condition photo: card_id=%s step=%s quality=%s blur=%.1f brightness=%.1f",
            card_id, step_name, quality, blur_score, brightness,
        )

        self._send_json({
            "quality": quality,
            "message": message,
            "blur_score": round(blur_score, 1),
            "brightness": round(brightness, 1),
            "step": step_name,
            "card_id": card_id,
        })

    def _send_condition_report(self, card_id):
        """Return the most recent condition assessment for a card.

        GET /condition/report/<card_id>

        Looks for cached condition results in the pending_scans directory
        and returns the most recent one matching the given card_id.
        If no cached report exists, returns a stub with the card metadata
        and instructions to run the capture workflow.
        """
        import tempfile as _tf

        # Search temp directories for the most recent condition assessment
        # that matches this card_id
        tmpdir_root = Path(_tf.gettempdir())
        best_report = None
        best_mtime = 0

        for d in tmpdir_root.glob("condition_*"):
            if not d.is_dir():
                continue
            front_path = d / "front.jpg"
            if not front_path.exists():
                # Also check for .jpeg extension
                front_path = d / "front.jpeg"
                if not front_path.exists():
                    continue

            # Check modification time
            mtime = front_path.stat().st_mtime
            if mtime > best_mtime:
                best_mtime = mtime
                best_report = d

        # Look up card metadata
        card_name = None
        set_name = None
        image_url = None
        nm_price = None

        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                row = session.execute(
                    sql_text("""
                        SELECT c.name, s.name as set_name, c.image_small
                        FROM dim_cards c
                        JOIN dim_sets s ON s.set_id = c.set_id
                        WHERE c.card_id = :cid
                    """),
                    {"cid": card_id},
                ).fetchone()
                if row:
                    card_name = row.name
                    set_name = row.set_name
                    image_url = row.image_small

                # Get latest NM price
                price_row = session.execute(
                    sql_text("""
                        SELECT market_price
                        FROM fact_prices
                        WHERE card_id = :cid
                        ORDER BY price_date DESC
                        LIMIT 1
                    """),
                    {"cid": card_id},
                ).fetchone()
                if price_row:
                    nm_price = float(price_row.market_price) if price_row.market_price else None
        except Exception as e:
            logger.warning("Condition report: card lookup failed: %s", e)

        local_url = _local_image_url(card_id)
        if local_url:
            image_url = local_url

        response = {
            "card_id": card_id,
            "card_name": card_name,
            "set_name": set_name,
            "image_url": image_url,
            "nm_price": nm_price,
            "has_report": False,
            "capture_url": f"/condition/capture/{card_id}",
        }

        # If we found a matching condition directory, try to re-derive the report
        # by running the assessor on the saved front image
        if best_report:
            front_path = best_report / "front.jpg"
            if not front_path.exists():
                front_path = best_report / "front.jpeg"

            if front_path.exists():
                try:
                    from cardprice.ml.condition_assessor import assess_condition
                    # Build images dict (assess_condition requires multi-photo)
                    images = {"front": str(front_path)}
                    for angle in ("back", "oblique", "edge"):
                        for ext in ("jpg", "jpeg", "png"):
                            p = best_report / f"{angle}.{ext}"
                            if p.exists():
                                images[angle] = str(p)
                                break
                    result = assess_condition(
                        images,
                        card_id=card_id,
                    )
                    overall_grade = result.get("overall_grade", "NM")
                    price_mult = result.get("price_multiplier", 1.0)

                    response["has_report"] = True
                    response["overall_grade"] = overall_grade
                    response["overall_confidence"] = result.get("overall_confidence", 0.0)
                    response["sub_scores"] = result.get("sub_scores", {})
                    response["modules_run"] = result.get("modules_run", [])
                    response["price_multiplier"] = price_mult

                    if nm_price is not None:
                        response["assessed_price"] = round(nm_price * price_mult, 2)
                    else:
                        response["assessed_price"] = None

                except Exception as e:
                    logger.warning("Condition report: assessment failed: %s", e)
                    response["error"] = f"Assessment failed: {e}"

        self._send_json(response)

    def _send_inventory(self):
        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                rows = session.execute(sql_text("""
                    SELECT ui.card_id, dc.name, dc.set_id, ds.name as set_name,
                           dc.tcg_product_id, dc.image_small,
                           ui.quantity, ui.condition, lp.market_price
                    FROM user_inventory ui
                    JOIN dim_cards dc ON dc.card_id = ui.card_id
                    JOIN dim_sets ds ON ds.set_id = dc.set_id
                    LEFT JOIN LATERAL (
                        SELECT market_price FROM fact_market_prices
                        WHERE card_id = ui.card_id
                        ORDER BY
                            CASE subtype_name WHEN 'Normal' THEN 0 WHEN 'Holofoil' THEN 1 ELSE 2 END,
                            price_date DESC
                        LIMIT 1
                    ) lp ON true
                    ORDER BY dc.name
                """)).fetchall()

                items = []
                for r in rows:
                    item = {
                        "card_id": r.card_id,
                        "name": r.name,
                        "set_id": r.set_id,
                        "set_name": r.set_name,
                        "quantity": r.quantity,
                        "condition": r.condition,
                        "market_price": (
                            float(r.market_price) if r.market_price else None
                        ),
                    }
                    _variant = r.card_id.rsplit("/", 1)[-1] if "/" in r.card_id else "normal"
                    cond_prices = _build_condition_prices(
                        r.market_price,
                        tcg_product_id=r.tcg_product_id,
                        variant=_variant,
                    )
                    if cond_prices:
                        item["condition_prices"] = cond_prices
                    if r.tcg_product_id:
                        item["tcgplayer_url"] = f"https://www.tcgplayer.com/product/{r.tcg_product_id}"
                    local_url = _local_image_url(r.card_id)
                    if local_url:
                        item["image_url"] = local_url
                    elif r.image_small:
                        item["image_url"] = r.image_small
                    items.append(item)
                self._send_json({"items": items, "count": len(items)})
        except Exception as e:
            logger.error("Inventory error: %s", e)
            self._send_json({"error": str(e)}, status=500)

    def _send_csv_export(self):
        """Export inventory as CSV with columns: card_id, name, set_name, variant, quantity, condition, market_price, total_value."""
        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                rows = session.execute(sql_text("""
                    SELECT ui.card_id, dc.name, ds.name as set_name, ui.quantity,
                           ui.condition, lp.market_price
                    FROM user_inventory ui
                    JOIN dim_cards dc ON dc.card_id = ui.card_id
                    JOIN dim_sets ds ON ds.set_id = dc.set_id
                    LEFT JOIN LATERAL (
                        SELECT market_price FROM fact_market_prices
                        WHERE card_id = ui.card_id
                        ORDER BY
                            CASE subtype_name WHEN 'Normal' THEN 0 WHEN 'Holofoil' THEN 1 ELSE 2 END,
                            price_date DESC
                        LIMIT 1
                    ) lp ON true
                    ORDER BY ds.name, dc.name
                """)).fetchall()

                # Extract variant from card_id (format: setnum-cardnum/variant)
                csv_buffer = io.StringIO()
                writer = csv.writer(csv_buffer)

                # Write header
                writer.writerow([
                    "card_id",
                    "name",
                    "set_name",
                    "variant",
                    "quantity",
                    "condition",
                    "market_price",
                    "total_value"
                ])

                # Write data rows
                for r in rows:
                    card_id = r.card_id or ""
                    variant = ""
                    if "/" in card_id:
                        variant = card_id.split("/", 1)[1]

                    market_price = float(r.market_price) if r.market_price else 0.0
                    total_value = market_price * r.quantity

                    writer.writerow([
                        card_id,
                        r.name or "",
                        r.set_name or "",
                        variant,
                        r.quantity or 0,
                        r.condition or "",
                        f"{market_price:.2f}" if market_price > 0 else "",
                        f"{total_value:.2f}" if total_value > 0 else ""
                    ])

                csv_content = csv_buffer.getvalue()
                csv_bytes = csv_content.encode("utf-8")

                # Send CSV response with attachment headers
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="inventory.csv"')
                self.send_header("Content-Length", str(len(csv_bytes)))
                self.end_headers()
                self.wfile.write(csv_bytes)

        except Exception as e:
            logger.error("CSV export error: %s", e)
            self._send_json({"error": str(e)}, status=500)

    def _send_pending(self):
        """List all pending scans awaiting identification."""
        pending = []
        for f in sorted(PENDING_DIR.glob("*.json")):
            data = json.loads(f.read_text())
            if data.get("status") == "pending":
                pending.append(data)
        self._send_json({"pending": pending, "count": len(pending)})

    def _send_history(self):
        """Return the last 50 scans (resolved and pending) sorted by timestamp desc."""
        scans = []
        for f in PENDING_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            scan_id = data.get("scan_id", f.stem)
            status = data.get("status", "unknown")

            # Derive timestamp from scan_id (format: YYYYMMDD_HHMMSS or page_YYYYMMDD_HHMMSS)
            ts_str = scan_id.replace("page_", "")
            try:
                ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                timestamp = ts.isoformat()
            except ValueError:
                # Fallback: use file mtime
                timestamp = datetime.fromtimestamp(
                    os.path.getmtime(str(f))
                ).isoformat()

            entry = {
                "scan_id": scan_id,
                "status": status,
                "card_id": data.get("card_id"),
                "card_name": data.get("card_name"),
                "market_price": data.get("market_price"),
                "method": data.get("method"),
                "timestamp": timestamp,
            }
            scans.append(entry)

        # Sort by timestamp descending, take last 50
        scans.sort(key=lambda s: s["timestamp"], reverse=True)
        scans = scans[:50]

        # Enrich resolved scans that have a card_id but no price data from DB
        resolved_ids = [
            s["card_id"]
            for s in scans
            if s["status"] == "resolved" and s["card_id"] and s["market_price"] is None
        ]
        if resolved_ids:
            try:
                from cardprice.db.session import SessionLocal
                from sqlalchemy import text as sql_text

                with SessionLocal() as session:
                    rows = session.execute(
                        sql_text(_PRICE_LOOKUP_BULK_SQL),
                        {"ids": resolved_ids},
                    ).fetchall()
                    db_lookup = {
                        r.card_id: {"name": r.name, "price": float(r.market_price) if r.market_price else None}
                        for r in rows
                    }
                    for s in scans:
                        if s["card_id"] in db_lookup:
                            info = db_lookup[s["card_id"]]
                            if s["card_name"] is None:
                                s["card_name"] = info["name"]
                            if s["market_price"] is None:
                                s["market_price"] = info["price"]
            except Exception as e:
                logger.debug("History DB enrichment skipped: %s", e)

        self._send_json({"scans": scans, "count": len(scans)})

    def _send_stats(self):
        """Return scanning statistics computed from pending_scans JSON files."""
        scans = []
        for f in PENDING_DIR.glob("*.json"):
            try:
                scans.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, OSError):
                continue

        total = len(scans)
        resolved = sum(1 for s in scans if s.get("status") == "resolved")
        pending = sum(1 for s in scans if s.get("status") == "pending")

        # Method breakdown and average confidence
        method_counts = {}
        method_conf_sums = {}
        method_conf_counts = {}
        for s in scans:
            method = s.get("method")
            if method:
                method_counts[method] = method_counts.get(method, 0) + 1
                conf = s.get("confidence")
                if conf is not None:
                    method_conf_sums[method] = method_conf_sums.get(method, 0.0) + conf
                    method_conf_counts[method] = method_conf_counts.get(method, 0) + 1

        avg_confidence_by_method = {}
        for m, total_conf in method_conf_sums.items():
            count = method_conf_counts[m]
            avg_confidence_by_method[m] = round(total_conf / count, 4)

        # ML index file sizes
        data_dir = Path("data")
        index_files = {
            "hash_db": data_dir / "hash_db.pkl",
            "dino_index": data_dir / "dino_index.faiss",
            "dino_card_ids": data_dir / "dino_card_ids.pkl",
            "clip_text_index": data_dir / "clip_text_index.pkl",
        }
        index_sizes = {}
        for name, path in index_files.items():
            if path.exists():
                size_bytes = path.stat().st_size
                index_sizes[name] = {
                    "bytes": size_bytes,
                    "human": (
                        f"{size_bytes / 1048576:.1f} MB"
                        if size_bytes >= 1048576
                        else f"{size_bytes / 1024:.1f} KB"
                    ),
                }

        # Count card images
        card_images_dir = data_dir / "card_images"
        image_count = 0
        if card_images_dir.exists():
            for entry in card_images_dir.iterdir():
                if entry.is_dir():
                    # set subdirectories contain the actual images
                    image_count += sum(
                        1 for _ in entry.iterdir() if _.is_file()
                    )
                elif entry.is_file():
                    image_count += 1

        self._send_json({
            "total_scans": total,
            "resolved": resolved,
            "pending": pending,
            "method_counts": method_counts,
            "avg_confidence_by_method": avg_confidence_by_method,
            "index_sizes": index_sizes,
            "card_image_count": image_count,
        })

    def _send_scan_stats(self):
        """Return accuracy summary from the scan logger."""
        try:
            summary = get_accuracy_summary()
            self._send_json(summary)
        except Exception as e:
            logger.error("Failed to get scan stats: %s", e)
            self._send_json({"error": str(e)}, status=500)

    def _send_card_ledger(self):
        """Render per-card accuracy table as HTML, sorted worst-first."""
        try:
            ledger = get_card_ledger()
        except Exception as e:
            logger.error("Failed to get card ledger: %s", e)
            self._send_html(f"<html><body><h1>Error</h1><p>{e}</p></body></html>")
            return

        total_cards = len(ledger)
        total_scans = sum(c["scans"] for c in ledger)
        total_correct = sum(c["correct"] for c in ledger)
        overall_acc = (total_correct / total_scans * 100) if total_scans > 0 else 0.0

        rows_html = ""
        for c in ledger:
            acc_pct = c["accuracy"] * 100
            if acc_pct >= 100:
                color = "#4CAF50"
                status = "OK"
            elif acc_pct >= 50:
                color = "#FFC107"
                status = "WARN"
            else:
                color = "#F44336"
                status = "FAIL"
            rows_html += (
                f'<tr>'
                f'<td style="font-family:monospace;font-size:12px">{c["card_id"]}</td>'
                f'<td>{c["card_name"]}</td>'
                f'<td style="text-align:center">{c["scans"]}</td>'
                f'<td style="text-align:center">{c["correct"]}</td>'
                f'<td style="text-align:center;font-weight:bold;color:{color}">{acc_pct:.0f}%</td>'
                f'<td style="text-align:center;color:{color};font-weight:bold">{status}</td>'
                f'</tr>'
            )

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Card Ledger</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #fff; }}
.summary {{ background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: inline-block; }}
.summary span {{ margin-right: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 8px 12px; border-bottom: 1px solid #333; text-align: left; }}
th {{ background: #0f3460; color: #fff; position: sticky; top: 0; }}
tr:hover {{ background: #16213e; }}
</style></head><body>
<h1>Card Accuracy Ledger</h1>
<div class="summary">
<span><b>Unique Cards:</b> {total_cards}</span>
<span><b>Total Scans:</b> {total_scans}</span>
<span><b>Correct:</b> {total_correct}</span>
<span><b>Overall Accuracy:</b> <b style="color:{'#4CAF50' if overall_acc >= 95 else '#FFC107' if overall_acc >= 80 else '#F44336'}">{overall_acc:.1f}%</b></span>
</div>
<table>
<thead><tr><th>Card ID</th><th>Card Name</th><th>Scans</th><th>Correct</th><th>Accuracy</th><th>Status</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
</body></html>"""
        self._send_html(html)

    def _send_result(self, scan_id):
        """Check result of a specific scan by scan_id."""
        meta_path = PENDING_DIR / f"{scan_id}.json"
        if not meta_path.exists():
            self.send_error(404, "Scan not found")
            return
        data = json.loads(meta_path.read_text())
        # If resolved, look up price
        if data.get("status") == "resolved" and data.get("card_id"):
            try:
                from cardprice.db.session import SessionLocal
                from sqlalchemy import text as sql_text
                with SessionLocal() as session:
                    row = session.execute(
                        sql_text(_PRICE_LOOKUP_SQL),
                        {"cid": data["card_id"]},
                    ).fetchone()
                    if row:
                        data["card_name"] = row.name
                        data["set_name"] = row.set_name
                        data["market_price"] = float(row.market_price) if row.market_price else None
                        data["image_url"] = row.image_small
                        data["local_image_url"] = _local_image_url(data["card_id"])
                        _variant = data["card_id"].rsplit("/", 1)[-1] if "/" in data["card_id"] else "normal"
                        cond_prices = _build_condition_prices(
                            row.market_price,
                            tcg_product_id=row.tcg_product_id,
                            variant=_variant,
                        )
                        if cond_prices:
                            data["condition_prices"] = cond_prices
            except Exception as e:
                logger.error("Result lookup error: %s", e)
        self._send_json(data)

    def _stream_sse(self, scan_id):
        """Stream Server-Sent Events for a pending scan until it resolves or times out."""
        meta_path = PENDING_DIR / f"{scan_id}.json"
        if not meta_path.exists():
            self.send_error(404, "Scan not found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        timeout = 5 * 60  # 5 minutes
        start = time.monotonic()
        last_keepalive = start

        try:
            while time.monotonic() - start < timeout:
                # Check current status
                try:
                    data = json.loads(meta_path.read_text())
                except (json.JSONDecodeError, OSError):
                    time.sleep(1)
                    continue

                if data.get("status") == "resolved":
                    # Enrich with DB info if needed
                    if data.get("card_id") and not data.get("card_name"):
                        try:
                            from cardprice.db.session import SessionLocal
                            from sqlalchemy import text as sql_text
                            with SessionLocal() as session:
                                row = session.execute(
                                    sql_text(_PRICE_LOOKUP_SQL),
                                    {"cid": data["card_id"]},
                                ).fetchone()
                                if row:
                                    data["card_name"] = row.name
                                    data["set_name"] = row.set_name
                                    data["market_price"] = float(row.market_price) if row.market_price else None
                                    data["image_url"] = row.image_small
                                    _variant = data["card_id"].rsplit("/", 1)[-1] if "/" in data["card_id"] else "normal"
                                    cond_prices = _build_condition_prices(
                                        row.market_price,
                                        tcg_product_id=row.tcg_product_id,
                                        variant=_variant,
                                    )
                                    if cond_prices:
                                        data["condition_prices"] = cond_prices
                                    data["local_image_url"] = _local_image_url(data["card_id"])
                        except Exception as e:
                            logger.debug("SSE DB enrichment error: %s", e)

                    payload = json.dumps(data)
                    self.wfile.write(f"event: resolved\ndata: {payload}\n\n".encode())
                    self.wfile.flush()
                    return

                # Send keepalive comment every 15 seconds
                now = time.monotonic()
                if now - last_keepalive >= 15:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    last_keepalive = now

                time.sleep(1)

            # Timeout reached
            self.wfile.write(b"event: timeout\ndata: {}\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("SSE client disconnected for scan %s", scan_id)

    def _send_card_image(self, card_path):
        """Serve a local card reference image as PNG.

        URL format: /card-image/bw5-107/normal
        Maps to:    data/card_images/bw5/bw5-107_normal.png

        The set_id is derived by stripping the trailing -<number> suffix from
        the card identifier (everything before the slash).
        """
        card_path = card_path.strip("/")
        if "/" not in card_path:
            self.send_error(400, "Expected format: <card_id>/<variant>")
            return

        base_id, variant = card_path.rsplit("/", 1)

        # Derive set_id: everything before the last '-'
        # e.g. "bw5-107" -> "bw5", "ecard3-H32" -> "ecard3", "swsh12pt5-160" -> "swsh12pt5"
        last_dash = base_id.rfind("-")
        if last_dash <= 0:
            self.send_error(400, "Cannot parse set from card_id")
            return
        set_id = base_id[:last_dash]

        # Build file path: data/card_images/<set_id>/<base_id>_<variant>.png
        filename = f"{base_id}_{variant}.png"
        image_path = CARD_IMAGES_DIR / set_id / filename

        if not image_path.is_file():
            self.send_error(404, f"Image not found: {set_id}/{filename}")
            return

        png_data = image_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png_data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(png_data)

    def _send_jp_card_image(self, jp_path):
        """Serve a Japanese card reference image.

        URL format: /jp-card-image/data/card_images_jp/.../<filename>.jpg
        The jp_path is the full relative path from the project root.
        """
        image_path = Path(jp_path)
        if not image_path.is_file():
            self.send_error(404, f"JP image not found: {jp_path}")
            return

        # Determine content type from extension
        ext = image_path.suffix.lower()
        content_type = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
        }.get(ext, "image/jpeg")

        img_data = image_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(img_data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(img_data)

    # Cache for variant overlay images: (card_id, frozenset(variants)) -> (bytes, content_type)
    _variant_image_cache = {}

    def _send_card_image_variant(self, card_id, variants_str):
        """Serve a card reference image with variant indicator overlays.

        URL format: /card-image-variant/<card_id>?variants=1st_edition,reverse_holo
        Loads the normal variant reference image, applies overlay badges for
        each detected variant, caches the result, and returns PNG.
        """
        card_id = card_id.strip("/")
        variant_list = [v.strip() for v in variants_str.split(",") if v.strip()]

        if not variant_list:
            # No variants requested — redirect to plain card-image
            ref_url = _local_image_url(card_id)
            if ref_url:
                self.send_response(302)
                self.send_header("Location", ref_url)
                self.end_headers()
            else:
                self.send_error(404, f"No reference image for {card_id}")
            return

        # Check cache
        cache_key = (card_id, frozenset(variant_list))
        cached = ScanHandler._variant_image_cache.get(cache_key)
        if cached:
            cached_data, cached_ctype = cached
            self.send_response(200)
            self.send_header("Content-Type", cached_ctype)
            self.send_header("Content-Length", str(len(cached_data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(cached_data)
            return

        # Variant-image fast path: serve a real photograph of the stamped/
        # holo/reverse-holo card directly when one is in variant_image_index.
        # The index was audited 2026-04-07: 19 mislabeled entries removed
        # (e.g. ex15-56 was a Rainbow Energy scan, ex15-12 was a 3-PSA
        # gallery), and 5 verified v2 photos added for the user's binder
        # (ex15-7/23/56 Dragon Frontiers, ex11-11/49 Delta Species).
        primary_variant = variant_list[0] if variant_list else None
        variant_ref = _ref_image_path(card_id, variant=primary_variant)
        if variant_ref and variant_ref != _ref_image_path(card_id):
            img_data = variant_ref.read_bytes()
            ext = variant_ref.suffix.lower()
            content_type = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp",
            }.get(ext, "image/jpeg")
            if len(ScanHandler._variant_image_cache) < 500:
                ScanHandler._variant_image_cache[cache_key] = (img_data, content_type)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(img_data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(img_data)
            return

        # Find the base reference image (normal variant)
        ref_path = _ref_image_path(card_id)
        if not ref_path:
            self.send_error(404, f"No reference image for {card_id}")
            return

        # No real variant image — apply overlay badges to the normal image
        from cardprice.ml.image_overlay import apply_variant_overlay
        base_data = ref_path.read_bytes()
        overlaid_data = apply_variant_overlay(base_data, variant_list, card_id=card_id)

        # Cache the result (limit cache size to prevent unbounded memory growth)
        if len(ScanHandler._variant_image_cache) < 500:
            ScanHandler._variant_image_cache[cache_key] = (overlaid_data, "image/png")

        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(overlaid_data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(overlaid_data)

    def _send_segment_image(self, rel_path):
        """Serve a segmented card image from data/inbox/.

        URL format: /segment-image/page_20260228_120000_cards/card_00.png
        """
        rel_path = rel_path.strip("/")
        # Security: prevent directory traversal
        if ".." in rel_path or rel_path.startswith("/"):
            self.send_error(400, "Invalid path")
            return

        image_path = UPLOAD_DIR / rel_path
        if not image_path.is_file():
            self.send_error(404, f"Segment image not found: {rel_path}")
            return

        img_data = image_path.read_bytes()
        ext = image_path.suffix.lower()
        ctype = "image/png" if ext == ".png" else "image/jpeg"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(img_data)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(img_data)

    def log_message(self, fmt, *args):
        logger.info(fmt, *args)


def warmup():
    """Pre-load all ML models and data files so the first request is fast.

    Loads resources in dependency order:
      1. RapidOCR (name + attack OCR) + dummy inference to trigger JIT compilation
      2. DINOv2 + dummy inference to trigger JIT/CUDA warmup
      3. FAISS index + card_ids
      4. Reference embeddings
      5. Card names JSON
      6. Card attacks JSON

    Each step is timed and logged.  Failures are logged as warnings but do
    not prevent the server from starting.

    IMPORTANT: CLIP is deliberately NOT loaded here.  Loading CLIP alongside
    PaddlePaddle causes a SIGSEGV (segmentation fault) due to conflicting
    protobuf/ONNX internals.  CLIP is not used in the v2 pipeline anyway.
    """
    total_start = time.time()
    logger.info("=== ML warmup starting ===")

    # --- 1. PaddleOCR (name OCR) + dummy inference ---

    def _warmup_paddleocr():
        from cardprice.ml.ocr_matcher import _paddle_ocr_name
        import numpy as np
        # Trigger PaddleOCR model load AND first inference (JIT warmup)
        dummy = np.zeros((100, 300, 3), dtype=np.uint8)
        _paddle_ocr_name(dummy, 100, 300)

    # --- 2. (EasyOCR removed — RapidOCR handles all OCR, saving ~800MB RAM + 15s warmup) ---

    # --- 3. DINOv2 + dummy inference ---

    def _warmup_dinov2():
        from cardprice.ml.dino_matcher import _load_model, _get_transform
        import torch
        from PIL import Image
        model, device = _load_model()
        # Run a dummy inference to trigger CUDA/JIT warmup
        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        tensor = _get_transform()(dummy_img).unsqueeze(0).to(device)
        with torch.no_grad():
            model(tensor)

    # --- 4. FAISS index + card_ids ---

    def _warmup_dino_faiss():
        from cardprice.ml import _get_dino_index
        _get_dino_index()

    # --- 5. Reference embeddings ---

    def _warmup_ref_embeddings():
        from cardprice.ml.ref_matcher import _load_ref_embeddings
        _load_ref_embeddings()

    # --- 6. Card names JSON ---

    def _warmup_card_names():
        from cardprice.ml.ocr_matcher import _load_card_names
        _load_card_names()

    # --- 7. Card attacks JSON ---

    def _warmup_attack_index():
        from cardprice.ml.attack_ocr import _load_attack_index
        _load_attack_index()

    # --- Supplementary (nice to have, not critical path) ---

    def _warmup_card_names_fallback():
        from cardprice.ml.ref_matcher import _load_card_names_fallback
        _load_card_names_fallback()

    def _warmup_card_metadata():
        from cardprice.ml import _get_card_metadata
        _get_card_metadata()

    def _warmup_hash_db():
        from cardprice.ml import _get_hash_db
        _get_hash_db()

    # Ordered: critical models first with dummy inference, then data files.
    # CLIP is deliberately excluded (SIGSEGV with PaddlePaddle).
    steps = [
        ("RapidOCR (PP-OCRv5)",        _warmup_paddleocr),
        ("DINOv2 ViT-B/14",           _warmup_dinov2),
        ("FAISS index (DINOv2)",       _warmup_dino_faiss),
        ("Ref embeddings (DINOv2)",    _warmup_ref_embeddings),
        ("Card names (DB/JSON)",       _warmup_card_names),
        ("Attack index",               _warmup_attack_index),
        ("Card names fallback (JSON)", _warmup_card_names_fallback),
        ("Card metadata (DB)",         _warmup_card_metadata),
        ("Hash DB",                    _warmup_hash_db),
    ]

    loaded = 0
    failed = 0
    for name, fn in steps:
        step_start = time.time()
        try:
            fn()
            elapsed = time.time() - step_start
            logger.info("  [OK] %-30s  %.1fs", name, elapsed)
            loaded += 1
        except Exception as e:
            elapsed = time.time() - step_start
            logger.warning("  [FAIL] %-30s  %.1fs — %s", name, elapsed, e)
            failed += 1

    total = time.time() - total_start
    logger.info("=== ML warmup complete: %d loaded, %d failed, %.1fs total ===",
                loaded, failed, total)


def run_server(host="0.0.0.0", port=8888):
    """Start the HTTP server."""
    global _server_port
    _server_port = port
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not os.environ.get("CARDPRICE_SKIP_WARMUP"):
        warmup()
    else:
        logger.info("Skipping warmup (CARDPRICE_SKIP_WARMUP set)")

    server = HTTPServer((host, port), ScanHandler)

    # SSL is available but disabled by default (self-signed certs cause
    # issues on Brave/Chrome mobile). Instead, use browser flags:
    #   brave://flags/#unsafely-treat-insecure-origin-as-secure
    #   Add http://<ip>:8888, enable, relaunch.
    # To enable SSL, pass --ssl flag.
    import ssl
    use_ssl = "--ssl" in sys.argv
    if use_ssl:
        cert_path = Path(__file__).resolve().parent.parent / "data" / "server.crt"
        key_path = Path(__file__).resolve().parent.parent / "data" / "server.key"
        if cert_path.is_file() and key_path.is_file():
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(str(cert_path), str(key_path))
            server.socket = ctx.wrap_socket(server.socket, server_side=True)

    protocol = "https" if use_ssl else "http"
    lan_ip = _get_lan_ip()
    print(f"Card scanner server running at {protocol}://{host}:{port}")
    print(f"LAN URL (for phone): {protocol}://{lan_ip}:{port}")

    # Auto-start Cloudflare tunnel for HTTPS (needed for getUserMedia/slide-scan)
    tunnel_url = None
    cloudflared_proc = None
    cloudflared_path = Path.home() / ".local" / "bin" / "cloudflared"
    if cloudflared_path.is_file() and not use_ssl:
        import subprocess, threading
        def _start_tunnel():
            nonlocal tunnel_url, cloudflared_proc
            try:
                cloudflared_proc = subprocess.Popen(
                    [str(cloudflared_path), "tunnel", "--url", f"http://localhost:{port}"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                for line in cloudflared_proc.stdout:
                    if "trycloudflare.com" in line:
                        # Extract URL
                        import re
                        m = re.search(r'(https://[^\s]+trycloudflare\.com)', line)
                        if m:
                            tunnel_url = m.group(1)
                            # Write tunnel URL to a file so the QR code JS can fetch it
                            tunnel_file = Path(__file__).resolve().parent.parent / "data" / "tunnel_url.txt"
                            tunnel_file.write_text(tunnel_url)
                            print(f"\n{'='*60}")
                            print(f"  HTTPS tunnel: {tunnel_url}")
                            print(f"  Slide scan:   {tunnel_url}/slide-scan-v7")
                            print(f"{'='*60}\n")
            except Exception as e:
                logger.warning("Cloudflare tunnel failed: %s", e)

        tunnel_thread = threading.Thread(target=_start_tunnel, daemon=True)
        tunnel_thread.start()

    print("Open this URL on your phone, or scan the QR code on the landing page")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        if cloudflared_proc:
            cloudflared_proc.terminate()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Card scanner HTTP server")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
