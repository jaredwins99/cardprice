"""One-off ingestion of Japanese Pokemon (TCGPlayer category 85) via TCGCSV.

Creates and populates parallel tables: dim_sets_jp, dim_cards_jp, fact_market_prices_jp.
JP set IDs are namespaced via the tcg_group_id (no collision with English set_id text).

Usage:
    python -m scripts.scrapers.tcgcsv_ingest_jp [--prices-only] [--limit N]
"""

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

# Ensure project root is on path so 'cardprice' package imports work under cron
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests
from sqlalchemy import create_engine, text

from cardprice.config import DATABASE_URL, TCGCSV_BASE_URL, POKEMON_JP_CATEGORY_ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("tcgcsv_jp")

BASE = f"{TCGCSV_BASE_URL}/tcgplayer/{POKEMON_JP_CATEGORY_ID}"
TCGPLAYER_MARKETPLACE_ID = 1
REQUEST_DELAY = 0.25
TODAY = date.today()


# ---------------------------------------------------------------------------
# Schema (idempotent)
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS dim_sets_jp (
    set_id        text PRIMARY KEY,           -- 'jp_<groupId>'
    tcg_group_id  integer UNIQUE NOT NULL,
    name          text NOT NULL,
    abbreviation  text,
    published_on  date,
    created_at    timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dim_cards_jp (
    card_id        text PRIMARY KEY,          -- 'jp_<productId>'
    tcg_product_id integer UNIQUE NOT NULL,
    set_id         text REFERENCES dim_sets_jp(set_id),
    name           text NOT NULL,
    clean_name     text,
    card_number    text,
    rarity         text,
    image_url      text,
    tcgplayer_url  text,
    ext_attrs      jsonb,
    created_at     timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_cards_jp_set     ON dim_cards_jp(set_id);
CREATE INDEX IF NOT EXISTS idx_cards_jp_product ON dim_cards_jp(tcg_product_id);

CREATE TABLE IF NOT EXISTS fact_market_prices_jp (
    id             bigserial PRIMARY KEY,
    tcg_product_id integer NOT NULL,
    marketplace_id integer NOT NULL,
    price_date     date NOT NULL,
    subtype_name   text,
    low_price      numeric(10,2),
    mid_price      numeric(10,2),
    high_price     numeric(10,2),
    market_price   numeric(10,2),
    direct_low     numeric(10,2),
    created_at     timestamptz DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_fmp_jp_product_date_subtype
    ON fact_market_prices_jp (tcg_product_id, price_date, subtype_name);
CREATE INDEX IF NOT EXISTS idx_fmp_jp_date ON fact_market_prices_jp (price_date);
"""


def ensure_schema(engine):
    with engine.begin() as conn:
        for stmt in DDL.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
    logger.info("Schema ensured (dim_sets_jp, dim_cards_jp, fact_market_prices_jp)")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_json(url: str) -> dict | list:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_groups() -> list[dict]:
    data = _get_json(f"{BASE}/groups")
    if isinstance(data, list):
        return data
    return data.get("results", data)


def fetch_products(group_id: int) -> list[dict]:
    data = _get_json(f"{BASE}/{group_id}/products")
    return data.get("results", []) if isinstance(data, dict) else data


def fetch_prices(group_id: int) -> list[dict]:
    data = _get_json(f"{BASE}/{group_id}/prices")
    return data.get("results", []) if isinstance(data, dict) else data


# ---------------------------------------------------------------------------
# Inserts
# ---------------------------------------------------------------------------

UPSERT_SET = text("""
    INSERT INTO dim_sets_jp (set_id, tcg_group_id, name, abbreviation, published_on)
    VALUES (:set_id, :tcg_group_id, :name, :abbreviation, :published_on)
    ON CONFLICT (set_id) DO UPDATE SET
        name = EXCLUDED.name,
        abbreviation = EXCLUDED.abbreviation,
        published_on = EXCLUDED.published_on
""")

UPSERT_CARD = text("""
    INSERT INTO dim_cards_jp
        (card_id, tcg_product_id, set_id, name, clean_name,
         card_number, rarity, image_url, tcgplayer_url, ext_attrs)
    VALUES
        (:card_id, :tcg_product_id, :set_id, :name, :clean_name,
         :card_number, :rarity, :image_url, :tcgplayer_url, CAST(:ext_attrs AS jsonb))
    ON CONFLICT (card_id) DO UPDATE SET
        name = EXCLUDED.name,
        rarity = EXCLUDED.rarity,
        card_number = EXCLUDED.card_number,
        image_url = EXCLUDED.image_url,
        tcgplayer_url = EXCLUDED.tcgplayer_url,
        ext_attrs = EXCLUDED.ext_attrs
""")

INSERT_PRICE = text("""
    INSERT INTO fact_market_prices_jp
        (tcg_product_id, marketplace_id, price_date, subtype_name,
         low_price, mid_price, high_price, market_price, direct_low)
    VALUES
        (:tcg_product_id, :marketplace_id, :price_date, :subtype_name,
         :low_price, :mid_price, :high_price, :market_price, :direct_low)
    ON CONFLICT (tcg_product_id, price_date, subtype_name) DO NOTHING
""")


def _ext_attrs_to_dict(ext) -> dict:
    """TCGCSV returns extendedData as a list of {name, displayName, value} dicts."""
    if not ext:
        return {}
    if isinstance(ext, dict):
        return ext
    out = {}
    for entry in ext:
        if isinstance(entry, dict) and "name" in entry:
            out[entry["name"]] = entry.get("value")
    return out


def _pick(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def upsert_group(conn, group: dict) -> str:
    gid = group["groupId"]
    set_id = f"jp_{gid}"
    published = _pick(group, "publishedOn", "publishedDate")
    if published and "T" in str(published):
        published = str(published).split("T")[0]
    conn.execute(UPSERT_SET, {
        "set_id": set_id,
        "tcg_group_id": gid,
        "name": group.get("name") or f"Group {gid}",
        "abbreviation": group.get("abbreviation"),
        "published_on": published or None,
    })
    return set_id


def upsert_products(conn, set_id: str, products: list[dict]) -> int:
    if not products:
        return 0
    import json as _json
    rows = []
    for p in products:
        pid = p["productId"]
        ext = _ext_attrs_to_dict(p.get("extendedData"))
        rows.append({
            "card_id": f"jp_{pid}",
            "tcg_product_id": pid,
            "set_id": set_id,
            "name": p.get("name") or f"Product {pid}",
            "clean_name": p.get("cleanName"),
            "card_number": ext.get("Number"),
            "rarity": ext.get("Rarity"),
            "image_url": p.get("imageUrl"),
            "tcgplayer_url": p.get("url"),
            "ext_attrs": _json.dumps(ext),
        })
    conn.execute(UPSERT_CARD, rows)
    return len(rows)


def insert_prices(conn, prices: list[dict], price_date: date) -> int:
    if not prices:
        return 0
    rows = [{
        "tcg_product_id": r["productId"],
        "marketplace_id": TCGPLAYER_MARKETPLACE_ID,
        "price_date": price_date,
        "subtype_name": r.get("subTypeName"),
        "low_price": r.get("lowPrice"),
        "mid_price": r.get("midPrice"),
        "high_price": r.get("highPrice"),
        "market_price": r.get("marketPrice"),
        "direct_low": r.get("directLowPrice"),
    } for r in prices]
    result = conn.execute(INSERT_PRICE, rows)
    return result.rowcount or len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N groups (debug)")
    parser.add_argument("--prices-only", action="store_true",
                        help="Skip product upserts (only fetch prices)")
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    ensure_schema(engine)

    groups = fetch_groups()
    logger.info("Fetched %d JP Pokemon groups", len(groups))
    if args.limit:
        groups = groups[: args.limit]

    total_products = 0
    total_prices = 0
    errors = 0

    for i, g in enumerate(groups, start=1):
        gid = g["groupId"]
        gname = g.get("name", str(gid))
        try:
            with engine.begin() as conn:
                set_id = upsert_group(conn, g)

                if not args.prices_only:
                    products = fetch_products(gid)
                    n_products = upsert_products(conn, set_id, products)
                    total_products += n_products
                    time.sleep(REQUEST_DELAY)

                prices = fetch_prices(gid)
                n_prices = insert_prices(conn, prices, TODAY)
                total_prices += n_prices

            logger.info(
                "[%d/%d] group=%s (%s) products+=%d prices+=%d  totals: p=%d pr=%d",
                i, len(groups), gid, gname[:40],
                0 if args.prices_only else n_products,
                n_prices, total_products, total_prices,
            )
        except Exception as e:
            errors += 1
            logger.exception("Failed group %s (%s): %s", gid, gname, e)
        time.sleep(REQUEST_DELAY)

    logger.info(
        "DONE. groups=%d products_upserted=%d price_rows_inserted=%d errors=%d",
        len(groups), total_products, total_prices, errors,
    )


if __name__ == "__main__":
    main()
