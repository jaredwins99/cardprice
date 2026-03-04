"""Run DDL to create all tables. Idempotent (uses IF NOT EXISTS)."""

from sqlalchemy import text

from cardprice.db.session import engine

DDL = """
CREATE TABLE IF NOT EXISTS dim_pokemon (
    pokemon_id    SERIAL PRIMARY KEY,
    name          TEXT NOT NULL,
    pokedex_num   INTEGER,
    types         TEXT[],
    hp_base       INTEGER,
    generation    SMALLINT,
    evolves_from  TEXT,
    evolves_to    TEXT[],
    created_at    TIMESTAMPTZ DEFAULT now()
    -- NOTE: plain UNIQUE(name, pokedex_num) doesn't work with NULLs.
    -- Use a functional unique index instead (created below).
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_pokemon_name_dex
    ON dim_pokemon(name, COALESCE(pokedex_num, -1));

CREATE TABLE IF NOT EXISTS dim_sets (
    set_id        TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    series        TEXT,
    tcg_group_id  INTEGER,
    total_cards   INTEGER,
    release_date  DATE,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dim_cards (
    card_id           TEXT PRIMARY KEY,
    tcg_product_id    INTEGER,
    name              TEXT NOT NULL,
    set_id            TEXT REFERENCES dim_sets(set_id),
    pokemon_id        INTEGER REFERENCES dim_pokemon(pokemon_id),
    card_number       TEXT,
    rarity            TEXT,
    supertype         TEXT,
    subtypes          TEXT[],
    types             TEXT[],
    variant           TEXT,
    hp                INTEGER,
    artist            TEXT,
    image_small       TEXT,
    image_large       TEXT,
    tcgplayer_url     TEXT,
    created_at        TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_cards_product ON dim_cards(tcg_product_id);
CREATE INDEX IF NOT EXISTS idx_cards_set ON dim_cards(set_id);
CREATE INDEX IF NOT EXISTS idx_cards_pokemon ON dim_cards(pokemon_id);

CREATE TABLE IF NOT EXISTS dim_marketplaces (
    marketplace_id  SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    source_system   TEXT
);

CREATE TABLE IF NOT EXISTS fact_market_prices (
    id              BIGSERIAL PRIMARY KEY,
    card_id         TEXT REFERENCES dim_cards(card_id),
    tcg_product_id  INTEGER,
    marketplace_id  INTEGER REFERENCES dim_marketplaces(marketplace_id),
    price_date      DATE NOT NULL,
    subtype_name    TEXT,
    low_price       NUMERIC(10,2),
    mid_price       NUMERIC(10,2),
    high_price      NUMERIC(10,2),
    market_price    NUMERIC(10,2),
    direct_low      NUMERIC(10,2),
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_fmp_card_date ON fact_market_prices(card_id, price_date);
CREATE INDEX IF NOT EXISTS idx_fmp_date ON fact_market_prices(price_date);
CREATE UNIQUE INDEX IF NOT EXISTS idx_fmp_product_date_subtype
    ON fact_market_prices(tcg_product_id, price_date, subtype_name);

-- Migration: add tcg_product_id if table already existed without it.
ALTER TABLE fact_market_prices ADD COLUMN IF NOT EXISTS tcg_product_id INTEGER;

-- Migration: add card-level types (may differ from species types, e.g. delta species).
ALTER TABLE dim_cards ADD COLUMN IF NOT EXISTS types TEXT[];

CREATE TABLE IF NOT EXISTS fact_sales (
    id              BIGSERIAL PRIMARY KEY,
    card_id         TEXT REFERENCES dim_cards(card_id),
    marketplace_id  INTEGER REFERENCES dim_marketplaces(marketplace_id),
    sale_date       TIMESTAMPTZ NOT NULL,
    sale_price      NUMERIC(10,2) NOT NULL,
    condition       TEXT,
    seller_info     JSONB,
    listing_url     TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_fs_card_date ON fact_sales(card_id, sale_date);

-- ============================================================
-- Phase 2: eBay transaction data support
-- ============================================================

-- New columns on fact_sales (idempotent via IF NOT EXISTS)
ALTER TABLE fact_sales ADD COLUMN IF NOT EXISTS source_item_id    TEXT;
ALTER TABLE fact_sales ADD COLUMN IF NOT EXISTS quantity           INTEGER DEFAULT 1;
ALTER TABLE fact_sales ADD COLUMN IF NOT EXISTS sale_type          TEXT;
ALTER TABLE fact_sales ADD COLUMN IF NOT EXISTS shipping_price     NUMERIC(10,2);
ALTER TABLE fact_sales ADD COLUMN IF NOT EXISTS grading_authority  TEXT;
ALTER TABLE fact_sales ADD COLUMN IF NOT EXISTS grade              TEXT;
ALTER TABLE fact_sales ADD COLUMN IF NOT EXISTS raw_title          TEXT;
ALTER TABLE fact_sales ADD COLUMN IF NOT EXISTS image_urls         TEXT[];
ALTER TABLE fact_sales ADD COLUMN IF NOT EXISTS match_confidence   REAL;

-- Dedup: same source_item_id should not appear twice
CREATE UNIQUE INDEX IF NOT EXISTS idx_fs_source_item_id
    ON fact_sales(source_item_id) WHERE source_item_id IS NOT NULL;

-- Scrape job tracking for observability and resumption
CREATE TABLE IF NOT EXISTS scrape_jobs (
    id              SERIAL PRIMARY KEY,
    source          TEXT,
    query           TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    items_found     INTEGER,
    items_matched   INTEGER,
    status          TEXT
);

-- Priority queue: which cards to scrape next
CREATE TABLE IF NOT EXISTS card_scrape_priority (
    card_id         TEXT PRIMARY KEY REFERENCES dim_cards(card_id),
    priority_score  REAL,
    last_scraped    TIMESTAMPTZ,
    scrape_count    INTEGER DEFAULT 0
);

-- ============================================================
-- Phase 3: User inventory and card scanning
-- ============================================================

CREATE TABLE IF NOT EXISTS user_inventory (
    id                  SERIAL PRIMARY KEY,
    card_id             TEXT REFERENCES dim_cards(card_id),
    quantity            INTEGER DEFAULT 1,
    condition           TEXT CHECK (condition IN ('NM', 'LP', 'MP', 'HP', 'DMG')),
    grade_authority     TEXT CHECK (grade_authority IN ('PSA', 'BGS', 'CGC')),
    grade               TEXT,
    acquisition_price   NUMERIC(10,2),
    acquisition_date    DATE,
    acquisition_source  TEXT CHECK (acquisition_source IN ('pulled', 'purchased', 'traded')),
    notes               TEXT,
    image_path          TEXT,
    scan_confidence     REAL,
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_inv_card ON user_inventory(card_id);
CREATE INDEX IF NOT EXISTS idx_inv_condition ON user_inventory(condition);

CREATE TABLE IF NOT EXISTS inventory_scans (
    id                  SERIAL PRIMARY KEY,
    image_path          TEXT NOT NULL,
    identified_card_id  TEXT REFERENCES dim_cards(card_id),
    identified_condition TEXT,
    confidence          REAL,
    model_used          TEXT,
    raw_response        JSONB,
    accepted            BOOLEAN,
    created_at          TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_scans_card ON inventory_scans(identified_card_id);

CREATE TABLE IF NOT EXISTS inventory_valuations (
    id                  SERIAL PRIMARY KEY,
    valuation_date      DATE NOT NULL,
    total_cards         INTEGER,
    total_value_low     NUMERIC(12,2),
    total_value_mid     NUMERIC(12,2),
    total_value_high    NUMERIC(12,2),
    breakdown           JSONB,
    created_at          TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_valuations_date ON inventory_valuations(valuation_date);

-- ============================================================
-- Phase 3: Pokemon features and competitive data
-- ============================================================

CREATE TABLE IF NOT EXISTS dim_pokemon_features (
    pokedex_num     INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    generation      TEXT,
    is_legendary    BOOLEAN DEFAULT FALSE,
    is_mythical     BOOLEAN DEFAULT FALSE,
    capture_rate    INTEGER,
    base_happiness  INTEGER,
    egg_groups      TEXT[],
    color           TEXT,
    shape           TEXT,
    habitat         TEXT,
    hp              INTEGER,
    attack          INTEGER,
    defense         INTEGER,
    sp_attack       INTEGER,
    sp_defense      INTEGER,
    speed           INTEGER,
    bst             INTEGER,
    height          INTEGER,
    weight          INTEGER,
    types           TEXT[],
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dim_smogon_usage (
    id              SERIAL PRIMARY KEY,
    pokemon_name    TEXT NOT NULL,
    format          TEXT NOT NULL,
    usage_weighted  REAL,
    usage_raw       REAL,
    usage_real      REAL,
    viability_gxe   REAL,
    count           INTEGER,
    lead_weighted   REAL,
    fetched_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(pokemon_name, format)
);
"""


def run():
    with engine.connect() as conn:
        conn.execute(text(DDL))
        conn.commit()
    print("Migration complete: all tables created.")


if __name__ == "__main__":
    run()
