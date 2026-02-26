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
    created_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE(name, pokedex_num)
);

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
"""


def run():
    with engine.connect() as conn:
        conn.execute(text(DDL))
        conn.commit()
    print("Migration complete: all tables created.")


if __name__ == "__main__":
    run()
