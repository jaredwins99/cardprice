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

-- ============================================================
-- Phase 3: Card condition assessment / grading
-- ============================================================

CREATE TABLE IF NOT EXISTS condition_scans (
    scan_id         SERIAL PRIMARY KEY,
    card_id         TEXT NOT NULL REFERENCES dim_cards(card_id),
    inventory_id    INTEGER REFERENCES user_inventory(id),
    overall_grade   REAL,
    tcg_condition   TEXT CHECK (tcg_condition IN ('NM', 'LP', 'MP', 'HP', 'DMG')),
    confidence      REAL,
    grade_ci_low    REAL,
    grade_ci_high   REAL,
    model_version   TEXT,
    raw_output      JSONB,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_cond_scans_card ON condition_scans(card_id);
CREATE INDEX IF NOT EXISTS idx_cond_scans_inventory ON condition_scans(inventory_id);
CREATE INDEX IF NOT EXISTS idx_cond_scans_grade ON condition_scans(overall_grade);

CREATE TABLE IF NOT EXISTS condition_images (
    image_id        SERIAL PRIMARY KEY,
    scan_id         INTEGER REFERENCES condition_scans(scan_id) ON DELETE CASCADE,
    angle_type      TEXT CHECK (angle_type IN (
                        'front', 'back',
                        'oblique_front', 'oblique_back',
                        'edge_top', 'edge_bottom', 'edge_left', 'edge_right',
                        'corner_tl', 'corner_tr', 'corner_bl', 'corner_br'
                    )),
    image_path      TEXT NOT NULL,
    image_quality   REAL,
    resolution_w    INTEGER,
    resolution_h    INTEGER,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_cond_images_scan ON condition_images(scan_id);

CREATE TABLE IF NOT EXISTS condition_scores (
    score_id        SERIAL PRIMARY KEY,
    scan_id         INTEGER REFERENCES condition_scans(scan_id) ON DELETE CASCADE,
    category        TEXT CHECK (category IN ('centering', 'corners', 'edges', 'surface')),
    score           REAL NOT NULL,
    confidence      REAL,
    defects         JSONB,
    source_images   INTEGER[],
    UNIQUE(scan_id, category)
);
CREATE INDEX IF NOT EXISTS idx_cond_scores_scan ON condition_scores(scan_id);

CREATE TABLE IF NOT EXISTS condition_calibration (
    id              SERIAL PRIMARY KEY,
    scan_id         INTEGER REFERENCES condition_scans(scan_id),
    grade_authority TEXT CHECK (grade_authority IN ('PSA', 'BGS', 'CGC')),
    actual_grade    REAL NOT NULL,
    actual_subgrades JSONB,
    cert_number     TEXT,
    predicted_grade REAL,
    UNIQUE(scan_id, grade_authority)
);
CREATE INDEX IF NOT EXISTS idx_cond_calib_scan ON condition_calibration(scan_id);
CREATE INDEX IF NOT EXISTS idx_cond_calib_cert ON condition_calibration(cert_number);
"""


def run():
    with engine.connect() as conn:
        conn.execute(text(DDL))
        conn.commit()
    print("Migration complete: all tables created.")


# ---------------------------------------------------------------------------
# Variant row synthesis: explode /normal rows into per-variant dim_cards rows
# and repoint fact_market_prices.card_id accordingly.
# ---------------------------------------------------------------------------

# Map fact_market_prices.subtype_name → variant suffix for dim_cards.card_id
SUBTYPE_TO_VARIANT = {
    "Normal":            "normal",
    "Holofoil":          "holofoil",
    "Reverse Holofoil":  "reverse_holofoil",
    "1st Edition Normal": "1st_edition",
    "1st Edition Holofoil": "1st_edition_holofoil",
    "Unlimited":         "unlimited",
    "Unlimited Holofoil": "unlimited_holofoil",
}


def synthesize_variants():
    """Create variant dim_cards rows from fact_market_prices.subtype_name.

    For every (base_card_id, subtype_name) pair in fact_market_prices where
    subtype_name != 'Normal', this migration:
      1. INSERTs a new dim_cards row with card_id = '{base}/{variant}'
         (copied from the /normal row).
      2. UPDATEs fact_market_prices rows to point to the new variant card_id.

    Normal rows are also updated to ensure card_id is set (for rows that were
    previously NULL or pointed at the wrong card).

    Runs in a single transaction with rollback on error.
    """
    from sqlalchemy import text as sa_text

    with engine.connect() as conn:
        try:
            # ---------------------------------------------------------------
            # Step 1: Discover all (base_id, subtype_name) pairs that need
            #         variant rows.  base_id is card_id with /normal stripped.
            # ---------------------------------------------------------------
            pairs = conn.execute(sa_text("""
                SELECT DISTINCT
                    CASE
                        WHEN dc.card_id LIKE '%%/normal'
                        THEN LEFT(dc.card_id, LENGTH(dc.card_id) - 7)
                        ELSE dc.card_id
                    END AS base_id,
                    fmp.subtype_name
                FROM fact_market_prices fmp
                JOIN dim_cards dc ON dc.card_id = fmp.card_id
                WHERE fmp.subtype_name IS NOT NULL
                  AND fmp.subtype_name != 'Normal'
            """)).fetchall()

            print(f"Found {len(pairs)} (base_id, subtype) pairs to synthesize.")

            # ---------------------------------------------------------------
            # Step 2: For each pair, INSERT a variant row (if not exists) by
            #         copying from the /normal row.
            # ---------------------------------------------------------------
            inserted = 0
            skipped = 0
            for base_id, subtype_name in pairs:
                variant_suffix = SUBTYPE_TO_VARIANT.get(subtype_name)
                if variant_suffix is None:
                    print(f"  WARNING: unknown subtype_name '{subtype_name}', skipping")
                    continue

                new_card_id = f"{base_id}/{variant_suffix}"
                normal_card_id = f"{base_id}/normal"

                # Check if variant row already exists
                exists = conn.execute(sa_text(
                    "SELECT 1 FROM dim_cards WHERE card_id = :cid"
                ), {"cid": new_card_id}).fetchone()

                if exists:
                    skipped += 1
                    continue

                # Copy from the /normal row
                result = conn.execute(sa_text("""
                    INSERT INTO dim_cards (
                        card_id, tcg_product_id, name, set_id, pokemon_id,
                        card_number, rarity, supertype, subtypes, types,
                        variant, hp, artist, image_small, image_large,
                        tcgplayer_url
                    )
                    SELECT
                        :new_card_id,
                        tcg_product_id,
                        name,
                        set_id,
                        pokemon_id,
                        card_number,
                        rarity,
                        supertype,
                        subtypes,
                        types,
                        :variant,
                        hp,
                        artist,
                        image_small,
                        image_large,
                        tcgplayer_url
                    FROM dim_cards
                    WHERE card_id = :normal_card_id
                """), {
                    "new_card_id": new_card_id,
                    "normal_card_id": normal_card_id,
                    "variant": variant_suffix,
                })

                if result.rowcount > 0:
                    inserted += 1
                else:
                    print(f"  WARNING: no /normal row for {normal_card_id}, "
                          f"cannot create {new_card_id}")

            print(f"Inserted {inserted} variant rows, skipped {skipped} "
                  f"(already exist).")

            # ---------------------------------------------------------------
            # Step 3: UPDATE fact_market_prices.card_id to point to the
            #         correct variant row.  We join through dim_cards to get
            #         the base_id and then build the target card_id.
            # ---------------------------------------------------------------
            for subtype_name, variant_suffix in SUBTYPE_TO_VARIANT.items():
                if subtype_name == "Normal":
                    continue  # Normal rows already point to /normal

                result = conn.execute(sa_text("""
                    UPDATE fact_market_prices fmp
                    SET card_id = LEFT(fmp.card_id, LENGTH(fmp.card_id) - 7)
                                  || '/' || :variant
                    FROM dim_cards dc_new
                    WHERE fmp.subtype_name = :subtype
                      AND fmp.card_id LIKE '%%/normal'
                      AND dc_new.card_id = LEFT(fmp.card_id, LENGTH(fmp.card_id) - 7)
                                           || '/' || :variant
                """), {
                    "subtype": subtype_name,
                    "variant": variant_suffix,
                })

                print(f"  Updated {result.rowcount:>9,} rows: "
                      f"{subtype_name} -> /{variant_suffix}")

            conn.commit()
            print("Variant synthesis complete.")

        except Exception:
            conn.rollback()
            print("ERROR: variant synthesis failed, transaction rolled back.")
            raise


if __name__ == "__main__":
    run()
