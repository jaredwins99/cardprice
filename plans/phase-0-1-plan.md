# Phase 0–1 Plan: Architecture & Data Collection

## Goal
Stand up the database, ingest the full Pokemon card catalog, and load historical + live TCGPlayer pricing data. By the end, we have a queryable star schema with every English Pokemon card variant and its price history from Feb 2024 to present.

## Context
- **Data sources**: TCGCSV (TCGPlayer market prices, free, no auth) + pokemontcg.io (card metadata, free, API key for 20k req/day)
- **Database**: Local PostgreSQL 16 in WSL2
- **Stack**: Python 3.11+, SQLAlchemy, requests, pandas
- **Refresh cadence**: Weekly (cron job later)

---

## 1. PostgreSQL Setup (WSL2)

```bash
sudo apt update && sudo apt install -y postgresql postgresql-contrib
sudo service postgresql start
sudo -u postgres createuser --superuser $USER
createdb cardprice
```

Verify: `psql cardprice -c "SELECT 1;"`

Config file: `~/.pgpass` for passwordless local access. No need for pgvector — embeddings go to parquet files.

---

## 2. Star Schema DDL

### Dimension Tables

```sql
-- Every unique Pokemon species (Charizard, Pikachu, etc.)
CREATE TABLE dim_pokemon (
    pokemon_id    SERIAL PRIMARY KEY,
    name          TEXT NOT NULL,             -- "Charizard"
    pokedex_num   INTEGER,                   -- 6
    types         TEXT[],                    -- {"Fire", "Flying"}
    hp_base       INTEGER,                   -- Base game HP (from first card appearance)
    generation    SMALLINT,                  -- 1
    evolves_from  TEXT,                      -- "Charmeleon"
    evolves_to    TEXT[],                    -- {}
    created_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE(name, pokedex_num)
);

-- Pokemon TCG sets (Base Set, Jungle, Fossil, etc.)
CREATE TABLE dim_sets (
    set_id        TEXT PRIMARY KEY,          -- pokemontcg.io set id: "base1"
    name          TEXT NOT NULL,             -- "Base Set"
    series        TEXT,                      -- "Base"
    tcg_group_id  INTEGER,                   -- TCGCSV groupId for joins
    total_cards   INTEGER,
    release_date  DATE,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Every unique card variant (the core dimension)
-- One row per printable card variant: "Charizard Base Set Holo" != "Charizard Base Set Non-Holo"
CREATE TABLE dim_cards (
    card_id           TEXT PRIMARY KEY,      -- pokemontcg.io card id: "base1-4"
    tcg_product_id    INTEGER,               -- TCGCSV productId for price joins
    name              TEXT NOT NULL,          -- "Charizard"
    set_id            TEXT REFERENCES dim_sets(set_id),
    pokemon_id        INTEGER REFERENCES dim_pokemon(pokemon_id),
    card_number       TEXT,                  -- "4" (set number)
    rarity            TEXT,                  -- "Rare Holo"
    supertype         TEXT,                  -- "Pokémon", "Trainer", "Energy"
    subtypes          TEXT[],               -- {"Stage 2"}
    variant           TEXT,                  -- "Holofoil", "Reverse Holofoil", "Normal"
    hp                INTEGER,
    artist            TEXT,
    image_small       TEXT,                  -- URL
    image_large       TEXT,                  -- URL
    tcgplayer_url     TEXT,
    created_at        TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_cards_product ON dim_cards(tcg_product_id);
CREATE INDEX idx_cards_set ON dim_cards(set_id);
CREATE INDEX idx_cards_pokemon ON dim_cards(pokemon_id);

-- Marketplace sources
CREATE TABLE dim_marketplaces (
    marketplace_id  SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,     -- "TCGPlayer", "eBay", "CardMarket"
    source_system   TEXT                      -- "tcgcsv", "pricecharting", etc.
);
```

### Fact Tables

```sql
-- Daily aggregate market prices from TCGCSV
-- Grain: one row per card variant per condition-subtype per day
CREATE TABLE fact_market_prices (
    id              BIGSERIAL PRIMARY KEY,
    card_id         TEXT REFERENCES dim_cards(card_id),
    marketplace_id  INTEGER REFERENCES dim_marketplaces(marketplace_id),
    price_date      DATE NOT NULL,
    subtype_name    TEXT,                    -- "Normal", "Holofoil", "Reverse Holofoil"
    low_price       NUMERIC(10,2),
    mid_price       NUMERIC(10,2),
    high_price      NUMERIC(10,2),
    market_price    NUMERIC(10,2),
    direct_low      NUMERIC(10,2),
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_fmp_card_date ON fact_market_prices(card_id, price_date);
CREATE INDEX idx_fmp_date ON fact_market_prices(price_date);

-- Individual completed sales (Phase 2 — PriceCharting/eBay)
-- Grain: one row per completed transaction
CREATE TABLE fact_sales (
    id              BIGSERIAL PRIMARY KEY,
    card_id         TEXT REFERENCES dim_cards(card_id),
    marketplace_id  INTEGER REFERENCES dim_marketplaces(marketplace_id),
    sale_date       TIMESTAMPTZ NOT NULL,
    sale_price      NUMERIC(10,2) NOT NULL,
    condition       TEXT,                    -- raw condition string from source
    seller_info     JSONB,                   -- flexible: rating, location, etc.
    listing_url     TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_fs_card_date ON fact_sales(card_id, sale_date);
```

### Key Design Decisions

**variant column in dim_cards**: pokemontcg.io treats variants as pricing tiers on a single card record (tcgplayer.prices has keys like "holofoil", "reverseHolofoil", "normal"). We explode these into separate dim_cards rows. One pokemontcg.io card ID like "base1-4" may produce up to 3 dim_cards rows: "base1-4/holofoil", "base1-4/reverseHolofoil", "base1-4/normal". The `/variant` suffix keeps IDs unique and traceable.

**tcg_product_id**: TCGCSV uses productId (integer). pokemontcg.io uses string IDs. We store both so we can join TCGCSV prices to our card dimension. The mapping is built during ingestion by matching on set + card number + variant.

**subtype_name in fact_market_prices**: TCGCSV prices have a `subTypeName` field ("Normal", "Holofoil", "Reverse Holofoil", "1st Edition Holofoil", etc.). This is the condition/variant axis in their data. We store it raw and map it to our variant taxonomy.

---

## 3. TCGCSV Ingestion Pipeline

### Architecture

```
tcgcsv.com/archive/ → download 7z → extract JSON → transform → PostgreSQL
tcgcsv.com/tcgplayer/3/{groupId}/prices → JSON → transform → PostgreSQL (weekly refresh)
```

### Steps

1. **Bootstrap historical data** (one-time):
   - Download all daily archives from `https://tcgcsv.com/archive/tcgplayer/prices-YYYY-MM-DD.ppmd.7z` (Feb 8 2024 → today)
   - Extract each archive (needs `p7zip-full` package)
   - Each archive contains `{categoryId}/{groupId}/prices` JSON files
   - Filter to categoryId=3 (Pokemon)
   - Parse JSON, map productId → card_id via dim_cards lookup
   - Insert into fact_market_prices with the archive date as price_date
   - Deduplicate: skip if (card_id, price_date, subtype_name) already exists

2. **Weekly refresh** (ongoing):
   - Hit live endpoint: `https://tcgcsv.com/tcgplayer/3/{groupId}/prices` for each Pokemon group
   - Get list of groups from `https://tcgcsv.com/tcgplayer/3/groups`
   - Insert new rows with today's date
   - Log: total cards priced, new prices inserted, skipped

3. **Product-to-card mapping**:
   - Fetch products: `https://tcgcsv.com/tcgplayer/3/{groupId}/products`
   - Match TCGCSV products to dim_cards by: set name → dim_sets.tcg_group_id, then card name + number
   - Store tcg_product_id in dim_cards
   - Flag unmatched products for manual review

### Dependencies
- `p7zip-full` (apt package for 7z extraction)
- `requests` (HTTP)
- `sqlalchemy` + `psycopg2-binary` (DB)

---

## 4. pokemontcg.io Card Catalog Ingestion

### Architecture

```
pokemontcg.io/v2/sets → dim_sets
pokemontcg.io/v2/cards → dim_cards + dim_pokemon
```

### Steps

1. **Ingest sets** (one-time + weekly refresh):
   - `GET https://api.pokemontcg.io/v2/sets`
   - Map to dim_sets: id→set_id, name, series, totalCards (as integer), releaseDate
   - Match to TCGCSV groupId by set name (fuzzy match may be needed)

2. **Ingest cards** (one-time, paginated):
   - `GET https://api.pokemontcg.io/v2/cards?pageSize=250&page=N`
   - ~46,000 total cards, 250 per page = ~184 requests
   - Rate limit: 20k/day with API key → comfortably fits in one run
   - For each card:
     a. Upsert dim_pokemon from nationalPokedexNumbers + name + types + HP
     b. Explode tcgplayer.prices keys into separate dim_cards rows (one per variant)
     c. Card ID format: `{pokemontcg_id}/{variant}` e.g. "base1-4/holofoil"
     d. Store image URLs, rarity, subtypes, artist

3. **Deduplication**:
   - Pokemon dimension: dedupe on (name, pokedex_num)
   - Cards: primary key is the composite card_id, no dupes possible

### Rate Limit Strategy
- 250 cards per request, ~184 requests total
- With API key: 20,000 req/day limit, no concern
- Add 0.5s delay between requests to be respectful → ~92 seconds total

---

## 5. Package Structure

```
cardprice/
├── plans/                        # Prometheus plans (this file)
├── scripts/
│   └── linear.sh                 # Linear API wrapper
├── cardprice/
│   ├── __init__.py
│   ├── config.py                 # DB connection string, API keys, constants
│   ├── db/
│   │   ├── __init__.py
│   │   ├── schema.py             # SQLAlchemy models matching DDL above
│   │   ├── session.py            # Engine + sessionmaker
│   │   └── migrate.py            # Create/update tables (raw SQL DDL)
│   ├── scrapers/
│   │   ├── __init__.py
│   │   ├── tcgcsv.py             # TCGCSV live + archive ingestion
│   │   ├── pokemontcg.py         # pokemontcg.io catalog ingestion
│   │   └── mapping.py            # Cross-source ID mapping (TCGCSV productId ↔ card_id)
│   ├── models/                   # ML models (Phase 3+)
│   │   └── __init__.py
│   └── features/                 # Feature engineering (Phase 3+)
│       └── __init__.py
├── data/                         # Local data cache
│   ├── archives/                 # Downloaded TCGCSV 7z files
│   ├── extracted/                # Extracted JSON from archives
│   └── embeddings/               # numpy/parquet embedding files (Phase 3+)
├── notebooks/                    # Exploration notebooks
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

## 6. Phase 1 Task Breakdown (Build Order)

### Step 1: Scaffold & DB (1 session)
- Create package structure above
- Write `config.py`, `db/session.py`, `db/migrate.py`
- Run DDL to create all tables
- Insert seed marketplace row: TCGPlayer
- Verify: `psql cardprice -c "SELECT * FROM dim_marketplaces;"`

### Step 2: Card Catalog Ingestion (1 session)
- Write `scrapers/pokemontcg.py`
- Ingest all sets → dim_sets
- Ingest all cards → dim_pokemon + dim_cards (with variant explosion)
- Verify: `SELECT count(*) FROM dim_cards;` should be ~50-80k (variants exploded)

### Step 3: TCGCSV Product Mapping (1 session)
- Write `scrapers/mapping.py`
- Fetch all TCGCSV Pokemon groups and products
- Map TCGCSV productId → dim_cards.tcg_product_id
- Log match rate. Target: >90% matched. Review unmatched.

### Step 4: Historical Price Backfill (1–2 sessions)
- Write `scrapers/tcgcsv.py` (archive mode)
- Install p7zip-full
- Download and extract archives from Feb 2024 → present
- Load into fact_market_prices
- Verify: `SELECT price_date, count(*) FROM fact_market_prices GROUP BY 1 ORDER BY 1;`

### Step 5: Weekly Refresh Pipeline (1 session)
- Add live-mode to `scrapers/tcgcsv.py`
- Fetch current prices from TCGCSV live endpoints
- Insert into fact_market_prices with today's date
- Test idempotency: running twice on same day should not duplicate

### Step 6: Validation & Smoke Tests (1 session)
- Query: top 10 most expensive cards today
- Query: price history for Charizard Base Set over time
- Query: cards with biggest price changes in last 30 days
- Verify no orphan fact rows (every fact row joins to a dim)
- Write a simple `notebooks/01_data_validation.ipynb`

---

## 7. Open Questions

1. **pokemontcg.io API key**: Need to register at their developer portal. Free, just need to do it.
2. **TCGCSV ↔ pokemontcg.io matching**: Set names may differ slightly ("Scarlet & Violet: Obsidian Flames" vs "Obsidian Flames"). Need fuzzy matching or a manual mapping table.
3. **Storage for 7z archives**: ~2 years of daily archives. Estimate ~50-100GB extracted. May want to delete extracted JSON after loading to save disk.
4. **Trainer and Energy cards**: These aren't Pokemon but have prices. Include them in dim_cards with pokemon_id=NULL. They still matter for pricing.

---

## 8. Definition of Done (Phase 1)

- [ ] PostgreSQL running locally with all tables created
- [ ] dim_pokemon populated with all Pokemon species (~1000 rows)
- [ ] dim_sets populated with all English TCG sets (~200+ rows)
- [ ] dim_cards populated with all card variants (~50-80k rows)
- [ ] dim_cards.tcg_product_id populated with >90% match rate to TCGCSV products
- [ ] fact_market_prices loaded with daily prices from Feb 2024 → present
- [ ] Weekly refresh script runs successfully and is idempotent
- [ ] Validation notebook confirms data integrity (no orphans, reasonable price distributions)
- [ ] Can answer: "What is the current market price of Charizard Base Set Holofoil?"
