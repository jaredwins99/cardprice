# Data Pipeline (scrapers/)

All scrapers live in `cardprice/scrapers/`. Each writes to specific DB tables and is invoked through the CLI.

## Scrapers

### pokemontcg.py -- Card Catalog

Loads the full Pokemon TCG card catalog into `dim_sets`, `dim_pokemon`, `dim_cards`.

- **Primary source**: pokemontcg.io API (v2)
- **Fallback**: GitHub mirror (`PokemonTCG/pokemon-tcg-data`) when API returns 504s
- Loads ~20k cards, filtered to "normal" variants only
- Handles retry with exponential backoff

```bash
python -m cardprice.cli refresh    # updates catalog + live prices
```

### tcgcsv.py -- Live Price Ingestion

Fetches current-day prices from TCGCSV (tcgcsv.com) into `fact_market_prices`.

- Pulls all Pokemon groups (category 3), then prices per group
- Deduplicates on `(tcg_product_id, price_date, subtype_name)`
- 0.3s delay between group fetches

### backfill.py -- Historical Archives

Downloads and ingests daily TCGCSV archive snapshots (Feb 2024 - present).

- Archives: ~3MB/day compressed, 193 price files per archive
- Idempotent: skips already-ingested dates
- Processes in batches of 7 days, restartable

```bash
python -m cardprice.cli backfill --start 2024-02-08 --end 2025-12-31
```

### mapping.py -- Product-to-Card Mapping

Links TCGCSV `tcg_product_id` values to `dim_cards.card_id`.

- Matches sets by name (fuzzy, threshold 0.80)
- Matches cards by set + name + card number
- Handles energy aliases, supplemental groups, shared groups
- Current rate: 99.8% match (166/171 sets, 15,473/20,078 cards)

```bash
python -m cardprice.cli mapping
```

### ebay.py -- Sold Listings Scraper

Scrapes completed/sold eBay listings for Pokemon cards.

- HTML scraping via BeautifulSoup (no API)
- Rotating user agents, 2-3s delay between pages
- Targets eBay category 183454 (Pokemon TCG)
- Writes to `fact_sales`

### ebay_title_parser.py -- Title Parser

Parses unstructured eBay titles into structured fields:
- Card name, set name, card number
- Grading authority + grade (PSA, BGS, CGC, SGC, ACE, AGS)
- Condition, edition, variant

### ebay_matcher.py -- Listing Matcher

Matches parsed eBay listings to `dim_cards` using fuzzy string matching.
- Confidence threshold: 0.55
- Writes matched sales to `fact_sales` with `match_confidence`

### pokeapi.py -- Species Metadata

Fetches Pokemon species data from PokeAPI into `dim_pokemon_features`.
- Stats (HP, ATK, DEF, SpA, SpD, Spe, BST), types, generation
- Legendary/mythical flags, capture rate, egg groups, habitat

```bash
python -m cardprice.cli pokeapi
```

### smogon.py -- Competitive Usage

Fetches Smogon usage stats from `pkmn.github.io` into `dim_smogon_usage`.
- Default formats: gen9ou, gen9uu, gen9ubers, gen9vgc2025
- Weighted/raw/real usage percentages, viability (GXE)

```bash
python -m cardprice.cli smogon --formats gen9ou,gen9uu
```

### priority.py -- Scrape Priority Queue

Ranks cards for scraping priority in `card_scrape_priority`.

Score weights:
| Component | Weight | Logic |
|-----------|--------|-------|
| market_price | 0.35 | Higher price = higher priority |
| volatility | 0.30 | Stddev of last 30 days |
| rarity | 0.20 | Rarer cards scored higher |
| set_recency | 0.15 | Newer sets scored higher |

```bash
python -m cardprice.cli priority --top-n 2000
```

### image_downloader.py -- Card Images

Downloads card images from pokemontcg.io URLs stored in `dim_cards`.
- Supports small/large sizes
- Skips already-downloaded images
- Retry with backoff

```bash
python -m cardprice.cli download-images --size small --output data/card_images
```
