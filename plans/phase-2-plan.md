# Phase 2 Plan: Real Transaction Data (eBay Sold Listings)

## Goal

Ingest real completed-sale transaction data from eBay into `fact_sales`, starting with the highest-value Pokemon cards. By the end of Phase 2, we have 1,000+ real eBay transactions matched to our card catalog with 70%+ accuracy, enabling true market price analysis beyond TCGPlayer aggregates.

## Why This Matters

TCGPlayer market prices (Phase 1) are useful but limited: they are aggregated mid/low/high values, not individual transactions. To build a pricing algorithm that captures trends, outliers, condition effects, and cross-marketplace arbitrage, we need granular sale-level data. eBay completed listings are the richest public source of this data.

---

## 1. Data Sources — Ranked

| Rank | Source | Type | Status | Notes |
|------|--------|------|--------|-------|
| 1 | **eBay Sold Listings** | Individual transactions | **Primary — Phase 2** | Public completed listings, richest dataset |
| 2 | **TCGPlayer Sales History** | Individual transactions | Future phase | Requires scraping product pages; more structured but less volume |
| 3 | **PriceCharting** | Aggregated sale prices | Monitoring | Good for validation/benchmarking; no individual sale data |
| 4 | **Mercari US** | Individual transactions | Deferred | No public API, requires Playwright + stealth; high effort |
| 5 | **Facebook Marketplace** | Individual transactions | Deferred | No API, anti-scraping measures, low signal-to-noise |

### Third-Party Aggregator APIs (Supplementary)

These are not primary sources but may be useful for validation, gap-filling, or bootstrapping:

- **PokemonPriceTracker** — Free tier available. Provides eBay PSA graded card price averages. Useful for validating our scraped eBay data against an independent source.
- **TCGAPIs** — Paid API with individual sale history from TCGPlayer. Could supplement our TCGPlayer data in a future phase without scraping.
- **PokeData.io** — Multi-source price aggregation service. Pulls from eBay, TCGPlayer, and others. Worth evaluating as a cross-reference.

---

## 2. eBay Approach

### Why Web Scraping (Not API)

- **eBay Finding API**: Deprecated / heavily restricted. Does not reliably return sold listings for non-partner developers.
- **eBay Marketplace Insights API**: Partner-only program. Requires application and approval process that is effectively closed to small projects.
- **eBay Browse API**: Returns active listings only, not completed sales.
- **Conclusion**: Web scraping of public sold listing search results is the only viable path.

### Technical Approach

**Stack**: `requests` + `BeautifulSoup4` (no browser automation needed for search results pages)

**Target URL pattern**:
```
https://www.ebay.com/sch/i.html?_nkw={search_query}&_sacat=183454&LH_Sold=1&LH_Complete=1&_ipg=240
```

Key parameters:
- `_sacat=183454` — eBay category for Pokemon TCG
- `LH_Sold=1&LH_Complete=1` — Completed/sold listings only
- `_ipg=240` — Max results per page
- `_sop=13` — Sort by end date (newest first)
- `rt=nc` — No redirect

**Data extracted per listing**:
- Title (freeform seller text)
- Sale price (including best offer accepted prices)
- Sale date
- Shipping price (when shown)
- Listing URL / item ID
- Condition tag (if present)
- Image URL
- Number of bids (auction) or "Buy It Now" indicator

### Rate Limiting & Anti-Detection

- **Delay**: 3-5 second random delay between requests
- **User-Agent rotation**: Pool of 5-10 common browser user-agents
- **No login**: All data accessed as anonymous public visitor
- **Session management**: Fresh `requests.Session()` per search batch, with cookie handling
- **Error handling**: Exponential backoff on 429/503 responses, circuit breaker after 5 consecutive failures
- **Daily cap**: 500 search pages per day maximum (covers ~120,000 listings)

---

## 3. Schema Changes

### Additions to `fact_sales`

The existing `fact_sales` table (created in Phase 1 DDL but unpopulated) needs additional columns:

```sql
ALTER TABLE fact_sales ADD COLUMN source_item_id    TEXT;           -- eBay item number, e.g. "314567890123"
ALTER TABLE fact_sales ADD COLUMN quantity           INTEGER DEFAULT 1;
ALTER TABLE fact_sales ADD COLUMN sale_type          TEXT;           -- "auction", "buy_it_now", "best_offer"
ALTER TABLE fact_sales ADD COLUMN shipping_price     NUMERIC(10,2); -- NULL if free or unknown
ALTER TABLE fact_sales ADD COLUMN grading_authority  TEXT;           -- "PSA", "BGS", "CGC", NULL if raw
ALTER TABLE fact_sales ADD COLUMN grade              TEXT;           -- "10", "9.5", etc.
ALTER TABLE fact_sales ADD COLUMN raw_title          TEXT NOT NULL;  -- Original eBay listing title
ALTER TABLE fact_sales ADD COLUMN image_urls         TEXT[];         -- Array of image URLs
ALTER TABLE fact_sales ADD COLUMN match_confidence   REAL;           -- 0.0-1.0 confidence of card_id match

CREATE INDEX idx_fs_source_item ON fact_sales(source_item_id);
CREATE INDEX idx_fs_confidence ON fact_sales(match_confidence);

-- Deduplicate constraint: same item from same source should not be inserted twice
ALTER TABLE fact_sales ADD CONSTRAINT uq_fact_sales_source
    UNIQUE (marketplace_id, source_item_id);
```

### New Tables

```sql
-- Track scrape job runs for observability and resumption
CREATE TABLE scrape_jobs (
    job_id          SERIAL PRIMARY KEY,
    source          TEXT NOT NULL,           -- "ebay"
    search_query    TEXT NOT NULL,           -- The query string used
    card_id         TEXT REFERENCES dim_cards(card_id),  -- Card this job targeted (if any)
    status          TEXT NOT NULL DEFAULT 'pending',     -- pending, running, completed, failed
    pages_scraped   INTEGER DEFAULT 0,
    listings_found  INTEGER DEFAULT 0,
    listings_matched INTEGER DEFAULT 0,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error_message   TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Priority queue: which cards to scrape next, ranked by value/importance
CREATE TABLE card_scrape_priority (
    card_id             TEXT PRIMARY KEY REFERENCES dim_cards(card_id),
    priority_score      REAL NOT NULL,           -- Higher = scrape first
    last_scraped_at     TIMESTAMPTZ,
    total_sales_found   INTEGER DEFAULT 0,
    next_scrape_after   TIMESTAMPTZ,             -- Rate limit per card
    search_query        TEXT,                     -- Pre-built eBay search string
    created_at          TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_csp_priority ON card_scrape_priority(priority_score DESC);
```

---

## 4. Title Parsing & Entity Resolution

This is the hardest part of Phase 2. eBay listing titles are freeform seller text with no standardization. Examples:

```
"Charizard Base Set Holo 4/102 PSA 10 GEM MINT Pokemon Card"
"1999 Pokemon Charizard #4 Base WOTC Holo Rare LP-NM"
"POKEMON CHARIZARD 4/102 BASE SET UNLIMITED HOLO **READ DESCRIPTION**"
"Lot of 5 Pokemon Cards Charizard Blastoise Venusaur Base Set"
```

### Matching Pipeline

1. **Pre-filter**: Skip listings with keywords indicating lots, bundles, custom items, fakes ("lot of", "bundle", "proxy", "custom", "fake", "repack")
2. **Normalize**: Lowercase, strip special characters, normalize whitespace
3. **Extract structured fields**:
   - Card name (fuzzy match against dim_cards.name)
   - Set name or set number (e.g., "base set", "4/102")
   - Variant: holo, reverse holo, 1st edition
   - Grading: PSA/BGS/CGC + numeric grade
   - Condition: NM, LP, MP, HP, DMG (for raw cards)
4. **Candidate generation**: Query dim_cards for candidates matching extracted card name + set
5. **Scoring**: Weighted confidence score based on:
   - Card name match (Levenshtein / token overlap): 40%
   - Set match: 25%
   - Card number match: 20%
   - Variant match: 15%
6. **Threshold**: Accept matches with confidence >= 0.65; flag 0.45-0.65 for review; reject < 0.45

### Libraries

- `rapidfuzz` — Fast fuzzy string matching (Levenshtein, token_sort_ratio, etc.)
- `regex` / `re` — Pattern extraction for set numbers, grades, conditions

### Known Challenges

- Multi-card lots (must be filtered or split)
- Japanese vs. English cards (filter to English-only initially)
- Misspellings ("Charzard", "Pikachu VMAX" vs "Pikachu V-MAX")
- Promo cards with non-standard numbering
- Graded vs. raw distinction changes effective price significantly

---

## 5. Priority Queue Strategy

We have ~20,000 cards in `dim_cards`, but we should not scrape all of them. Most bulk commons have negligible transaction volume on eBay.

### Priority Score Formula

```
priority_score = log(market_price + 1) * rarity_weight * recency_weight
```

Where:
- `market_price` = latest TCGPlayer market price from `fact_market_prices`
- `rarity_weight` = multiplier based on rarity (Rare Holo = 2.0, Ultra Rare = 3.0, etc.)
- `recency_weight` = newer sets get slight boost (more eBay activity)

### Initial Target

- **Top 500 cards** by priority score for MVP
- Scale to **2,000 cards** once pipeline is validated
- Each card gets re-scraped at most once per week
- Estimated eBay volume: ~20-50 sold listings per top card per week

---

## 6. Legal Considerations

### Precedent

**hiQ Labs v. LinkedIn (2022, 9th Circuit)**: Scraping publicly accessible data does not violate the Computer Fraud and Abuse Act (CFAA). This is the leading US case law on web scraping of public data.

### Mitigations

- **No login required**: All eBay sold listings are publicly visible without authentication
- **Rate limiting**: 3-5 second delays between requests; well below any reasonable server impact threshold
- **No circumvention**: We do not bypass any access controls, CAPTCHAs, or robots.txt blocks
- **Respectful behavior**: Identify as a standard browser, respect HTTP 429 responses, cap daily volume
- **Data use**: Internal analysis only; we do not republish raw eBay data

### robots.txt Compliance

eBay's `robots.txt` allows crawling of `/sch/` (search) paths for most user agents. We will verify this before launching and honor any `Crawl-delay` directives.

---

## 7. MVP Definition of Done

- [ ] Schema migration applied: `fact_sales` extended, `scrape_jobs` and `card_scrape_priority` tables created
- [ ] eBay scraper fetches completed Pokemon card listings and parses price, date, title, item ID
- [ ] Title parser extracts card name, set, variant, grade from freeform eBay titles
- [ ] Entity resolution matches eBay listings to `dim_cards` with confidence scoring
- [ ] **1,000+ real eBay transactions** loaded into `fact_sales`
- [ ] **70%+ title match accuracy** (measured by manual spot-check of 100 random matches)
- [ ] Deduplication works: re-running scraper does not create duplicate `fact_sales` rows
- [ ] Priority queue populated from Phase 1 market prices; top 500 cards targeted
- [ ] Scrape job tracking logs all runs with status, counts, errors
- [ ] CLI command: `python -m cardprice.cli scrape-ebay` runs the full pipeline
- [ ] Marketplace row inserted: `INSERT INTO dim_marketplaces (name, source_system) VALUES ('eBay', 'ebay_scraper')`

---

## 8. Build Order (5-8 Sessions)

### Session 1: Schema Migration
- Write Alembic or raw SQL migration for `fact_sales` columns, `scrape_jobs`, `card_scrape_priority`
- Apply migration to local DB
- Insert eBay marketplace row
- Verify all tables/columns exist

### Session 2: eBay Scraper Core
- Write `cardprice/scrapers/ebay.py`
- Implement search URL builder for Pokemon TCG sold listings
- Parse search results page: extract title, price, date, item ID, shipping, image
- Handle pagination (multi-page results)
- Rate limiting, user-agent rotation, error handling
- Unit test with saved HTML fixtures

### Session 3: Title Parser & Entity Resolution
- Write `cardprice/scrapers/title_parser.py`
- Implement title normalization and field extraction (name, set, number, variant, grade)
- Implement candidate generation from `dim_cards`
- Implement confidence scoring with `rapidfuzz`
- Test against 50+ real eBay title examples
- Filter logic for lots, bundles, non-English cards

### Session 4: Priority Queue
- Write `cardprice/scrapers/priority.py`
- Populate `card_scrape_priority` from Phase 1 market prices
- Build eBay search queries for top N cards
- Implement scheduling logic (next_scrape_after, re-scrape cadence)

### Session 5: Integration & Pipeline
- Wire scraper + parser + priority queue into end-to-end pipeline
- Write `scrape-ebay` CLI command in `cardprice/cli.py`
- Implement scrape job tracking (create/update `scrape_jobs` rows)
- Handle the full flow: pick card from queue -> search eBay -> parse results -> match to dim_cards -> insert fact_sales

### Session 6: Validation & Tuning
- Run pipeline against top 50 cards
- Manually review 100 random matched sales for accuracy
- Tune confidence thresholds and matching weights
- Fix systematic parsing errors (common patterns the title parser misses)
- Measure: total sales ingested, match rate, confidence distribution

### Sessions 7-8: Scale & Harden (if needed)
- Scale to top 500 cards
- Handle edge cases: best offer prices, lots that slipped through filters, international listings
- Add retry logic for failed scrape jobs
- Performance: batch inserts, connection pooling
- Reach 1,000+ transaction target

---

## 9. File Structure (New/Modified)

```
cardprice/
├── scrapers/
│   ├── ebay.py              # NEW — eBay sold listing scraper
│   ├── title_parser.py      # NEW — Freeform title parsing + entity resolution
│   ├── priority.py          # NEW — Card scrape priority queue management
│   ├── tcgcsv.py            # Existing
│   ├── pokemontcg.py        # Existing
│   └── mapping.py           # Existing
├── db/
│   └── migrate.py           # MODIFIED — Add Phase 2 migration
├── cli.py                   # MODIFIED — Add scrape-ebay command
└── config.py                # MODIFIED — Add eBay scraper config (delays, caps, etc.)
```

---

## 10. Dependencies (New)

```
rapidfuzz>=3.0        # Fast fuzzy string matching
beautifulsoup4>=4.12  # HTML parsing (likely already installed)
```

No browser automation (Playwright/Selenium) is needed for Phase 2. eBay search results render server-side.

---

## 11. Open Questions

1. **eBay HTML stability**: eBay search result HTML structure changes periodically. The scraper will need CSS selector maintenance. Consider extracting structured data from `<script type="application/ld+json">` blocks if available.
2. **Best Offer prices**: Some eBay listings show "Price: $X or Best Offer — Sold". The actual accepted price may differ from the listed price. Need to determine if the accepted price is visible on sold listings.
3. **Graded vs. raw pricing**: PSA 10 Charizard and raw NM Charizard are fundamentally different products. Phase 2 will track the grade but match both to the same `card_id`. Phase 3 may need a graded-card dimension.
4. **TCGPlayer scraping**: If eBay coverage proves insufficient, TCGPlayer product pages show recent sales. This would be Session 9+ work.
5. **PokemonPriceTracker API**: Worth integrating as a validation source once we have our own eBay data to compare against.
