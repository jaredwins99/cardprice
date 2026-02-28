# Cardprice Development Process

How this project has been built, what worked, and how to replicate/improve it.

## The Loop

Every phase follows the same cycle:

```
1. PLAN    → Define tables, APIs, acceptance criteria
2. SCAFFOLD → DDL first, then empty modules with signatures
3. INGEST   → Get real data flowing, even if ugly
4. MAP      → Link data sources together (this is where 80% of bugs live)
5. VALIDATE → Spot-check with real examples ("Darkrai EX 107/108")
6. FIX      → Categorize failures into patterns, fix systematically
7. MEASURE  → Report match rates, row counts, coverage %
8. REPEAT   → Until diminishing returns, then move on
```

## Phase Execution Pattern

### Phase 1: Data Foundation
What we did:
- Designed star schema (dim_pokemon, dim_sets, dim_cards, fact_market_prices)
- Loaded card catalog from pokemontcg.io (fell back to GitHub mirror when API was down)
- Backfilled 751 days of TCGCSV price archives (30.9M rows)
- Mapped TCGCSV product IDs to our card catalog

Key lesson: **The mapping is the hard part.** Raw ingestion is straightforward.
Card catalog had 20,078 entries. TCGCSV had 24,179 products. Getting them to
agree required 6 matching passes and iterative debugging.

### Phase 2: Enrichment + Scrapers
What we did:
- eBay sold listings scraper (real transaction prices)
- PokeAPI species metadata (base stats, types, legendaries)
- Smogon competitive usage stats
- Card scrape priority queue
- Image downloader pipeline

Key lesson: **Write the schema migration first**, then the ingestion code.
Having the table structure forces you to think about what data you actually need.

### Phase 3: ML + Models
What we did:
- Researched card recognition landscape (closed apps, no open models)
- Built tiered approach: hash → DINOv2 → CLIP → Claude vision
- Baseline price predictor (R²=0.517)
- Condition pricing model with multipliers
- Inventory system

Key lesson: **Always look for the lazy solution first.**
DINOv2 embeddings + FAISS gives 90-95% accuracy with zero training.
Claude Haiku gives ~98% at $0.0015/card. Neither requires collecting
training data or running GPU training jobs.

## Mapping Strategy (the most reusable part)

The TCGCSV-to-catalog mapping went through these stages:

| Stage | Match Rate | What Changed |
|-------|-----------|--------------|
| v1    | 77%       | Exact name + number match |
| v2    | 86%       | Fuzzy matching, set name normalization |
| v3    | 97%       | Prefix stripping, parenthetical cleaning |
| v4    | 99.3%     | Number-only pass, supplemental groups, claimed-card tracking |
| v5    | 99.8%     | Energy aliases, a-suffix post-pass |

### The fix pattern that always works:

1. **Count failures**: `SELECT count(*) FROM dim_cards WHERE tcg_product_id IS NULL`
2. **Categorize failures**: Group unmapped cards by pattern (a-suffix, RC-prefix, no set group, etc.)
3. **Fix the biggest category first**: Usually 1 regex or alias table fixes 50+ cards
4. **Re-run and re-count**: Verify the fix, check for regressions
5. **Stop when remaining failures are genuinely unmappable**

The 36 remaining unmapped cards are: trainer kits without TCGCSV groups (37),
Japan-only promos (4), SVP cards too new for TCGCSV (9), etc. These aren't
bugs — they're data that doesn't exist upstream.

### Common mapping failure patterns (reusable across any card data source):

| Pattern | Example | Fix |
|---------|---------|-----|
| Parenthetical qualifiers | "Leafeon V (Full Art)" vs "Leafeon V" | Regex strip `_PAREN_QUALIFIER_RE` |
| Abbreviated names | "Blend Energy GFPD" vs "Blend Energy GrassFirePsychicDarkness" | `ENERGY_NAME_ALIASES` dict |
| Alternate art suffixes | card #121a vs product #121 | Post-pass: copy base card's product ID |
| Supplemental groups | RC cards in Legendary Treasures | `SUPPLEMENTAL_GROUPS` dict |
| Shared groups | Trainer Kit halves | `SHARED_GROUPS` dict |
| Number format mismatch | "074a/147" vs "74" | `_parse_card_number()` strips leading zeros + alpha suffix |

## Agent Delegation Pattern

Heavy use of Claude Code subagents for parallelism:

### What works well for agents:
- **Research tasks**: "Analyze all 329 unmapped cards and categorize by failure pattern"
- **Independent module creation**: "Write an eBay title parser" (clear inputs/outputs)
- **API interactions**: "Create 8 Linear tickets with these descriptions"
- **Data analysis**: "Run these SQL queries and report findings"

### What doesn't work for agents:
- **Multi-step DB operations**: Agents get blocked on bash permissions
- **Code that depends on other agents' output**: Sequential dependencies
- **Large refactors across many files**: Context gets fragmented

### The optimal delegation pattern:
1. Do the architecture/schema yourself in the main thread
2. Spawn agents for independent module implementation (5-10 at a time)
3. Apply agent outputs from the main thread (run migrations, verify imports)
4. Spawn more agents for the next wave of independent tasks

## Validation Playbook

### Spot-checking (do this constantly):
```sql
-- Pick a specific card you know the price of
SELECT c.card_id, c.name, c.tcg_product_id, p.market_price, p.price_date
FROM dim_cards c
LEFT JOIN fact_market_prices p ON c.tcg_product_id = p.tcg_product_id
WHERE c.card_id LIKE 'bw5-107%'
ORDER BY p.price_date DESC LIMIT 5;
```

### Coverage metrics (run after every change):
```sql
SELECT
    (SELECT count(*) FROM dim_cards) as total_cards,
    (SELECT count(*) FROM dim_cards WHERE tcg_product_id IS NOT NULL) as mapped,
    (SELECT count(*) FROM fact_market_prices WHERE card_id IS NOT NULL) as linked;
```

### Failure categorization (when match rate stalls):
```sql
SELECT
    CASE
        WHEN card_number ~ '[0-9]+a$' THEN 'a-suffix'
        WHEN card_number ~ '^RC' THEN 'RC-prefix'
        WHEN set_id IN ('tk1a','tk1b','tk2b','fut20') THEN 'no-tcgcsv-group'
        ELSE 'other'
    END as category,
    count(*)
FROM dim_cards WHERE tcg_product_id IS NULL
GROUP BY 1 ORDER BY 2 DESC;
```

## Environment Notes

- **WSL2**: PostgreSQL dies on WSL restart. Always check `sudo service postgresql status` first.
- **pokemontcg.io**: Intermittently down (504s). GitHub mirror at `PokemonTCG/pokemon-tcg-data` is the fallback.
- **TCGCSV**: Reliable, free, daily archives. The best price data source.
- **Linear API**: Use `curl` directly, not the MCP server (hangs in WSL2).

## What to Improve Next

1. **Automated regression tests**: After every mapping.py change, run mapping and assert match rate >= 99.8%
2. **Data freshness monitoring**: Alert if TCGCSV archives stop appearing or pokemontcg.io stays down
3. **Agent output validation**: Before applying agent-generated code, run import checks automatically
4. **Incremental mapping**: Don't re-map already-mapped cards, only process new/unmapped ones
5. **Price anomaly detection**: Flag cards where market_price changes >50% in a day
