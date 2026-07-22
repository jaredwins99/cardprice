# pricing_model

**Isolated sub-project** (like `set_population/`): fair-price + direction models
for buy-decision support. Own `scripts/` + `data/`; reads main-project data
stores (Postgres, SQLite, set_population/pokemon_likeability JSONs) but never
edits main-project code.

## Purpose

Given a (card, printing), answer two questions the owner actually asks before
buying a specific card:

1. **Fair value** — what should this card cost given fundamentals (set print
   population, per-card PSA pops, species demand, artist, rarity, age,
   printing)? Residual vs the actual price = over/under-priced signal.
   Trained on LP/MP-adjusted prices (NM is the ceiling, not the floor).
2. **Direction** — given where the price has been (trailing returns,
   volatility, liquidity, pop velocity) and the fundamentals residual, did
   cards like this rise over the next 3–6 months? Trained on historical
   windows of the 2024–2026 daily price history, validated strictly forward
   in time.

Point estimate + rough band; a sanity-checking tool, not a trading system.

## Scripts

- `scripts/collect_card_pops.py` — per-card, per-variant PSA graded
  populations for the whole catalog (~20k cards) from the GradedMetrics/
  PokeMetrics community mirror. Expands set_population's chase-only
  collection; reuses its solved set-ID mapping. Includes 8-week pop history
  (grading velocity). Card matching = (card number, name-verified) with a
  year guard. Raw set docs cached in `data/psa_raw/`.

## Data

- `data/card_graded_pops.json` — output of collect_card_pops.py
- `data/psa_raw/` — cached GradedMetrics set docs (gitignored)

## Status

- 2026-07-21: sub-project started. Card-pop collection built (blocker #1:
  card-grain credible supply). English TCGCSV price feed found to be
  **unscheduled since inception** (the 2024-02→2026-02 "daily" history was a
  one-shot backfill); archives re-downloaded with a browser UA (tcgcsv.com
  401s `python-requests`) and backfilled via the unmodified
  `python -m cardprice.cli backfill`.
