# Cardprice Documentation

Pokemon card pricing engine: ingest market data, identify cards from photos via ML, predict prices, and manage inventory.

## Architecture

```
  Phone (iOS Shortcut / Android)
        |
        | photo
        v
  +--- Syncthing ---> data/inbox/ ---> watch_folder.py ---+
  |                                                        |
  | HTTP POST /scan                                        |
  v                                                        v
server.py ---------> ML Cascade -------> Price Lookup ---> Response
  ^                     |                     |
  |                     v                     v
telegram_bot.py    dim_cards            fact_market_prices
                                              |
            +------ Data Pipeline ------------+
            |            |            |
            v            v            v
       scrapers/     mapping.py   backfill.py
       (catalog,     (99.8%       (30.9M rows,
        eBay,         match)       751 days)
        Smogon,
        PokeAPI)
```

### ML Cascade (card identification from photo)

```
  photo
    |
    v
  hash_matcher.py ----> exact match? ---yes---> done (cost: $0)
    |no
    v
  dino_matcher.py ----> DINOv2+FAISS ---yes---> done (cost: $0, ~93% acc)
    |no (low confidence)
    v
  clip_matcher.py ----> CLIP match? ----yes---> done (cost: $0)
    |no
    v
  claude_scanner.py --> Claude Haiku ----------> done (cost: ~$0.0015/card, ~98% acc)
```

## Documentation Tree

```
docs/
|
|-- README.md                 <-- you are here (index + architecture)
|
|-- data-pipeline.md          Data ingestion and mapping
|   |-- pokemontcg.py            Card catalog (20k cards, 171 sets)
|   |-- tcgcsv.py                Live daily prices from TCGPlayer
|   |-- backfill.py              Historical price archives (Feb 2024 - present)
|   |-- mapping.py               Link TCGCSV products to card catalog (99.8%)
|   |-- ebay.py                  Sold listings scraper + title parser + matcher
|   |-- pokeapi.py               Species metadata (stats, types, legendaries)
|   |-- smogon.py                Competitive usage stats
|   |-- priority.py              Scrape priority queue (price/volatility/rarity)
|   +-- image_downloader.py      Bulk card image downloads
|
|-- process.md                Development process and methodology
|   |-- The Loop (plan/scaffold/ingest/map/validate/fix/measure)
|   |-- Phase execution patterns (1: data, 2: enrichment, 3: ML)
|   |-- Mapping strategy (77% -> 99.8% in 5 stages)
|   |-- Agent delegation patterns
|   +-- Validation playbook (SQL spot-checks)
|
|-- syncthing-setup.md        Phone-to-WSL2 file sync
|   |-- WSL2 install + systemd service
|   |-- Phone setup (Android Syncthing / iOS Mobius Sync)
|   +-- WSL2 networking (relay vs port-forward vs mirrored)
|
+-- ios-shortcut.md           One-tap card scanning from iPhone
    |-- Step-by-step Shortcuts app setup (9 actions)
    |-- Expected server JSON response format
    +-- Network access (Tailscale / WireGuard / Cloudflare Tunnel)
```

### Component Map

```
cardprice/
|-- cli.py                    CLI entry point (12 commands)
|-- config.py                 DB connection, API keys, paths
|-- server.py                 HTTP scan server (port 8888)
|-- telegram_bot.py           Telegram photo scanning bot
|
|-- scrapers/                 Data ingestion layer
|   |-- pokemontcg.py            Card catalog loader
|   |-- tcgcsv.py                Live price fetcher
|   |-- backfill.py              Historical archive ingester
|   |-- mapping.py               TCGCSV-to-catalog linker
|   |-- ebay.py                  eBay sold listings scraper
|   |-- ebay_title_parser.py     Parse eBay titles into fields
|   |-- ebay_matcher.py          Match eBay listings to dim_cards
|   |-- pokeapi.py               Pokemon species metadata
|   |-- smogon.py                Competitive usage stats
|   |-- priority.py              Scrape priority queue
|   |-- image_downloader.py      Bulk image downloads
|   +-- watch_folder.py          Auto-scan photos from inbox/
|
|-- ml/                       Card recognition (cascade)
|   |-- hash_matcher.py          Perceptual hash (free, exact match)
|   |-- dino_matcher.py          DINOv2 + FAISS (free, ~93%)
|   |-- clip_matcher.py          CLIP similarity (free, fallback)
|   +-- claude_scanner.py        Claude Haiku vision (paid, ~98%)
|
|-- models/                   Pricing models
|   |-- condition_pricing.py     Condition-based multipliers
|   |-- price_predictor.py       GradientBoosting predictor (R2=0.517)
|   +-- card_embeddings.py       Card feature embeddings
|
|-- db/                       Database layer (PostgreSQL, star schema)
|-- features/                 Feature engineering
+-- utils/                    Shared utilities
```

## Quick Start

```bash
# 1. Set up database and load card catalog
python -m cardprice.cli migrate && python -m cardprice.cli refresh

# 2. Map prices to cards and backfill history
python -m cardprice.cli mapping && python -m cardprice.cli backfill

# 3. Start the scan server (phone uploads photos here)
python -m cardprice.server --port 8888

# 4. Or scan a single card from the command line
python -m cardprice.cli scan --image path/to/card.jpg
```

## Key Numbers

| Metric | Value |
|--------|-------|
| Cards in catalog | 20,078 |
| Sets mapped | 166 / 171 |
| Card mapping rate | 99.8% |
| Price rows | 30.9M |
| Date range | Feb 2024 - Feb 2026 |
| Price subtypes | 7 |

## Scan Interfaces (4 ways to identify a card)

| Interface | How it works | Doc |
|-----------|-------------|-----|
| **HTTP server** | `POST /scan` with image, get JSON back | [ios-shortcut.md](ios-shortcut.md) |
| **CLI** | `python -m cardprice.cli scan --image photo.jpg` | -- |
| **Watch folder** | Drop images in `data/inbox/`, auto-scanned | [syncthing-setup.md](syncthing-setup.md) |
| **Telegram bot** | Send photo to bot, get card name + price | -- |

## Phone Integration

To scan cards from your phone, you need two things:

1. **Get photos to the server**: Either upload directly via HTTP ([iOS Shortcut](ios-shortcut.md)) or sync photos automatically ([Syncthing](syncthing-setup.md))
2. **Server running**: `python -m cardprice.server --port 8888` or `python -m cardprice.cli watch`

See [syncthing-setup.md](syncthing-setup.md) for phone-to-WSL2 file sync and [ios-shortcut.md](ios-shortcut.md) for one-tap scanning from iPhone.
