# Phase 3: Pokemon Latent Space / Feature Engineering

## Goal

Build a 32-dimensional species embedding for every Pokemon that captures
competitive relevance, popularity, game stats, and market behavior. These
embeddings feed into the price prediction model in Phase 4.

---

## 1. Data Sources for Species Features

### 1.1 PokeAPI (pokeapi.co)

Primary source for in-game metadata. All endpoints are public, no auth required.

| Feature | Endpoint | Notes |
|---|---|---|
| Types (primary/secondary) | `/pokemon/{id}` | 18 types |
| Base stats (HP/Atk/Def/SpA/SpD/Spe) | `/pokemon/{id}` | Also compute BST |
| Evolution chains | `/evolution-chain/{id}` | Derive `evo_stage` (0=basic, 1=stage1, 2=stage2) |
| Legendary / Mythical / Ultra Beast | `/pokemon-species/{id}` | `is_legendary`, `is_mythical` flags |
| Egg groups | `/pokemon-species/{id}` | 15 groups |
| Capture rate | `/pokemon-species/{id}` | 0-255 scale |
| Generation | `/pokemon-species/{id}` | 1-9 |

Coverage: ~1025 species (through Gen 9). Rate limit is lenient (100 req/min)
but we should cache aggressively and use the bulk CSV dumps where possible.

### 1.2 Smogon Usage Stats (smogon.com/stats/)

Competitive tier placement and usage percentages from the official Smogon
ladder. Available as monthly text/JSON dumps at `smogon.com/stats/YYYY-MM/`.

- **Tier**: OU, UU, RU, NU, PU, ZU, Untiered, Uber, AG (ordinal encode)
- **Usage %**: float, varies by month; use latest available month
- Pokemon not in any tier get usage = 0 and tier = Untiered

Parse the `chaos/` JSON files for structured data. Fall back to the plain-text
usage files if needed.

### 1.3 Pokemon of the Year 2020 Vote

Google-run poll with 6.6M votes ranking 890 species. This is the single best
public signal for general popularity (not just competitive players).

- Source: Bulbapedia wiki page / various archived lists
- Normalize rank to 0-1 scale (rank 1 = 1.0, rank 890 = ~0.0)
- Pokemon not in the vote (Gen 8 DLC, Gen 9) get imputed as median rank

### 1.4 Card-Market-Derived Features (from our own data)

Computed from `fact_prices` and `dim_cards` after Phase 1/2 backfill:

| Feature | Computation |
|---|---|
| `num_cards` | Count of distinct card printings per species |
| `avg_price` | Mean market price across all printings |
| `price_volatility` | Std dev of log-returns across printings |

These are recomputed on each model training run so they stay current.

### 1.5 Pre-trained Embeddings (optional baseline)

Max Woolf's nomic embeddings on HuggingFace (`minimaxir/pokemon-embeddings`)
provide a 768-dim vector per species derived from text descriptions.
Useful as a sanity-check baseline or as an auxiliary input, but too high-dimensional
to use directly. PCA down to 16-32 dims if incorporated.

---

## 2. Architecture: Entity Embeddings

### 2.1 Categorical Features -> Learned Embedding Layers

| Feature | Cardinality | Embedding dim |
|---|---|---|
| Type (primary) | 18 | 8 |
| Type (secondary) | 18 + 1 (none) | 8 |
| Generation | 9 | 4 |
| Evolution stage | 3 (basic/stage1/stage2) | 2 |
| Legendary class | 4 (normal/legendary/mythical/ultra_beast) | 2 |
| Egg group (primary) | 15 + 1 (none) | 4 |
| Smogon tier | 9 | 4 |

Total categorical embedding width: **32 dims**

### 2.2 Numerical Features -> Batch Normalization

All numerical inputs pass through a BatchNorm1d layer before concatenation:

- 6 base stats (HP, Atk, Def, SpA, SpD, Spe)
- BST (base stat total)
- Capture rate
- Smogon usage %
- Popularity rank (normalized)
- num_cards
- avg_price (log-transformed)
- price_volatility

Total numerical width: **13 dims**

### 2.3 Combined Model

```
[categorical embeddings (32d)] + [batch-normed numericals (13d)]
    -> Linear(45, 64) -> ReLU -> Dropout(0.2)
    -> Linear(64, 32) -> ReLU
    -> species_embedding (32d)
    -> Linear(32, 1) -> price prediction head
```

The 32-dim bottleneck layer is the species embedding. It is learned jointly
with the price prediction objective (log market price).

### 2.4 Training Details

- Loss: MSE on log(price)
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Batch size: 512
- Validation: 20% held-out cards, stratified by species
- Early stopping on val loss, patience=10

---

## 3. Optional: Contrastive Refinement

After the supervised embedding is trained, optionally refine with a
contrastive objective:

- **Positive pairs**: species whose card price time-series have Pearson r > 0.7
- **Negative pairs**: species with r < 0.1
- **Loss**: InfoNCE / NT-Xent on the 32-dim embeddings
- **Goal**: species with correlated market movements should be close in latent space

This step is secondary. Only pursue if the supervised embeddings show poor
clustering in UMAP visualization.

---

## 4. Schema Additions

### 4.1 `dim_pokemon_features`

```sql
CREATE TABLE dim_pokemon_features (
    pokemon_id        INTEGER PRIMARY KEY,  -- national dex number
    name              TEXT NOT NULL,
    type1             TEXT NOT NULL,
    type2             TEXT,                  -- NULL if mono-type
    generation        SMALLINT NOT NULL,
    evo_stage         SMALLINT NOT NULL,     -- 0=basic, 1=stage1, 2=stage2
    bst               SMALLINT NOT NULL,     -- base stat total
    hp                SMALLINT NOT NULL,
    attack            SMALLINT NOT NULL,
    defense           SMALLINT NOT NULL,
    sp_attack         SMALLINT NOT NULL,
    sp_defense        SMALLINT NOT NULL,
    speed             SMALLINT NOT NULL,
    capture_rate      SMALLINT NOT NULL,
    is_legendary      BOOLEAN NOT NULL DEFAULT FALSE,
    is_mythical       BOOLEAN NOT NULL DEFAULT FALSE,
    egg_group1        TEXT,
    egg_group2        TEXT,
    smogon_tier       TEXT,
    smogon_usage      REAL,                  -- 0.0-1.0
    popularity_rank   SMALLINT,              -- 1-890 from POTY vote
    num_cards         INTEGER,               -- computed from our data
    updated_at        TIMESTAMP NOT NULL DEFAULT NOW()
);
```

### 4.2 `pokemon_embeddings`

```sql
CREATE TABLE pokemon_embeddings (
    pokemon_id        INTEGER NOT NULL REFERENCES dim_pokemon_features(pokemon_id),
    model_version     TEXT NOT NULL,         -- e.g. 'v1-supervised'
    embedding         REAL[] NOT NULL,       -- 32-element float array
    created_at        TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (pokemon_id, model_version)
);
```

---

## 5. Build Order

### Step 1: Pull PokeAPI metadata (~1025 species)

- Write `cardprice/ingest/pokeapi.py`
- Fetch all species + pokemon endpoints, cache raw JSON to `data/pokeapi/`
- Parse into `dim_pokemon_features` rows
- Estimated time: 1-2 hours (API calls + parsing)

### Step 2: Scrape Smogon usage stats

- Write `cardprice/ingest/smogon.py`
- Download latest month's `chaos/*.json` files
- Parse tier + usage % per species
- Update `smogon_tier` and `smogon_usage` columns in `dim_pokemon_features`
- Estimated time: 1 hour

### Step 3: Compute market-derived features

- Query `fact_prices` + `dim_cards` for per-species aggregates
- Fill `num_cards`, and compute `avg_price` / `price_volatility` at training time
- Estimated time: 30 min (SQL + light Python)

### Step 4: Build entity embedding model

- **4a**: Baseline with sklearn/XGBoost on hand-crafted features (no embeddings).
  Establishes a performance floor.
- **4b**: PyTorch entity embedding model as described in Section 2.
  Train on card-level price data with species features as input.
- **4c**: Extract 32-dim embeddings from the bottleneck layer, write to
  `pokemon_embeddings` table.
- Estimated time: 3-4 hours

### Step 5: Validate embeddings

- UMAP 2D projection, colored by type / generation / legendary status
- Nearest-neighbor sanity checks (e.g., Pikachu near Raichu, starters clustered)
- Compare XGBoost baseline vs embedding model on held-out price prediction
- Estimated time: 1-2 hours

---

## 6. Dependencies

- Python: `requests`, `torch`, `scikit-learn`, `xgboost`, `umap-learn`, `matplotlib`
- Data: Phase 1/2 must be complete (card catalog + historical prices loaded)
- Compute: CPU-only is fine; the model is small (~50k parameters)

## 7. Success Criteria

- `dim_pokemon_features` populated for 1000+ species
- Embedding model val MSE on log(price) beats XGBoost baseline by >= 5%
- UMAP visualization shows meaningful clusters (types, legendaries, generations separate)
- Nearest-neighbor queries return intuitively similar Pokemon
