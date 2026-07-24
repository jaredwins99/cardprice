# Card pricing / valuation models — a survey + what we've tried

Two parts: (1) **models people have used** for card pricing/grading historically
(the external landscape), and (2) **models tried in THIS project** with results.

---

## Part 1 — What people have used before

### A. Price prediction / valuation

**Academic (note: NO peer-reviewed hedonic/factor study exists specific to
Pokémon/TCG — the rigorous analogs are coins & stamps, §C):**
- **Pawlicki, Polin & Zhang (2014, Stanford CS229)** — MTG. **Regularized
  logistic regression (L1/L2) + SVM as binary "buy/don't-buy"
  classification**, not regression. 28 features (lagged prices, price diffs,
  tournament-usage share, sales-volume diffs, mana cost, days-to-rotation,
  price variance), 13.6k Rare/Mythic samples. **LR: 11% test error, 93% of
  max profit.** The most fully documented academic card forecaster.
- **Sakaji et al. (2019, Springer AISC 993)** — the one academic *TCG-pricing*
  paper; supervised regression on card price level, novelty = **card
  rules/effect text as an NLP feature**. Quant results paywalled.
- **Wood (2024, IJSRM)** — **CNN over card image+text** (image-based hedonic
  ML) on >1.2M Pokémon auction records; "more accurate than traditional
  hedonic models, less precise than expert estimates."
- **CalState NBA-card thesis / ACM 2024** — FNN+MLP+LSTM+SVM ensemble +
  **RoBERTa news sentiment**; **>91% on price-*trend* (direction)**, weak on
  extreme/limited cards.

**Documented hobby regression projects (concrete metrics — note the ceiling):**
- Jaffe (MTG): LassoCV + degree-2 poly. **R² 0.296, MAE 1.50.**
- Nason (MTG): GBM on log price, **test R² 0.626**; a Decision Tree beat the
  ensembles on small data.
- "Cokorie" (TCG marketplace): Random Forest, **MAE $2.85**, big errors on
  hype cards.
- Davidson (MTG): Keras LSTM on daily returns — "real power on clean series,
  fails under noise."

**Named tools / price-tracker methodologies (almost all are aggregation, NOT
prediction):**
- **Didier Lopes' Pokémon estimator (Claude Code)** — a **lookup**, not a
  model: Claude-vision card ID → scrape Pokellector aggregate. No modeling.
- **TCGPlayer "Market Price"** — **time-weighted rolling average of completed
  checkouts** (not listings), per condition & treatment, outlier-filtered.
  Proprietary weights.
- **PriceCharting** — proprietary algo over eBay + own *sold* data: recent
  sale, median, **age-weighted average**, single-outlier drop (~5× sale
  excluded), LLM junk-listing filter, separate raw/graded tracks.
- **Card Ladder (sports)** — most rigorous documented: per-card = "last sold"
  (avg on most recent sale day); index = Σ(last-sold) ÷ count nightly, a
  **price-weighted Dow-Jones-style construction**; CL50 = "S&P 500 for
  cards"; between-sales estimate = last sale × relevant index % change.
- **Market Movers / Collectr / 130Point** — sold-comp aggregators; no
  disclosed formula (130Point's edge is **recovering hidden accepted-Best-
  Offer prices** eBay masks).

**Simple baselines practitioners actually use:**
- **Comps averaging** — **median of last 5–10 sold**, 30–90d window, manual
  outlier drop. No formal statistical trim standard in the hobby.
- **Population-scarcity scoring** — **Gem Rate = PSA10 ÷ total × 100**;
  non-linear pop→price multiplier (pop 1–10 → 5–20× over PSA9; 1000+ →
  1.1–1.3×). Only pays paired with demand.
- **Pull-rate EV math** — sealed EV = Σ(price × pull rate); modern boxes
  usually negative-EV at MSRP.

### B. Condition grading / computer vision

- **Per-component transfer-learning CNNs** are the convergent recipe (NOT one
  end-to-end grader): **DenseNet201** corners (83%, +confidence calibration &
  human-in-loop), **ResNet50** edges (93%), VGG/InceptionV3 corners
  (DeepCornerNet 78%). Academic cluster = Griffith U + U Newcastle.
- **Commercial**: Ximilar (one model per dimension, geomean aggregation;
  centering = pixel-margin ratios), TAG (**photometric stereo** surface-normal
  depth maps), AGS (**3D laser depth**), PSA/Genamint (CNN+RL+RandomForest+
  OpenCV, patent = N×N tile segmentation per subgrade). Consumer pre-graders
  claim ~70–94% within ±0.5 grade. **Surface micro-defects invisible in
  photos are the dominant error source** — why flat-photo tools cap at ±0.5
  and the leaders use specialized hardware.
- **Card identification** — field has **converged on embedding + approximate-
  nearest-neighbor, explicitly rejecting flat classification at 20k+
  classes**: PSA/Collectors documents pHash→SIFT/ORB→CNN-embeddings+ANN;
  Ximilar CNN similarity ("97%+ exact-match") + a CARD-VLM fallback; academic
  ArcFace metric learning (512-dim, cosine). OSS is stuck on CLIP+FAISS+OCR;
  **no mature DINOv2 card-ID repo exists** — our scanner is ahead of open
  source here.

### C. Economics/finance framing that transfers

- **Hedonic regression** — `ln(Price)=α+Σβ·X+Σγ·Time+ε`; Rosen (1974, JPE) is
  the seminal theory. Applied to **art** (Renneboog & Spaenjers 2013, >1M
  sales; Garay 2022 **time-varying attribute premiums** ≈ hype), **wine**
  (Ashenfelter's Bordeaux equation: a few *right* covariates → R² 0.83), and
  most relevantly **coins** — *"Returns from rare coins: a machine learning
  approach"* (J. Cultural Economics 2025): **hedonic + cross-fit LASSO for
  feature selection; grade/mint/age dominate, 87.2% of price variance
  explained.** This is the closest structural analog to graded cards.
  General finding: **RF/GBM beat OLS hedonic**, most when attributes are poor
  substitutes.
- **Repeat-sales (Case-Shiller) indices** — use only items sold ≥2×, so
  differencing cancels unobserved quality: Bailey-Muth-Nourse (1963) →
  Case-Shiller **weighted repeat-sales** (down-weight long holds) →
  Sotheby's **Mei-Moses** art index. **Cards are an unusually CLEAN fit** —
  a specific PSA cert / SKU resells frequently, making "same asset" exact
  (cleaner than art). Card Ladder is the practitioner instance.
- **Factor models / selection bias** — Dimson & Spaenjers (2014) treat
  collectibles as an asset class (an "emotional dividend" lowers financial
  return). **Korteweg, Kräussl & Verwijmeren (2016, RFS)** is the key
  warning: correcting for the fact that only items that *chose* to sell
  appear in an index **cuts art returns 8.7%→6.3% and Sharpe 0.27→0.11.**
  Any card returns claim needs this selection-bias + transaction-cost
  correction or it overstates performance.

---

## Part 2 — Models tried in THIS project

### Pricing / valuation (the pricing_model subproject)

| Model | Type | Task | Result |
|---|---|---|---|
| **Fair-value GBM** (`train_fair_value.py`) | LightGBM on log(LP-adjusted price), 5-fold CV **grouped by set** (out-of-set OOF) | Cross-sectional fair value per (card, printing) | **MALE 0.578, 69% within 2×**, n=16,304. Honest finding: it's a *ranking* signal, not a price forecaster (see below). |
| Stale baseline (`cardprice/models/price_predictor.py`) | sklearn GradientBoostingRegressor, huber loss, un-grouped KFold | Latest market_price | Pre-existing, mixed printings, leaky CV — superseded. |
| **Factor backtest** (`backtest_factors.py`) | Cross-sectional quintile long-short: **value** (OOF residual), **momentum** (trailing 3m return), **size** (log price); monthly | Do characteristics predict forward returns? | **Value +8%/qtr, 23/23 months positive** (skip-month robust). Momentum/size ≈ 0. But adversarial audit → only ~25-35% survives as tradable (mark-vs-trade). |
| **Characteristic factors** (`backtest_characteristics.py`) | **Hedged long-short via cell-demeaning** within era×rarity×printing — long the trait, short a matched basket | Artist / species / era factor returns | Artist is real: **Asako Ito +7.4%/mo hedged (t=6.2)**; Nishida +0.36%/mo (t=2.5). Species dispersion large. Era/printing/rarity ≈ 0 by construction (null check passes). |
| **Temporal out-of-sample test** | Model-FV MALE vs random-walk MALE at forward horizons; residual-persistence regression | Is FV a forecaster? | **No.** Random walk beats FV ~2× at predicting future price; residuals **~93% permanent** (b≈0.93). FV ranks within comps, doesn't forecast levels. |
| **Feature attribution** (`explain_fair_values.py`) | LightGBM `pred_contrib` (SHAP-style) per row | Explain each FV | Top drivers: species-LOO price, PSA pop, rarity, set age, sibling-Normal anchor. Powers the alert explanations. |

### Supply-side / set-population (the set_population subproject)

| Model | Type | Task | Result |
|---|---|---|---|
| **Grading-rate model** (`fit_grading_rate.py`) | **Mechanistic** log-linear: `log(grading_rate) = α[era] + βₚ·log(chase_value) + β_y·log(age)`, only era intercepts fitted (L-BFGS-B), βₚ pinned; calibrated to a dated TPC cumulative-checkpoint ladder | Relative & absolute set print population | v3 english-only. Within-era *relative* rankings are the trustworthy output; absolutes are ±3× order-of-magnitude. rel_pop → the pricing model's supply feature. |
| v1 popularity divisor (superseded) | `exp(z(log(Google Trends)))` as the demand divisor | Same | 1/10 anchors within 2× — blind to grading dynamics; replaced. |
| **Likeability Elo** (`ratings.json`) | **Elo / pairwise-comparison** rating over species | Personal species-demand score | Sparse (max 3 votes/species); used as a weak feature. Elo is the Bradley-Terry family. |

### Condition / label modeling

- **LP/NM ratio model** — pooled + card-specific Lightly-Played-to-Near-Mint
  ratios (clamped [0.45, 1.0]), used to build the LP-adjusted training
  label so the model prices the played-copy market, not the NM ceiling.

### Market-structure / time-series analysis (not price models, but stats)

- **Set/era rotation lead-lag** — cross-era monthly-return correlation
  matrices (raw + market-adjusted), release-adjacency event study.
  Finding: momentum/persistence within an old-vintage block, no clean
  baton-pass; adjacency adds nothing beyond era membership.
- **Adversarial audits** — trade-anchored re-run, mark-freshness partition,
  survivorship audit (all in `audit_backtest_claims.py`); the discipline
  that demoted the value factor from "+8%/qtr" to "~2-3%/qtr harvestable".

### Scanner-side ML (prior work, for identification not pricing — from memory)

DINOv2 embeddings + nearest-neighbor (the working matcher), RapidOCR + fuzzy
name matching, and a graveyard of **confirmed-useless** methods: CLIP name
matching (19%), perceptual hashing (0%), template matching (0%), set-symbol
matching (regresses accuracy), card-number OCR, DINOv2 global search (15% on
20k). See `ml-learnings.md` in memory.

---

## Part 3 — What the credible approaches converge on, and how we map to them

1. **Production pricing is aggregation, not forecasting.** Every credible
   commercial number (TCGPlayer, PriceCharting, Card Ladder) is a
   recency/age-weighted median/average of *sold* comps, per grade, outlier-
   filtered. → **We independently rediscovered this**: our fair-value GBM is
   a ranking tool, not a forecaster (random walk beats it), and the terminal
   check that actually works is `true_floors.py` + the sales-ladder comps.
2. **Attribute valuation = log-linear hedonic + regularized (LASSO) feature
   selection; tree ensembles (GBM/RF) beat OLS.** The coin paper (grade-driven,
   87% variance) is the template. → **Our LightGBM-on-log-price IS a
   gradient-boosted hedonic model.** MALE 0.578 / 69%-within-2× sits squarely
   in the documented hobby band (R² ~0.3–0.63). Two universal facts we also
   hit: **direction is far easier than level** (Stanford MTG framed it as
   buy/don't-buy classification — exactly our alert framing), and **every
   model regresses to the mean and fails on rare/hype cards** (our model
   over-prices famous species ~5×).
3. **Market trend = weighted repeat-sales (Case-Shiller) index; cards fit it
   cleaner than art.** → **We have NOT built a true WRS index** — our
   set/era return indices in the rotation study are median-of-normalized-
   prices, closer to Card Ladder's construction. A real repeat-sales index
   (same tcg_product_id+condition sold twice) is a documented upgrade.
4. **Returns claims need selection-bias + transaction-cost correction
   (Korteweg cut art Sharpe 0.27→0.11).** → **We did exactly this**: the
   adversarial audits + observation-conditioning demoted the value factor
   from +8%/qtr to ~2–3% harvestable — the same discipline the literature
   demands.
5. **Grading CV = per-component transfer-learning CNNs + calibration + HITL;
   card ID = embedding+ANN.** → Our scanner already uses embedding+ANN
   (DINOv2), which the field converged on and OSS hasn't matched. We do **no**
   grading CV (out of scope).

### What we haven't tried but the survey says we should
- **A true weighted-repeat-sales index** per (product, condition) — the
  credible way to state "the vintage market is up X%" without the
  survivorship bias our median-index carries.
- **A promo / print-scale feature** — the model's ~5× over-pricing of cheap
  promos of famous species is the "regresses to mean, fails on rare/hype"
  failure the literature names; a print-scale or is-promo covariate is the
  fix, mirroring how the coin model leans on grade.
- **LASSO feature selection over high-cardinality categoricals** (set, artist,
  species) as a hedonic baseline to sanity-check the GBM — the coin paper's
  exact recipe. **CatBoost** (native categorical handling) is the untested
  ensemble.
- **Direction/trend classification as a first-class model** — the academic
  consensus is it's much more tractable than level (88–93% vs R² 0.3–0.63),
  and it's literally what the alert system wants ("will this be worth more").
  The direction model we scoped but didn't build.

## Honest bottom line (this project)

The **value factor is real but mostly not harvestable** at these frictions;
the **fair-value model ranks within comparable groups but does not forecast
price levels** (residuals are ~93% permanent) and **over-prices its own
extreme picks ~5× even against live listings** (promos, Rare Ultra, famous
species with no print-scale feature). The genuinely useful outputs are: the
**supply-side rel_pop**, the **hedged artist/species factors**, and the
**verification tooling** (`true_floors.py`, sales-ladder checks) that stops
frozen quotes and foreign-print listings from masquerading as bargains. The
strongest tradable signal found is not a price model at all — it's the
**artist/species characteristic factor** matched to the owner's domain
knowledge.
