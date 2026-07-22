# Methodology: Set Print-Population Estimation

> This document is the authoritative spec for `scripts/combine.py` (v2 model).
> Every magic constant in the model is listed here and flagged as an estimate.
> The model is **defensible on relatives, order-of-magnitude on absolutes** —
> see the "Honesty" section at the bottom and `results.md` for the realised
> numbers.

Read `prior_art.md` first. The core idea (method #5 from prior_art, with the
grading-rate bias modelled out) is:

```
relative_population(set)  ∝  numerator(set) × pull_denominator(set)
                              / predicted_grading_rate(set)
```

where the divisor is now a **mechanistic grading-rate model** (§3), not the v1
Google-Trends popularity composite. The three factors are built independently
below; §3 replaces v1's popularity divisor entirely. Trends, pageviews and
sales velocity move to validation-only (§6).

---

## 1. Numerator — chase-card graded population

Source: `data/chase_graded_pop.json` (PSA pop mirror). Per set we have up to 3
"chase" cards, each with a `psa_total` and a `match_confidence` in
`{high, med, low}`.

**Rule.** Keep only chase cards with `match_confidence ∈ {high, med}` **and**
`psa_total` not null. The numerator is the **mean** `psa_total` over the kept
cards. We use the mean (not sum) so that sets backed by 1 vs 3 chase cards are
on the same scale (a set's chase cards are all roughly the apex of that set, so
their mean is a stable "apex graded pop" statistic; summing would penalise sets
where we only resolved 1 card).

- `n_chase_used` records how many chase cards backed each set.
- Sets with `n_chase_used == 0` get **no estimate** (dropped from output's
  scored sets, listed under `flags`).
- `low`-confidence matches are excluded because a wrong card→pop match poisons
  the numerator far worse than a missing one (same logic as the anchor warning).

**Why PSA only.** CGC/BGS totals are sparse in the source and PSA dominates the
graded market; mixing partial CGC/BGS would add noise, not signal. Flagged as a
simplification.

---

## 2. Pull-rate denominator — "packs opened per graded chase copy"

A chase holo is one observable card sitting on top of a much larger printed
population. The denominator `D` converts "graded copies of the chase card" into
"copies of the set printed" terms by accounting for **how rare the chase pull
is**. A 1-in-3-rare-slot WOTC holo represents far fewer packs-opened-per-copy
than a 1-in-200-packs modern secret/alt-art.

`D` = (approx. packs that must be opened to yield one copy of the chase card) ×
(a within-era normaliser). It is the multiplicative bridge that makes a vintage
holo pop and a modern secret-rare pop comparable in "packs opened" units.

We bucket by **era** (from `dim_sets.release_date`) × **chase rarity tier**
(the `tier` from `chase_cards.json`, 0–5; see `select_chase_cards.py`).

### Era buckets (by release_date)

| Era    | Start (inclusive) | Examples            |
|--------|-------------------|---------------------|
| WOTC   | 1999-01-01        | base, gym, neo      |
| ECARD  | 2002-09-01        | ecard1-3            |
| EX     | 2003-06-01        | ex1-ex16            |
| DP     | 2007-04-01        | dp, pl              |
| HGSS   | 2010-02-01        | hgss, col           |
| BW     | 2011-04-01        | bw, dv              |
| XY     | 2013-10-01        | xy, g1, dc          |
| SM     | 2017-02-01        | sm, sma             |
| SWSH   | 2020-02-01        | swsh, cel, pgo      |
| SV     | 2023-03-01        | sv, sv1+            |

Bucketing is by date thresholds, so any set_id maps cleanly even if its prefix
is unusual.

### Pull-rate / packs-per-chase-copy table (ALL VALUES ARE ESTIMATES)

`D[era][tier]`. Higher = the chase card is rarer per pack ⇒ each graded copy
implies more packs opened ⇒ bigger upward correction to the set population.
These are anchored to community pull-rate data (see `prior_art.md` #3) and are
**order-of-magnitude priors, not measurements**. Every cell is flagged
`pull_rate_estimate` in the output.

Baseline reasoning:
- **WOTC holo (tier 3)**: holo sits in the rare slot ~1-in-3 packs, ~16 holos
  in Base ⇒ ~1-in-48 packs per specific holo. Normalised to `D=1.0` (the unit).
- Rarer tiers within an era scale `D` up roughly with the inverse pull rate.
- Modern apex tiers (secret/alt/hyper, tier 5) run ~1:100–1:1000 packs ⇒ large
  `D`. We cap the spread at ~3 orders of magnitude to avoid runaway estimates
  from a single noisy pop count.

The literal table (packs-per-copy, relative units, WOTC tier-3 holo = 1.0):

```
                tier0  tier1  tier2  tier3   tier4    tier5
WOTC            1.0    1.0    1.5    1.0     3.0      6.0
ECARD           1.0    1.0    1.5    1.1     3.5      7.0
EX              1.0    1.0    1.5    1.2     4.0      9.0
DP              1.0    1.0    1.5    1.2     4.5     11.0
HGSS            1.0    1.0    1.5    1.2     5.0     13.0
BW              1.0    1.0    1.5    1.3     5.5     15.0
XY              1.0    1.0    1.5    1.3     6.0     18.0
SM              1.0    1.0    1.5    1.4     7.0     22.0
SWSH            1.0    1.0    1.5    1.5     8.0     28.0
SV              1.0    1.0    1.5    1.6     9.0     35.0
```

Interpretation: a tier-5 SV chase (e.g. a Special Illustration Rare at ~1:130
packs) implies ~35× more packs opened per graded copy than a WOTC holo.

**Why ~35× and not ~200× (the raw pull-rate inverse).** Two reasons, both
documented in `prior_art.md`: (a) prior art puts modern sets at "10–100× larger
per set" than vintage, not 1000×, and ~60% of all cards were printed since
FY2020-21 — an early draft with tier-5 = 200 made the modern era soak up ~80% of
the calibrated 75 B and pushed the WOTC anchors ~1000× too low; (b) graded
**rate** rises steeply for rarer/more-valuable cards (people grade the chase
harder), so the raw pull-rate inverse double-counts a value-driven grading
inflation that the popularity divisor (§3) is *supposed* to remove. The
denominators are therefore deliberately conservative. **This table is the single
biggest lever on cross-era results and is entirely estimates** — flagged
`pull_rate_estimate` in code and called out in `results.md`.

`tier_mixed` sets (top-N spans >1 tier) use the **max tier** present and carry a
`tier_mixed` flag; their denominator is slightly less reliable.

---

## 3. Grading-rate divisor — mechanistic model (v2)

Goal: divide out the *grading-rate* bias, i.e. how big a fraction of each set's
printed cards actually end up in PSA's slabs. This is what v1's popularity
divisor was *trying* to capture, but Google Trends is blind to the dominant
dynamic: **grading rate is super-linear in chase-card value and varies by
era**. A $3 K vintage holo gets graded ~10-20× harder than a $100 modern SIR
regardless of who's searching for the set.

### 3.1 The model

We model the per-set grading rate (graded chase copies per printed card) as

```
log( grading_rate(set) ) = alpha[era]
                        + beta_p * log( chase_value(set) / $100 )
                        + beta_y * log( years_since_release(set) / 10y )
```

with `chase_value = mean(chase prices)` from `data/chase_cards.json`,
`years_since_release` from `dim_sets.release_date`, and `era` from the same
buckets as §2.

### 3.2 Fitted coefficients (commit `data/grading_rate_model.json`)

Both slopes are **pinned to priors** (not fit) because of severe data sparsity
— see §3.3:

| coefficient | value | source |
|-------------|------:|--------|
| `beta_p`    | 0.5   | Prior: sqrt scaling of grading rate in chase value. Consistent with PSA value-driven grading dynamics; conservative. |
| `beta_y`    | 0.0   | Prior: age effect absorbed into era intercept (eras already map to date windows). Kept as a parameter for future refinement. |

Per-era intercepts `alpha[era]` (the actual fitted values; log grading rate
at chase_value = $100):

| era    | alpha   | baseline rate |
|--------|--------:|--------------:|
| WOTC   | −12.421 | 4.0e-06 |
| ECARD  | −11.520 | 9.9e-06 |
| EX     | −10.700 | 2.3e-05 |
| DP     | −10.092 | 4.1e-05 |
| HGSS   |  −9.575 | 6.9e-05 |
| BW     |  −9.103 | 1.1e-04 |
| XY     |  −8.667 | 1.7e-04 |
| SM     |  −8.526 | 2.0e-04 |
| SWSH   |  −8.469 | 2.1e-04 |
| SV     |  −8.592 | 1.9e-04 |

Median predicted rate at *actual* chase values (across each era's sets) is
slightly different because real chase values vary; see `data/grading_rate_model.json`
`per_era.median_predicted_grading_rate` for the per-era median rates.

### 3.3 How alphas are fit

`scripts/fit_grading_rate.py` minimises (via L-BFGS-B):

```
L = sum_a w_a * ( log_rate_predicted(a) - log_rate_observed(a) )^2     # anchor terms
  + W_TPC * sum_cp ( log(sum_predicted_pop_in_window(cp)) - log(cp.value) )^2
  + W_SMOOTH * sum_i ( alpha[i+1] - alpha[i] )^2                       # era smoothness
  + W_RIDGE * sum_i alpha[i]^2                                          # mild ridge
```

with weights `w_a ∈ {official=4.0, well-sourced-estimate=2.0,
hobbyist-guess=0.5}` per anchor credibility, `W_TPC = 8.0`,
`W_SMOOTH = 0.5`, `W_RIDGE = 0.001`. The TPC checkpoints (23.6 B 2017 → 75 B
2025) provide the cross-era global constraints; per-set anchors constrain
their specific era's alpha.

**Per-set anchor exclusions** (see `model.anchor_exclusions` in the JSON):

- `print_variant ∈ {1st_edition, shadowless}` — anchor targets a variant
  subset, but `mean_chase_psa` is reported across all variants of the set, so
  the implied rate is biased upward.
- `set_id == "neo3"` — the 12 B figure is a cumulative checkpoint context, not
  Neo Revelation's print run.

> **[SUPERSEDED by v3 — see addendum]** v3 fits **17 per-set anchors** spanning 6 eras (8 WOTC + ex7 + xy12 + sm115 + 5 SWSH + 2 SV), all hobbyist-guess tier, unit-converted from all-languages. The text below describes the v2 state.

After exclusions, **8 per-set anchors** remain (7 WOTC + 1 EX). Other eras
(ECARD, DP, HGSS, BW, XY, SM, SWSH, SV) have **zero** per-set anchors; their
intercepts come from TPC windowed sums + smoothness.

### 3.4 Why we PIN beta_p instead of fitting

With 7 WOTC anchors and 1 EX anchor, attempting to also fit `beta_p` either
overfits or returns a *negative* coefficient (more value → less grading),
which is physically wrong. Per `prompt rules` we refuse to ship a backwards
model. Pinning `beta_p = 0.5` is a conservative documented prior; the run
script raises if `beta_p <= 0`. The model is fully reproducible from the JSON.

### 3.5 What replaces what

| v1                                  | v2                                              |
|-------------------------------------|-------------------------------------------------|
| Google Trends z-score → exp() composite divisor | `predicted_grading_rate` from the mechanistic model |
| Trends had weight 1.0 in divisor    | Trends/pageviews/sales: VALIDATION only (§6)    |
| `popularity_imputed` flag           | replaced by `no_grading_rate` (only if chase value or PSA pop missing) |

---

## 4. Relative population (primary deliverable)

```
rel_pop_score(set) = numerator(set) × pull_denominator(set)
                     / predicted_grading_rate(set)
```

This is a unitless cross-set score. Ratios between sets are the meaningful
output (set A's `rel_pop` 3× set B's ⇒ ~3× the estimated print population). We
additionally report it normalised so the max set = 100 for readability.

Note that `numerator × pull_denominator / predicted_grading_rate` is precisely
the un-scaled implied print run from the anchor equation — the post-fit scale
to TPC 75 B is therefore close to 1.0 (currently ~1.08). **[SUPERSEDED: v3.1 uses a credibility-weighted geometric-mean scale over all usable rungs — currently 1.019 — instead of an exact pin; see addendum.]**

---

## 5. Absolute calibration (stretch — wide bands)

We pin the **windowed sum** of `rel_pop` to an official TPC cumulative
checkpoint (`known_print_runs.json`, `set_id == "GLOBAL"`,
`estimate_type == "total_print_run"`). Procedure:

1. Pick the latest checkpoint whose date ≤ today, with a dated `value_mid`
   (the 75 B / 64.9 B / 52.9 B / 43.2 B / 23.6 B ladder). **[SUPERSEDED: the constant `TPC_CHECKPOINTS` no longer exists; v3 loads a dated ladder from `known_print_runs.json` via `model_constants_v3.load_checkpoints`, 64.9 B was corrected to 64.8 B (archive-verified), and rungs newer than the scored roster supports are excluded from the fit.]**
2. Sum `rel_pop` over only the sets **released before that checkpoint date**.
3. `scale = checkpoint_value / windowed_rel_pop_sum`.
4. `abs_estimate_mid(set) = rel_pop(set) × scale` for every scored set.

This converts the ranking into absolute bands "for free" using the single
hardest anchor, without trusting any per-set hobbyist number.

**Uncertainty bands.** The two dominant multiplicative uncertainties are
grading rate (how big a fraction of printed copies get graded — varies maybe
3×) and pull rate (our `D` table — maybe 2–3× off per cell). We propagate a
combined multiplicative factor `BAND_FACTOR = 3.0`:

```
abs_low(set)  = abs_estimate_mid / BAND_FACTOR
abs_high(set) = abs_estimate_mid × BAND_FACTOR
```

These bands are honest order-of-magnitude rails, not confidence intervals.

**Anchor cross-check.** For each non-GLOBAL anchor with a per-set
`total_print_run` `value_mid`, report `estimate/anchor` ratio (sensitivity
table in `results.md`). In v2 the grading-rate model is *fit jointly* against
these per-set anchors (weighted by credibility) AND the TPC checkpoints, so
the ratios are a fit-quality measure rather than a fully held-out check.
The exclusions in §3.3 still apply: 1st-Ed/Shadowless and `neo3` anchors are
reported but excluded from the fit and from the "within Nx" tally.

---

## 6. Validation — Spearman vs held-out signals

Three signals are held out of the v2 divisor and become Spearman cross-checks:

| signal | source | role |
|--------|--------|------|
| sales velocity (`sales_per_month`) | `data/set_sales_velocity.json` (TCGPlayer) | Endogenous: more print → more supply → more sales. Expect positive but <1. |
| Google Trends interest | `data/set_trends.json` | Demand-side, exogenous to print. Expect positive: popular sets get printed in larger runs. |
| Wikipedia pageviews | `data/set_pageviews.json` | Mostly shared-mascot articles (152/171 sets); expect weak/noisy. Reported but not relied on. |

A value near 0 or negative would flag a bug; ~1.0 would mean we just re-derived
the held-out signal. We report each ρ and the biggest rank disagreements vs
sales (since sales has the most coverage). Implemented in scipy if available,
else by hand (average-rank ties + Pearson on ranks).

---

## Honesty / caveats (see results.md for the frank version)

- **Relatives**: defensible. Numerator is directly counted; the corrections
  (pull-rate, grading-rate) are transparent and monotonic.
- **Absolutes**: order-of-magnitude only. The windowed-sum calibration inherits
  every error in the `D` table and the grading-rate model, scaled to one
  official number. Bands are ±3× and even those are optimistic for under-anchored
  eras.
- **Per-era anchor coverage**: WOTC has 7 anchors → its alpha is data-driven.
  EX has 1 → its alpha is essentially that one anchor. ECARD/DP/HGSS/BW/XY/SM/SWSH/SV
  have **zero** per-set anchors; their alphas are determined by TPC checkpoint
  windowed sums + smoothness across consecutive eras. Per-set absolutes in
  those eras are therefore order-of-magnitude.
- **Pinned slopes**: `beta_p = 0.5` and `beta_y = 0.0` are priors, not fitted.
  Fitting them on the current 8 anchors flips the price-slope sign — physically
  wrong. With more anchors (especially modern), we'd un-pin and refit.
- **Known unmodelled biases**: WOTC variant splits (1st-Ed/Shadowless/Unlimited
  graded separately but pop-merged here), JP vs EN prints, attrition,
  crack-and-resubmit inflation, precon-deck dilution. All flagged in prior_art.
- Every numeric constant above lives in `combine.py` / `fit_grading_rate.py` as a
  named, commented constant and is echoed into `data/grading_rate_model.json`
  + the output JSON's `notes`/`flags`.

---

## v3 addendum (2026-07-21): English-only units + dated checkpoint ladder

v2's absolute calibration had a unit inconsistency, exposed by the 2026-07-21
deep-research pass (`docs/anchor_research_2026-07-21.md`): English-catalog PSA
pops were compared against ALL-LANGUAGES anchors and GLOBAL cumulative
checkpoints — an implicit `english_share = 1.0`. Every WOTC per-set anchor was
also provenance-traced to a single 2018 Elite Fourum post of self-described
all-languages "wild ass guesses". v3 changes (all in
`scripts/model_constants_v3.py`, `scripts/fit_grading_rate.py`,
`scripts/combine.py`):

1. **Language scope**: all outputs are ENGLISH-ONLY projected lifetime
   production. `english_share(date)` = 0.40 pre-2020 / 0.35 after (±0.10),
   evidence documented in `known_print_runs.json → conversion_evidence`.
   Absolutes scale linearly in this assumption.
2. **Checkpoint ladder**: 5 hardcoded checkpoints → 16 dated rungs (12
   official/archive-verified) loaded from `known_print_runs.json` GLOBAL
   anchors, each weighted by credibility (official 8, well-sourced 4).
3. **Anchor unit conversion**: `unit: cards_all_languages` anchors are
   converted at the set's release-date share before use, in both the fit and
   the sensitivity table (variant-subset rows excluded from the headline).
4. **Independent corroboration**: audited Hasbro SEC revenue ($500M/1999,
   $568M/2000, ~$100M/2001) ÷ wholesale pack price (40–55% × $3.29 primary-
   sourced MSRP) × 11 cards/pack × 0.65 EN-of-West → a 4.2–7.4B English
   1999–2001 window fit as an `english_window_total` anchor. The fitted model
   lands within band — two independent derivations of WOTC-era English volume
   agree.
5. **Production ramp**: a checkpoint at date T credits a set released d days
   earlier with min(1, d/ramp) of its lifetime run; ramp = 365 d for pre-2003
   sets (documented 2001 overproduction writeoffs → front-loaded printing),
   730 d after.

Known residual tensions are documented in `results.md` ("Known residual
tensions"): the 2005/2006 rungs sit ~1.5× over (crash-era conflict between
checkpoint deltas and graded-pop structure), base1 is likely inflated by the
Charizard grading premium, and BW-era estimates violate one uncorroborated
community ordinal while matching sealed-box market prices.

The v2 text above is retained for history; where it conflicts with this
addendum (e.g. the "~1.08 post-fit scale" claim, the 5-checkpoint table), the
addendum and the current scripts are authoritative.

### v3.1 (2026-07-22, post-adversarial-audit)

A 4-agent adversarial audit of the first v3 build found and led to these fixes:

1. **Share-on-increments** — applying english_share to cumulative totals made
   the EN ladder non-monotonic across the 2020 switch; now applied to
   per-regime increments with G(switch) interpolated from the ladder.
2. **Geomean calibration** — the v2-style exact pin to the latest rung
   concentrated a real slope misfit into a 1.38x scale inflating every earlier
   rung; replaced with a credibility-weighted geometric mean over usable rungs
   (now 1.019). Residuals are reported per rung (crash-era ~1.4x over, modern
   ~0.75x under — the model's main open tension).
3. **Rung capping** — rungs newer than (pop snapshot − 730 d) are excluded so
   unscored 2025-26 sets and pop-lagged recent sets cannot push their
   production onto older sets (was inflating the 2023 cohort ~2x).
4. **Orphaned anchor** — Shining Fates was tagged `swsh4pt5`; the catalog id
   is `swsh45`. Fixed; anchor tally is now n=17.
5. **Rung credibility corrections** — 13B@2005 and 43.2B@2022 are
   transcription-tier (forum/blog-relayed), downgraded from `official`.
6. **Reliability flags** — `numerator_unreliable` (non-booster/promo products:
   absolutes suppressed as floor artifacts), `subset_set` (Trainer Galleries,
   cel25c: excluded from checkpoint windows, absolutes suppressed),
   `pop_lag_underestimate` (released < 24 mo before the pop snapshot),
   `icon_premium_suspect` (base1).
7. **beta_p retest** — freely fit on the 17-anchor set: ≈0.04 (not
   sign-flipped, but guess-fitted); the 0.5 physical prior is retained.
8. Wording fixes in results.md: the SEC-revenue window is a consistency check
   between assumption-sharing derivations (its agreement partly informed the
   0.40 share choice), not independent corroboration; evidence tiers stated
   precisely; 'over X' floors noted.
