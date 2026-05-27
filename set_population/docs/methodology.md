# Methodology: Set Print-Population Estimation

> This document is the authoritative spec for `scripts/combine.py`. Every magic
> constant in the model is listed here and flagged as an estimate. The model is
> **defensible on relatives, order-of-magnitude on absolutes** — see the
> "Honesty" section at the bottom and `results.md` for the realised numbers.

Read `prior_art.md` first. The core idea (method #5 with #1's popularity bias
divided out) is:

```
relative_population(set)  ∝  numerator(set) × pull_denominator(set) / popularity(set)
```

The three factors are built independently below.

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

## 3. Popularity divisor — exogenous demand

Goal: divide out the grading-rate-vs-popularity bias. Popularity must be
**exogenous** to print count.

Inputs and weights:

| Signal                         | Weight | Rationale |
|--------------------------------|--------|-----------|
| Google Trends `interest_rescaled` | 1.0  | Primary. Search interest is demand-side, independent of how many were printed. |
| Wikipedia pageviews            | 0.0    | Near-useless here: only Base Set resolves to a real *set* article; 152/171 fall back to a shared **mascot card** article (e.g. many sets all map to "Pikachu"), so pageviews measure the mascot, not the set. Included in output for transparency, weighted 0. |
| Sales velocity                 | —      | **ENDOGENOUS** (sales volume rises with population). NOT in the divisor. Reserved for validation (§6). |

**Composite.** We log-transform Trends (`log1p`) to compress its heavy right
tail (Base=100 dwarfs the median), then z-score across sets:

```
pop_z(set)        = zscore( log1p(interest_rescaled) )      # Trends
popularity(set)   = exp( W_TRENDS * pop_z + W_PAGEVIEWS * pageview_z )
```

with `W_TRENDS = 1.0`, `W_PAGEVIEWS = 0.0`. Exponentiating the weighted z keeps
`popularity` strictly positive and multiplicative (a set 1σ above mean Trends
gets ~e× the divisor). `interest_rescaled == 0` is treated as a real (very low)
popularity, not missing.

**Imputation.** A set with `interest_rescaled == null` (Trends not yet
collected) gets the **era-median** `pop_z` and is flagged
`popularity_imputed: true`. The script is fully re-runnable: when the background
Trends fill completes, re-running picks up the new non-null values
automatically. If a set has no era peers with Trends either, it falls back to
the global-median `pop_z`.

---

## 4. Relative population (primary deliverable)

```
rel_pop_score(set) = numerator(set) × pull_denominator(set) / popularity(set)
```

This is a unitless cross-set score. Ratios between sets are the meaningful
output (set A's `rel_pop` 3× set B's ⇒ ~3× the estimated print population). We
additionally report it normalised so the max set = 100 for readability.

---

## 5. Absolute calibration (stretch — wide bands)

We pin the **windowed sum** of `rel_pop` to an official TPC cumulative
checkpoint (`known_print_runs.json`, `set_id == "GLOBAL"`,
`estimate_type == "total_print_run"`). Procedure:

1. Pick the latest checkpoint whose date ≤ today, with a dated `value_mid`
   (the 75 B / 64.9 B / 52.9 B / 43.2 B / 23.6 B ladder; checkpoint dates are
   listed in `combine.py`'s `TPC_CHECKPOINTS`).
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
table in `results.md`). We do NOT fit to these (most are `hobbyist-guess`); they
are a sanity rail per the prior-art recommendation.

---

## 6. Validation — Spearman vs sales velocity

Sales velocity (`set_sales_velocity.json`, `sales_per_month`) is endogenous, so
we held it out of the model. We compute **Spearman rank correlation** between
`rel_pop_score` and `sales_per_month` across scored sets (implemented by hand if
scipy absent). Expectation: **positive but < 1** — more printed ⇒ more supply ⇒
more sales, but popularity/age/price break the tie, so it must not be 1.0. Gross
disagreement (≈0 or negative) would flag a bug. We report the coefficient and
the biggest rank outliers with plausible explanations.

---

## Honesty / caveats (see results.md for the frank version)

- **Relatives**: defensible. Numerator is directly counted; the two corrections
  (pull-rate, popularity) are transparent and monotonic.
- **Absolutes**: order-of-magnitude only. The windowed-sum calibration inherits
  every error in the `D` table and the popularity divisor, scaled to one
  official number. Bands are ±3× and even those are optimistic for vintage.
- **Known unmodelled biases**: WOTC variant splits (1st-Ed/Shadowless/Unlimited
  graded separately but pop-merged here), JP vs EN prints, attrition,
  crack-and-resubmit inflation, precon-deck dilution. All flagged in prior_art.
- Every numeric constant above lives in `combine.py` as a named, commented
  constant and is echoed into the output JSON's `notes`/`flags`.
</content>
</invoke>
