# set_population — Pokémon set print-population estimation

> **CURRENT STATE (v3.1, 2026-07-22):** the model described below evolved.
> Estimates are now ENGLISH-ONLY projected lifetime production:
> `rel_pop = mean_chase_psa x pull_D / predicted_grading_rate`, with the
> grading-rate model (`scripts/fit_grading_rate.py`) fit against a 16-rung
> dated TPC checkpoint ladder converted global->English via a documented
> `english_share` layer, unit-corrected community anchors, and an SEC-revenue
> WOTC window. See `docs/methodology.md` (v3 addendum), `docs/results.md`,
> and `docs/anchor_research_2026-07-21.md`. Rows carry reliability flags
> (`numerator_unreliable`, `subset_set`, `pop_lag_underestimate`,
> `icon_premium_suspect`); flagged absolutes are suppressed. The v1
> popularity-divisor text below is historical.

> **This is a self-contained analysis sub-project, isolated from the rest of
> the cardprice repo.** It does NOT touch the card scanner, the pricing
> pipeline, the server, or the ML code. Nothing here is imported by the main
> application. It reads the shared `dim_sets` table and `data/tcgplayer_sales.db`
> read-only; everything it produces lives under `set_population/`.
>
> If you're working on the scanner / pricing / server, you can ignore this
> directory entirely.

## Goal

Estimate the **relative** print population of every Pokémon set — i.e. roughly
how many copies of each set were printed, *relative to each other*. Absolute
numbers are a stretch goal with wide error bands; the primary deliverable is a
defensible ranking and ratio between sets.

## Method (summary)

Graded-card population reports (PSA/CGC/BGS) of each set's top chase cards are
the anchor signal. But grading *rate* varies by set — popular sets get graded
far more — so raw graded pop overstates popular sets. We correct for that with
an **exogenous popularity index**:

```
relative_population(set)  ∝  mean_chase_graded_pop(set) / popularity(set)
```

where `popularity(set)` is a composite of demand proxies that are independent
of how many copies were printed.

See `docs/methodology.md` for the full derivation and `docs/prior_art.md` for
how others have estimated Pokémon print runs (so we don't reinvent the wheel).

## Components

| Signal | Role | Source | Output |
|---|---|---|---|
| Google Trends | popularity divisor (exogenous) | pytrends | `data/set_trends.json` |
| Wikipedia/Bulbapedia pageviews | popularity divisor (exogenous) | public pageview API | `data/set_pageviews.json` |
| TCGPlayer sales velocity | validation / endogenous cross-check | `data/tcgplayer_sales.db` | `data/set_sales_velocity.json` |
| Known print-run anchors | calibration + sensitivity analysis | research (published estimates) | `data/known_print_runs.json` |
| Chase-card graded pop | the core anchor signal | PSA pop reports | `data/chase_graded_pop.json` (future) |

## Status

Bootstrapping. Each signal is collected by an independent script under
`scripts/`, writing a JSON under `data/`. A final `combine.py` (future) joins
them into `data/set_population_estimates.json` with low/mid/high bands.

## Conventions

- Every script is standalone, argparse-driven, dry-run-friendly, and writes a
  single JSON under `set_population/data/`.
- Sets are keyed by the `set_id` used in the main `dim_sets` table (e.g.
  `base1`, `neo1`, `sv4`). Each output JSON maps `set_id -> {...}`.
- No script writes outside `set_population/`. No script mutates shared state.
- All shared-data access (`dim_sets`, `tcgplayer_sales.db`) is read-only.
