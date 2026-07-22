# Results: Set Print-Population Estimates

_Generated 2026-07-22T04:26:10.845682+00:00 — model v3-english-only._

> **Model v3 (english-only + dated checkpoint ladder).** ALL absolute numbers are ENGLISH-ONLY projected lifetime production. rel_pop = mean_chase_psa × pull_D / predicted_grading_rate; the grading-rate fit targets a dated TPC cumulative-checkpoint ladder (6 rungs archive-verified, the current 85B live-page, the rest official-claim or transcription-tier and down-weighted accordingly) converted global→English via a documented `english_share` layer applied to per-regime INCREMENTS (0.40 pre-2020 / 0.35 after, ±0.10), unit-corrected community anchors (every per-set anchor is an all-languages community guess — provenance traced 2026-07-21), and an SEC-revenue-derived WOTC 1999–2001 English window (a consistency check that shares the assumption layer — see caveats). Checkpoint windows use a production ramp (12 mo boom-era / 24 mo after); subset products are excluded from windows. Trends / pageviews / sales velocity remain held-out VALIDATION signals (pageviews has n=1 usable pair and is not reported). See `docs/anchor_research_2026-07-21.md` for the evidence base.

- Sets scored: **153** / 170 (17 unscored, no confident chase pop or grading rate).
- Absolute calibration: credibility-weighted geometric-mean scale over **14 usable ladder rungs** (latest: 64,800,000,000 global @ 2024-03-31 → 24,160,655,738 English, increment-correct share). Rungs after (pop snapshot − 730 d) are excluded: the scored roster lacks 2025-26 sets and recent sets' graded pops lag. scale=1.019; no single rung is pinned exactly — per-rung residuals are in the ladder table below.

## Grading-rate model (v3 divisor)

`log(grading_rate(set)) = alpha[era] + beta_p * log(chase_value / $100) + beta_y * log(years_since_release / 10y)`

- `beta_p = 0.5` (pinned prior — grading rate rises with chase value; sqrt scaling).
- `beta_y = 0.0` (pinned; age absorbed into era intercept).
- Per-era intercept `alpha[era]` (log grading rate at chase_value=$100):

| era | alpha | baseline rate | median predicted rate (across sets) |
|-----|------:|--------------:|-----------------------------:|
| WOTC | -11.496 | 1.02e-05 | 2.16e-05 |
| ECARD | -10.801 | 2.04e-05 | 4.64e-05 |
| EX | -10.234 | 3.59e-05 | 6.30e-05 |
| DP | -9.801 | 5.54e-05 | 5.95e-05 |
| HGSS | -9.034 | 1.19e-04 | 1.61e-04 |
| BW | -8.187 | 2.78e-04 | 3.33e-04 |
| XY | -7.300 | 6.76e-04 | 7.63e-04 |
| SM | -6.999 | 9.13e-04 | 9.10e-04 |
| SWSH | -7.240 | 7.17e-04 | 4.30e-04 |
| SV | -7.504 | 5.51e-04 | 3.64e-04 |

## Top 20 sets by estimated print population

| # | set_id | name | era | n_chase | mean PSA | pull_D | chase $ | grading rate | rel_pop(norm100) | abs_mid | abs_low–high | flags |
|---|--------|------|-----|--------:|---------:|------:|--------:|-------------:|-----------------:|--------:|-------------|-------|
| 1 | base1 | Base | WOTC | 3 | 46,396 | 1.0 | $268 | 1.67e-05 | 100.0 | 2,839,656,517 | 946,552,172–8,518,969,552 | icon_premium_suspect |
| 2 | sv3 | Obsidian Flames | SV | 3 | 16,962 | 35.0 | $44 | 3.64e-04 | 58.4892 | 1,660,891,197 | 553,630,399–4,982,673,592 | — |
| 3 | swsh12pt5 | Crown Zenith | SWSH | 3 | 15,466 | 28.0 | $20 | 3.21e-04 | 48.4472 | 1,375,733,812 | 458,577,937–4,127,201,435 | tier_mixed,tier_fallback |
| 4 | sv3pt5 | 151 | SV | 3 | 26,009 | 35.0 | $192 | 7.63e-04 | 42.8331 | 1,216,314,163 | 405,438,054–3,648,942,488 | — |
| 5 | neo3 | Neo Revelation | WOTC | 3 | 3,986 | 6.0 | $750 | 2.79e-05 | 30.8126 | 874,973,174 | 291,657,725–2,624,919,522 | tier_mixed,tier_fallback |
| 6 | base4 | Base Set 2 | WOTC | 3 | 10,416 | 1.0 | $174 | 1.34e-05 | 27.8955 | 792,136,506 | 264,045,502–2,376,409,517 | — |
| 7 | cel25c | Celebrations: Classic Collection | SWSH | 3 | 18,739 | 28.0 | $101 | 7.21e-04 | 26.138 | n/a | n/a | subset_set |
| 8 | xy12 | Evolutions | XY | 3 | 6,062 | 18.0 | $5 | 1.54e-04 | 25.5042 | 724,232,042 | 241,410,681–2,172,696,127 | — |
| 9 | sv4pt5 | Paldean Fates | SV | 3 | 14,399 | 35.0 | $292 | 9.41e-04 | 19.2217 | 545,830,666 | 181,943,555–1,637,491,997 | — |
| 10 | swsh35 | Champion's Path | SWSH | 3 | 16,772 | 28.0 | $150 | 8.79e-04 | 19.1883 | 544,880,484 | 181,626,828–1,634,641,451 | — |
| 11 | swsh12pt5gg | Crown Zenith Galarian Gallery | SWSH | 3 | 16,354 | 28.0 | $146 | 8.67e-04 | 18.9581 | n/a | n/a | subset_set |
| 12 | neo4 | Neo Destiny | WOTC | 3 | 3,824 | 6.0 | $1871 | 4.40e-05 | 18.7158 | 531,463,452 | 177,154,484–1,594,390,357 | — |
| 13 | swsh45 | Shining Fates | SWSH | 3 | 3,322 | 28.0 | $8 | 1.98e-04 | 16.8645 | 478,892,750 | 159,630,917–1,436,678,250 | tier_mixed,tier_fallback |
| 14 | sv1 | Scarlet & Violet | SV | 3 | 4,160 | 35.0 | $35 | 3.24e-04 | 16.1116 | 457,514,065 | 152,504,688–1,372,542,195 | — |
| 15 | swsh9 | Brilliant Stars | SWSH | 3 | 6,719 | 28.0 | $35 | 4.21e-04 | 16.0297 | 455,187,997 | 151,729,332–1,365,563,991 | — |
| 16 | sm115 | Hidden Fates | SM | 3 | 10,463 | 22.0 | $41 | 5.81e-04 | 14.2123 | 403,580,613 | 134,526,871–1,210,741,838 | tier_mixed,tier_fallback |
| 17 | sv4 | Paradox Rift | SV | 3 | 3,342 | 35.0 | $34 | 3.20e-04 | 13.13 | 372,847,810 | 124,282,603–1,118,543,431 | — |
| 18 | base3 | Fossil | WOTC | 3 | 5,476 | 1.0 | $254 | 1.62e-05 | 12.1314 | 344,490,179 | 114,830,060–1,033,470,538 | — |
| 19 | base5 | Team Rocket | WOTC | 3 | 7,126 | 1.0 | $451 | 2.16e-05 | 11.8427 | 336,292,076 | 112,097,359–1,008,876,227 | tier_fallback |
| 20 | sv7 | Stellar Crown | SV | 3 | 2,661 | 35.0 | $30 | 3.03e-04 | 11.0332 | 313,304,310 | 104,434,770–939,912,930 | pop_lag_underestimate |

## Bottom 20 sets by estimated print population

| # | set_id | name | era | n_chase | mean PSA | pull_D | chase $ | grading rate | rel_pop(norm100) | abs_mid | abs_low–high | flags |
|---|--------|------|-----|--------:|---------:|------:|--------:|-------------:|-----------------:|--------:|-------------|-------|
| 1 | mcd16 | McDonald's Collection 2016 | XY | 3 | 1 | 1.0 | $33 | 3.90e-04 | 0.0001 | n/a | n/a | numerator_unreliable |
| 2 | xy0 | Kalos Starter Set | XY | 3 | 13 | 1.0 | $17 | 2.75e-04 | 0.0017 | n/a | n/a | numerator_unreliable |
| 3 | swshp | SWSH Black Star Promos | SM | 1 | 98 | 1.0 | $397 | 1.82e-03 | 0.0019 | n/a | n/a | numerator_unreliable |
| 4 | mcd15 | McDonald's Collection 2015 | XY | 3 | 24 | 1.0 | $22 | 3.14e-04 | 0.0027 | n/a | n/a | numerator_unreliable |
| 5 | mcd12 | McDonald's Collection 2012 | BW | 3 | 8 | 1.0 | $11 | 9.22e-05 | 0.0032 | n/a | n/a | numerator_unreliable |
| 6 | mcd14 | McDonald's Collection 2014 | XY | 3 | 33 | 1.0 | $20 | 2.99e-04 | 0.004 | n/a | n/a | numerator_unreliable |
| 7 | mcd17 | McDonald's Collection 2017 | SM | 3 | 36 | 1.0 | $11 | 3.01e-04 | 0.0043 | n/a | n/a | numerator_unreliable |
| 8 | mcd18 | McDonald's Collection 2018 | SM | 3 | 66 | 1.0 | $32 | 5.14e-04 | 0.0046 | n/a | n/a | numerator_unreliable |
| 9 | xyp | XY Black Star Promos | XY | 3 | 168 | 1.0 | $369 | 1.30e-03 | 0.0046 | n/a | n/a | numerator_unreliable |
| 10 | mcd19 | McDonald's Collection 2019 | SM | 3 | 65 | 1.0 | $18 | 3.91e-04 | 0.0059 | n/a | n/a | numerator_unreliable |
| 11 | pop4 | POP Series 4 | EX | 3 | 65 | 1.5 | $343 | 6.65e-05 | 0.0526 | n/a | n/a | numerator_unreliable |
| 12 | pop6 | POP Series 6 | DP | 3 | 18 | 1.5 | $9 | 1.68e-05 | 0.0587 | n/a | n/a | numerator_unreliable |
| 13 | bw11 | Legendary Treasures | XY | 2 | 104 | 18.0 | $262 | 1.09e-03 | 0.0612 | 1,737,287 | 579,096–5,211,860 | tier_mixed,tier_fallback |
| 14 | bw2 | Emerging Powers | BW | 3 | 41 | 5.5 | $17 | 1.13e-04 | 0.0715 | n/a | n/a | tier_mixed,tier_fallback,numerator_unreliable |
| 15 | mcd22 | McDonald's Collection 2022 | SWSH | 3 | 225 | 1.0 | $2 | 1.01e-04 | 0.08 | n/a | n/a | numerator_unreliable |
| 16 | dv1 | Dragon Vault | BW | 3 | 249 | 1.3 | $18 | 1.19e-04 | 0.0976 | n/a | n/a | numerator_unreliable |
| 17 | pop9 | POP Series 9 | DP | 3 | 48 | 1.5 | $17 | 2.30e-05 | 0.1115 | n/a | n/a | numerator_unreliable |
| 18 | ru1 | Pokémon Rumble | DP | 3 | 362 | 1.0 | $372 | 1.07e-04 | 0.1217 | n/a | n/a | numerator_unreliable |
| 19 | pop8 | POP Series 8 | DP | 3 | 43 | 1.5 | $10 | 1.79e-05 | 0.1284 | n/a | n/a | numerator_unreliable |
| 20 | col1 | Call of Legends | HGSS | 3 | 718 | 1.2 | $379 | 2.32e-04 | 0.1331 | n/a | n/a | numerator_unreliable |

## Anchor sensitivity (estimate / published anchor, ENGLISH units)

Per-set `total_print_run` anchors from `known_print_runs.json`, converted to English cards at the set's release-date share when tagged `cards_all_languages` (all of them are — the 2026-07-21 provenance audit traced every one to all-languages community guesses). The grading-rate model is fit *jointly* against these (weight 0.5 for hobbyist-guess) and the checkpoint ladder, so this is a fit-quality measure, not held-out. Variant-subset rows (1st Ed/Shadowless) are excluded from the headline — the estimate covers all variants of a set.

**5/17 anchors within 2×, 10/17 within 3×** (v2 counted 8/18 within 2× against UN-converted all-language anchors — not comparable; v1 had 1/10).

| set | variant | credibility | anchor raw | anchor EN | estimate_mid | est/anchor | headline |
|-----|---------|-------------|-----------:|----------:|-------------:|-----------:|----------|
| base1 (Base Set) | 1st_edition | hobbyist-guess | 1,500,000 | 600,000 | 2,839,656,517 | 4732.76× | excluded |
| swsh12pt5 (Crown Zenith) | na | hobbyist-guess | 700,000,000 | 245,000,000 | 1,375,733,812 | 5.62× | yes |
| sm115 (Hidden Fates) | na | hobbyist-guess | 200,000,000 | 80,000,000 | 403,580,613 | 5.04× | yes |
| xy12 (XY Evolutions) | na | hobbyist-guess | 400,000,000 | 160,000,000 | 724,232,042 | 4.53× | yes |
| base4 (Base Set 2) | na | hobbyist-guess | 600,000,000 | 240,000,000 | 792,136,506 | 3.30× | yes |
| swsh35 (Champion's Path) | na | hobbyist-guess | 500,000,000 | 175,000,000 | 544,880,484 | 3.11× | yes |
| base1 (Base Set) | unlimited | hobbyist-guess | 3,000,000,000 | 1,200,000,000 | 2,839,656,517 | 2.37× | yes |
| swsh45 (Shining Fates) | na | hobbyist-guess | 600,000,000 | 210,000,000 | 478,892,750 | 2.28× | yes |
| ex7 (Team Rocket Returns) | na | hobbyist-guess | 80,000,000 | 32,000,000 | 51,610,938 | 1.61× | yes |
| sv4pt5 (Paldean Fates) | na | hobbyist-guess | 1,200,000,000 | 420,000,000 | 545,830,666 | 1.30× | yes |
| cel25 (Celebrations) | na | hobbyist-guess | 400,000,000 | 140,000,000 | 150,243,644 | 1.07× | yes |
| sv3pt5 (Pokemon 151) | na | hobbyist-guess | 4,000,000,000 | 1,400,000,000 | 1,216,314,163 | 0.87× | yes |
| base5 (Team Rocket) | na | hobbyist-guess | 1,100,000,000 | 440,000,000 | 336,292,076 | 0.76× | yes |
| base3 (Fossil) | na | hobbyist-guess | 1,750,000,000 | 700,000,000 | 344,490,179 | 0.49× | yes |
| neo1 (Neo Genesis) | na | hobbyist-guess | 700,000,000 | 280,000,000 | 117,378,331 | 0.42× | yes |
| base2 (Jungle) | na | hobbyist-guess | 1,750,000,000 | 700,000,000 | 253,442,288 | 0.36× | yes |
| gym1 (Gym Heroes) | na | hobbyist-guess | 850,000,000 | 340,000,000 | 107,092,735 | 0.31× | yes |
| swsh7 (Evolving Skies) | na | hobbyist-guess | 2,000,000,000 | 700,000,000 | 210,528,657 | 0.30× | yes |

## Checkpoint-ladder fit (English targets)

| date | global | share | EN target | predicted | ratio | credibility |
|------|-------:|------:|----------:|----------:|------:|-------------|
| 2001-12-31 | 12.0B | 0.40 | 4.8B | 5.2B | 1.09 | well-sourced-estimate |
| 2005-03-31 | 13.0B | 0.40 | 5.2B | 7.5B | 1.44 | well-sourced-estimate |
| 2006-03-31 | 14.0B | 0.40 | 5.6B | 7.8B | 1.40 | well-sourced-estimate |
| 2013-11-15 | 20.0B | 0.40 | 8.0B | 9.8B | 1.23 | well-sourced-estimate |
| 2015-03-31 | 21.5B | 0.40 | 8.6B | 9.9B | 1.15 | official |
| 2017-03-31 | 23.6B | 0.40 | 9.4B | 10.3B | 1.09 | official |
| 2018-03-31 | 25.7B | 0.40 | 10.3B | 10.8B | 1.05 | official |
| 2019-03-31 | 27.2B | 0.40 | 10.9B | 11.2B | 1.03 | official |
| 2019-09-30 | 28.8B | 0.40 | 11.5B | 11.4B | 0.99 | well-sourced-estimate |
| 2020-03-31 | 30.4B | 0.35 | 12.1B | 11.6B | 0.96 | official |
| 2021-03-31 | 34.1B | 0.35 | 13.4B | 12.4B | 0.92 | official |
| 2022-03-31 | 43.2B | 0.35 | 16.6B | 13.7B | 0.82 | well-sourced-estimate |
| 2023-03-31 | 52.9B | 0.35 | 20.0B | 15.2B | 0.76 | official |
| 2024-03-31 | 64.8B | 0.35 | 24.2B | 17.9B | 0.74 | official |

Revenue-derived window **WOTC_EN_WINDOW_1999_2001** (1999-01-01→2001-12-31): predicted **5.2B** vs target 5.7B [4.2–7.4] — **WITHIN band** (ratio 0.92). Honesty note: this is a CONSISTENCY CHECK, not independent corroboration — the window is a term in the same joint fit, its prediction is the same quantity the end-2001 rung constrains, and the two target derivations (TPC totals × share vs SEC dollars ÷ wholesale price × EN-of-West share) use disjoint primary documents but SHARE the community-assumption share layer, whose 0.40 value was itself chosen partly for this agreement.

## Per-era summary

| era | n sets | median rate | median print run | sum print run |
|-----|------:|-----------:|----------------:|--------------:|
| WOTC | 13 | 2.16e-05 | 253,442,288 | 6,599,816,634 |
| ECARD | 3 | 4.64e-05 | 123,071,611 | 429,112,646 |
| EX | 24 | 6.30e-05 | 53,901,780 | 1,727,242,727 |
| DP | 16 | 5.95e-05 | 38,456,735 | 796,658,374 |
| HGSS | 5 | 1.61e-04 | 43,037,410 | 220,911,802 |
| BW | 12 | 3.33e-04 | 6,777,686 | 112,339,870 |
| XY | 20 | 7.63e-04 | 8,843,239 | 1,060,932,541 |
| SM | 21 | 9.10e-04 | 30,545,036 | 1,210,042,106 |
| SWSH | 26 | 4.30e-04 | 167,971,940 | 6,752,085,042 |
| SV | 13 | 3.64e-04 | 268,755,140 | 5,401,291,498 |

## Validation: rel_pop vs held-out signals (Spearman)

| signal | ρ | n | interpretation |
|--------|--:|--:|----------------|
| sales velocity (TCGPlayer) | 0.565 | 153 | endogenous: bigger print -> more supply -> more sales (expect +) |
| Google Trends interest | 0.596 | 153 | demand-side: popular sets get printed more (expect +, weaker than sales) |

Expectation: **positive but < 1** — more printed ⇒ more supply/demand traffic, but popularity, age and price break the tie. A value near 0 or negative would flag a bug; ~1.0 would mean we just re-derived demand.

Biggest rank disagreements vs sales (plausibly real, not bugs):

| set_id | name | rel_pop_rank | sales_rank | gap |
|--------|------|-------------:|-----------:|----:|
| sv8pt5 | Prismatic Evolutions | 111 | 10 | +101 |
| neo3 | Neo Revelation | 5 | 103 | -98 |
| bw2 | Emerging Powers | 140 | 43 | +97 |
| neo4 | Neo Destiny | 12 | 106 | -94 |
| xy5 | Primal Clash | 133 | 40 | +93 |
| cel25c | Celebrations: Classic Collection | 7 | 98 | -91 |
| ex10 | Unseen Forces | 26 | 117 | -91 |
| swshp | SWSH Black Star Promos | 151 | 61 | +90 |
| ecard2 | Aquapolis | 23 | 112 | -89 |
| bw11 | Legendary Treasures | 141 | 54 | +87 |

_Positive gap = ranks much higher in population than in sales (printed big but trades slowly — old/cheap bulk). Negative = trades hot for its print size (small but in demand)._

## Confidence & caveats (frank)

### What changed in v3 (2026-07-21/22 deep-research + adversarial audit)
1. **Units fixed.** v2 compared English graded pops against ALL-LANGUAGES anchors and global checkpoints — an implicit english_share of 1.0, smearing Japanese-only volume across English sets. v3 outputs are English-only: the share (0.40 pre-2020 / 0.35 after, ±0.10) is applied to per-regime production INCREMENTS (applying it to cumulative totals made the EN ladder non-monotonic — caught in audit), and every per-set community anchor was provenance-traced and converted.
2. **5 checkpoints → dated ladder.** Rungs by evidence tier: 6 archive-verified (21.5B@2015, 25.7B@2018, 27.2B@2019, 30.4B@2020, 34.1B@2021, 64.8B@2024 — the last correcting v2's 64.9B), the 85B@2026 live page, other official-claim rungs, and transcription-tier rungs (13B@2005 via forum-relayed press releases, 12B@2001, 14B@2006, 20B@2013, 43.2B@2022) at half weight. Rungs newer than the scored roster supports are EXCLUDED from the fit (currently capped at Mar-2024) so recent unscored production cannot be redistributed onto older sets.
3. **SEC-revenue window.** Evidence tiers, precisely: $568M/2000 is derived from an audited-period Hasbro 10-K405 disclosure (the 15% sentence itself sits outside the auditor's opinion); ~$500M/1999 is an ICv2 trade-press derivation across filings (well-sourced-estimate); 2001 has only an official ≤$286M ceiling with ~$100M inferred. ÷ wholesale pack price (40–55% of the $3.29 MSRP, WOTC's own archived store) × 11 cards/pack × 0.65 EN-of-West → 4.2–7.4B EN window, fit jointly (a consistency check — see the ladder section's honesty note).
4. **Production ramp + subset exclusion.** Windows ramp over 12 mo (boom era, motivated by the documented 2001 overproduction writeoffs AND tuned partly to reduce the Mar-2005 rung overshoot — both true) / 24 mo (modern); subset products (Trainer Galleries, cel25c) no longer double-claim parent production. Absolutes are projected LIFETIME production.
5. **Reliability flags** (`numerator_unreliable`, `pop_lag_underestimate`, `subset_set`, `icon_premium_suspect`) mark rows where the framework's assumptions fail; absolutes are SUPPRESSED for non-booster/subset rows rather than published as floor artifacts.

### Why we PIN beta_p instead of fitting it
All 17 per-set anchors are hobbyist guesses (weight 0.5). Re-tested 2026-07-22 with the expanded 6-era anchor set: a free fit gives beta_p≈0.04 (no longer sign-flipped as with the v2 anchor set, but a slope learned from guess-tier data; ≈0 would claim chase value doesn't drive grading propensity within an era, contradicting observable PSA submission behavior). We keep the 0.5 physical prior and only fit per-era intercepts. The run is deterministic.

### Known residual tensions (documented, not hidden)
- **The ladder has a residual SLOPE misfit**: the fit over-predicts crash/mid-era rungs and under-predicts the steep post-2021 English increments (FY23 alone implies ~4B EN of new production at share 0.35). Under the geomean calibration the residuals are spread rather than hidden in a pinned rung. Worst rung: 1.44× @ 2005-03-31; full residual profile in the ladder table above. Unresolved candidate causes: english_share >0.35 post-2020, modern reprint tails longer than 24 mo, transcription-tier crash-era rungs being loose, or genuinely larger modern per-set runs than community English estimates (~1B/set).
- **base1 is flagged `icon_premium_suspect`** (est ≈2.5× its halved WAG anchor). The Charizard-icon grading premium exceeds what beta_p=0.5 corrects; base1's within-era relative is likely inflated ~2×.
- **The 12B end-2001 rung and the WOTC per-set WAGs are NOT independent** — the WAGs were constructed to sum to that checkpoint, so jointly fitting both partly double-counts one source (both are low-weight).
- **All cumulative figures are 'over X' floors** used here as point targets; true values sit above each rung by an unknown margin.
- **BW-era estimates are very low** (median ~7M EN/set), violating the single-poster ordinal 'BW > HGSS' — but matching the sealed-box market (BW boxes price above many WOTC boxes). We report, not force, the ordinal.
- **Era bucketing is date-mechanical**: bw11 Legendary Treasures lands in era XY, swshp in era SM. Cosmetic for flagged rows, but visible in tables.

### What's still order-of-magnitude
- **english_share is THE load-bearing assumption** — no official language split exists; absolutes scale linearly in it (±0.10 ⇒ ±25–29%). Evidence: Japan market ≈ US market; TPCi/TPC production split; language count 11→16.
- **Pull-rate `D` table** unchanged — still the second-biggest lever, still estimates.
- **Unmodelled biases:** WOTC 1st-Ed/Shadowless/Unlimited pop-merging; attrition; crack-and-resubmit inflation; precon dilution (~30% of EX-era prints were theme decks per community estimate); grading-rate drift.
- **Bands ±3×** are order-of-magnitude rails, not confidence intervals.

### Honest read
Within-era relatives among UNFLAGGED mainline booster sets remain the strongest output (flagged rows — subsets, promos, icon-premium, pop-lagged — are exactly where relatives break too). Absolutes are now unit-consistent and ladder-dense: for mid-band mainline sets of ~2003–2022 the ±3× band is a fair claim; for WOTC (base1 especially), 2023+ mega-sets, and anything flagged, treat the numbers as directional only. Everything scales linearly in english_share. The EX–XY dead zone is now constrained by real checkpoints instead of smoothness alone; no NEW conclusive per-set evidence for those eras was found by the 2026-07-21 research pass — the existing community per-set numbers remain unsourced guesses (and two of them, ex7 and xy12, are used here at guess weight).
