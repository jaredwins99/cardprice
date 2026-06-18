# Results: Set Print-Population Estimates

_Generated 2026-06-18T15:37:54.848089+00:00 — model v2-grading-rate._

> **Model v2 (grading-rate divisor).** rel_pop = mean_chase_psa × pull_D / predicted_grading_rate. Grading rate is a mechanistic function of era + chase value (see `methodology.md` and `data/grading_rate_model.json`). Trends / pageviews / sales velocity are now VALIDATION signals, not divisor inputs.

- Sets scored: **153** / 170 (17 unscored, no confident chase pop or grading rate).
- Absolute calibration anchor: TPC **75,000,000,000** cards cumulative @ 2025-03-31 (windowed over 153 pre-checkpoint sets; post-fit scale=1.163).

## Grading-rate model (v2 divisor)

`log(grading_rate(set)) = alpha[era] + beta_p * log(chase_value / $100) + beta_y * log(years_since_release / 10y)`

- `beta_p = 0.5` (pinned prior — grading rate rises with chase value; sqrt scaling).
- `beta_y = 0.0` (pinned; age absorbed into era intercept).
- Per-era intercept `alpha[era]` (log grading rate at chase_value=$100):

| era | alpha | baseline rate | median predicted rate (across sets) |
|-----|------:|--------------:|-----------------------------:|
| WOTC | -12.658 | 3.18e-06 | 6.76e-06 |
| ECARD | -11.746 | 7.92e-06 | 1.80e-05 |
| EX | -10.829 | 1.98e-05 | 3.47e-05 |
| DP | -9.988 | 4.59e-05 | 4.94e-05 |
| HGSS | -9.141 | 1.07e-04 | 1.45e-04 |
| BW | -8.307 | 2.47e-04 | 2.95e-04 |
| XY | -7.485 | 5.61e-04 | 6.34e-04 |
| SM | -7.404 | 6.09e-04 | 6.07e-04 |
| SWSH | -8.227 | 2.67e-04 | 1.60e-04 |
| SV | -8.702 | 1.66e-04 | 1.10e-04 |

## Top 20 sets by estimated print population

| # | set_id | name | era | n_chase | mean PSA | pull_D | chase $ | grading rate | rel_pop(norm100) | abs_mid | abs_low–high | flags |
|---|--------|------|-----|--------:|---------:|------:|--------:|-------------:|-----------------:|--------:|-------------|-------|
| 1 | base1 | Base | WOTC | 3 | 46,396 | 1.0 | $268 | 5.21e-06 | 100.0 | 10,353,941,343 | 3,451,313,781–31,061,824,029 | — |
| 2 | sv3 | Obsidian Flames | SV | 3 | 16,962 | 35.0 | $44 | 1.10e-04 | 60.5816 | 6,272,588,430 | 2,090,862,810–18,817,765,291 | — |
| 3 | sv3pt5 | 151 | SV | 3 | 26,009 | 35.0 | $192 | 2.30e-04 | 44.3655 | 4,593,581,526 | 1,531,193,842–13,780,744,579 | — |
| 4 | swsh12pt5 | Crown Zenith | SWSH | 3 | 15,466 | 28.0 | $20 | 1.20e-04 | 40.651 | 4,208,983,383 | 1,402,994,461–12,626,950,150 | tier_mixed,tier_fallback |
| 5 | neo3 | Neo Revelation | WOTC | 3 | 3,986 | 6.0 | $750 | 8.71e-06 | 30.8127 | 3,190,325,693 | 1,063,441,898–9,570,977,080 | tier_mixed,tier_fallback |
| 6 | base4 | Base Set 2 | WOTC | 3 | 10,416 | 1.0 | $174 | 4.19e-06 | 27.8955 | 2,888,283,435 | 962,761,145–8,664,850,306 | — |
| 7 | cel25c | Celebrations: Classic Collection | SWSH | 3 | 18,739 | 28.0 | $101 | 2.69e-04 | 21.9318 | 2,270,810,281 | 756,936,760–6,812,430,842 | — |
| 8 | sv4pt5 | Paldean Fates | SV | 3 | 14,399 | 35.0 | $292 | 2.84e-04 | 19.9094 | 2,061,405,546 | 687,135,182–6,184,216,639 | — |
| 9 | neo4 | Neo Destiny | WOTC | 3 | 3,824 | 6.0 | $1871 | 1.38e-05 | 18.7158 | 1,937,817,823 | 645,939,274–5,813,453,469 | — |
| 10 | sv1 | Scarlet & Violet | SV | 3 | 4,160 | 35.0 | $35 | 9.80e-05 | 16.688 | 1,727,867,385 | 575,955,795–5,183,602,155 | — |
| 11 | swsh35 | Champion's Path | SWSH | 3 | 16,772 | 28.0 | $150 | 3.28e-04 | 16.1005 | 1,667,032,115 | 555,677,372–5,001,096,346 | — |
| 12 | swsh12pt5gg | Crown Zenith Galarian Gallery | SWSH | 3 | 16,354 | 28.0 | $146 | 3.23e-04 | 15.9073 | 1,647,036,845 | 549,012,282–4,941,110,536 | — |
| 13 | swsh45 | Shining Fates | SWSH | 3 | 3,322 | 28.0 | $8 | 7.38e-05 | 14.1506 | 1,465,146,535 | 488,382,178–4,395,439,604 | tier_mixed,tier_fallback |
| 14 | sv4 | Paradox Rift | SV | 3 | 3,342 | 35.0 | $34 | 9.66e-05 | 13.5998 | 1,408,112,021 | 469,370,674–4,224,336,063 | — |
| 15 | swsh9 | Brilliant Stars | SWSH | 3 | 6,719 | 28.0 | $35 | 1.57e-04 | 13.4502 | 1,392,622,431 | 464,207,477–4,177,867,294 | — |
| 16 | base3 | Fossil | WOTC | 3 | 5,476 | 1.0 | $254 | 5.07e-06 | 12.1314 | 1,256,078,453 | 418,692,818–3,768,235,359 | — |
| 17 | base5 | Team Rocket | WOTC | 3 | 7,126 | 1.0 | $451 | 6.76e-06 | 11.8427 | 1,226,185,993 | 408,728,664–3,678,557,980 | tier_fallback |
| 18 | sv7 | Stellar Crown | SV | 3 | 2,661 | 35.0 | $30 | 9.15e-05 | 11.4279 | 1,183,236,163 | 394,412,054–3,549,708,490 | — |
| 19 | sv5 | Temporal Forces | SV | 3 | 2,816 | 35.0 | $46 | 1.13e-04 | 9.803 | 1,014,992,663 | 338,330,888–3,044,977,989 | — |
| 20 | xy12 | Evolutions | XY | 3 | 6,062 | 18.0 | $5 | 1.28e-04 | 9.6035 | 994,341,147 | 331,447,049–2,983,023,440 | — |

## Bottom 20 sets by estimated print population

| # | set_id | name | era | n_chase | mean PSA | pull_D | chase $ | grading rate | rel_pop(norm100) | abs_mid | abs_low–high | flags |
|---|--------|------|-----|--------:|---------:|------:|--------:|-------------:|-----------------:|--------:|-------------|-------|
| 1 | mcd16 | McDonald's Collection 2016 | XY | 3 | 1 | 1.0 | $33 | 3.24e-04 | 0.0 | 4,778 | 1,593–14,335 | — |
| 2 | xy0 | Kalos Starter Set | XY | 3 | 13 | 1.0 | $17 | 2.28e-04 | 0.0006 | 64,473 | 21,491–193,419 | — |
| 3 | swshp | SWSH Black Star Promos | SM | 1 | 98 | 1.0 | $397 | 1.21e-03 | 0.0009 | 93,982 | 31,327–281,947 | — |
| 4 | mcd15 | McDonald's Collection 2015 | XY | 3 | 24 | 1.0 | $22 | 2.61e-04 | 0.001 | 105,478 | 35,159–316,433 | — |
| 5 | mcd12 | McDonald's Collection 2012 | BW | 3 | 8 | 1.0 | $11 | 8.19e-05 | 0.0011 | 118,264 | 39,421–354,793 | — |
| 6 | mcd14 | McDonald's Collection 2014 | XY | 3 | 33 | 1.0 | $20 | 2.49e-04 | 0.0015 | 155,895 | 51,965–467,685 | — |
| 7 | xyp | XY Black Star Promos | XY | 3 | 168 | 1.0 | $369 | 1.08e-03 | 0.0017 | 181,002 | 60,334–543,007 | — |
| 8 | mcd17 | McDonald's Collection 2017 | SM | 3 | 36 | 1.0 | $11 | 2.01e-04 | 0.002 | 208,550 | 69,517–625,650 | — |
| 9 | mcd18 | McDonald's Collection 2018 | SM | 3 | 66 | 1.0 | $32 | 3.42e-04 | 0.0022 | 224,067 | 74,689–672,201 | — |
| 10 | mcd19 | McDonald's Collection 2019 | SM | 3 | 65 | 1.0 | $18 | 2.61e-04 | 0.0028 | 288,229 | 96,076–864,687 | — |
| 11 | pop6 | POP Series 6 | DP | 3 | 18 | 1.5 | $9 | 1.40e-05 | 0.0221 | 2,291,007 | 763,669–6,873,021 | — |
| 12 | bw11 | Legendary Treasures | XY | 2 | 104 | 18.0 | $262 | 9.08e-04 | 0.023 | 2,385,225 | 795,075–7,155,676 | tier_mixed,tier_fallback |
| 13 | bw2 | Emerging Powers | BW | 3 | 41 | 5.5 | $17 | 1.01e-04 | 0.0252 | 2,608,233 | 869,411–7,824,699 | tier_mixed,tier_fallback |
| 14 | pop4 | POP Series 4 | EX | 3 | 65 | 1.5 | $343 | 3.67e-05 | 0.0298 | 3,089,917 | 1,029,972–9,269,750 | — |
| 15 | dv1 | Dragon Vault | BW | 3 | 249 | 1.3 | $18 | 1.06e-04 | 0.0344 | 3,561,429 | 1,187,143–10,684,286 | — |
| 16 | pop9 | POP Series 9 | DP | 3 | 48 | 1.5 | $17 | 1.91e-05 | 0.042 | 4,353,398 | 1,451,133–13,060,194 | — |
| 17 | ru1 | Pokémon Rumble | DP | 3 | 362 | 1.0 | $372 | 8.86e-05 | 0.0459 | 4,750,941 | 1,583,647–14,252,823 | — |
| 18 | col1 | Call of Legends | HGSS | 3 | 718 | 1.2 | $379 | 2.09e-04 | 0.0463 | 4,797,802 | 1,599,267–14,393,407 | — |
| 19 | pop8 | POP Series 8 | DP | 3 | 43 | 1.5 | $10 | 1.48e-05 | 0.0484 | 5,013,494 | 1,671,165–15,040,482 | — |
| 20 | xy5 | Primal Clash | XY | 3 | 60 | 18.0 | $17 | 2.29e-04 | 0.0527 | 5,452,897 | 1,817,632–16,358,692 | — |

## Anchor sensitivity (estimate / published anchor)

Per-set `total_print_run` anchors from `known_print_runs.json`. The grading-rate model is fit *jointly* against these (with low weight for `hobbyist-guess` credibility) and the official TPC cumulative checkpoints. This sensitivity is a fit-quality measure, not held-out.

**8/18 anchors within 2×, 10/18 within 3× (model v1 had 1/10 within 2×, 1/10 within 3×).**

| set | variant | credibility | anchor_mid | estimate_mid | est/anchor |
|-----|---------|-------------|-----------:|-------------:|-----------:|
| base1 (Base Set) | 1st_edition | hobbyist-guess | 1,500,000 | 10,353,941,343 | 6902.63× |
| swsh12pt5 (Crown Zenith) | na | hobbyist-guess | 700,000,000 | 4,208,983,383 | 6.01× |
| base4 (Base Set 2) | na | hobbyist-guess | 600,000,000 | 2,888,283,435 | 4.81× |
| sm115 (Hidden Fates) | na | hobbyist-guess | 200,000,000 | 690,485,873 | 3.45× |
| base1 (Base Set) | unlimited | hobbyist-guess | 3,000,000,000 | 10,353,941,343 | 3.45× |
| swsh35 (Champion's Path) | na | hobbyist-guess | 500,000,000 | 1,667,032,115 | 3.33× |
| xy12 (XY Evolutions) | na | hobbyist-guess | 400,000,000 | 994,341,147 | 2.49× |
| sv4pt5 (Paldean Fates) | na | hobbyist-guess | 1,200,000,000 | 2,061,405,546 | 1.72× |
| ex7 (Team Rocket Returns) | na | hobbyist-guess | 80,000,000 | 106,762,440 | 1.33× |
| cel25 (Celebrations) | na | hobbyist-guess | 400,000,000 | 459,662,325 | 1.15× |
| sv3pt5 (Pokemon 151) | na | hobbyist-guess | 4,000,000,000 | 4,593,581,526 | 1.15× |
| base5 (Team Rocket) | na | hobbyist-guess | 1,100,000,000 | 1,226,185,993 | 1.11× |
| base3 (Fossil) | na | hobbyist-guess | 1,750,000,000 | 1,256,078,453 | 0.72× |
| neo1 (Neo Genesis) | na | hobbyist-guess | 700,000,000 | 427,984,314 | 0.61× |
| base2 (Jungle) | na | hobbyist-guess | 1,750,000,000 | 924,099,269 | 0.53× |
| gym1 (Gym Heroes) | na | hobbyist-guess | 850,000,000 | 390,480,369 | 0.46× |
| swsh7 (Evolving Skies) | na | hobbyist-guess | 2,000,000,000 | 644,101,068 | 0.32× |
| neo3 (Neo Revelation (cumulative checkpoint context)) | na | well-sourced-estimate | 12,000,000,000 | 3,190,325,693 | 0.27× |

## Per-era summary

| era | n sets | median rate | median print run | sum print run |
|-----|------:|-----------:|----------------:|--------------:|
| WOTC | 13 | 6.76e-06 | 924,100,015 | 24,064,218,722 |
| ECARD | 3 | 1.80e-05 | 361,286,095 | 1,259,692,880 |
| EX | 24 | 3.47e-05 | 111,501,710 | 3,572,989,923 |
| DP | 16 | 4.94e-05 | 52,862,839 | 1,095,090,996 |
| HGSS | 5 | 1.45e-04 | 54,628,703 | 280,410,123 |
| BW | 12 | 2.95e-04 | 8,708,729 | 144,346,827 |
| XY | 20 | 6.34e-04 | 12,141,414 | 1,456,617,964 |
| SM | 21 | 6.07e-04 | 52,259,484 | 2,070,260,337 |
| SWSH | 26 | 1.60e-04 | 513,901,061 | 20,657,638,835 |
| SV | 13 | 1.10e-04 | 1,014,991,407 | 20,398,733,392 |

## Validation: rel_pop vs held-out signals (Spearman)

| signal | ρ | n | interpretation |
|--------|--:|--:|----------------|
| sales velocity (TCGPlayer) | 0.586 | 153 | endogenous: bigger print -> more supply -> more sales (expect +) |
| Google Trends interest | 0.616 | 153 | demand-side: popular sets get printed more (expect +, weaker than sales) |

Expectation: **positive but < 1** — more printed ⇒ more supply/demand traffic, but popularity, age and price break the tie. A value near 0 or negative would flag a bug; ~1.0 would mean we just re-derived demand.

Biggest rank disagreements vs sales (plausibly real, not bugs):

| set_id | name | rel_pop_rank | sales_rank | gap |
|--------|------|-------------:|-----------:|----:|
| bw2 | Emerging Powers | 141 | 43 | +98 |
| neo3 | Neo Revelation | 5 | 103 | -98 |
| neo4 | Neo Destiny | 9 | 106 | -97 |
| xy5 | Primal Clash | 134 | 40 | +94 |
| cel25c | Celebrations: Classic Collection | 7 | 98 | -91 |
| swshp | SWSH Black Star Promos | 151 | 61 | +90 |
| bw11 | Legendary Treasures | 142 | 54 | +88 |
| ecard2 | Aquapolis | 24 | 112 | -88 |
| ex10 | Unseen Forces | 34 | 117 | -83 |
| ex16 | Power Keepers | 47 | 129 | -82 |

_Positive gap = ranks much higher in population than in sales (printed big but trades slowly — old/cheap bulk). Negative = trades hot for its print size (small but in demand)._

## Confidence & caveats (frank)

### What changed in v2
v1 used `exp(z(log(Google Trends)))` as the divisor. That captures search-popularity but is *blind* to grading-rate dynamics: a $3,000 vintage card gets graded ~10–20× harder than a $100 modern SIR, regardless of who's searching. v1 anchors landed 1/10 within 2× (vintage anchors ~30–1000× off). v2 swaps in a mechanistic `grading_rate = exp(alpha[era] + 0.5·log(chase_value/$100))` fit jointly to per-set anchors AND the official TPC cumulative checkpoints.

### Why we PIN beta_p instead of fitting it
Of the 15 anchors in `known_print_runs.json`, only 8 are usable per-set print-run estimates after excluding variant subsets and the `neo3` cumulative checkpoint. **Seven of those 8 are WOTC and one is EX.** Fitting log_price/log_yrs slopes on that distribution either over-fits or returns negative coefficients (more value → less grading), which is physically wrong. We pin `beta_p=0.5` (square-root scaling, consistent with PSA's value-driven grading-rate dynamics) and only fit per-era intercepts. **The model is reproducible from the JSON; the scripts/fit_grading_rate.py run is deterministic.**

### What's still order-of-magnitude
- **Single-era anchors drive their era's intercept.** WOTC has 7 anchors → WOTC alpha is data-driven. EX has 1 → EX alpha is essentially that one anchor. ECARD, DP, HGSS, BW, XY, SM, SWSH, SV have ZERO per-set anchors; their alphas are determined by TPC checkpoint windowed sums + cross-era smoothness. Per-set absolutes in those eras are order-of-magnitude.
- **Pull-rate `D` table unchanged from v1** — still the second-biggest lever and still all estimates. See `methodology.md` for the table.
- **Unmodelled biases unchanged:** WOTC 1st-Ed/Shadowless/Unlimited graded separately but pop-merged; JP vs EN separate prints; attrition; crack-and-resubmit pop inflation; precon-deck dilution; grading-rate drift over time.
- **Bands ±3×** are honest order-of-magnitude rails, not confidence intervals.

### Honest read
Within-era relatives remain the strongest output (the numerator and pull_D are unchanged; the divisor change mostly affects cross-era). Cross-era relatives and all absolutes are still order-of-magnitude — better than v1 but not tight. To tighten the absolutes meaningfully we need more per-set anchors, especially in DP/HGSS/BW/XY/SM/SWSH/SV (currently zero per-set anchors in those eras).
