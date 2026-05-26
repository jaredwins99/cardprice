# Prior Art: How People Estimate Pokémon TCG Print Runs

> Purpose: survey existing methodologies for estimating Pokémon set print-run /
> population sizes so we don't reinvent the wheel, and reconcile them with this
> sub-project's plan (`relative_population ∝ chase_graded_pop / popularity`).
> Companion machine-readable anchors live in `../data/known_print_runs.json`.

## TL;DR

- **No per-set print run is officially published.** The Pokémon Company only
  releases a single global cumulative total ("Pokémon in Figures") plus annual
  totals. Everything per-set is estimated by hobbyists.
- The best community work (Elite Fourum) **starts from the official global
  cumulative checkpoints and divides downward** — the inverse of what we want,
  but it gives us calibration anchors and a method to cross-check our ranking.
- **Graded population reports (PSA/CGC/BGS) are the only directly countable
  per-set signal**, which is exactly why this project uses them. But grading
  *rate* is driven by popularity/value, so raw pop overstates popular sets —
  the bias we correct with a popularity divisor.
- WOTC-era (1999–2003) and modern-era (post-2020) need different handling:
  modern sets are 10–100× larger and TPCi releases more aggregate data.

---

## Methodologies

### 1. Official global figures ("Pokémon in Figures") — top-down anchor
**How it works.** TPCi publishes a cumulative worldwide card count and annual
totals. As of FY2024-25 (Mar 2025): **75 billion cards ever printed**, with
~44.6 B (≈60%) printed since FY2020-21. Earlier checkpoints: 23.6 B (Mar 2017),
~43.2 B (Mar 2022), 52.9 B (Mar 2023), 64.9 B (Mar 2024). Annual: 9.7 B (FY22),
11.9 B (FY23, record), 10.2 B (FY24).
**Data needed.** Just the published figures.
**Accuracy.** The figures themselves are authoritative. They do **not**
disaggregate by set, language, or variant, so per-set use requires assumptions.
**When it applies.** As a hard *upper bound on the sum* of all per-set
estimates, and to confirm the modern era dwarfs the vintage era.
**Source.** TPC corporate "Pokémon in Figures"; PokeBeach annual recaps.
- https://corporate.pokemon.co.jp/en/aboutus/figures/
- https://www.pokebeach.com/2025/05/pokemon-tcg-printed-10-2-billion-cards-in-2024-lower-than-the-previous-year

### 2. Checkpoint division (top-down, per-set) — Elite Fourum method
**How it works.** Take the official cumulative figure at a date, subtract the
prior checkpoint to get cards printed in an interval, then divide across the
sets released in that interval (often assuming a 50/50 English/non-English
split and weighting by perceived demand). Example: "12 B cumulative by end of
2001 ⇒ ≈1.2 B per WOTC set on average."
**Data needed.** Official checkpoints + a set release timeline + demand weights.
**Accuracy.** Order-of-magnitude. The per-set split is the weak link — it's
essentially a popularity-weighted guess, the same thing we're trying to *solve*.
**When it applies.** Vintage/early-modern sets where checkpoints bracket a small
number of releases. Best used for the *sum across a window*, not single sets.
**Source.** https://www.elitefourum.com/t/an-elaborate-attempt-at-print-run-estimation-wip-5-8-18/20273

### 3. Pull-rate × packs-sold (bottom-up) — most rigorous when inputs exist
**How it works.** Cards of rarity *r* per set ≈
`packs_sold × (cards_per_pack) × P(rarity r per pack) ÷ (#unique cards of r)`.
Inverted, you can solve for packs from a known per-card population. The Elite
Fourum thread runs this for Team Rocket Returns: 80 M cards × 1/9 × 1/36 ≈
247 K booster boxes ⇒ ≈41 K of each Gold Star.
**Data needed.** Pull rates (community-aggregated; TPCi doesn't publish them),
pack/box composition, and either packs-sold or a per-card population to anchor.
**Accuracy.** Pull-rate variance is large: ±30–50% at 500 packs, ±15–25% at
1,000, ±5–10% only past 3,000 packs. So you need big community sample sizes per
set. Modern SIR/hyper-rare rates run 1:91 to 1:1,260 packs.
**When it applies.** Modern sets with thousands of logged openings; vintage only
where pull rates are well established (Base holo = 1-in-3 of the rare slot).
**Gotcha.** Theme/precon decks distort it — ~30% of each EX-era set's print run
was precon (60 cards/deck vs 11/pack), so booster-only math overcounts unless
you subtract decks.
**Source.** https://www.elitefourum.com/t/pokemon-tcg-production-numbers-pull-rates-more/56264 ;
https://cardchill.com/article/pokemon-tcg-pull-rates-explained-what-to-expect-from-booster-packs

### 4. Print-sheet / uncut-sheet analysis — fine-grained, data-starved
**How it works.** Cards are printed on large sheets with a fixed number of
slots; a rarity's print quantity depends on how many slots it occupies vs. how
many unique cards share that rarity. WOTC holo sheets are reported as ~110
cards. Surviving sheets carry printer metadata (printer name, date, "Form #",
job #) documenting individual runs (e.g. "Unlimited Base Uncommon – 8th Run,
Yaquinto, 30-7-99, Form 5, Job #1739"), which lets you count *number of runs*.
**Data needed.** Actual sheet photos, layouts, and run logs.
**Accuracy.** Potentially the most precise for *relative* within-set rarity,
but starved of data: layouts vary by printer (USA/Belgium/UK/Australia) and few
modern sheets ever surfaced ("TPCi's stricter policies"). Cannot give absolute
totals without knowing how many times each sheet was run.
**When it applies.** WOTC-era relative-rarity nuance (why some holos in a set
are scarcer than others on the same sheet). Rarely usable for whole-set totals.
**Source.** https://pokemuseum.weebly.com/uncut-sheets.html

### 5. Graded-population extrapolation (bottom-up, per-card) — our core signal
**How it works.** Count cards graded by PSA + CGC + BGS for a set's chase
card(s); optionally gross up by an assumed grading rate and an attrition factor
(~20% destroyed) to back into a surviving/total population.
**Data needed.** Pop reports (PSA, CGC, BGS; aggregated by GemRate, Pikawiz,
PriceCharting, PokeMetrics, gemtracker).
**Accuracy.** The *count* is exact and directly observable. The *extrapolation*
is not: grading rate is driven by card value/popularity, varies wildly by set,
drifts over time (Base Charizard pop grew only 3–4%/yr post-2020), and is
inflated by crack-and-resubmit. Doubling-the-grade-count heuristics are crude.
**When it applies.** Every era, but only as a *relative* signal after correcting
for grading-rate bias — precisely this project's design.
**Source.** https://www.pokemonpricetracker.com/blog/posts/pokemon-card-population-report-guide-rarity-analysis-2026 ;
https://www.gemrate.com/universal-search ; https://www.pikawiz.com/cards/pop-report/baseset

### 6. Sealed-product / market-supply analysis — indirect
**How it works.** Infer scarcity from how many sealed boxes/cases surface, how
fast sealed inventory clears, and dealer allocation patterns. Vintage 1st-Ed
Base is "limited printing, sold out before Pokémania" → small run.
**Accuracy.** Qualitative; good for *ordering* runs (1st Ed < Shadowless <
Unlimited) but not for numbers. Vulnerable to resealing fraud.
**When it applies.** Vintage variant ranking and direction-of-bias checks.

---

## Era split: WOTC (1999–2003) vs. modern (post-2020)

| Aspect | WOTC era | Modern era |
|---|---|---|
| Per-set size | ~0.6–3 B cards (hobbyist est.) | 10–100× larger; FY23 alone = 11.9 B |
| Official data | Only old cumulative checkpoints | Annual + cumulative totals published |
| Variants | 1st Ed / Shadowless / Unlimited are **separate runs** | reprint waves, but no Shadowless-style split |
| Pull-rate data | sparse, must use known slot odds | thousands of logged openings |
| Print sheets | some surfaced w/ run logs | almost none surfaced |
| Grading rate | very high (high value) → pop overstates | lower per-card, but huge raw numbers |

Implication: WOTC sets must be split by `print_variant` (Base = 3 runs);
modern sets need the post-2020 production explosion baked into priors (the
old "EX≈200 M, XY≈150 M per set" averages are a *floor*, not current).

---

## Known gotchas affecting cross-set comparison

1. **Variant splits as separate runs.** Base 1st-Ed, Shadowless, and Unlimited
   are three populations; never merge them. Our anchors keep them distinct.
2. **Regional (JP vs EN) separate prints.** Japanese sets are printed and graded
   independently; English-set print runs ≠ global. Pop reports often mix them.
   Japan shipped 87 M cards in Oct'96–Mar'97 alone, before any English release.
3. **Reprints / multi-wave runs.** WOTC Unlimited Base = print runs 2–7; modern
   chase reprints (e.g. "first wave" with ~10–15% better SIR odds) inflate later.
4. **Theme/precon decks.** ~30% of EX-era print runs were precon (different card
   counts) — distorts booster-only pull-rate math.
5. **Grading-rate drift & crack-resubmit.** Pop reports double-count resubmitted
   cards and grow non-uniformly over time; popular cards get graded at far higher
   rates. This is the *exact bias* our popularity divisor targets.
6. **Resealing fraud.** Corrupts sealed-supply signals; ignore sealed counts for
   anything quantitative.
7. **Attrition.** Vintage survival is far below print run (cards were "toys");
   modern survival is high. Attrition is itself age- and popularity-correlated.

---

## Recommended approach (reconciling prior art with our plan)

Our plan — `relative_population(set) ∝ mean_chase_graded_pop(set) / popularity(set)`
— is **method #5 with #1's popularity bias explicitly divided out**. Prior art
validates the core choice (graded pop is the only countable per-set signal) and
the core worry (grading rate ∝ popularity). Recommendations:

1. **Keep graded pop as the signal, popularity as the divisor** — sound and
   matches what serious estimators implicitly wish they could do.
2. **Anchor, don't fit, on the official global total (#1).** Constrain the
   *sum* of our scaled relative estimates to ≤75 B (and respect the WOTC-window
   checkpoints). This converts our ranking into absolute bands "for free" without
   trusting any single hobbyist per-set number. **This is the highest-value
   method to adopt that isn't yet in the plan.**
3. **Use checkpoint division (#2) only as a coarse sanity rail**, not an input —
   its per-set split is the same popularity-weighted guess we're replacing.
4. **Add pull-rate × packs (#3) as an independent cross-check for modern sets**,
   where logged-opening volume is high; flag low confidence below ~3,000 packs.
5. **Split every WOTC set by `print_variant`** and treat JP/EN separately;
   don't let pop reports silently merge regions.
6. **Treat `known_print_runs.json` anchors by tier**: hard-anchor only on
   `official`; use `well-sourced-estimate` (the graded-pop totals) as ratio
   checks; use `hobbyist-guess` only for order-of-magnitude rails.
7. **Model grading-rate drift over time** if comparing sets graded across
   different periods (Base Charizard's 3–4%/yr growth shows pop reports are
   not stationary).

### Adopt that isn't in the current plan
- **Global-total normalization constraint (#2 above).** Pin the sum of scaled
  relative estimates to the official 75 B (windowed by checkpoint). Cheap,
  authoritative, turns a ranking into calibrated absolute bands.
- **Pull-rate × packs cross-validation for modern sets (#4 above).** A second,
  methodologically independent estimator to triangulate the modern era where
  graded pop is a noisy fraction of a huge print run.

---

## Sources
- TPC, "Pokémon in Figures": https://corporate.pokemon.co.jp/en/aboutus/figures/
- PokeBeach FY24 production: https://www.pokebeach.com/2025/05/pokemon-tcg-printed-10-2-billion-cards-in-2024-lower-than-the-previous-year
- Elite Fourum, elaborate print-run estimation (WIP): https://www.elitefourum.com/t/an-elaborate-attempt-at-print-run-estimation-wip-5-8-18/20273
- Elite Fourum, production numbers & pull rates: https://www.elitefourum.com/t/pokemon-tcg-production-numbers-pull-rates-more/56264
- Pull rates explained: https://cardchill.com/article/pokemon-tcg-pull-rates-explained-what-to-expect-from-booster-packs
- Pokemuseum uncut sheets: https://pokemuseum.weebly.com/uncut-sheets.html
- Pop report guide: https://www.pokemonpricetracker.com/blog/posts/pokemon-card-population-report-guide-rarity-analysis-2026
- GemRate pop aggregator: https://www.gemrate.com/universal-search
- OG Cards, 1st-Ed Charizard pop: https://ogcards.com/blogs/pokemon-cards/how-many-1st-edition-charizards-are-there
- OG Cards, Shadowless Charizard pop: https://ogcards.com/blogs/pokemon-cards/how-many-shadowless-charizards
- Pokemonpricing, Base Set runs: https://pokemonpricing.com/how-many-cards-were-printed-in-each-pokemon-base-set-run/
