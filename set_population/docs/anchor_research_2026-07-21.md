# Print-Run Anchor Deep Research — 2026-07-21

Method: 5-angle web-search fan-out → 19 sources fetched → 88 claims extracted →
25 claims 3-vote adversarially verified (23 confirmed, 2 refuted) → 12 synthesized
findings. 101 agents. Integrated into `data/known_print_runs.json` (36 → 45 anchors)
same day; see `_meta.notes_2026_07_21` and `qualitative_constraints` there.

## Summary

The research materially upgraded the calibration anchor set in three ways. First, the official cumulative-production timeline is now far denser and better dated: newly verified checkpoints include over 14B (as of ~2004, boilerplate through 2010), over 21.5B (~2014/15), 25.7B (Mar 2018, archive-verified official), 27.2B (Mar 2019, archive-verified), 28.8B (Sep 2019), 34.1B (Mar 2021, +3.7B), a correction of the Mar 2024 figure from 64.9B to 64.8B, and a firm dating of the 85B checkpoint to end-March 2026 (implying a ~10B FY2025 plateau at maximum printing capacity). Second, audited Hasbro SEC filings yield hard dollar-revenue anchors for the WOTC era — ~$568M Pokemon TCG wholesale revenue in CY2000, ~$500M (range $450-514M) of WOTC's $669M 1999 revenue, a <=$286M ceiling (likely ~$100M actual) for 2001 — which bound card volumes once wholesale pack pricing is assumed. Third, the provenance of every currently-held WOTC per-set round number (Base 3B, Jungle/Fossil 1.75B, etc.) was traced to the 2018 Elite Fourum thread, where they are explicitly ALL-LANGUAGES "wild ass guesses" calibrated to sum to the 12B end-2001 checkpoint — meaning English-only use requires roughly halving them. The biggest stated gaps (per-set EX/DP/HGSS/BW/XY figures, per-set SWSH/SV figures, documented 1st Edition/Shadowless runs) remain numerically unfilled; only ordinal era-ranking constraints and negative findings (no conclusive WOTC per-set data exists anywhere) were obtained.

## Findings

### 1. [high] vote 3-0 (claims 0, 1, 8 merged; each 3-0)

OFFICIAL ANCHOR (new dating + delta): Cumulative Pokemon TCG production exceeded 85 billion cards as of end of March 2026 (16 languages, 90 countries/regions), per TPC's official Pokemon in Figures page. Differencing against the 75B (Mar 2025) checkpoint gives a derived FY2025 annual production of ~10B cards (Dexerto's 11.7%-of-all-cards figure implies ~9.95B), confirming production plateaued near 10B/yr after the 11.9B FY23 peak rather than continuing to grow. Anchor shape: value_low=85B (stated as a floor), as_of=2026-03-31, unit=cards (all languages), tier=official; annual delta tier=derived-from-official.

**Evidence:** Raw HTML of the official page (fetched 2026-07-21) contains 'As of the end of March 2026' and 'Total production ... over {"count":85} billion cards' (counter-animation prop explains why text scrapes show blank). Corroborated independently by PokeBeach, Kotaku, GameRant, PokeGuardian, Dexerto (which quotes 'as of March 31, 2026' and the internally consistent 11.7% annual share). Arithmetic cross-check: 64.9B (Mar 2024, actually 64.8B per correction below) + 10.2B FY24 = ~75.1B at Mar 2025, so the 85B floor implies FY25 >= ~9.9B.

Sources: https://corporate.pokemon.co.jp/en/aboutus/figures/, https://www.pokebeach.com/2026/05/pokemon-tcg-printed-10-billion-cards-in-2025-as-overwhelming-demand-outpaced-production-capacity

### 2. [high] vote 3-0 (claims 9, 10, 18 merged; each 3-0)

OFFICIAL/DERIVED ANCHORS (new pre-2017 and intermediate checkpoints, filling a stated gap): over 14 billion cards cumulative worldwide, figure first issued 2004 and reused as boilerplate through 2010 (30-40 countries) — anchor to ~2004, not 2010; over 21.5 billion cumulative ~2014/15 (10 languages, 74 countries, one source ties it to end-March-2015); over 25.7 billion as of end March 2018 (archive-verified on the official TPC page: web.archive.org/web/20181201135654, tier=official, implying ~2.1B in FY ending Mar 2018); over 27.2 billion as of end March 2019 (archive-verified: web.archive.org/web/20190802044404, tier=official, +1.5B); over 28.8 billion as of Sep 2019 (+1.6B; no direct archive snapshot found — tier=derived-from-official, transcribed); 34.1B Mar 2021 with explicit +3.7B annual delta (archive-verified snapshot 20210604071148).

**Evidence:** The Dec 2018 Wayback snapshot of the official page reads verbatim 'Total shipments / over 25.7 billion cards worldwide / 11 languages / 74 countries and regions / (as of the end of March 2018)'; the Aug 2019 snapshot shows 27.2B (end Mar 2019), also independently cited by an Elite Fourum post in Aug 2019. The 14B and 21.5B figures come from TPCi press emails/boilerplate per PokeBeach's provenance note ('first appeared in emails from Pokemon USA to PokeBeach in 2004 and continued through 2010'); PokeBeach itself warns TPC 'only started keeping track more accurately in 2017'. Full sequence is monotone and arithmetically coherent: 12B (end 2001) -> >=14B (2004) -> >=21.5B (2014/15) -> 23.6B (Mar 2017) -> 25.7B -> 27.2B -> 28.8B -> 30.4B (Mar 2020) -> 34.1B (Mar 2021). Note official wording shifted from 'total shipments' (<=2020) to 'total production' (>=2021).

Sources: https://www.pokebeach.com/2026/05/pokemon-tcg-printed-10-billion-cards-in-2025-as-overwhelming-demand-outpaced-production-capacity, https://web.archive.org/web/20181201135654/https://www.pokemon.co.jp/corporate/en/services/, https://web.archive.org/web/20190802044404/https://www.pokemon.co.jp/corporate/en/services/, https://www.elitefourum.com/t/an-elaborate-attempt-at-print-run-estimation-wip-5-8-18/20273

### 3. [high] vote 3-0 (part of claim 10)

CORRECTION to an existing anchor: the official March 2024 cumulative figure is 64.8 billion, not the currently-held 64.9B. Verdict: 64.9B anchor CONTRADICTED; replace with 64.8B (tier=official, archive-verified).

**Evidence:** Wayback snapshot 20240718172040 of the official corporate figures page reads 'Total production over 64.8 billion cards'. Arithmetic agrees with known annual figures: 52.9B (Mar 2023) + 11.9B (FY23) = 64.8B. PokeBeach's checkpoint table lists the same 64.8B (+11.9B).

Sources: https://web.archive.org/web/20240718172040/https://corporate.pokemon.co.jp/en/aboutus/figures/, https://www.pokebeach.com/2026/05/pokemon-tcg-printed-10-billion-cards-in-2025-as-overwhelming-demand-outpaced-production-capacity

### 4. [high] vote 3-0 (claims 2, 11 merged; each 3-0)

OFFICIAL REVENUE ANCHOR, CY2000 (audited SEC filing): the Pokemon trading card line contributed 15% of Hasbro's FY2000 consolidated net revenues of $3,787,215 thousand, i.e. ~$568M wholesale revenue (rounding range $549-587M). This is worldwide, all-language WOTC TCG revenue, so it upper-bounds English-only volume; converting to cards requires an assumed wholesale price per pack and cards per pack. Anchor: era=WOTC peak (Base Set 2 / Team Rocket / Gym / Neo Genesis retail window), unit=USD wholesale revenue, as_of=FY2000 (ended 2000-12-31), tier=official.

**Evidence:** Verified verbatim in Hasbro's 10-K405: 'During the 2000 fiscal year, revenues from the POKEMON trading card line of products contributed 15% of consolidated net revenues of the Company' with FY2000 net revenues of $3,787,215K in the audited statements; 0.15 x 3,787,215 = $568.1M. The filing separately discloses Pokemon toys, confirming the 15% is TCG-specific. ICv2's contemporary reporting independently computes the same $568M. Caveat: the 15% sentence sits in Item 1 (SEC-filed disclosure) outside the auditor's opinion, which strictly covers only the revenue base.

Sources: https://www.sec.gov/Archives/edgar/data/46080/000091205702012556/a2073921z10-k405.htm, https://icv2.com/articles/games/view/373/hasbro-results-cite-pokemon-decline

### 5. [high] vote 3-0 (claims 5, 12, 13 merged; votes 3-0, 3-0, 2-1)

OFFICIAL REVENUE ANCHORS, 1999 (SEC filings + trade-press arithmetic): WOTC total revenue was ~$237M in Q4 FY1999 alone (post-acquisition, stated in Hasbro's 8-K; cross-checked as 14% of the $1,936,100K Games segment = $238.4M) and ~$669M for full-year 1999 ($271M Q4 + $398M pro-forma pre-acquisition delta from the 10-K). The Pokemon-specific slice of 1999 WOTC revenue is ~$500M (ICv2 derivation: $669M minus ~$155M 1998 pro-forma WOTC baseline, both endpoints reproduced from primary SEC filings; refined range $450-514M since Magic also grew in 1999). Anchor: era=Base/Jungle/Fossil retail window (NOT Neo — English Neo Genesis released Dec 2000; Base Set 2/Team Rocket released after the Q4-1999 window closed), unit=USD wholesale revenue, tier=official ($237M/$669M inputs) and derived-from-official (~$500M Pokemon slice).

**Evidence:** 8-K Exhibit 99 verbatim: WOTC 'acquired in September and contributed approximately $237 million of revenue'. FY1999 10-K verbatim: Wizards 'accounted for 14% of Games segment revenues'; pro forma FY1999 revenues $4,630,368K vs actual $4,232,263K (+$398.1M = WOTC Jan-Sep). The $155M 1998 WOTC baseline reproduces exactly from differencing the FY1999 and FY1998 10-K pro formas. $271M + $398M = $669M; $669M - $155M = $514M ~ 'half a billion' as ICv2 states, with the article's own cross-check (~30% of acquisition growth was non-Pokemon) trimming the Pokemon slice to ~$450-470M. Verifiers confirmed no unit/print-run figures exist anywhere in these filings — revenue only.

Sources: https://www.sec.gov/Archives/edgar/data/0000046080/000004608000000001/0000046080-00-000001.txt, https://www.sec.gov/Archives/edgar/data/46080/0000046080-00-000003.txt, https://icv2.com/articles/games/view/373/hasbro-results-cite-pokemon-decline

### 6. [high] vote 3-0 (claims 3, 4 merged; each 3-0)

OFFICIAL REVENUE ANCHORS, 2001 crash (audited SEC filing): FY2001 had no product line above 10% of Hasbro's $2,856,339K net revenues, placing an official <=$286M ceiling on 2001 Pokemon TCG revenue (>=49.7% collapse from 2000's $568M). The same filing quantifies the crash: ~$469M Games-segment decline from Pokemon products 'primarily trading card games' plus ~$288M International decline (Pokemon+Furby combined), implying actual 2001 Pokemon TCG revenue on the order of $100M. Also notes obsolescence writeoffs from 'overproduction of certain trading card games, primarily POKEMON related' — direct official evidence that late-WOTC sets were overprinted relative to sell-through. Constrains print volume of the FY2001 English window (Neo Discovery/Revelation plus carryover; Neo Destiny and Expedition are fiscal-2002 releases). Tier=official.

**Evidence:** All quotes verified verbatim in the primary filing, with units confirmed as thousands of dollars. Internal arithmetic consistent: Games segment fell $734.9M total = ~$194.3M Infogrames divestiture + $469M Pokemon + $54M Furby, offset by Magic growth. Caveat on the residual: $568M is company-wide while $469M is Games-segment-only, so the ~$100M residual estimate is an upper-ish bound direction ('$100M or less' holds a fortiori).

Sources: https://www.sec.gov/Archives/edgar/data/46080/000091205702012556/a2073921z10-k405.htm, https://icv2.com/articles/games/view/373/hasbro-results-cite-pokemon-decline

### 7. [medium] vote 2-1 (claim 6)

SUPPORTING OFFICIAL INPUT (weaker): Hasbro Games segment external net revenues for the nine months ended 2000-10-01 were $1,384,267K vs $906,615K in the 1999 comparable (which ended before the WOTC close, so contains zero WOTC revenue) — a ~$478M delta attributed primarily to WOTC trading card and role-playing games. Usable as an upper-bound input for 9-month-2000 WOTC (and hence Pokemon TCG) wholesale revenue. Tier=official; use the accession-level URL, as the per-document URL 404s.

**Evidence:** Segment table line 'Games 1,384,267 55,215 906,615 51,766' and the MD&A attribution sentence verified verbatim in the 10-Q. Ceiling logic is sound but not airtight: the delta equals WOTC 9-month revenue plus net change in legacy Games lines (Furby declined, interactive software grew), and Pokemon is a subset of WOTC. Split verifier vote (2-1) and a sibling claim about Q3-2000 Pokemon shipments rising was refuted — treat this as a bounding input only, not a Pokemon-specific figure.

Sources: https://www.sec.gov/Archives/edgar/data/46080/000004608000000016/0000046080-00-000016.txt

### 8. [high] vote 3-0 (claim 17)

PROVENANCE FOUND + UNIT CORRECTION for existing WOTC per-set anchors: the currently-held round numbers (Base Set 3B, Jungle 1.75B, Fossil 1.75B, Base Set 2 600M, Team Rocket 1.1B, Gym Heroes 850M, Gym Challenge 850M, Neo Genesis/Discovery/Revelation 700M each) originate verbatim from gottaketchumall's 2018-05-08 Elite Fourum thread, where they are explicitly labeled ALL LANGUAGES (not English-only), self-described as 'wild ass guesses', and calibrated so the ten WOTC figures sum to exactly 12.0B matching the end-2001 cumulative checkpoint, within a framework that assumed a 50/50 English/non-English split of the official 23.6B Mar-2017 total. Verdict on these anchors: origin CONFIRMED at tier=documented community estimate (not better), but if used as English-only anchors they must be roughly HALVED per the thread's own split assumption — the current model likely double-counts by treating all-language totals as English populations.

**Evidence:** Raw Discourse post 2 (2018-05-08T13:30:29Z) contains the exact list under 'My WOTC estimates (All languages)'. Post 1 states the derivation: '23.6/2 or 11.8 billion English cards printed across 72 sets' and cites the official pokemon.co.jp figure. Author's own caveat: 'Initial estimates are wild ass guesses that attempt to mesh with the chart above'. The exact numeric match with the currently-held anchors establishes origin.

Sources: https://www.elitefourum.com/t/an-elaborate-attempt-at-print-run-estimation-wip-5-8-18/20273, https://www.elitefourum.com/raw/20273/2

### 9. [high] vote 3-0 (claims 7, 14, 20 merged; each 3-0)

NEGATIVE FINDINGS (protective — no numeric anchor justified): (a) As of Dec 2020, and substantively still as of 2026, no statistically conclusive per-set 1st Edition vs Unlimited print figures exist for any WOTC set ('people have been asking this question for twenty years and there remain no statistically conclusive answers'); former-WOTC-employee interviews (Chris Nitz) discuss process but never quantities; therefore ALL circulating WOTC per-set 1E/Shadowless numbers, including the held base1-1st-Edition-1.5M-cards anchor, are guesses — verdict: no new evidence, keep flagged bare-guess. (b) The Skyridge 'single print run' story is marked [citation needed] on Bulbapedia and the Elite Fourum thread on it confirms no primary documentation exists — warrants only a qualitative smaller-than-peers prior, no number. (c) Wikipedia's The Pokemon Company article contains zero production figures and cannot corroborate any anchor.

**Evidence:** All three verified by direct fetch: the Elite Fourum quote is verbatim (qwachansey, 2020-12-17) with only satirical ratios in-thread; the May 2025 successor estimation thread self-labels as 'a fun thought experiment and not a definitive answer'; Bulbapedia's Skyridge sentence carries [citation needed] and the dedicated forum thread found no primary documentation; exhaustive grep of the Wikipedia article's wikitext found no card-count figures (only yen-denominated financials).

Sources: https://www.elitefourum.com/t/estimated-print-runs-1st-edition-unlimited/31119, https://bulbapedia.bulbagarden.net/wiki/Wizards_of_the_Coast, https://www.elitefourum.com/t/skyridge-print-runs/37434, https://en.wikipedia.org/wiki/The_Pok%C3%A9mon_Company

### 10. [high] vote 3-0 and 2-1 (claims 21, 22 merged)

BARE-GUESS-TIER RATIO (usable only as a wide prior, disputed in its own thread): a secondhand, from-memory report of an unnamed Konami rep's 'industry standard' that ~30% of a first print run is 1st Edition implies ~2.33 Shadowless cards per 1st Edition Base Set card (70/30). The ratio is not WOTC-specific, has zero independent input, no external corroboration exists for a 30% industry standard, and it is directly disputed in-thread (hammr7 cites a seller who moved ~200 1st Ed Base sets vs ~100 Shadowless sets, i.e. Shadowless possibly RARER than 1st Ed). Direction of consensus (Shadowless volume > 1st Ed) is itself contested. Tier=bare guess; if used at all, use as a ratio prior spanning roughly 0.5x to 2.3x Shadowless:1stEd.

**Evidence:** Both posts verified verbatim (cullers post 6 for the 30% rep claim; EnlightenedBulbasaur post 10 for the 2.33 derivation; hammr7 post 19 for the counter-evidence; note the 'will likely never be public' quote is from a third poster, jkanly, post 8). Confidence is high that this is what the source says and that bare-guess is the correct tier — not high confidence in the ratio itself.

Sources: https://www.elitefourum.com/t/shadowless-vs-1st-edition-print-runs/27547

### 11. [high] vote 3-0 (claims 15, 16 merged; each 3-0)

MODERN-ERA CAPACITY CONSTRAINTS (structural, not numeric): Throughout 2025 TPCi repeatedly stated it was printing at maximum capacity, and PokeBeach insiders identify printing capacity itself as the bottleneck — so calendar-2025 global production (~10B/yr per official figures) can be treated as approximately equal to the pre-expansion capacity ceiling, a hard constraint on any 2025 per-set English estimate (sets share a fixed pie; Prismatic Evolutions et al. cannot sum beyond it). Separately, Millennium Print Group (TPCi's printing subsidiary) signed a 1.27M sq ft lease at the Spark campus in Morrisville NC — the largest US manufacturing lease of 2025 per the developers — with the new 866K sq ft facility completing 2027 and full operations late 2028, implying roughly a doubling of capacity and therefore that per-set populations for 2028+ sets should NOT be extrapolated from the 2025 plateau.

**Evidence:** Tenant identity triple-sourced: WRAL (people with knowledge of the deal), the developer's own press release (1.27M sq ft, largest-2025 superlative), and MPG's official email confirmation to press. Max-capacity statements are TPCi's own (Jan-Mar 2025 Prismatic Evolutions statements), post-hoc corroborated by TPC's May 2026 admission that demand outpaced production capacity. Caveats: capacity was mildly time-varying during 2025; TPC figures are global all-language, so English share needs a separate allocation step; MPG also prints other TCGs.

Sources: https://www.pokebeach.com/2025/12/pokemons-millennium-print-group-signs-largest-u-s-manufacturing-lease-of-2025-will-occupy-1-27-million-square-foot-campus-to-print-cards, https://www.wral.com/business/pokemon-card-maker-expands-north-carolina-factory-2025/, https://www.trinitycapitaladvisors.com/news/spark-ls-secures-largest-u-s-manufacturing-lease-in-2025-1-27m-sf-establishing-major-regional-employment-hub, https://www.pokebeach.com/2025/12/millennium-print-group-officially-confirms-massive-new-printing-campus-expanding-capacity-for-pokemon-tcg-fans

### 12. [medium] vote 3-0 (claim 19; verifier evidence medium)

ORDINAL CONSTRAINTS ONLY for the 2007-2022 gap eras (no numeric per-set figures found anywhere for EX/DP/HGSS/BW/XY, per-set SWSH, or per-set SV): collector-observation ranking from a July 2026 Elite Fourum thread — HGSS printed more than DP; BW more than HGSS; early XY < late XY; SM overall high but late SM considerably lower; SWSH 'uber printed'. Two of five orderings are independently corroborated (SWSH via official 9.7B/11.9B annual figures vs ~1.5-2B/yr pre-2019; late-SM scarcity via the documented 2021 Cosmic Eclipse reprint necessitated by $200+ booster boxes). HGSS>DP and BW>HGSS are uncorroborated single-poster 'vibes'. Verdict on held era-average anchors (EX/DP 200M, HGSS 200M, BW 200M, XY 150M, SM 485M, SWSH 1.5B, SV 2.5B): no new numeric evidence; the ordinal ranking is partially inconsistent with holding HGSS=BW=DP=200M equal and suggests DP < HGSS < BW ordering within that band.

**Evidence:** Quote verified verbatim (poster thsigma, 2026-07-09, self-described 'crapshoot... vibes'). Verifier assigned medium; a companion claim that the thread contained no quantified analysis was refuted 0-3, so the thread should be re-checked for later numeric posts. Use only as soft ordering constraints with low weight, exactly as the source frames itself.

Sources: https://www.elitefourum.com/t/print-run-estimations-by-set-for-2010-2020/63246, https://www.pokebeach.com/2020/11/new-cosmic-eclipse-print-run-next-year-in-line-with-japan, https://www.pokebeach.com/2023/05/pokemon-tcg-sold-record-9-7-billion-cards-in-2022

## Caveats

Adversarial verification status: the SEC-filing anchors (claims on FY1999-2001 Hasbro/WOTC revenue) and the archive-verified cumulative checkpoints (25.7B, 27.2B, 34.1B, 64.8B correction, 85B dating) were verified against primary documents and are the strongest material; treat them as adversarially verified. Single-source or transcription-dependent items: the 28.8B (Sep 2019) and 25.7B-in-PokeBeach-table figures rest partly on PokeBeach's transcription (the PokeBeach article itself 403s to fetchers and was read via search-index copies — 25.7B is independently archive-verified but 28.8B is not); the 14B and 21.5B pre-2017 figures are stale press boilerplate reused for years and must be anchored to the START of their windows and treated as floors; PokeBeach itself notes TPC only tracked accurately from 2017. All official cumulative figures are 'over X' floors and are GLOBAL all-language production (post-2021 wording) or shipments (pre-2020 wording) — converting any of them to English-only per-set populations requires an English-share assumption that no source documents (the community's 50/50 WOTC-era split is itself an assumption). Revenue-to-volume conversion of the Hasbro anchors requires undocumented 1999-2001 wholesale pack pricing and returns/channel-glut timing adjustments (the 2001 writeoffs mean shipped != sold-through). The held WOTC per-set round numbers are now known to be all-language WAGs summing to the 12B end-2001 checkpoint — internally consistent but only one tier above pure guesses, and currently mis-unitized if the model treats them as English-only. Two claims were refuted in verification (a Q3-2000 shipment-growth claim and a claim that the 2010-2020 forum thread contained no quantified analysis) — do not rely on either direction of those without re-checking. Nothing numeric was found for the largest gaps: per-set EX through XY, per-set SWSH/SV (including Evolving Skies 2B, 151 4B, Celebrations 400M, Evolutions 400M, Team Rocket Returns 80M — all remain unsourced hobbyist numbers with no new evidence). Time-sensitivity: the 85B/Mar-2026 checkpoint is current as of 2026-07-21; the MPG capacity expansion means post-2027 production will break any plateau-based extrapolation.

## Open questions

- What is the actual English vs non-English production split by era? Every all-language-to-English conversion currently rests on the Elite Fourum 50/50 WOTC-era assumption; distributor documents, WOTC localization records, or language-share statements from TPCi would materially re-scale every WOTC anchor.
- Can documented 1999-2001 wholesale pack pricing (distributor price sheets, ICv2 trade data, court filings from the WOTC v. Nintendo era) be found to convert the audited $568M (2000), ~$500M (1999), and ~$100-286M (2001) revenue anchors into card-count ranges with defensible error bars?
- Does an archive.org or press-release copy of the official page exist confirming the 28.8B (Sep 2019) checkpoint directly, closing the last transcription-only gap in the 2017-2020 sequence?
- Do any per-set numeric figures for EX/DP/HGSS/BW/XY or per-set SWSH/SV exist in sources not yet mined — e.g., TPCi press releases with production statements, distributor allocation leaks, Millennium Print Group employment/output data, or the possibly-quantified later posts in the 2010-2020 Elite Fourum thread (whose emptiness claim was refuted 0-3)?

## Refuted during verification (do not rely on either direction)

- (1-2) Hasbro states that shipments of Pokemon-related product in Q3 2000 increased over Q3 1999 across all segments, enough to more than offset the decline in Star Wars revenue — confirming Pokemon TCG/product volume was still rising through at least Q3 2000 (the Base Set 2 / Team Rocket / Gym Heroes / Gy... — https://www.sec.gov/Archives/edgar/data/0000046080/000004608000000016/0001.txt
- (0-3) As of July 2026, no one in the Elite Fourum community has produced a quantified per-set print run analysis for 2010-2020 sets; the thread opener asks whether anyone has attempted it and announces an intent to try, with no numeric results posted in the thread. This means all per-set BW/XY/SM-era numb... — https://www.elitefourum.com/t/print-run-estimations-by-set-for-2010-2020/63246

---

## Appendix: post-report anchor-file changes (2026-07-21/22)

This report describes the state when `known_print_runs.json` grew 36 → 45
anchors. Later same-day and next-day changes it does NOT cover:

- **+3 anchors** beyond the 45: the 20B @ ~Nov-2013 rung and the 14B → Mar-2006
  re-dating both come from the follow-up Elite Fourum re-read (single sourcing
  pass, NOT 3-vote verified like the material above); 52.9B @ Mar-2023 was
  promoted from a notes field to a standalone anchor; the SEC-revenue-derived
  `english_window_total` anchor was constructed from this report's findings.
- **Credibility downgrades (2026-07-22 audit):** 13B @ Mar-2005 and 43.2B @
  Mar-2022 are transcription-tier (forum/blog-relayed press figures), not
  archive-verified — both downgraded from `official` to
  `well-sourced-estimate`.
- **ID fix:** the Shining Fates anchor was tagged `swsh4pt5`; the catalog id is
  `swsh45`. It had been silently dropped from every fit until 2026-07-22.
- **Unit tags:** all 18 per-set community anchors now carry
  `unit: cards_all_languages`.
