# True Variant Detection Reference

True variants share the same base artwork and card_id stem but differ in physical treatment (stamps, holo finish, border printing). They require visual detection because the reference image cannot distinguish them.

This document does NOT cover full art, alt art, rainbow rare, gold, BREAK, GX, mega, IR, SIR, etc. Those have different artwork and are already separate `card_id` entries in the database.

---

## Table of Contents

1. [1st Edition Stamp](#1st-edition-stamp)
2. [Shadowless vs Unlimited](#shadowless-vs-unlimited)
3. [1999-2000 Copyright (4th Print)](#1999-2000-copyright-4th-print)
4. [Holo vs Non-Holo](#holo-vs-non-holo)
5. [Reverse Holo](#reverse-holo)
6. [EX-Era Set Logo Stamp (Reverse Holo)](#ex-era-set-logo-stamp)
7. [Prerelease Stamp](#prerelease-stamp)
8. [Staff Stamp](#staff-stamp)
9. [Cosmos Holo](#cosmos-holo)
10. [Cracked Ice Holo](#cracked-ice-holo)
11. [McDonald's Confetti Holo](#mcdonalds-confetti-holo)
12. [Build & Battle / Pokemon Center Stamp](#build--battle--pokemon-center-stamp)
13. [Promo Stamp (Black Star)](#promo-stamp-black-star)
14. [Holo Swirl](#holo-swirl)

---

## 1st Edition Stamp

**What it is:** Small black circle containing "1" with "EDITION" text below, stamped on the card between the artwork and the text box on the left side.

**Sub-variants:**
- **Thick stamp** (early print runs): Bolder, darker circle and text
- **Thin stamp** (later print runs): Thinner line weight, slightly smaller
- **Grey stamp** (transitional/error): Lighter grey ink instead of solid black

**Detection region:** Left side, x: 2-24%, y: 44-65% (wide search). Tight: x: 3-15%, y: 53-67%.

**What to look for:**
- OCR for "1st" and/or "EDITION" text
- Dark circular blob (circularity >= 0.65) as supporting evidence
- HoughCircles on tight region for small dark circles

**Sets:** base1, base2, base3, base5 (Team Rocket), gym1, gym2, neo1-neo4

**Price impact:** Massive. 1st Edition Base Set holos command 10-100x Unlimited prices. 1st Edition Neo sets 2-5x. Thick vs thin stamp matters for grading but not for pricing category.

**Detection method:** `stamp_ocr` (RapidOCR on region, upscaled 3x with padding). Implemented in `cardprice/ml/stamp_detection.py::_check_1st_edition()`. Confidence: OCR both tokens = 0.95, one token = 0.85, circle + digit = 0.70.

**Current status:** Implemented and working.

---

## Shadowless vs Unlimited

**What it is:** Base Set cards printed between the 1st Edition and standard Unlimited runs lack the dark drop shadow on the right and bottom borders of the card frame.

**Detection region:**
- Right edge strip: x: 90-100%, y: 0-100%
- Bottom edge strip: x: 0-100%, y: 90-100%

**What to look for:** Compare mean brightness of rightmost 3% of card vs next 3% inward. Unlimited (shadow) cards show a >15 pixel value drop in the outer strip. Shadowless cards have uniform border brightness.

**Sets:** base1 only

**Price impact:** High. Shadowless Charizard is worth 5-20x Unlimited. Common/uncommon Shadowless cards 2-5x Unlimited.

**Detection method:** `edge_gradient` -- pixel brightness comparison on right/bottom border strips. Defined in `data/stamp_positions.json` but detection function not yet fully implemented as a standalone checker.

**Current status:** Region and method defined. Needs a dedicated `_check_shadowless()` function in `stamp_detection.py`.

---

## 1999-2000 Copyright (4th Print)

**What it is:** Base Set 4th print run has a copyright line reading "1999-2000" instead of just "1999". These are sometimes called "4th print" or "UK print" cards. Same artwork, same border shadow as Unlimited.

**Detection region:** Bottom footer area, x: 5-50%, y: 93-100%

**What to look for:** OCR the copyright line at the bottom of the card. Look for "1999-2000" vs "1999" text.

**Sets:** base1 only (4th print run)

**Price impact:** Low. Generally 1.1-1.5x standard Unlimited. Mostly a collector curiosity rather than a major price driver.

**Detection method:** `ocr` -- RapidOCR on the footer region looking for the date string. Text is small and may require 3-4x upscaling for reliable detection at binder photo resolution.

**Current status:** Not implemented.

---

## Holo vs Non-Holo

**What it is:** Same card number and artwork, but the holo version has a holographic foil pattern on the artwork area. The non-holo version is flat-printed.

**Detection region:** Artwork area, x: 10-90%, y: 12-56%

**What to look for:**
- Saturation variance in artwork region (holo: sat_std >= 33, non-holo: sat_std ~15-30)
- Hue spatial noise via Laplacian (holo: ~50-150+, non-holo: ~5-30)
- Hue spread across multiple bins at high saturation

**Sets by era:**
- **WotC (base1-base5, gym1-gym2, neo1-neo4):** Rare holos had both holo and non-holo printings (same card number). 1st Ed holos exist.
- **Team Rocket (base5):** Cards #1-14 are holo, #18-31 are non-holo versions of the same Pokemon with same art
- **Expedition/Aquapolis/Skyridge (ecard1-ecard3):** H-numbered cards are holo versions of regular-numbered cards
- **EX era through SV era:** All main sets have holo rares paired with non-holo commons/uncommons (different card numbers = different DB entries, NOT a variant). Reverse holo is the variant here.

**Price impact:** Varies enormously. WotC holo rares are 5-50x their non-holo counterparts. Modern era holo vs non-holo is typically 1.5-3x.

**Detection method:** `holo_detector` -- HSV analysis comparing artwork region vs body region. Implemented in `cardprice/ml/holo_detector.py::detect_holo_type()`. Measures sat_std, hue_std, hue_spread, and spatial_noise.

**Limitations:** Single-photo analysis is fundamentally limited. Holo effects are angle-dependent and may not be visible in a binder page photo taken with even lighting. "Normal" at high confidence is more reliable than "holofoil" detection. See `holo_detector.py` module docstring for full limitations.

**Current status:** Implemented. Confidence values documented. Works best under fluorescent or angled lighting.

---

## Reverse Holo

**What it is:** Holographic foil applied to the card body (border, name bar, text box) while the artwork area itself is NOT holographic. The reverse of standard holofoil.

**Detection region:**
- Body regions: name bar (y: 3-11%), text box (y: 58-92%), left border (x: 2-10%), right border (x: 90-98%)
- Artwork region measured for comparison (should be LOW holo signal)

**What to look for:**
- Body region shows HIGH holo signal (sat_std >= 33, hue_std >= 22, combined_score >= 6.0)
- Artwork region shows LOW holo signal
- Body/art score ratio exceeds 1.25

**Pattern variations by era:**
- **Legendary Collection (base6, 2002):** Unique "fireworks" holographic pattern
- **E-Card (ecard1-ecard3, 2002-2003):** "Cosmic" holographic pattern
- **EX era (ex1-ex16, 2003-2007):** Standard parallel-line holo. ex7-ex16 add set logo stamp (see next section)
- **DP through SM (2007-2019):** Standard reverse holo pattern
- **SWSH (2020-2023):** Standard reverse holo
- **SV (2023+):** New reverse holo pattern, slightly different texture

**Sets:** Almost all main expansion sets from base6 (Legendary Collection) onward. NOT present in: base1-base5, gym1-gym2, neo1-neo4 (pre-reverse-holo era).

**Price impact:** Typically 1.5-3x normal for commons/uncommons. For rares, reverse holo can be worth less, equal, or more than regular holo depending on supply. Legendary Collection reverse holos are premium (5-20x) due to limited print run and distinctive pattern.

**Detection method:** `holo_detector` -- same HSV analysis as regular holo, but comparing body vs artwork regions. Implemented in `cardprice/ml/holo_detector.py`. Body-dominant signal = reverse_holofoil.

**Current status:** Implemented. Same single-photo limitations as regular holo detection.

---

## EX-Era Set Logo Stamp

**What it is:** Semi-transparent set name text stamped on the bottom-right of the artwork area on reverse holo cards from EX Team Rocket Returns through EX Power Keepers.

**Detection region:** Artwork bottom-right, x: 50-90%, y: 30-58%

**What to look for:** OCR for set-specific text:
| Set | Stamp Text |
|-----|-----------|
| ex7 | TEAM ROCKET RETURNS |
| ex8 | DEOXYS |
| ex9 | EMERALD |
| ex10 | UNSEEN FORCES |
| ex11 | DELTA SPECIES |
| ex12 | LEGEND MAKER |
| ex13 | HOLON PHANTOMS |
| ex14 | CRYSTAL GUARDIANS |
| ex15 | DRAGON FRONTIERS |
| ex16 | POWER KEEPERS |

**Sets:** ex7-ex16 only. Earlier EX sets (ex1-ex6) have unstamped reverse holos.

**Price impact:** Stamped reverse holos are the standard for these sets -- unstamped would be an error card. The stamp itself does not create a price premium but confirms the card is reverse holo vs normal.

**Detection method:** `stamped_detector` -- RapidOCR on artwork bottom-right region matching against known set words. Implemented in `cardprice/ml/stamp_detection.py::_check_ex_set_stamp()`. Confidence: 2+ set words = 0.90, 1 word = 0.80, generic stamp words = 0.75.

**Current status:** Implemented and working.

---

## Prerelease Stamp

**What it is:** "PRERELEASE" text stamp overlaid on card artwork. Given to participants at set prerelease events (Build & Battle tournaments).

**Detection region:** Artwork bottom-right, x: 55-95%, y: 30-58%. Alt position (less common): left side x: 5-45%, y: 30-58%.

**What to look for:**
- OCR for "PRERELEASE" or "PRE-RELEASE" text
- Known OCR confusions: "PRENELEMEE", "PRERELEAS", "PRE RELEASE"
- Fuzzy matching threshold >= 0.70

**Sets:** All eras from EX onward. WotC prerelease promos exist but are separate card_ids (e.g., promo Aerodactyl, Misty's Seadra, Dark Gyarados).

**Era-specific appearance:**
- **EX era (2003-2007):** Large gold text stamp
- **DP/Platinum (2007-2010):** Gold text stamp with set logo
- **HGSS/BW (2010-2013):** Gold text stamp
- **XY (2014-2016):** Set logo stamp on artwork (replaces text in some sets)
- **SM (2017-2019):** Build & Battle prerelease stamp
- **SWSH (2020-2023):** Build & Battle stamp with pokeball + trainer silhouette
- **SV (2023+):** Build & Battle stamp, similar to SWSH

**Price impact:** 2-10x normal card price depending on the card and era. Staff variants (see below) are 5-50x.

**Detection method:** `ocr` -- RapidOCR with 6 preprocessing strategies (raw, unsharp, CLAHE, adaptive threshold, inverted, Otsu). Implemented in `scripts/detect_prerelease.py` with 100% precision and 66.7% recall on test set. Faint/transparent stamps on busy artwork backgrounds are the main failure mode.

**Current status:** Implemented as standalone script. Integrated into stamp_positions.json. Recall limited by OCR difficulty with semi-transparent stamps.

---

## Staff Stamp

**What it is:** Gold "STAFF" text stamp on prerelease/event cards given to tournament staff. Always appears alongside or near a PRERELEASE stamp.

**Detection region:** Upper-right artwork area, x: 55-95%, y: 20-45%

**What to look for:** OCR for "STAFF" text in gold coloring, positioned above or near the PRERELEASE stamp.

**Sets:** Any set that has prerelease cards (DP era onward). Much rarer than standard prerelease.

**Price impact:** Very high. Staff prerelease cards are typically 5-50x the price of standard prerelease cards, and 20-200x the normal card. Low supply drives prices.

**Detection method:** `ocr` -- RapidOCR looking for "STAFF" text. Should be checked whenever a prerelease stamp is detected. Defined in `data/stamp_positions.json`.

**Current status:** Region and method defined in stamp_positions.json. Not yet implemented as a detection function.

---

## Cosmos Holo

**What it is:** A distinctive holographic pattern of scattered sparkle dots (galaxy/cosmos pattern) across the artwork area. Different from standard parallel-line holographic foil.

**Detection region:** Full artwork area, x: 5-95%, y: 8-52%

**What to look for:**
- Scattered high-saturation sparkle points (not continuous holo lines)
- HSV hue spread analysis showing dispersed bright points
- Pattern is more "random dots" than "parallel lines"

**Sets by era:**
- **XY (2014-2016):** Blister-exclusive cards
- **SM (2017-2019):** Blister/product exclusive cards
- **SWSH (2020-2023):** Product exclusive cards
- **SV (2023+):** All rare cards in main sets use cosmos holo pattern

**Price impact:** Low in SV era (standard for rares). In XY/SM/SWSH eras, cosmos holo blister exclusives can be 1.5-3x the standard holo version.

**Detection method:** `holo_detector` -- HSV hue spread analysis adapted for scattered sparkle pattern. Defined in `data/stamp_positions.json`. Currently uses the same holo_detector as standard holo, which may not distinguish cosmos from standard holo patterns.

**Current status:** Defined in stamp_positions.json. The holo_detector does not currently differentiate between cosmos and standard holo patterns -- both register as "holofoil". Distinguishing them would require texture analysis (dot pattern vs line pattern).

---

## Cracked Ice Holo

**What it is:** A distinctive "cracked ice" or "shattered glass" holographic pattern used on theme deck exclusive holos. Different texture from standard cosmos/parallel holo.

**Detection region:** Artwork area, x: 10-90%, y: 12-56%

**What to look for:**
- Holographic pattern with angular, geometric facets (like cracked ice)
- Differs from smooth parallel-line holo and scattered cosmos dots
- Only appears on specific theme deck holo cards

**Sets:** HGSS through BW era theme decks (2010-2013). Some XY theme deck holos also use this pattern.

**Price impact:** Generally 0.5-1.5x the standard holo version. Sometimes worth less than standard holo (theme deck cards are perceived as lower quality), sometimes comparable.

**Detection method:** Would require texture classification to distinguish from standard holo. The current holo_detector would classify these as "holofoil" without distinguishing the pattern type.

**Current status:** Not implemented. Would need a texture classifier (CNN or template-based) to distinguish cracked ice from standard parallel/cosmos holo patterns.

---

## McDonald's Confetti Holo

**What it is:** A unique confetti-pattern holographic foil used exclusively on McDonald's Happy Meal promotional Pokemon cards. The pattern features small, irregular holographic shapes scattered across the entire card (not just artwork).

**Detection region:** Full card, x: 0-100%, y: 0-100% (confetti pattern covers entire card surface)

**What to look for:**
- Holographic pattern covering the ENTIRE card (not just artwork or just body)
- Small, irregularly shaped holographic elements (confetti/sprinkle pattern)
- Both artwork AND body regions show elevated holo signal

**Sets:** mcd11, mcd12, mcd14-mcd19, mcd21, mcd22 (McDonald's collections, various eras)

**Price impact:** Low individually (most cards $1-5). Complete sets or sealed packs can be more valuable.

**Detection method:** The current holo_detector would detect elevated signal in BOTH artwork and body regions (unlike reverse holo where only body is hot). This "both hot, ambiguous" case currently defaults to "holofoil" at low confidence. A McDonald's set detection (via set_id) combined with full-card holo signal would be more accurate.

**Current status:** Not specifically implemented. The holo_detector's "both regions hot" path partially handles it but without McDonald's-specific classification.

---

## Build & Battle / Pokemon Center Stamp

**What it is:** Modern stamps used on special distribution cards:
- **Build & Battle (SWSH/SV):** Pokeball + trainer silhouette stamp on prerelease Build & Battle box promos
- **Pokemon Center exclusive (SV):** Pokemon Center logo stamp on cards exclusive to Pokemon Center retail

**Detection region:**
- Build & Battle: Same as prerelease stamp region, x: 55-95%, y: 30-58%
- Pokemon Center: Varies; typically bottom area of card near set symbol

**What to look for:**
- Build & Battle: Pokeball icon with trainer silhouette, distinct from plain text "PRERELEASE" stamps
- Pokemon Center: Pokeball-shaped logo with "Pokemon Center" text

**Sets:**
- Build & Battle: SWSH and SV main sets
- Pokemon Center: Select SV sets (sv1+)

**Price impact:**
- Build & Battle: 2-5x normal card (same as modern prerelease)
- Pokemon Center: 1.5-3x normal, varies by card popularity

**Detection method:** OCR + template matching. Build & Battle stamps can be detected by the existing prerelease OCR pipeline. Pokemon Center stamps would need a dedicated template or OCR matcher.

**Current status:** Build & Battle partially covered by prerelease detection. Pokemon Center stamp not implemented.

---

## Promo Stamp (Black Star)

**What it is:** A black five-pointed star symbol that replaces the normal set symbol on promotional cards. Various eras use different promo identifiers.

**Detection region:** Set symbol area, x: 42-62%, y: 86-97% (bottom center of card)

**What to look for:**
- **WotC era:** Large black star with "PROMO" text, low solidity contour (star shape ~0.25-0.40 solidity)
- **EX/DP/BW/XY/SM era:** Star or promo marking in set symbol position
- **SWSH/SV era:** Stylized promo set symbol (SWSHP/SVP)

**Sets:** basep, np, dpp, hsp, bwp, xyp, smp, swshp, svp (all promo sets)

**Price impact:** Varies enormously by individual promo. Some promos are worthless ($0.10), others are extremely valuable ($500+). The promo stamp itself just identifies the card as a promo -- price depends on which promo it is.

**Detection method:** `promo_detector` / `symbol_match` -- combination of OCR for "PROMO" text and dark star-shaped contour detection. Implemented in `cardprice/ml/stamp_detection.py` with checkers for `_check_black_star_promo()`, `_check_modern_promo()`, and `_check_promo_stamp()`.

**Current status:** Implemented for all eras. Multiple detection strategies (OCR, blob detection, HoughCircles) with era-appropriate routing.

---

## Holo Swirl

**What it is:** A visible swirl pattern in the holographic foil, caused by the manufacturing process. Not an intentional variant, but a manufacturing artifact that collectors value. The swirl appears as a concentrated spiral of the holographic lines.

**Detection region:** Artwork area (holo cards only), x: 10-90%, y: 12-56%

**What to look for:**
- Spiral/circular concentration of holographic line pattern within the artwork
- Only present on holo cards (not reverse holo or non-holo)
- Position within the artwork varies randomly (manufacturing artifact)

**Sets:** Any set with holographic cards, all eras. More common/visible on certain print runs.

**Price impact:** 1.5-3x for cards with a prominent, well-centered swirl. A swirl positioned directly on the Pokemon's face/body commands the highest premium. Subtle or edge swirls have minimal impact.

**Detection method:** Not trivially detectable from binder photos. Would require:
1. First confirming the card is holofoil (via holo_detector)
2. Analyzing the holo pattern in the artwork region for spiral/circular line concentrations
3. Phase correlation or Fourier analysis to detect circular frequency patterns

**Current status:** Not implemented. Would be extremely difficult to detect from single binder page photos due to angle/lighting dependency. Higher-resolution close-up photos would be needed.

---

## Detection Priority Matrix

Ordered by price impact and detection feasibility:

| Priority | Variant | Price Impact | Detection Feasibility | Status |
|----------|---------|-------------|----------------------|--------|
| P0 | 1st Edition stamp | 2-100x | High (OCR) | Implemented |
| P0 | Shadowless | 5-20x | High (pixel analysis) | Region defined, needs function |
| P0 | Holo vs Non-holo | 5-50x | Medium (lighting dependent) | Implemented |
| P1 | Reverse holo | 1.5-3x | Medium (lighting dependent) | Implemented |
| P1 | Prerelease stamp | 2-10x | Medium (OCR, faint stamps hard) | Implemented |
| P1 | Staff stamp | 5-50x | Medium (OCR) | Defined, not implemented |
| P2 | EX set logo stamp | Confirms rev holo | High (OCR) | Implemented |
| P2 | Promo stamp | Identifies card | High (multiple methods) | Implemented |
| P2 | Cosmos vs standard holo | 1.5-3x (older eras) | Low (texture analysis needed) | Not implemented |
| P3 | Cracked ice holo | 0.5-1.5x | Low (texture classifier needed) | Not implemented |
| P3 | McDonald's confetti | Low | Low (need full-card holo check) | Not implemented |
| P3 | 1999-2000 copyright | 1.1-1.5x | Medium (footer OCR) | Not implemented |
| P3 | Build & Battle stamp | 2-5x | Medium (OCR/template) | Partially via prerelease |
| P4 | Holo swirl | 1.5-3x | Very low (needs close-up) | Not implemented |

---

## Key Implementation Files

- `cardprice/ml/variant_detector.py` -- Full variant detection pipeline orchestrator
- `cardprice/ml/stamp_detection.py` -- Era-gated stamp detection (1st Ed, EX stamp, promo)
- `cardprice/ml/holo_detector.py` -- Holographic type detection (normal/holo/reverse)
- `cardprice/ml/variant_tree.py` -- Query variant tree JSON for era/set applicability
- `cardprice/ml/era_detector.py` -- Map set_id to era number (1-9)
- `data/variant_tree.json` -- Full variant definitions by era with detection methods
- `data/stamp_positions.json` -- Stamp types, regions, and detection details
- `scripts/detect_prerelease.py` -- Standalone prerelease stamp detection script
