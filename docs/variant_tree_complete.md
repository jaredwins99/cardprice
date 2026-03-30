# Pokemon TCG Variant Tree -- Complete Reference

Comprehensive variant catalog covering all 10 eras (1999-present).
Structured for implementation: each variant includes detection region,
visual signature, applicable sets, and pricing impact.

Last updated: 2026-03-30

---

## How to Read This Document

Each variant entry follows this schema:

```
ERA -> Set Range -> Variant Type
  Detection region:      [x0, y0, x1, y1] as fractions of card dimensions
  Visual signature:      What to look for in the image
  Applicable sets:       Which set IDs have this variant
  Detection method:      Algorithm/module to use
  TCGCSV subtype:        Whether this variant has its own price row (affects pricing)
  Price impact:          Relative price premium over base variant
  Detection feasibility: Whether this variant can be detected from binder scans
```

**Detection Feasibility Key**:
- **Detectable** -- reliably detectable from standard binder page scans
- **Maybe** -- depends on image quality, lighting, sleeve type, or card positioning
- **Not detectable** -- requires close-up photography, physical inspection, or video

Coordinate system: origin at top-left of card image. x increases rightward,
y increases downward. Values are fractions of card width/height (0.0 to 1.0).

Standard card regions for reference:
- Name bar: y=0.02-0.10, x=0.06-0.94
- Artwork: y=0.10-0.56, x=0.10-0.90
- Text box: y=0.58-0.92, x=0.08-0.92
- Border strips: outer 6% on each side
- Set symbol: y=0.86-0.97, x=0.42-0.62

---

## TCGCSV Subtype Reference

Pokemon TCG uses exactly 7 TCGCSV subtypes that drive separate pricing:

| TCGCSV SubtypeName     | Our Variant Key          | Notes                              |
|------------------------|--------------------------|------------------------------------|
| Normal                 | `normal`                 | Flat print, no foil                |
| Holofoil               | `holofoil`               | Holo artwork only                  |
| Reverse Holofoil       | `reverse_holofoil`       | Holo on border/text, not artwork   |
| 1st Edition            | `1st_edition`            | 1st Ed stamp, non-holo             |
| 1st Edition Holofoil   | `1st_edition_holofoil`   | 1st Ed stamp + holo artwork        |
| Unlimited              | `unlimited`              | Explicitly Unlimited (= normal)    |
| Unlimited Holofoil     | `unlimited_holofoil`     | Unlimited holo (= holofoil)        |

Visual-only variants NOT tracked as separate TCGCSV subtypes (share a price row):
- `shadowless` / `shadowless_holofoil` -- Base Set only, massive price premium
- `full_art` -- BW+ era, priced under Holofoil subtype
- `gold` -- SM+ era, priced under Holofoil subtype
- `rainbow_rare` -- SM+ era, priced under Holofoil subtype
- `promo` -- priced as its own product (different TCGCSV productId)

---

## ERA 1: WotC Base (1999-2000)

**Sets**: base1 (Base Set), base2 (Jungle), base3 (Fossil), base4 (Base Set 2), base5 (Team Rocket)
**Set IDs**: `base1`, `base2`, `base3`, `base4`, `base5`, `basep`

### 1.1 1st Edition Stamp

```
Variant:           1st_edition / 1st_edition_holofoil
Detection region:  [0.02, 0.44, 0.24, 0.65]  (wide search)
Tight region:      [0.03, 0.53, 0.15, 0.67]  (focused)
Applicable sets:   base1, base2, base3, base5
TCGCSV subtype:    YES -- "1st Edition" / "1st Edition Holofoil"
Price impact:      5x-100x over Unlimited depending on card
Detection feasibility: Maybe -- stamp is small, OCR works ~85% at binder resolution
```

**Visual signature**: Small black circle (~30-40px at binder resolution) containing
the numeral "1" with "EDITION" text below it. Located on the left side of the card,
just below the artwork frame, to the left of the HP text. The stamp center sits at
approximately 8% from left edge, 59% from top.

**Detection method**: `stamp_ocr` (primary) + `_has_dark_circular_blob` (secondary)
1. PaddleOCR on stamp region looking for "1st" and/or "edition" text
2. Dark circular blob detection via contour analysis (circularity >= 0.65, area 3-30% of region)
3. HoughCircles on tight region for small dark circles
4. Confidence: OCR match = 0.95, blob+hough = 0.90, blob only = 0.85

**OCR confusion substitutions**: "l" -> "1", "|" -> "1" (stamp text is small, low-contrast)

**1st Edition stamp variants by era**:
- **Thick stamp** (early Base Set print runs): Bolder, thicker lines in the "1" and circle
- **Thin stamp** (later print runs and Jungle/Fossil): Thinner, more refined lines
- **Grey stamp** (some Team Rocket): Lighter grey instead of solid black

These sub-variants are NOT separately priced but affect grading/collector value.

### 1.2 Shadowless

```
Variant:           shadowless / shadowless_holofoil
Detection region:  Right edge: [0.90, 0.00, 1.00, 1.00]
                   Bottom edge: [0.00, 0.90, 1.00, 1.00]
Applicable sets:   base1 ONLY
TCGCSV subtype:    NO -- shares Normal/Holofoil subtype
Price impact:      3x-50x over Unlimited (Charizard shadowless is >$5000)
Detection feasibility: Maybe -- edge gradient analysis works but sleeve edges can interfere
```

**Visual signature**: No dark gradient/shadow on right and bottom card borders.
Unlimited cards have a visible dark shadow cast to the right and below the card
frame edges. Shadowless cards have clean, evenly-colored borders with no gradient.

**Detection method**: `edge_gradient`
- Compare mean brightness of rightmost 3% vs next 3% inward
- Shadow cards show a >15 pixel value drop in the outer strip
- Shadowless cards have <5 pixel value difference

**Historical context**: Printed between the 1st Edition run and standard Unlimited.
Only exists for Base Set (base1). The shadow was added starting with Unlimited printing
and continued through all subsequent WotC sets.

**Additional shadowless tells** (for confirming ambiguous edge gradient):
- Copyright line: shadowless cards show "1999" only; unlimited show "1999-2000"
  - Region: bottom 5% of card, x=0.10-0.90
- Font weight: shadowless HP text is slightly thinner
- Border color: shadowless has a slightly different yellow tone

### 1.3 1999-2000 Copyright Date

```
Variant:           Sub-variant of Unlimited (distinguishes early vs late Unlimited)
Detection region:  Copyright line: ~[0.10, 0.95, 0.90, 1.00]
Applicable sets:   base1 ONLY
TCGCSV subtype:    NO -- same product as Unlimited
Price impact:      None (informational only, helps confirm shadowless status)
Detection feasibility: Not detectable -- copyright text too small at binder resolution
```

**Visual signature**: Copyright text at card bottom reads either:
- "1999" only = 1st Edition or Shadowless print
- "1999-2000" = Unlimited print (later run)

**Detection method**: OCR on bottom 5% of card. Look for "2000" text presence.

### 1.4 Unlimited vs 1st Edition

```
Variant:           unlimited / unlimited_holofoil
Detection region:  Same as 1st Edition (absence of stamp)
Applicable sets:   base1, base2, base3, base5
TCGCSV subtype:    YES -- "Unlimited" / "Unlimited Holofoil"
Price impact:      Base price (1st Edition is the premium)
Detection feasibility: Detectable -- default when no 1st Ed stamp found
```

**Detection method**: Default -- if no 1st Edition stamp detected and not shadowless,
the card is Unlimited. For base1, must also rule out Shadowless via edge gradient.

### 1.5 Holofoil (WotC pattern)

```
Variant:           holofoil
Detection region:  Artwork area: [0.10, 0.10, 0.90, 0.55]
Applicable sets:   All WotC sets (rare cards only)
TCGCSV subtype:    YES -- "Holofoil"
Price impact:      2x-20x over Normal for same card
Detection feasibility: Not detectable -- holo shimmer invisible through binder sleeves
```

**Visual signature**: Holographic "star" or "cosmos" pattern visible on the artwork
area only. Border and text box remain flat/matte. The pattern creates rainbow
prismatic reflections that shift with viewing angle.

**Detection method**: `holo_detector`
- Measure hue spread (distinct hue bins at high saturation) in artwork region
- Measure hue spatial noise (Laplacian of hue channel in non-edge flat regions)
- Combined score = hue_spread * (noise / threshold)
- Artwork score must exceed body score by ratio >= 1.3

**Critical limitation**: Holofoil detection is UNRELIABLE through binder sleeves.
Plastic dampens prismatic micro-reflections. Detection works on direct card photos
but fails at ~0% accuracy through sleeves. See open_problems.md Problem 3.

**Alternative detection (reverse holo only)**: High-frequency RGB channel decorrelation
in the name bar region. Threshold: hf_decorr >= 0.055. Achieves 100% accuracy for
reverse holo vs not-reverse-holo on binder scans. Does NOT help distinguish holofoil
from normal. Module: `detect_reverse_holo.py`.

### 1.6 Red Cheeks Pikachu (base1 only)

```
Variant:           Visual sub-variant of base1-58
Detection region:  Cheek area of artwork: ~[0.35, 0.25, 0.65, 0.45]
Applicable sets:   base1 ONLY (card #58)
TCGCSV subtype:    NO -- same product
Price impact:      Red cheeks: slight premium ($5-15 vs $3-8 for yellow cheeks)
Detection feasibility: Maybe -- color difference is subtle but measurable in artwork
```

**Visual signature**: Pikachu's cheek circles are red instead of yellow.
Early Base Set printings used red; later printings used yellow to match the
correct game character design.

**Detection method**: Not implemented. Would require:
- Identify card as base1-58 Pikachu
- Extract cheek region from artwork
- Measure hue: red cheeks have H=0-10, yellow cheeks have H=20-35

### 1.7 Vulpix HP Error (base2 Jungle)

```
Variant:           Error card (base2 Vulpix #68)
Detection region:  HP text area: ~[0.70, 0.03, 0.95, 0.10]
Applicable sets:   base2 (Jungle)
TCGCSV subtype:    NO -- same product
Price impact:      Error version: $10-30 premium
Detection feasibility: Not detectable -- HP text too small at binder resolution
```

**Visual signature**: HP printed as "50" instead of correct "60" on early print runs.

**Detection method**: Not implemented. Would require OCR on HP region.

### 1.8 No-Symbol Jungle Error (base2)

```
Variant:           Error card (various base2 cards)
Detection region:  Set symbol area: [0.42, 0.86, 0.62, 0.97]
Applicable sets:   base2 (Jungle)
TCGCSV subtype:    NO -- same product
Price impact:      $5-50 depending on card rarity
Detection feasibility: Maybe -- set symbol absence detectable if resolution is adequate
```

**Visual signature**: Jungle set symbol (palm tree/leaf) is missing from the card.
Early Jungle print run omitted the set symbol on some cards.

**Detection method**: Not implemented. Would require set symbol detection
and absence verification.

### 1.9 Fossil Errors (base3)

```
Variant:           Various misprints in Fossil set
Detection region:  Varies by error type
Applicable sets:   base3 (Fossil)
TCGCSV subtype:    NO -- same product
Price impact:      $5-30 depending on error
Detection feasibility: Not detectable -- requires close-up inspection
```

Known Fossil errors:
- Missing/wrong set symbol on some cards
- Zapdos HP error on early prints
- Various text/energy errors on specific cards

### 1.10 Holo Swirl

```
Variant:           Visual collector variant (any holo card)
Detection region:  Artwork area: [0.10, 0.10, 0.90, 0.55]
Applicable sets:   Any holo card from any era
TCGCSV subtype:    NO -- same product
Price impact:      $5-50 premium for visible swirl pattern
Detection feasibility: Not detectable -- requires direct angled lighting
```

**Visual signature**: A spiral/swirl pattern visible in the holographic foil,
caused by the foil manufacturing process. Not all holo cards have visible swirls;
it depends on where the foil sheet was cut relative to the swirl pattern.

**Detection method**: Not implemented. Would require:
- Confirm card is holofoil
- Detect spiral/circular pattern in holo foil texture
- Extremely difficult from binder scans

### 1.11 Base Set 2 Cosmos Holo (base4)

```
Variant:           holofoil (different holo pattern than base1)
Detection region:  Artwork area: [0.10, 0.10, 0.90, 0.55]
Applicable sets:   base4 (Base Set 2)
TCGCSV subtype:    YES -- "Holofoil"
Price impact:      Generally worth less than base1 holos
Detection feasibility: Not detectable -- same holo limitation through sleeves
```

**Visual signature**: "Cosmos" holographic pattern (small scattered dots/sparkles)
instead of the Base Set's "star" pattern. The cosmos pattern creates a galaxy-like
sparkle effect across the artwork.

**Detection method**: `holo_detector` -- same as standard holo. The pattern difference
(star vs cosmos) is not distinguished by the current detector.

### 1.12 Machamp Always-1st-Edition (base1)

```
Variant:           1st_edition_holofoil (base1-8/1st_edition_holofoil)
Detection region:  1st Ed stamp region: [0.02, 0.44, 0.24, 0.65]
Applicable sets:   base1 ONLY (card #8, Machamp)
TCGCSV subtype:    YES -- "1st Edition Holofoil"
Price impact:      Minimal -- almost all Machamps are 1st Ed ($5-15)
Detection feasibility: Maybe -- stamp is present but "true" 1st Ed requires card stock analysis
```

**Special case**: Machamp was only distributed in Base Set Starter Decks, which
were ALL stamped 1st Edition regardless of print run (even Unlimited-era decks).
A "true" 1st Edition Machamp must be from the actual 1st Edition print run
(thicker stamp, different card stock) but this is nearly impossible to verify
from scans. For pricing purposes, all Base Set Machamps are 1st Edition.

---

## ERA 2: WotC Gym/Rocket (2000)

**Sets**: gym1 (Gym Heroes), gym2 (Gym Challenge), base5 (Team Rocket)
**Set IDs**: `gym1`, `gym2`, `base5`

### 2.1 Holo/Non-Holo Pairs (Team Rocket)

```
Variant:           holofoil vs normal for same card number
Detection region:  Artwork area: [0.10, 0.10, 0.90, 0.55]
Applicable sets:   base5 (Team Rocket)
TCGCSV subtype:    YES -- separate products for holo vs non-holo
Price impact:      Holo version 3x-10x over non-holo
Detection feasibility: Not detectable -- holo vs non-holo indistinguishable through sleeves
```

**Visual signature**: Team Rocket rare cards exist as both holo and non-holo
versions with IDENTICAL card numbers. The ONLY visual difference is the
holographic pattern on artwork.

**Key cards**: Dark Arbok, Dark Blastoise, Dark Charizard, Dark Dragonite,
Dark Dugtrio, Dark Golbat, Dark Gyarados, Dark Hypno, Dark Machamp,
Dark Magneton, Dark Slowbro, Dark Vileplume, Dark Weezing, Here Comes Team Rocket!

**Detection method**: `holo_detector` -- must detect holo shimmer. Through binder
sleeves this is unreliable (see ERA 1 limitations). DB lookup can narrow:
if card number is in the holo/non-holo pair list, both variants exist.

### 2.2 Dark Raichu Secret Rare (base5)

```
Variant:           Secret rare (card #83 in a 82-card set)
Detection region:  Card number region: ~[0.60, 0.90, 0.95, 0.98]
Applicable sets:   base5 (Team Rocket)
TCGCSV subtype:    YES -- separate product
Price impact:      $100-500+ (rare secret card)
Detection feasibility: Detectable -- DINOv2 matches the specific card reference
```

**Visual signature**: Card number 83/82 (exceeds set total). Holofoil.
Features Raichu in Team Rocket artwork style.

**Detection method**: Card identification via name OCR + DINOv2 matching against
the specific reference. Card number OCR could confirm (83/82) but text is too
small at binder resolution.

### 2.3 Dark Dragonite Variants (base5)

```
Variant:           6 distinct printings of Dark Dragonite
Applicable sets:   base5 (Team Rocket)
TCGCSV subtype:    Partially -- 1st Ed Holo, 1st Ed Non-Holo, Unl Holo, Unl Non-Holo
Price impact:      1st Ed Holo: $50-200, Non-Holo: $5-20
Detection feasibility: Maybe -- 1st Ed stamp detectable, holo/non-holo not through sleeves
```

**Variant matrix**:
1. 1st Edition Holofoil (card 5/82)
2. 1st Edition Non-Holo (card 5/82)
3. Unlimited Holofoil (card 5/82)
4. Unlimited Non-Holo (card 5/82)
5. 1st Edition Holofoil with "Rocket's" error text
6. Corrected version without error

**Detection**: Requires 1st Edition stamp detection + holo detection + OCR for
error text. Extremely challenging from binder scans.

### 2.4 Gym Trainer Pairs (gym1, gym2)

```
Variant:           Owner-specific Pokemon cards
Detection region:  Name bar: [0.06, 0.02, 0.94, 0.10]
Applicable sets:   gym1 (Gym Heroes), gym2 (Gym Challenge)
TCGCSV subtype:    YES -- each is a separate product
Price impact:      Varies widely by trainer and Pokemon
Detection feasibility: Detectable -- name OCR reads trainer prefix
```

**Visual signature**: Card name includes trainer prefix: "Misty's Seadra",
"Brock's Onix", "Lt. Surge's Pikachu", "Blaine's Charizard", etc.
Same Pokemon may appear under different trainers.

**Detection challenge**: The possessive prefix is on a separate line from the
Pokemon name. OCR reads each line independently. See open_problems.md Problem 1.
Possessive fragment concatenation is implemented but fragile.

### 2.5 Blaine's Charizard Energy Error (gym2)

```
Variant:           Error card (Blaine's Charizard)
Detection region:  Energy symbols in attack cost: ~[0.08, 0.58, 0.40, 0.72]
Applicable sets:   gym2 (Gym Challenge)
TCGCSV subtype:    NO -- same product
Price impact:      Error version: slight premium
Detection feasibility: Not detectable -- energy symbols too small at binder resolution
```

**Visual signature**: Wrong energy symbol printed in attack cost area on early
print runs. Corrected in later printings.

**Detection method**: Not implemented. Would require energy symbol classification.

### 2.6 1st Edition for Gym/Rocket Sets

Same mechanism as ERA 1 1st Edition stamp. All gym1, gym2, base5 cards exist
in both 1st Edition and Unlimited printings.

---

## ERA 3: WotC Neo (2000-2002)

**Sets**: neo1 (Neo Genesis), neo2 (Neo Discovery), neo3 (Neo Revelation), neo4 (Neo Destiny)
**Set IDs**: `neo1`, `neo2`, `neo3`, `neo4`

### 3.1 Neo 1st Edition

```
Variant:           1st_edition / 1st_edition_holofoil
Detection region:  [0.02, 0.44, 0.24, 0.65]
Applicable sets:   neo1, neo2, neo3, neo4
TCGCSV subtype:    YES -- "1st Edition" / "1st Edition Holofoil"
Price impact:      2x-20x over Unlimited
Detection feasibility: Maybe -- same as ERA 1 (~85% OCR accuracy)
```

Same detection as ERA 1. Neo Destiny (neo4) was the LAST Pokemon TCG set to
have a 1st Edition print run.

### 3.2 Shining Pokemon (Neo Revelation + Neo Destiny)

```
Variant:           Special rarity cards with "Shining" prefix
Detection region:  Full card -- distinctive silver/holographic border
Applicable sets:   neo3 (Shining Gyarados, Shining Magikarp)
                   neo4 (Shining Celebi, Shining Charizard, Shining Kabutops,
                         Shining Mewtwo, Shining Noctowl, Shining Raichu,
                         Shining Steelix, Shining Tyranitar)
TCGCSV subtype:    YES -- separate products
Price impact:      $50-500+ (chase cards of the era)
Detection feasibility: Detectable -- name OCR reads "Shining" prefix
```

**Visual signature**: Name starts with "Shining". Card has a distinctive
silver holographic border and full-art holographic treatment. The entire card
shimmers, not just the artwork area.

**Detection method**: Name OCR identifies "Shining" prefix. DINOv2 matching
against specific reference images. The silver border is a strong visual signal
but not currently used for detection.

### 3.3 Crystal Pokemon (Aquapolis/Skyridge -- e-Card era)

```
Variant:           Special rarity "Crystal Type" cards
Detection region:  Full card -- crystal/rainbow holographic treatment
Applicable sets:   ecard2 (Aquapolis: Crystal Kingdra, Crystal Lugia, Crystal Nidoking)
                   ecard3 (Skyridge: Crystal Celebi, Crystal Charizard, Crystal Crobat,
                           Crystal Golem, Crystal Ho-Oh, Crystal Kabutops)
TCGCSV subtype:    YES -- separate products
Price impact:      $100-2000+ (extremely rare, low print runs)
Detection feasibility: Detectable -- name OCR reads "Crystal" prefix
```

**Visual signature**: Name includes "Crystal" prefix. Card has a unique
holographic treatment where the Pokemon changes color depending on viewing angle.
Distinctive from standard holofoil.

**Detection method**: Name OCR for "Crystal" prefix. These are distinct card
products in the database.

### 3.4 Legendary Collection Fireworks Reverse Holo (base6)

```
Variant:           reverse_holofoil (unique "fireworks" pattern)
Detection region:  Border/text regions (NOT artwork): body regions as defined in holo_detector
Applicable sets:   base6 (Legendary Collection) ONLY
TCGCSV subtype:    YES -- "Reverse Holofoil"
Price impact:      5x-50x over normal version (highly collectible pattern)
Detection feasibility: Detectable -- hf_decorr on name bar works at 100% for reverse holo
```

**Visual signature**: Unique "fireworks" holographic pattern on everything
EXCEPT the artwork. Unlike standard reverse holo (uniform rainbow sheen),
Legendary Collection reverse holos have a starburst/fireworks explosion pattern
emanating from points across the card surface. This pattern is exclusive to
Legendary Collection and is the most visually distinctive reverse holo ever made.

**Detection method**: `holo_detector` -- detected as reverse_holofoil. The
fireworks vs standard distinction is not currently classified but could be via
texture analysis of the holo pattern (starburst shapes vs linear/rainbow).

**Reverse holo detection signals** (from `detect_reverse_holo.py`):
- Primary: name_bar high-frequency channel decorrelation (hf_decorr) >= 0.055
- Supporting: art_window hf_decorr < 0.026 (reverse holos have NO foil on artwork)
- Supporting: left_strip color_hp < 0.65
- Supporting: name_bar hp_energy < 5.5
- All 4 features show perfect class separation between reverse holo and other variants

### 3.5 e-Reader Dot Codes (ecard1, ecard2, ecard3)

```
Variant:           Visual feature on e-Card era cards
Detection region:  Bottom edge strip: ~[0.05, 0.93, 0.95, 1.00]
                   Left edge strip: ~[0.00, 0.10, 0.05, 0.93]
Applicable sets:   ecard1 (Expedition), ecard2 (Aquapolis), ecard3 (Skyridge)
TCGCSV subtype:    NO -- same product (dot codes are on all e-Card cards)
Price impact:      None (all cards in these sets have dot codes)
Detection feasibility: Maybe -- dot pattern visible but not needed for variant detection
```

**Visual signature**: Machine-readable dot code pattern along the bottom and/or
left edge of the card, designed to be scanned by the e-Reader peripheral for the
Game Boy Advance. The dots are small black marks in a specific binary pattern.

**Detection method**: Not needed for variant detection (all cards in these sets
have them). Could be used for SET identification if needed.

### 3.6 e-Card a/b Variants (ecard2, ecard3)

```
Variant:           Two different dot code patterns per card
Detection region:  Dot code area (bottom/left edge)
Applicable sets:   ecard2 (Aquapolis), ecard3 (Skyridge)
TCGCSV subtype:    NO -- same product
Price impact:      None (identical pricing)
Detection feasibility: Not detectable -- dot patterns too fine at binder resolution
```

**Visual signature**: Some cards exist with two different dot code patterns
(labeled "a" and "b" by collectors). The card artwork and text are identical;
only the dot code pattern differs. These unlocked different minigames on the
e-Reader.

### 3.7 e-Card Cosmic Reverse Holo (ecard1, ecard2, ecard3)

```
Variant:           reverse_holofoil (different pattern from LC fireworks)
Detection region:  Border/text regions
Applicable sets:   ecard1, ecard2, ecard3
TCGCSV subtype:    YES -- "Reverse Holofoil"
Price impact:      2x-10x over normal
Detection feasibility: Detectable -- hf_decorr method works for reverse holo detection
```

**Visual signature**: "Cosmic" holographic pattern on everything except artwork.
Different from Legendary Collection's fireworks -- this pattern has a smoother,
galaxy-like appearance with circular rainbow halos.

### 3.8 Dark/Light Pokemon (neo4)

```
Variant:           Name prefix variants ("Dark" or "Light")
Applicable sets:   neo4 (Neo Destiny)
TCGCSV subtype:    YES -- separate products
Price impact:      Varies by card
Detection feasibility: Detectable -- name OCR reads "Dark"/"Light" prefix
```

**Visual signature**: Cards with "Dark" or "Light" prefix to the Pokemon name.
Dark Pokemon have darker artwork tones; Light Pokemon have brighter, more
ethereal artwork.

**Detection method**: Name OCR identifies "Dark" or "Light" prefix.

---

## ERA 4: EX Era (2003-2007)

**Sets**: ex1-ex16, np (Nintendo Promos), pop1-pop5
**Set IDs**: `ex1` through `ex16`, `np`, `pop1`-`pop5`

### 4.1 Unstamped Reverse Holo (ex1-ex6)

```
Variant:           reverse_holofoil (no set logo stamp on artwork)
Detection region:  Border/text regions
Applicable sets:   ex1 (Ruby & Sapphire), ex2 (Sandstorm), ex3 (Dragon),
                   ex4 (Team Magma vs Team Aqua), ex5 (Hidden Legends),
                   ex6 (FireRed & LeafGreen)
TCGCSV subtype:    YES -- "Reverse Holofoil"
Price impact:      1.5x-3x over Normal
Detection feasibility: Detectable -- hf_decorr method
```

**Visual signature**: Standard reverse holographic foil on border and text areas.
NO set logo stamp on the artwork. Clean foil appearance across non-artwork regions.

**Detection method**: `holo_detector` for reverse holo classification. Stamp
absence confirmed by default (only ex7+ have stamps).

### 4.2 Stamped Reverse Holo (ex7-ex16)

```
Variant:           reverse_holofoil with set logo stamp
Detection region:  Stamp on artwork: [0.50, 0.30, 0.90, 0.58]
                   (bottom-right quadrant of artwork area)
Applicable sets:   ex7 (Team Rocket Returns), ex8 (Deoxys), ex9 (Emerald),
                   ex10 (Unseen Forces), ex11 (Delta Species), ex12 (Legend Maker),
                   ex13 (Holon Phantoms), ex14 (Crystal Guardians),
                   ex15 (Dragon Frontiers), ex16 (Power Keepers)
TCGCSV subtype:    YES -- same as "Reverse Holofoil" (stamp does not create separate subtype)
Price impact:      Same as reverse holo; stamped status is visual confirmation of reverse holo
Detection feasibility: Maybe -- stamp OCR at 68.8% accuracy on binder scans
```

**Visual signature**: Semi-transparent set logo and set name text overlaid on
the card artwork in the bottom-right area. The stamp text varies by set:

| Set ID | Stamp Text           |
|--------|----------------------|
| ex7    | TEAM ROCKET RETURNS  |
| ex8    | DEOXYS               |
| ex9    | EMERALD              |
| ex10   | UNSEEN FORCES        |
| ex11   | DELTA SPECIES        |
| ex12   | LEGEND MAKER         |
| ex13   | HOLON PHANTOMS       |
| ex14   | CRYSTAL GUARDIANS    |
| ex15   | DRAGON FRONTIERS     |
| ex16   | POWER KEEPERS        |

**Detection method**: `stamped_detector`
- Primary: PaddleOCR on artwork bottom-right region, match against known set name text
- Secondary: DINOv2 features + logistic regression (91.7% on training data, 68.8% LOO on binder scans)
- Edge density ratio between stamp region and control region
- Currently 68.8% accuracy on binder scans (see open_problems.md Problem 2)

### 4.3 Gold Star Cards (27 total across multiple EX sets)

```
Variant:           Special rarity with gold star next to name
Detection region:  Name bar area, right of name: ~[0.60, 0.03, 0.80, 0.10]
Applicable sets:   ex5, ex7, ex8, ex10, ex11, ex12, ex13, ex14, ex15, ex16
TCGCSV subtype:    YES -- separate products
Price impact:      $50-3000+ (chase cards, extremely collectible)
Detection feasibility: Detectable -- DINOv2 matches distinct artwork
```

**Visual signature**: A gold star symbol appears next to the Pokemon's name in
the name bar. The card also has extended/full artwork that bleeds further than
standard cards. 27 Gold Star cards total were released across the EX era.

**Detection method**: Name OCR may capture star symbol as garbage character.
DINOv2 matching against specific Gold Star reference images is the primary method.
These are distinct card products in the database.

### 4.4 Pokemon-ex (lowercase "ex")

```
Variant:           Mechanic-based card type (not a visual variant)
Detection region:  Name bar: "ex" suffix after Pokemon name
Applicable sets:   ex1 through ex16
TCGCSV subtype:    YES -- separate products
Price impact:      Varies (2x-20x over regular version)
Detection feasibility: Detectable -- name OCR reads "ex" suffix
```

**Visual signature**: Card name ends in "ex" (lowercase). Card has a distinct
border style (often silver/metallic). Two-prize card mechanic indicated on the card.

**Detection method**: Name OCR identifies "ex" suffix. These are distinct
database entries.

### 4.5 Prerelease Stamp (EX era)

```
Variant:           PRERELEASE event stamp on card artwork
Detection region:  [0.55, 0.30, 0.95, 0.58] (artwork bottom-right)
Applicable sets:   Various EX sets (one prerelease card per set)
TCGCSV subtype:    YES -- separate products
Price impact:      $5-50 over regular version
Detection feasibility: Maybe -- OCR reads stamp text but garbling is common
```

**Visual signature**: Gold/yellow "PRERELEASE" text stamped on the card artwork,
typically in the bottom-right area of the illustration.

**Detection method**: PaddleOCR for "PRERELEASE" text. Known OCR confusions:
"PRENELEMEE", "PRERELEAS", "PRE RELEASE". Use fuzzy matching with threshold 0.70.
Confirmed bug: "PRENELEMEE" fuzzy-matched to "peerless edge" at 0.636 before
threshold was raised from 0.60 to 0.70.

### 4.6 POP Series

```
Variant:           normal / holofoil only (no reverse holo)
Applicable sets:   pop1, pop2, pop3, pop4, pop5
TCGCSV subtype:    YES -- separate products per POP set
Price impact:      Moderate (limited distribution)
Detection feasibility: Detectable -- identified via set/card matching
```

**Visual signature**: Standard card layout. POP Series cards have their own
set symbol. No reverse holofoil variant exists for POP Series.

---

## ERA 5: Diamond & Pearl / Platinum (2007-2010)

**Sets**: dp1-dp7, pl1-pl4, dpp (DP Promos), pop6-pop9
**Set IDs**: `dp1`-`dp7`, `pl1`-`pl4`, `dpp`, `pop6`-`pop9`

### 5.1 LV.X Cards

```
Variant:           Special card type with "LV.X" in name
Detection region:  Name bar -- "LV.X" text after Pokemon name
Applicable sets:   dp1 through pl4
TCGCSV subtype:    YES -- separate products
Price impact:      $10-200+ depending on Pokemon
Detection feasibility: Detectable -- name OCR reads "LV.X" suffix
```

**Visual signature**: Card name includes "LV.X" suffix. Card has a distinctive
metallic silver border and extended artwork. The level-up mechanic is indicated
on the card.

**Detection method**: Name OCR identifies "LV.X" suffix.

### 5.2 Reverse Holo (DP/Platinum style)

```
Variant:           reverse_holofoil (no set logo stamp, clean rainbow sheen)
Detection region:  Border/text regions
Applicable sets:   dp1-dp7, pl1-pl4
TCGCSV subtype:    YES -- "Reverse Holofoil"
Price impact:      1.5x-3x over Normal
Detection feasibility: Detectable -- hf_decorr method
```

**Visual signature**: Clean rainbow holographic sheen on border and text areas.
No set logo stamp (unlike EX era ex7-ex16). Non-patterned rainbow effect, more
subtle than BW-era type-symbol reverse holos.

**Detection method**: `holo_detector` -- standard reverse holo detection.

### 5.3 SH Shiny Cards (12 total)

```
Variant:           Shiny (alternate color) Pokemon cards
Detection region:  Full card -- shiny Pokemon have different coloring
Applicable sets:   pl3 (Supreme Victors: Milotic, Relicanth, Swablu,
                        Lotad, Vulpix, Yanma)
                   pl4 (Arceus: Bagon, Ponyta, Shinx, Drifloon,
                        Duskull, Voltorb)
TCGCSV subtype:    YES -- separate products (SH number prefix)
Price impact:      $20-100+
Detection feasibility: Detectable -- DINOv2 matches distinct shiny artwork
```

**Visual signature**: Pokemon depicted in its shiny/alternate color palette
(e.g., red Gyarados instead of blue). Card number has "SH" prefix.

**Detection method**: DINOv2 matching against specific reference images.
Card number OCR for "SH" prefix (unreliable at binder resolution).

### 5.4 RT Rotom Forms (6 total)

```
Variant:           Rotom form cards
Applicable sets:   pl3 (Supreme Victors) and pl4 (Arceus)
TCGCSV subtype:    YES -- separate products
Price impact:      $5-30
Detection feasibility: Detectable -- distinct artwork per form
```

**Visual signature**: Rotom in various appliance forms (Wash, Heat, Fan, Frost, Mow).
Card number has "RT" prefix.

### 5.5 AR Arceus Forms (9 total)

```
Variant:           Different Arceus type forms
Applicable sets:   pl4 (Arceus)
TCGCSV subtype:    YES -- separate products
Price impact:      $5-50
Detection feasibility: Detectable -- distinct type-colored backgrounds
```

**Visual signature**: Arceus depicted with different type-colored backgrounds.
Card number has "AR" prefix. Each form corresponds to a different Pokemon type.

### 5.6 Secret Rares

```
Variant:           Cards with numbers exceeding set total
Detection region:  Card number area: ~[0.60, 0.90, 0.95, 0.98]
Applicable sets:   Various DP/Platinum sets
TCGCSV subtype:    YES -- separate products
Price impact:      $20-200+
Detection feasibility: Detectable -- DINOv2 matches distinct secret rare artwork
```

**Detection method**: Identified by card number exceeding set size (e.g., 131/130).
Card number OCR is unreliable at binder resolution; rely on DINOv2 matching.

### 5.7 Prerelease/Staff Stamps

```
Variant:           PRERELEASE and STAFF event stamps
Detection region:  [0.55, 0.30, 0.95, 0.58] (artwork bottom-right)
                   STAFF: [0.55, 0.20, 0.95, 0.45] (above prerelease)
Applicable sets:   Various DP/Platinum sets
TCGCSV subtype:    YES -- separate products
Price impact:      Prerelease: $5-30; Staff: $20-200+
Detection feasibility: Maybe -- stamp OCR works but text is small
```

Same detection as ERA 4 prerelease stamps. Staff stamp is separate:
- Region: [0.55, 0.20, 0.95, 0.45] (upper-right of artwork, above prerelease)
- OCR for "STAFF" text

### 5.8 DP Promos (dpp)

```
Variant:           normal / holofoil / promo
Applicable sets:   dpp
TCGCSV subtype:    YES -- separate products
Price impact:      Varies
Detection feasibility: Detectable -- identified via set/card matching
```

---

## ERA 6: HGSS / Black & White (2010-2013)

**Sets**: hgss1-hgss4, hsp, col1, bw1-bw11, bwp, dv1, dc1
**Set IDs**: `hgss1`-`hgss4`, `hsp`, `col1`, `bw1`-`bw11`, `bwp`, `dv1`, `dc1`

### 6.1 Pokemon PRIME (HGSS)

```
Variant:           Special rarity with "PRIME" text
Detection region:  Below name bar or in card text
Applicable sets:   hgss1-hgss4
TCGCSV subtype:    YES -- separate products
Price impact:      $5-50
Detection feasibility: Detectable -- name OCR reads "PRIME" text
```

**Visual signature**: "PRIME" text below the Pokemon name. Card has a distinctive
metallic/prismatic border treatment. Artwork is slightly more detailed than
standard versions.

**Detection method**: Name OCR for "PRIME" text. DINOv2 matching against references.

### 6.2 LEGEND Cards (HGSS)

```
Variant:           Two-card panoramic Pokemon
Detection region:  Full card -- distinctive landscape format
Applicable sets:   hgss1, hgss2, hgss4
TCGCSV subtype:    YES -- separate products (top half and bottom half)
Price impact:      $10-100+ per half; $50-300+ for pairs
Detection feasibility: Detectable -- distinctive half-card layout + name OCR for "LEGEND"
```

**Visual signature**: LEGEND cards span TWO physical cards that combine to form
a single panoramic image. Each half has "LEGEND" text and shows the top or bottom
portion of the artwork. Card names include "&" (e.g., "Lugia & Ho-Oh LEGEND").

**Detection method**: Name OCR for "LEGEND" text and "&" connector. These are
unique database entries. The two-card nature means they may appear in adjacent
binder slots.

### 6.3 First Full Art EX (BW)

```
Variant:           full_art (artwork extends to card edges)
Detection region:  Edge strips: outer 5% on each side
Applicable sets:   bw1-bw11
TCGCSV subtype:    NO -- shares Holofoil subtype
Price impact:      2x-20x over regular version
Detection feasibility: Detectable -- edge strip saturation analysis
```

**Visual signature**: Artwork extends to card edges with no visible border.
Full art EX cards and full art Supporters introduced in this era. The entire
card surface is covered in illustration.

**Detection method**: `full_art_detector`
- Measure mean saturation, hue std dev, and colorful pixel fraction on 4 edge strips
- Each strip passes when: mean_sat >= 65, hue_std >= 18, colorful_frac >= 0.35
- Require 3/4 edge strips passing
- Era-gated to era >= 5 (BW onward)
- Edge strip width: 5% of card dimensions (`FULL_ART_EDGE_FRAC = 0.05`)

### 6.4 Full Art Trainers (BW)

```
Variant:           full_art Trainer/Supporter cards
Detection region:  Same as full art EX
Applicable sets:   bw1-bw11 (introduced mid-BW era)
TCGCSV subtype:    NO -- shares Holofoil subtype
Price impact:      $10-200+
Detection feasibility: Detectable -- edge strip saturation analysis
```

**Visual signature**: Supporter cards with full-art illustration extending to
card edges. Characters depicted in detailed, dynamic poses.

### 6.5 ACE SPEC Cards (BW)

```
Variant:           Special trainer mechanic cards
Detection region:  ACE SPEC text/symbol in card text area
Applicable sets:   bw7-bw10 (Boundaries Crossed through Plasma Blast)
TCGCSV subtype:    YES -- separate products
Price impact:      $5-50
Detection feasibility: Detectable -- DINOv2 matches distinct card design
```

**Visual signature**: "ACE SPEC" designation with a distinctive gold ACE SPEC
symbol on the card. Only one ACE SPEC card allowed per deck.

**Detection method**: Name/text OCR for "ACE SPEC" or DINOv2 matching.

### 6.6 Cracked Ice Theme Deck Holos (BW/XY)

```
Variant:           holofoil with "cracked ice" holo pattern
Detection region:  Artwork area: [0.10, 0.10, 0.90, 0.55]
Applicable sets:   Various BW and XY theme deck exclusives
TCGCSV subtype:    YES -- "Holofoil" (same subtype as standard holo)
Price impact:      Generally lower than booster pack holos ($2-10)
Detection feasibility: Not detectable -- holo pattern distinction invisible through sleeves
```

**Visual signature**: Instead of the standard cosmos/star holographic pattern,
these cards have a "cracked ice" pattern that looks like fractured ice or crystal
shards across the artwork. Available exclusively in theme decks.

**Detection method**: `holo_detector` detects as holofoil. The cracked ice vs
standard holo pattern distinction is not currently implemented.

### 6.7 Radiant Collection (BW11 Legendary Treasures)

```
Variant:           Special subset within Legendary Treasures
Detection region:  Full card -- distinctive cute/chibi art style
Applicable sets:   bw11 (Legendary Treasures)
TCGCSV subtype:    YES -- separate products (RC-numbered cards)
Price impact:      $2-100+ (RC Full Art Meloetta, etc.)
Detection feasibility: Detectable -- DINOv2 matches distinct artwork
```

**Visual signature**: Cards numbered with "RC" prefix (e.g., RC1, RC25).
Feature cute, chibi-style artwork of various Pokemon and characters.

### 6.8 Shiny Vault (BW era)

```
Variant:           Shiny Pokemon cards
Applicable sets:   bw9 (Plasma Freeze), bw11 (Legendary Treasures)
TCGCSV subtype:    YES -- separate products
Price impact:      $10-300+
Detection feasibility: Detectable -- DINOv2 matches distinct shiny artwork
```

**Visual signature**: Pokemon depicted in shiny/alternate color palette.
These cards appear in the secret rare section of the set.

### 6.9 Cosmos Holo (Theme Deck / Promo)

```
Variant:           holofoil with cosmos pattern (distinct from standard)
Detection region:  Artwork area
Applicable sets:   Various theme deck cards, some promos
TCGCSV subtype:    YES -- "Holofoil"
Price impact:      Usually lower than booster holo ($1-5)
Detection feasibility: Not detectable -- holo pattern invisible through sleeves
```

**Visual signature**: Galaxy/cosmos sparkle pattern across artwork instead of
standard holographic lines. Small scattered dots of light rather than continuous
rainbow streaks.

### 6.10 Reverse Holo Patterns

```
BW era reverse holo patterns:
  bw1:        Plain type-colored sheen (no pattern)
  bw2-bw11:   Type symbol pattern (small type symbols repeated)
  bw11:       Includes Radiant Collection subset with RC numbering
```

---

## ERA 7: XY (2014-2016)

**Sets**: xy0-xy12, xyp (XY Promos), g1 (Generations)
**Set IDs**: `xy0`-`xy12`, `xyp`, `g1`

### 7.1 Mega EX Cards

```
Variant:           Mega Evolution card type
Detection region:  Name bar -- "M" prefix and "EX" suffix
Applicable sets:   xy1-xy12, g1
TCGCSV subtype:    YES -- separate products
Price impact:      $5-100+
Detection feasibility: Detectable -- name OCR reads "M" prefix + "-EX" suffix
```

**Visual signature**: Card name starts with "M " and ends with "-EX" (e.g.,
"M Charizard-EX"). Distinctive blue/red border depending on version. Often
has extended artwork.

**Detection method**: Name OCR for "M" prefix + "-EX" suffix.

### 7.2 BREAK Cards (Landscape!)

```
Variant:           BREAK Evolution cards in landscape orientation
Detection region:  Full card -- 90-degree rotated layout
Applicable sets:   xy8 (BREAKthrough), xy9 (BREAKpoint), xy12 (Evolutions)
TCGCSV subtype:    YES -- separate products
Price impact:      $3-30
Detection feasibility: Maybe -- landscape orientation detectable but rotation handling needed
```

**Visual signature**: Card is LANDSCAPE oriented (rotated 90 degrees from normal).
Gold/yellow border with "BREAK" text. Features the Pokemon breaking through a
shattered surface effect.

**Detection method**: Aspect ratio analysis -- BREAK cards are wider than tall
when in their natural landscape orientation. If placed in a binder vertically,
the card will appear rotated. Name OCR for "BREAK" suffix.

**Implementation note**: BREAK cards in binder slots may be rotated. The
segmenter should handle 90-degree rotated cards. If width > height after
segmentation, the card may be a BREAK card.

### 7.3 Full Art (XY pattern)

```
Variant:           full_art EX, Mega, BREAK, Trainer cards
Detection region:  Edge strips: outer 5%
Applicable sets:   xy1-xy12, g1
TCGCSV subtype:    NO -- shares Holofoil subtype
Price impact:      2x-20x over regular version
Detection feasibility: Detectable -- edge strip saturation analysis
```

Same detection as ERA 6 full art.

### 7.4 Secret Rare Gold Cards (XY)

```
Variant:           Gold-colored secret rare cards
Detection region:  Full card -- gold border and gold color dominance
Applicable sets:   xy1-xy12
TCGCSV subtype:    NO -- shares Holofoil subtype
Price impact:      $10-100+
Detection feasibility: Detectable -- HSV gold color analysis
```

**Visual signature**: Entire card has gold-tinted borders and accents. Card
number exceeds set total. Energy cards and select trainer items.

**Detection method**: `gold_detector` (era-gated to >= 7 in current code, but
XY golds exist). HSV analysis for gold-dominant hue (15-45 range) covering >40%
of card surface.

**Implementation gap**: Current code era-gates gold detection to SM+ (era 7+).
XY gold secret rares (era 6) would be missed. Consider lowering gate to era 6.

### 7.5 Generations Radiant Collection (g1)

```
Variant:           Special subset with RC numbering
Detection region:  Full card -- distinctive art style
Applicable sets:   g1 (Generations)
TCGCSV subtype:    YES -- separate products (RC-numbered)
Price impact:      $5-200+ (Radiant Collection FA cards)
Detection feasibility: Detectable -- DINOv2 matches distinct artwork
```

**Visual signature**: Cards numbered with "RC" prefix. Cute/chibi art style
featuring Fairy-type Pokemon and characters. Includes full art Radiant Collection
cards which are highly collectible.

### 7.6 Prerelease/Staff Stamps

Same mechanism as ERA 4/5. Region: [0.55, 0.30, 0.95, 0.58]

---

## ERA 8: Sun & Moon (2017-2019)

**Sets**: sm1-sm12, sm35, sm75, sm115, smp, sma, det1, mcd18, mcd19
**Set IDs**: `sm1`-`sm12`, `sm35`, `sm75`, `sm115`, `smp`, `sma`, `det1`

### 8.1 GX Cards

```
Variant:           Pokemon-GX card type
Detection region:  Name bar -- "GX" suffix
Applicable sets:   sm1-sm12 (all SM main sets)
TCGCSV subtype:    YES -- separate products
Price impact:      $3-200+
Detection feasibility: Detectable -- name OCR reads "-GX" suffix
```

**Visual signature**: Card name ends in "-GX". Cards have a holographic GX
symbol and typically have a GX attack (powerful one-use attack with rainbow
energy-colored text). Border style varies (standard, full art, rainbow rare).

**Detection method**: Name OCR for "-GX" suffix.

### 8.2 Full Art GX

```
Variant:           full_art version of GX cards
Detection region:  Edge strips: outer 5%
Applicable sets:   sm1-sm12
TCGCSV subtype:    NO -- shares Holofoil subtype (different product from regular GX)
Price impact:      2x-10x over regular GX
Detection feasibility: Detectable -- edge strip saturation analysis
```

**Visual signature**: Full art illustration extending to card edges. Textured
foil surface. Character depicted in dynamic pose against detailed background.

### 8.3 Rainbow Rare (Hyper Rare)

```
Variant:           rainbow_rare
Detection region:  Full card surface
Applicable sets:   sm1-sm12 (first era with rainbow rares)
TCGCSV subtype:    NO -- shares Holofoil subtype
Price impact:      2x-20x over regular version
Detection feasibility: Detectable -- multi-hue saturation analysis
```

**Visual signature**: Full art card with rainbow/prismatic foil covering the
entire surface. The illustration appears in multi-colored rainbow hues rather
than natural colors. High saturation across all hue segments.

**Detection method**: `rainbow_detector`
- Measure high saturation spread across 6 hue segments (red, orange, yellow, green, blue, purple)
- Rainbow rare has significant pixels in 4+ of 6 segments
- Era-gated to >= 7 (SM onward)
- Must distinguish from colorful full art (rainbow has UNIFORM saturation distribution)

### 8.4 Gold Secret Rare

```
Variant:           gold
Detection region:  Full card surface, especially borders
Applicable sets:   sm1-sm12
TCGCSV subtype:    NO -- shares Holofoil subtype
Price impact:      $10-200+
Detection feasibility: Detectable -- HSV gold color analysis
```

**Visual signature**: Entire card dominated by gold color including borders.
Gold Pokemon-GX, gold Trainer items, gold Energy cards. Textured foil with
gold coloring.

**Detection method**: `gold_detector`
- HSV analysis: hue 15-45 (gold range) covering >40% of card surface
- Border region must also be gold (distinguishes from gold-tinted artwork)
- Era-gated to >= 7 (SM onward)

### 8.5 Prism Star Cards

```
Variant:           Special mechanic cards with prism star symbol
Detection region:  Name bar -- prism star symbol after name
Applicable sets:   sm5 (Ultra Prism) through sm11 (Unified Minds)
TCGCSV subtype:    YES -- separate products
Price impact:      $2-20
Detection feasibility: Detectable -- DINOv2 matches distinct card design
```

**Visual signature**: A prismatic/rainbow star symbol appears after the card name.
Cards have a distinctive rainbow border treatment. Limited to one per deck.

**Detection method**: Name OCR may capture prism star as special character.
DINOv2 matching against references.

### 8.6 Tag Team GX

```
Variant:           Two-Pokemon GX cards
Detection region:  Name bar -- two Pokemon names connected by "&"
Applicable sets:   sm9 (Team Up) through sm12 (Cosmic Eclipse)
TCGCSV subtype:    YES -- separate products
Price impact:      $5-500+ (alt art Tag Teams are chase cards)
Detection feasibility: Detectable -- name OCR reads "&" connector + "-GX" suffix
```

**Visual signature**: Card name contains two Pokemon names with "&" connector
and "GX" suffix (e.g., "Pikachu & Zekrom-GX"). Extended artwork showing both
Pokemon together.

**Detection method**: Name OCR for "&" connector + "-GX" suffix.

### 8.7 Alternate Art (Late SM Era)

```
Variant:           Alternate illustration for existing cards
Detection region:  Full card -- different artwork from standard version
Applicable sets:   sm9-sm12
TCGCSV subtype:    YES -- separate products
Price impact:      $20-500+ (highly collectible)
Detection feasibility: Detectable -- DINOv2 distinguishes alt art from regular
```

**Visual signature**: Same Pokemon name and card text but completely different
artwork. Often depicts Pokemon in natural/story settings rather than battle poses.
Full art treatment.

**Detection method**: DINOv2 matching. Alt arts are distinct database entries.
The artwork difference from regular versions is large enough for DINOv2 to
distinguish reliably.

### 8.8 Character Rare (Cosmic Eclipse)

```
Variant:           Cards featuring Pokemon with human characters
Detection region:  Full card -- unique art style with trainer character
Applicable sets:   sm12 (Cosmic Eclipse)
TCGCSV subtype:    YES -- separate products
Price impact:      $5-50
Detection feasibility: Detectable -- DINOv2 matches distinct artwork
```

**Visual signature**: Card artwork features a Pokemon alongside a known Pokemon
trainer character from the games/anime. Distinctive art style different from
standard card illustrations.

### 8.9 Reverse Holo (SM pattern)

```
Variant:           reverse_holofoil (type symbol pattern)
Detection region:  Border/text regions
Applicable sets:   sm1-sm12
TCGCSV subtype:    YES -- "Reverse Holofoil"
Price impact:      1.2x-2x over Normal
Detection feasibility: Detectable -- hf_decorr method
```

**Visual signature**: Horizontal rows of small type symbols (fire, water, grass,
etc.) with one large type symbol in the bottom-right corner of the text box.
The type matches the Pokemon's type. Holographic sheen over the pattern.

**Detection method**: `holo_detector` for reverse holo classification. The
type symbol pattern is a visual identifier for SM-era cards but not currently
used for automated detection.

### 8.10 Cosmos Holo / Prerelease / Staff

Same patterns as prior eras. Cosmos holo appears on theme deck and some promo cards.
Prerelease/Staff stamps in same position as ERA 4+.

---

## ERA 9: Sword & Shield (2020-2023)

**Sets**: swsh1-swsh12, swsh12pt5, swshp, plus sub-sets (tg, sv, gg)
**Set IDs**: `swsh1`-`swsh12`, `swsh12pt5`, `swshp`, `swsh45sv`, `cel25`, `cel25c`, `pgo`, etc.

### 9.1 V / VMAX / VSTAR Cards

```
Variant:           Pokemon V card types
Detection region:  Name bar -- "V", "VMAX", or "VSTAR" suffix
Applicable sets:   swsh1-swsh12, swsh12pt5
TCGCSV subtype:    YES -- separate products
Price impact:      V: $2-50, VMAX: $5-100, VSTAR: $5-50
Detection feasibility: Detectable -- name OCR reads V/VMAX/VSTAR suffix
```

**Visual signature**:
- **V**: Card name ends in "V". Standard full-art or half-art layout.
- **VMAX**: Card name ends in "VMAX". Larger card frame, Dynamax/Gigantamax artwork. Often features the Pokemon at enormous scale.
- **VSTAR**: Card name ends in "VSTAR". Star-themed border treatment with VSTAR Power ability.

### 9.2 Alternate Art (Alt Art)

```
Variant:           Alternate illustration V/VMAX/VSTAR cards
Detection region:  Full card -- unique scene-based artwork
Applicable sets:   swsh5 (Battle Styles) onward through swsh12
TCGCSV subtype:    YES -- separate products
Price impact:      $20-500+ (Moonbreon VMAX alt art: $200-400)
Detection feasibility: Detectable -- DINOv2 distinguishes alt art from regular
```

**Visual signature**: Same Pokemon name and attacks as regular version but with
completely different artwork depicting the Pokemon in a natural or story setting.
Full art treatment with distinctive artistic styles.

**Detection method**: DINOv2 matching. Alt arts are distinct database entries.
The artwork difference from regular versions is large enough for DINOv2 to
distinguish reliably.

### 9.3 Rainbow Rare

```
Variant:           rainbow_rare (continuing from SM era)
Detection region:  Full card surface
Applicable sets:   swsh1-swsh12
TCGCSV subtype:    NO -- shares Holofoil subtype
Price impact:      $5-100+
Detection feasibility: Detectable -- multi-hue saturation analysis
```

Same detection as ERA 8 rainbow rare.

### 9.4 Gold Secret Rare

```
Variant:           gold (continuing from SM era)
Detection region:  Full card surface
Applicable sets:   swsh1-swsh12
TCGCSV subtype:    NO -- shares Holofoil subtype
Price impact:      $10-100+
Detection feasibility: Detectable -- HSV gold color analysis
```

Same detection as ERA 8 gold.

### 9.5 Trainer Gallery (TG subset)

```
Variant:           Character rare subset cards
Detection region:  Full card -- character + Pokemon art
Applicable sets:   swsh9tg, swsh10tg, swsh11tg, swsh12tg
TCGCSV subtype:    YES -- separate products (TG-numbered)
Price impact:      $3-50
Detection feasibility: Detectable -- DINOv2 matches distinct artwork
```

**Visual signature**: Cards numbered with "TG" prefix. Feature Pokemon alongside
trainer characters in artistic settings. Full art or half-art treatment.

**Detection method**: These are distinct database entries. TG-numbered cards are
separate products in the database.

### 9.6 Amazing Rare

```
Variant:           Special rarity with rainbow border
Detection region:  Border region -- distinctive rainbow/prismatic border
Applicable sets:   swsh4 (Vivid Voltage), swsh5 (Battle Styles)
TCGCSV subtype:    YES -- separate products
Price impact:      $5-30
Detection feasibility: Detectable -- rainbow border color analysis + DINOv2
```

**Visual signature**: Rainbow-colored border with a unique stained-glass-like
appearance. The card's border swirls through all colors. Very different from
standard yellow/silver borders. Features Legendary Pokemon.

**Detection method**: Border color analysis -- Amazing Rare borders have very
high hue diversity in the border region specifically. DINOv2 matching against references.

### 9.7 Radiant Cards

```
Variant:           "Radiant" prefix cards with rainbow foil treatment
Detection region:  Name bar -- "Radiant" prefix
Applicable sets:   swsh10 (Astral Radiance), swsh11 (Lost Origin),
                   swsh12 (Silver Tempest)
TCGCSV subtype:    YES -- separate products
Price impact:      $5-50
```

**Visual signature**: Name starts with "Radiant". Card has a rainbow foil
treatment across the entire card. Depicts the Pokemon's shiny form. Limited
to one Radiant card per deck.

### 9.8 Shiny Vault (swsh45sv)

```
Variant:           Shiny Pokemon subset
Detection region:  Full card -- shiny color palette
Applicable sets:   swsh45sv (Shining Fates Shiny Vault)
TCGCSV subtype:    YES -- separate products (SV-numbered)
Price impact:      $5-300+ (Shiny Charizard VMAX)
```

**Visual signature**: Pokemon depicted in shiny/alternate color palette.
Cards numbered with "SV" prefix. Baby shiny cards have a distinctive black
star border.

### 9.9 Peelable Ditto (pgo)

```
Variant:           Cards with Ditto peel-off layer
Detection region:  Full card surface -- Ditto card hidden underneath
Applicable sets:   pgo (Pokemon GO)
TCGCSV subtype:    YES -- separate products
Price impact:      $3-20
```

**Visual signature**: Normal-looking cards that have a peelable top layer.
When peeled, a Ditto card is revealed underneath. In a binder scan, these
appear as normal cards (the peel-off is intact). Cannot be detected from scans.

**Detection method**: Not possible from scans. Requires physical inspection
or database lookup (specific card numbers in Pokemon GO set have Ditto versions).

### 9.10 Celebrations Classic Collection (cel25c)

```
Variant:           Holofoil reprints of classic cards
Detection region:  Full card -- distinctive 25th anniversary border
Applicable sets:   cel25c
TCGCSV subtype:    YES -- separate products
Price impact:      $5-100 (Classic Charizard, Classic Umbreon)
```

**Visual signature**: Reprints of iconic cards from past eras with a distinctive
25th anniversary Celebrations border treatment. All cards are holofoil with
cosmos holo pattern. Card names and artwork match the original versions.

### 9.11 McDonald's Confetti Holo

```
Variant:           holofoil with confetti pattern
Detection region:  Artwork area
Applicable sets:   mcd21 (McDonald's 2021), mcd22 (McDonald's 2022)
TCGCSV subtype:    YES -- "Holofoil"
Price impact:      Holo: $3-15, Non-holo: $1-5
```

**Visual signature**: Confetti-shaped holographic pattern (small triangles,
circles, squares scattered across artwork). Different from standard cosmos
or parallel-line holo patterns.

### 9.12 Reverse Holo (SWSH chevron pattern)

```
Variant:           reverse_holofoil (chevron/type symbol pattern)
Detection region:  Border/text regions
Applicable sets:   swsh1-swsh12
TCGCSV subtype:    YES -- "Reverse Holofoil"
Price impact:      1.2x-2x over Normal
```

**Visual signature**: Upward-facing chevrons containing the Pokemon's type symbol,
repeated across the card border and text areas. This pattern was widely criticized
for making text hard to read.

### 9.13 Build & Battle Stamp / Prerelease / Staff

```
Variant:           Build & Battle event cards with stamps
Detection region:  [0.55, 0.30, 0.95, 0.58] (artwork bottom-right)
                   STAFF: [0.55, 0.20, 0.95, 0.45] (above prerelease)
Applicable sets:   Various SWSH sets (one per set)
TCGCSV subtype:    YES -- separate products
Price impact:      Prerelease: $3-20, Staff: $10-100+
```

**Visual signature**: SWSH+ era prerelease cards come from "Build & Battle" boxes.
The prerelease stamp is smaller and more subtle than older eras but in the same
general position. Staff versions have an additional "STAFF" stamp.

---

## ERA 10: Scarlet & Violet (2023+)

**Sets**: sv1-sv10+, sve (Energy), svp (Promos)
**Set IDs**: `sv1`-`sv10`, `sve`, `svp`

### 10.1 Pokemon ex (lowercase)

```
Variant:           Pokemon ex card type (SV-era mechanic)
Detection region:  Name bar -- "ex" suffix (lowercase)
Applicable sets:   sv1-sv10+
TCGCSV subtype:    YES -- separate products
Price impact:      $2-200+
```

**Visual signature**: Card name ends in lowercase "ex" (e.g., "Charizard ex").
Distinguished from EX-era uppercase "EX" by the capitalization. Cards have a
distinctive new border style with the ex designation.

### 10.2 Double Rare

```
Variant:           Two-star rarity cards
Detection region:  Rarity symbols: ~[0.75, 0.92, 0.95, 0.98]
Applicable sets:   sv1-sv10+
TCGCSV subtype:    YES -- separate products (double black star rarity)
Price impact:      $2-20
```

**Visual signature**: Two black star symbols in the card number/rarity area.
These are the new "base rarity" for ex Pokemon and other notable cards.
Replaced the old single-star rare rarity.

### 10.3 Illustration Rare (IR)

```
Variant:           full_art illustration rare (single gold star rarity)
Detection region:  Edge strips: outer 5%
Applicable sets:   sv1-sv10+
TCGCSV subtype:    YES -- separate products
Price impact:      $5-100+
```

**Visual signature**: Single gold star rarity symbol. Full art illustration
with Pokemon depicted in natural environments or story scenes. Artwork extends
to card edges. More subdued foil treatment compared to older full arts.

**Detection method**: `full_art_detector` for the full art visual signature.
The gold star vs black star rarity distinction requires rarity symbol detection
(currently not implemented).

### 10.4 Special Illustration Rare (SIR)

```
Variant:           Chase full art cards (double gold star rarity)
Detection region:  Edge strips: outer 5%
Applicable sets:   sv1-sv10+
TCGCSV subtype:    YES -- separate products
Price impact:      $20-500+
```

**Visual signature**: Double gold star rarity symbol. Premium full art with
distinctive glitter/sparkle foil treatment. Highly detailed artwork often
with unusual compositions or perspectives. These are the main chase cards
of the SV era.

### 10.5 Hyper Rare Gold (Triple Star)

```
Variant:           gold (triple gold star rarity)
Detection region:  Full card -- gold borders and accents
Applicable sets:   sv1-sv10+
TCGCSV subtype:    YES -- separate products (different from standard gold)
Price impact:      $10-200+
```

**Visual signature**: Triple gold star rarity. Full art with gilded/gold borders
and accents. High-end chase cards featuring gold-tinted artwork.

**Detection method**: `gold_detector` -- same as ERA 8/9 gold detection.

### 10.6 ACE SPEC (SV era return)

```
Variant:           ACE SPEC trainer cards (returning from BW era)
Detection region:  ACE SPEC marking on card
Applicable sets:   sv4 (Paradox Rift) onward
TCGCSV subtype:    YES -- separate products
Price impact:      $5-50
```

**Visual signature**: Distinctive gold ACE SPEC border/marking. Only one ACE SPEC
card allowed per deck. Trainer items with powerful effects.

### 10.7 Shiny Rare (SV Paldean Fates and beyond)

```
Variant:           Shiny Pokemon cards
Detection region:  Full card -- shiny color palette
Applicable sets:   sv4pt5 (Paldean Fates), sv8pt5 (future shiny sets)
TCGCSV subtype:    YES -- separate products
Price impact:      $3-200+
```

**Visual signature**: Pokemon depicted in shiny/alternate color palette.
Similar to previous shiny vaults but with SV-era border styling.

### 10.8 Stellar Tera Cards

```
Variant:           Tera-type Pokemon with crystal/stellar treatment
Detection region:  Full card -- crystalline border effect
Applicable sets:   sv6 (Twilight Masquerade) onward
TCGCSV subtype:    YES -- separate products
Price impact:      $5-100+
```

**Visual signature**: Card has a crystalline/gem-like border treatment reflecting
the Terastallization mechanic. The Pokemon appears to be encased in or emerging
from crystal structures.

### 10.9 Trainer's Pokemon (sv7+)

```
Variant:           Pokemon associated with specific trainers
Detection region:  Name bar -- trainer name prefix
Applicable sets:   sv7 (Stellar Crown) onward
TCGCSV subtype:    YES -- separate products
Price impact:      $5-50+
```

**Visual signature**: Card name includes trainer prefix (similar to Gym era but
with modern card design). Features Pokemon alongside their trainer character.

**Detection method**: Name OCR for trainer name prefix. Same possessive prefix
challenge as ERA 2 (see open_problems.md Problem 1).

### 10.10 Type-Specific Reverse Holos (SV pattern)

```
Variant:           reverse_holofoil (type-specific pattern)
Detection region:  Border/text regions
Applicable sets:   sv1-sv10+
TCGCSV subtype:    YES -- "Reverse Holofoil"
Price impact:      1.2x-2x over Normal
```

**Visual signature**: Each Pokemon type has its own unique reverse holo pattern:
- Lightning: lightning bolt shapes
- Grass: leaf patterns
- Fire: flame-like shapes
- Water: wave/water drop patterns
- Psychic: swirl/spiral patterns
- Fighting: rocky/angular shapes
- Metal: geometric/industrial patterns
- Dark: shadow/dark swirl patterns
- Fairy/Dragon: respective thematic patterns

Pebble-like shapes with type symbol embedded in circles. Much more visually
appealing than SWSH-era chevrons.

### 10.11 Build & Battle Stamp

```
Variant:           Build & Battle promo cards with event stamp
Detection region:  [0.55, 0.30, 0.95, 0.58]
Applicable sets:   Various SV sets
TCGCSV subtype:    YES -- separate products
Price impact:      $3-20
```

Same detection as SWSH era Build & Battle stamps.

### 10.12 Pokemon Center Stamp

```
Variant:           Pokemon Center exclusive cards with stamp
Detection region:  Artwork area -- Pokemon Center logo stamp
Applicable sets:   Select svp promos
TCGCSV subtype:    YES -- separate products
Price impact:      $5-30 premium over regular promo
```

**Visual signature**: Pokemon Center logo stamp overlaid on card artwork.
Similar in style to EX-era set stamps but with the Pokemon Center branding.

**Detection method**: OCR for "POKEMON CENTER" text or template matching for
the Pokemon Center logo.

---

## Detection Method Summary

### Implemented and Working

| Method              | Module               | Function           | Accuracy       | Notes                                        |
|---------------------|----------------------|--------------------|----------------|----------------------------------------------|
| `holo_detector`     | `holo_detector.py`   | `detect_holo_type` | 100% rev holo  | hf_decorr >= 0.055 on name bar               |
|                     |                      |                    | ~0% holofoil   | Holofoil undetectable through binder sleeves  |
| `stamp_ocr`         | `variant_detector.py`| `_check_1st_edition`| ~85%          | PaddleOCR + blob + HoughCircles              |
| `stamped_detector`  | `variant_detector.py`| `detect_stamped`   | 68.8% binder   | EX-era set stamps, needs training data        |
| `full_art_detector` | `variant_detector.py`| `detect_variant`   | Not measured   | Edge strip sat/hue analysis, era >= 5         |
| `gold_detector`     | `variant_detector.py`| `detect_variant`   | Not measured   | HSV gold hue >40% coverage, era >= 7          |
| `rainbow_detector`  | `variant_detector.py`| `detect_variant`   | Not measured   | 4+/6 hue segments saturated, era >= 7         |
| `edge_gradient`     | `variant_detector.py`| `detect_variant`   | Not measured   | Right/bottom border brightness comparison     |
| `promo_detector`    | `variant_detector.py`| `detect_promo_stamp`| Not measured  | Star contour solidity 0.25-0.40               |

### Detection Thresholds Reference

```python
# Holo detection
HOLO_HUE_SPREAD_THRESHOLD = 20       # bins (out of 36)
HOLO_SPATIAL_NOISE_THRESHOLD = 70.0   # Laplacian mean
HOLO_COMBINED_THRESHOLD = 60.0        # spread * noise_factor
ART_HOLO_RATIO = 1.3                  # art must exceed border by this
BORDER_HOLO_RATIO = 1.2               # border must exceed art by this

# Reverse holo (name bar hf_decorr method)
REVERSE_HOLO_HF_DECORR = 0.055       # primary threshold
REVERSE_HOLO_ART_HFD_MAX = 0.026     # art must be below this

# Full art detection
FULL_ART_EDGE_FRAC = 0.05            # edge strip width
FULL_ART_MEAN_SAT_THRESHOLD = 65.0   # mean saturation
FULL_ART_HUE_STD_THRESHOLD = 18.0    # hue standard deviation
FULL_ART_COLORFUL_FRAC_THRESHOLD = 0.35  # fraction of colorful pixels
FULL_ART_MIN_EDGES_PASSING = 3       # out of 4 edges

# Gold detection
GOLD_HUE_RANGE = (15, 45)            # HSV hue range for gold
GOLD_COVERAGE_THRESHOLD = 0.40       # fraction of card surface

# 1st Edition stamp
STAMP_OCR_REGION = [0.02, 0.44, 0.24, 0.65]
STAMP_TIGHT_REGION = [0.03, 0.53, 0.15, 0.67]
STAMP_BLOB_CIRCULARITY = 0.65        # minimum circularity
STAMP_BLOB_AREA_MIN = 0.03           # fraction of region
STAMP_BLOB_AREA_MAX = 0.30           # fraction of region

# Shadowless
SHADOW_BRIGHTNESS_DROP = 15          # pixel value difference threshold

# Prerelease stamp
PRERELEASE_FUZZY_THRESHOLD = 0.70    # minimum fuzzy match score
```

### Not Implemented (Needs Development)

| Variant                  | Proposed Method                              | Priority |
|--------------------------|----------------------------------------------|----------|
| Cracked ice vs cosmos    | Holo texture pattern classification          | Low      |
| Holo swirl               | Spiral pattern detection in holo texture     | Low      |
| Red cheeks Pikachu       | Hue analysis in cheek region of base1-58     | Low      |
| Error cards (HP, symbol) | OCR on specific card regions                 | Low      |
| Rarity symbol (star)     | Small symbol classification at card bottom   | Medium   |
| BREAK landscape detect   | Aspect ratio analysis post-segmentation      | Medium   |
| Peelable Ditto           | Not possible from scans                      | N/A      |
| Shiny color palette      | Color histogram comparison vs normal palette | Medium   |
| Type-specific rev holo   | Pattern recognition for type symbols         | Low      |
| Cracked ice theme deck   | Holo texture classification                  | Low      |
| Holofoil through sleeve  | Multi-frame video or DB constraint           | High     |

---

## Key Detection Regions Reference

```
Standard Pokemon Card Layout (fractional coordinates):

   0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0
    |    |    |    |    |    |    |    |    |    |    |
0.0 +----+-------------------------------------------+----+
    |    |         NAME BAR (0.02-0.10)               |    |
0.1 |    +-------------------------------------------+    |
    |    |                                           |    |
    |    |              A R T W O R K                |    |
    |    |            (0.10-0.56 y)                  |    |
    |    |            (0.10-0.90 x)                  |    |
0.5 |    |                                           |    |
    | B  |  [1st Ed stamp: 0.02-0.24 x, 0.44-0.65 y]|    |
    | O  +-------------------------------------------+ B  |
    | R  |                                           | O  |
    | D  |           T E X T  B O X                  | R  |
    | E  |         (0.58-0.92 y)                     | D  |
    | R  |         (0.08-0.92 x)                     | E  |
    |    |                                           | R  |
0.9 |    +-------------------------------------------+    |
    |    |    SET SYMBOL (0.42-0.62 x, 0.86-0.97 y)  |    |
1.0 +----+-------------------------------------------+----+
```

### Stamp/Mark Positions by Era

| Stamp Type        | Position (x, y ranges)           | Era Applicability       |
|-------------------|----------------------------------|-------------------------|
| 1st Edition       | [0.02-0.24, 0.44-0.65]           | WotC (base1-neo4)       |
| 1st Edition tight | [0.03-0.15, 0.53-0.67]           | WotC (base1-neo4)       |
| Shadowless edge   | [0.90-1.00, 0.00-1.00] (right)   | base1 ONLY              |
|                   | [0.00-1.00, 0.90-1.00] (bottom)  |                         |
| EX set stamp      | [0.50-0.90, 0.30-0.58]           | EX (ex7-ex16)           |
| Prerelease        | [0.55-0.95, 0.30-0.58]           | All eras                |
| Staff             | [0.55-0.95, 0.20-0.45]           | All eras                |
| Black star promo  | [0.42-0.62, 0.86-0.97]           | WotC (basep)            |
| Modern promo      | [0.42-0.62, 0.86-0.97]           | SWSH/SV (swshp, svp)    |
| Cosmos holo       | [0.05-0.95, 0.08-0.52]           | SV (sv1-sv10)           |
| League stamp      | [0.55-0.95, 0.30-0.55]           | WotC/EX/DP              |
| Pokemon Center    | Artwork area (varies)            | svp promos              |

---

## Era-to-Set Mapping (Complete)

| Era | Era Key      | Set IDs                                                              |
|-----|--------------|----------------------------------------------------------------------|
| 1   | `wotc_base`  | base1-base6, basep, gym1, gym2, neo1-neo4, ecard1-ecard3, bp, si1   |
| 2   | `ex_era`     | ex1-ex16, np, pop1-pop5, tk1a, tk1b, tk2a, tk2b                     |
| 3   | `dp_era`     | dp1-dp7, dpp, pl1-pl4, pop6-pop9                                    |
| 4   | `hgss_era`   | hgss1-hgss4, hsp, col1, ru1                                         |
| 5   | `bw_era`     | bw1-bw11, bwp, dv1, dc1                                             |
| 6   | `xy_era`     | xy0-xy12, xyp, g1, me1, me2, me2pt5                                 |
| 7   | `sm_era`     | sm1-sm12, sm35, sm75, sm115, smp, sma, det1, mcd18, mcd19           |
| 8   | `swsh_era`   | swsh1-swsh12, swsh12pt5, swshp, swsh35, swsh45, swsh45sv, cel25,    |
|     |              | cel25c, pgo, fut20, mcd21, mcd22, swsh9tg-swsh12tg, swsh12pt5gg     |
| 9   | `sv_era`     | sv1-sv10, sv3pt5, sv4pt5, sv6pt5, sv8pt5, sve, svp                   |

---

## Valid Variants by Set (Quick Reference)

| Set Pattern            | Valid Variants                                                          |
|------------------------|-------------------------------------------------------------------------|
| base1                  | normal, holofoil, 1st_ed, 1st_ed_holo, unlimited, unl_holo, shadowless, shadowless_holo |
| base2, base3, base5    | normal, holofoil, 1st_ed, 1st_ed_holo, unlimited, unl_holo            |
| base4                  | normal, holofoil                                                        |
| base6                  | normal, holofoil, reverse_holofoil (fireworks)                          |
| gym1, gym2             | normal, holofoil, 1st_ed, 1st_ed_holo, unlimited, unl_holo            |
| neo1-neo4              | normal, holofoil, 1st_ed, 1st_ed_holo, unlimited, unl_holo            |
| ecard1-ecard3          | normal, holofoil, reverse_holofoil (cosmic)                            |
| basep, np, dpp, hsp    | normal, holofoil, promo                                                |
| ex1-ex6                | normal, holofoil, reverse_holofoil (unstamped)                         |
| ex7-ex16               | normal, holofoil, reverse_holofoil (stamped)                           |
| dp1-dp7, pl1-pl4       | normal, holofoil, reverse_holofoil                                     |
| hgss1-hgss4, col1      | normal, holofoil, reverse_holofoil                                     |
| bw1-bw11               | normal, holofoil, reverse_holofoil, full_art                           |
| xy1-xy12, g1           | normal, holofoil, reverse_holofoil, full_art                           |
| sm1-sm12               | normal, holofoil, reverse_holofoil, full_art, gold, rainbow_rare       |
| swsh1-swsh12           | normal, holofoil, reverse_holofoil, full_art, gold, rainbow_rare       |
| sv1-sv10               | normal, holofoil, reverse_holofoil, full_art, gold, rainbow_rare       |
| pop1-pop9              | normal, holofoil                                                        |
| Trainer kits           | normal                                                                  |
| McDonald's sets        | normal, holofoil                                                        |
| Shiny vaults           | normal, holofoil                                                        |
| Trainer Galleries      | normal, holofoil                                                        |
| Energy (sve)           | normal                                                                  |
| bp (Best of Game)      | holofoil                                                                |
| si1 (Southern Islands) | normal, holofoil                                                        |
| dv1 (Dragon Vault)     | normal, holofoil                                                        |
| dc1 (Double Crisis)    | normal, holofoil                                                        |
| cel25 (Celebrations)   | normal, holofoil (all cosmos holo)                                      |
| cel25c (Classic Coll.) | holofoil (all reprints)                                                 |
| ru1 (Pokemon Rumble)   | normal (all with Rumble stamp)                                          |
| det1 (Det. Pikachu)    | normal, holofoil                                                        |

---

## Implementation Priority

For building detection code, implement in this order:

1. **Reverse holo** (all eras) -- hf_decorr on name bar >= 0.055. Already working at 100%.
2. **1st Edition stamp** (WotC only) -- PaddleOCR + blob detection. ~85% accuracy, high price impact.
3. **Full art** (BW+ era) -- edge strip saturation/hue analysis. Enables correct OCR strategy.
4. **Prerelease/Staff stamps** (all eras) -- PaddleOCR for text. Moderate price impact.
5. **Gold/Rainbow** (SM+ era) -- HSV color analysis. Visual-only variant, not separate TCGCSV subtype.
6. **EX set stamp** (ex7-ex16) -- OCR + classifier. 68.8% accuracy needs improvement.
7. **Shadowless** (base1 only) -- edge gradient analysis. High price impact, narrow scope.
8. **Holofoil from binder scans** -- Multi-frame video or DB constraints. Currently ~0% accuracy.
9. **Shiny color detection** -- color histogram comparison. Multiple eras, moderate priority.
10. **Rarity symbol** (SV era) -- small symbol classification. Enables IR/SIR/HR distinction.

---

## Known Issues and Gotchas

1. **Holofoil vs Normal is undetectable through binder sleeves** -- the single biggest variant detection gap. Must use DB constraints (some card numbers are holo-only) or multi-frame video.

2. **EX-era stamp detection at 68.8%** -- DINOv2+logistic regression struggles with binder scan quality. OCR on stamp text is more promising but stamps are semi-transparent.

3. **Gold detection era gate too high** -- current code gates at era >= 7 (SM) but XY (era 6) has gold secret rares. Needs lowering.

4. **PRERELEASE OCR garbling** -- "PRENELEMEE" is a common OCR misread. Fuzzy threshold 0.70 prevents false matches but may miss valid stamps at lower OCR quality.

5. **Possessive prefix problem** -- affects Gym/Rocket era and SV Trainer's Pokemon. Owner name on separate line from Pokemon name.

6. **BREAK card rotation** -- landscape cards in portrait binder slots need rotation handling in segmenter.

7. **Peelable Ditto** -- fundamentally undetectable from scans. Must use DB lookup.

8. **Same-set art variants** -- DINOv2 can confuse similar artwork variants (e.g., Mew regular vs Mew alt art from same set). 1 known failure in eval.
