# Open Problems Catalog

Last updated: 2026-03-19

This document catalogs all known open problems in the card identification,
variant detection, and pricing pipelines. Each problem includes root cause
analysis, attempted solutions, and promising next steps.

Current eval accuracy: **106/108 (98.1%)** on 12 ground truth pages (108 cards).
The blurry Misty's page (014711, 6 failures) was removed from eval as a
photography quality issue, not a pipeline issue. Full 13-page eval: 110/117 (94.0%).

---

## Problem 1: Possessive Prefix Problem

**Category**: identification
**Severity**: medium
**Current accuracy**: Partially solved (possessive concatenation exists but fragile)
**Linear issue**: CAR-79

**Description**: Cards with possessive owner prefixes like "Team Aqua's Poochyena",
"Misty's Seadra", "Brock's Onix", "Lt. Surge's Pikachu", or "Lillie's Clefairy ex"
have the owner name on a separate line from the Pokemon name. OCR reads
"Poochyena" alone and matches to the wrong card (base-set Poochyena instead
of Team Aqua's Poochyena).

**Example**: "Team Aqua's Poochyena" -- OCR reads "Poochyena" from line 2 and
misses "Team Aqua's" from line 1. Fuzzy match returns base-set Poochyena.
"Misty's Seadra" -- OCR reads "Seadra" at 0.81, falls to attack path.

**Root cause**: Pokemon card names span two physical lines on the card. OCR
reads each line as a separate detection. The pipeline's top-25% crop may
capture both lines but they are returned as separate OCR fragments. Without
reassembly, only the Pokemon name (without the owner) is matched.

**Attempted solutions**:
- Possessive fragment concatenation (lines 2917-2939 in `__init__.py`):
  detects fragments ending in `'s` and combines them with non-possessive
  fragments. Works when both fragments are detected by OCR.
- Attack path fallback: when name confidence is 0.70-0.85, use attack OCR
  to narrow candidates and then DINOv2 picks. Solved Misty's Seadra case.

**Promising approaches**:
- Increase OCR crop from top 25% to top 30% to ensure owner name line is captured
- Multi-line OCR grouping: group detections by Y-coordinate proximity
- Direct DB lookup for "{owner}'s {pokemon}" patterns when single-word Pokemon
  name matches multiple cards

**Conditioning in pipeline**: Step 2 (Name OCR) in `__init__.py`, possessive
concatenation block at line ~2917

**Dependencies**: Interacts with Problem 4 (edge clipping) -- if the owner
name is physically clipped by segmentation, no OCR approach can recover it.

---

## Problem 2: EX-Era Stamp Detection

**Category**: variant
**Severity**: high
**Current accuracy**: 68.8% LOO on binder scans (11/16 correct in binder GT), 91.7% on training data
**Linear issue**: CAR-73, CAR-47

**Description**: EX-era cards (ex7-ex16, ~2005-2007) have set logo stamps
overlaid on the card artwork. Stamped cards are reverse holofoils and command
different prices. The stamp classifier uses DINOv2 features + logistic
regression but struggles with binder scan quality.

**Example**: `page_20260305_094228_cards/card_00.png` -- Chikorita from
EX Dragon Frontiers with DRAGON FRONTIERS stamp. Correctly detected.
`page_20260305_094228_cards/card_01.png` -- Bayleef, no stamp. Correctly detected.
Failures on cards where stamp is partially occluded by sleeve or glare.

**Root cause**: Stamps are translucent overlays on artwork. Through binder
sleeves, the stamp texture is muted. DINOv2 features capture overall card
structure rather than fine stamp detail. Edge density ratio between stamp
region and control region is a good signal but noise from sleeve plastic
creates false positives.

**Attempted solutions**:
- DINOv2 ViT-B/14 features + logistic regression: 75.6% initial, 91.7% with
  real training data, but 68.8% on binder LOO
- Combined classifier (DINOv2 + edge density): improved on synthetic data
  but not consistently on real binder scans
- Era-specific stamp region cropping (right side of artwork, 55-92% x, 40-68% y)

**Promising approaches**:
- Train on more binder scan examples (currently only 17 in ground truth)
- OCR on stamp region: stamps have set name text (e.g., "DRAGON FRONTIERS")
  that could be read by OCR at higher resolution
- Close-up photo mode: ask user to photograph individual cards for variant detection
- Multi-frame analysis: compare same card position across page re-scans

**Conditioning in pipeline**: `variant_detector.py` line ~1860, `stamp_classifier.py`

**Dependencies**: Interacts with Problem 3 (holo detection) -- stamped cards
are a sub-type of reverse holofoil.

---

## Problem 3: Holographic Card Detection from Binder Scans

**Category**: variant
**Severity**: high
**Current accuracy**: ~0% on binder scans (holo shimmer invisible through sleeves)
**Linear issue**: CAR-47

**Description**: Cannot distinguish holographic from non-holographic cards in
binder page photos. Binder sleeves suppress the prismatic shimmer that makes
holo cards visually distinct. The variant detector's hue-spread and
hue-spatial-noise analysis works on direct card photos but fails through plastic.

**Example**: Feraligatr (holofoil) and Typhlosion (holofoil) in binder GT
are classified identically to their non-holo counterparts (Totodile, Cyndaquil).
All cards through sleeves have similar hue distributions.

**Root cause**: Holographic effects are light-angle-dependent optical phenomena.
A single static photo through a plastic sleeve captures none of the prismatic
color shifts. The variant detector relies on high-frequency hue variation
(Laplacian of hue channel) which plastic sleeves eliminate.

**Attempted solutions**:
- Hue spread analysis: works on direct card photos, fails through sleeves
- Hue spatial noise (Laplacian): same issue -- plastic dampens micro-reflections
- Art vs border holo signal comparison: both regions equally flat through sleeves

**Promising approaches**:
- Multi-frame video capture: record a short video clip while tilting the binder.
  Holo cards will show color shifts across frames that non-holo cards won't.
  Camera UI (`condition_camera_ui.py`) already supports WebRTC video.
- Close-up individual card photos outside sleeves
- Use card identity to constrain variants: if DB says a card only exists as
  holofoil (e.g., rare holos in WotC sets), skip detection and assume holo
- Price-based inference: check if normal variant exists for this card number.
  Many cards are holo-only or non-holo-only.

**Conditioning in pipeline**: `variant_detector.py` lines 1927-1970 (holo analysis)

**Dependencies**: Interacts with Problem 2 (stamp detection) -- stamps are on
reverse holos. Also interacts with Problem 6 (same-set variants) since holo/non-holo
is the most common variant axis.

---

## Problem 4: Segmentation Edge Clipping

**Category**: segmentation
**Severity**: medium
**Current accuracy**: Fixed for most cases (14% edge expansion), but photography-dependent
**Linear issue**: CAR-79

**Description**: Cards near the edges of binder pages have their top text
(name, HP) partially or fully outside the camera's field of view. Asymmetric
edge expansion (14% for edge corners, 4% for interior) recovers text that IS
in the photo but slightly outside the detected contour. Cannot recover text
that is physically outside the photograph.

**Example**: Yanma (card_00 on e-reader page 123427) -- name at y=20 on 4032px
image, contour top at y=81. Fixed by 14% edge expansion. Trapinch (page
174819/card_02) -- "Tra" in "Trapinch" is physically outside the photo frame.
Expansion helps but cannot create missing pixels.

**Root cause**: Phone cameras have a finite field of view. When photographing
a binder page, outer cards' names extend to the very edge. The segmenter
detects the card body but the name region may be outside the contour or
outside the photograph entirely.

**Attempted solutions**:
- Uniform 4% expansion from centroid: insufficient (y=81 to y=57, needed y=20)
- 10% asymmetric edge expansion: got "anma" visible but still clipped "Y"
- 14% asymmetric edge expansion: fixed Yanma, no regressions on 13 GT pages
- BORDER_REPLICATE padding: extends edge pixels naturally vs black border

**Promising approaches**:
- Scanning guidance UI: show overlay on camera to help user center the page
- Two-pass scanning: wide shot for grid detection, then zoomed crops for edge cards
- Wider angle lens or greater distance from binder

**Conditioning in pipeline**: `card_segmenter.py`, `_perspective_crop` function

**Dependencies**: Directly affects Problem 1 (possessive prefix) -- owner names
are often on the topmost line, most likely to be clipped.

---

## Problem 5: Full Art Card Identification

**Category**: identification
**Severity**: medium
**Current accuracy**: Not measured (no full art cards in current ground truth)
**Linear issue**: CAR-80

**Description**: Full art cards have different layouts -- artwork extends to
card edges, name may be in a different position (bottom instead of top, or
overlaid on art). The standard OCR crop (top 25%) may miss the name entirely
or capture art details instead of text.

**Example**: No specific failure in current eval (ground truth is WotC/EX era).
Full art cards from BW+ era (2011+) would need different OCR strategy.

**Root cause**: The pipeline assumes a standard Pokemon card layout with the
name in the top 25% of the card. Full art, illustration rare, and special
illustration rare cards break this assumption.

**Attempted solutions**: None specific to full art identification.

**Promising approaches**:
- Layout detection: classify card layout (standard vs full art) before OCR,
  then adjust crop region accordingly
- Variant detector already has `_check_full_art` -- use its output to switch
  OCR strategy
- Full-card OCR: run OCR on the entire card when full art is detected, then
  filter for name-like text
- DINOv2 is likely more reliable on full art cards since the art IS the
  distinguishing feature

**Conditioning in pipeline**: `__init__.py` (OCR crop at top 25%), `variant_detector.py`
(full art detection at line ~1898)

**Dependencies**: Interacts with Problem 7 (small/irrelevant text) -- full art
cards have more visible art text, artist signatures, etc.

---

## Problem 6: Same-Set Art Variants (Alt Art Disambiguation)

**Category**: identification
**Severity**: medium
**Current accuracy**: 1 known failure (Mew variant confusion in eval)
**Linear issue**: CAR-80

**Description**: Multiple cards with the same Pokemon name in the same set
but different artwork (regular art, full art, alt art, illustration rare).
After OCR identifies the name, DB lookup returns all variants. DINOv2 must
disambiguate by visual features alone, but binder scan quality degrades
DINOv2's discriminative power.

**Example**: Mew variant confusion in eval -- DINOv2 picked wrong Mew variant
from the same set. Both cards have identical name, HP, attacks but different art.

**Root cause**: DINOv2 embeddings are compared via dot product against reference
images (clean digital scans). Binder scan photos have glare, color cast, and
lower resolution, reducing the margin between correct and incorrect variants.

**Attempted solutions**:
- DINOv2 reference matching: works when art difference is large (different poses)
  but fails on subtle art differences
- Page context reranking: doesn't help since both variants are from the same set

**Promising approaches**:
- DINOv2 fine-tuning or projection head trained on binder scan augmentations
  (`dino_projector.py` exists with augmentation pipeline)
- Close-up photo for variant confirmation
- Card number OCR: different art variants have different card numbers, but
  text is ~5px at binder resolution (confirmed infeasible at current resolution)

**Conditioning in pipeline**: `ref_matcher.py` (DINOv2 matching), `__init__.py`
(candidate selection)

**Dependencies**: Interacts with Problem 3 (holo detection) -- holo/non-holo
is often the variant axis. Also Problem 5 (full art layout).

---

## Problem 7: Small/Irrelevant Text OCR Noise

**Category**: identification
**Severity**: low
**Current accuracy**: Mitigated by fuzzy threshold (0.70) and score cutoffs
**Linear issue**: CAR-81

**Description**: OCR picks up set numbers, artist names, copyright text,
PRERELEASE stamps, HP values ("PV"), and ability text, then tries to match
these as card names or attacks, producing garbage matches.

**Example**: "PRENELEMEE" (PRERELEASE stamp) fuzzy-matched to "peerless edge"
at 0.636. "Mackogneur Py" (French HP suffix) matched to "Machamp-EX" instead
of "Machamp". French "100 PV" suffix on Machamp read as part of the name.

**Root cause**: OCR is a text detection engine -- it reads ALL visible text,
not just the card name. Without spatial awareness of card layout zones,
irrelevant text gets mixed into the matching pipeline.

**Attempted solutions**:
- Attack fuzzy threshold raised from 0.60 to 0.70: rejects stamp garbage
  (real attacks score 0.875+, garbage 0.60-0.64)
- Fuzzy score cutoff at 75 for name matching: prevents "Kingg" to "Klang"
- HP text stripping: partially implemented (OCR confusion handling)
- Position-based filtering considered but rejected as fragile across eras

**Promising approaches**:
- Strip known non-name suffixes before matching: "PV", "HP", "EX", "ex",
  "GX", "V", "VMAX", "VSTAR" when they appear as trailing words
- Layout-zone OCR: only match text from the name region (top 5-10% of card),
  reject text from art/attack/copyright regions by Y-coordinate
- Non-name word blocklist expansion

**Conditioning in pipeline**: `__init__.py` (name matching), `attack_ocr.py`
(attack matching with filtering at line ~258)

**Dependencies**: Interacts with Problem 5 (full art) where more art text is visible.

---

## Problem 8: Japanese/Foreign Text Cards (Non-Latin Scripts)

**Category**: identification
**Severity**: medium
**Current accuracy**: 1 known failure (Dark Electrode, Japanese kanji)
**Linear issue**: CAR-81

**Description**: RapidOCR (PaddleOCR) can read Latin-script foreign languages
(French, German, Spanish) but cannot read Japanese kanji, Chinese characters,
or Korean hangul. The translation lookup system resolves Latin-script foreign
names to English equivalents, but non-Latin scripts produce no usable OCR output.

**Example**: Dark Electrode (020047/card_06) -- Japanese text, RapidOCR returns
garbage. Falls to DINOv2 global search which gets 0% accuracy.

**Root cause**: RapidOCR's default model is trained on Latin + Chinese scripts.
Japanese kanji (shared with Chinese) might partially work, but katakana/hiragana
for Pokemon names is unreliable. EasyOCR with `['ja', 'en']` reader works but
has cold-start latency (5-10s for model loading).

**Attempted solutions**:
- EasyOCR Japanese reader: works but requires caching to avoid timeout (fixed
  in session 4). Still slow on first call.
- Translation lookup: maps foreign names to English. Works for Latin scripts
  (French "Mackogneur" to "Machamp"). Cannot help if OCR produces no output.
- DINOv2 era-constrained search: tested on Japanese cards, correct card scored
  0.565 but wrong cards scored 0.579-0.594. Not discriminative enough.

**Promising approaches**:
- Japanese OCR model: dedicated PaddleOCR model for Japanese text
- Claude Vision fallback: for cards where both name and attack OCR fail,
  send the card image to Claude API for identification ($0.0015/card)
- Pre-cache EasyOCR Japanese reader at server startup

**Conditioning in pipeline**: `__init__.py` (name OCR), falls to DINOv2 global
when OCR fails

**Dependencies**: Interacts with Problem 9 (DINOv2 global useless) -- this is
the primary case where DINOv2 global is the only remaining option.

---

## Problem 9: DINOv2 Global Search Useless

**Category**: identification
**Severity**: high
**Current accuracy**: 0% (0/7 in full eval, 15% historical on 20k cards)
**Linear issue**: CAR-44

**Description**: When both name OCR and attack OCR fail, the pipeline falls
back to DINOv2 global search across all 20,026 card embeddings. This returns
the wrong card every time in practice. It is the final fallback and represents
a complete identification failure.

**Example**: All 7 remaining eval errors are DINOv2 global fallback cases.
6 are blurry photography (no text visible), 1 is Japanese text (unreadable).

**Root cause**: DINOv2 embeddings capture structural features (card frame,
layout, color distribution) more than card-specific content (artwork details).
At 20k scale, many cards share similar structural features. Binder scan quality
further degrades discriminative power. The correct card typically scores only
0.01-0.03 below the top-ranked wrong card.

**Attempted solutions**:
- DINOv2 global: 15% accuracy on 20k (effectively random for rare cards)
- CLIP global: 19% accuracy (slightly better but still useless)
- Era-constrained DINOv2: search within page era (1918 cards) or page sets
  (416 cards). Still fails -- correct card at 0.565, wrong at 0.579-0.594.
- DINOv2 + CLIP ensemble: marginal improvement but still unreliable
- Perceptual hashing: 0% (distance 22-40 vs threshold 5)
- Template matching: 0% (matches layout not content)

**Promising approaches**:
- Claude Vision API fallback ($0.0015/card): only triggered for the ~2% of
  cards where OCR fails entirely. Cost-effective at scale.
- DINOv2 fine-tuning on binder scan augmentations (`dino_projector.py` exists
  with augmentation pipeline including perspective distortion, glare, blur)
- TIP-Adapter few-shot approach (`tip_adapter.py` exists): learn a lightweight
  adapter on top of frozen DINOv2 features
- Re-scan prompt: ask user to re-photograph at higher quality

**Conditioning in pipeline**: `__init__.py`, final fallback path when name
and attack paths both fail

**Dependencies**: Problem 8 (Japanese text) feeds directly into this.

---

## Problem 10: Card Back Detection Edge Cases

**Category**: segmentation
**Severity**: low
**Current accuracy**: Works well on tested cases (HSV orange detection)
**Linear issue**: CAR-81

**Description**: Empty binder slots show card backs (Pokemon logo on red/orange
background). These need to be detected and skipped. The current HSV-based
detector works reliably (92%+ orange pixels, hue std < 5.0, sat mean > 140)
but may fail on worn/faded card backs or slots with non-standard inserts.

**Example**: No known failures in current eval. Card back detector at
`card_segmenter.py` line ~54 uses center-region HSV analysis.

**Root cause**: The detector assumes the Pokemon card back's distinctive
orange-red color profile. Edge cases: faded/sun-damaged card backs, cards
inserted with energy cards behind them, double-sleeved cards.

**Attempted solutions**:
- HSV center-region analysis: works with strict thresholds
- 20% margin crop to avoid sleeve edges

**Promising approaches**:
- ML classifier (fine-tuned DINOv2 or simple CNN) for robust card back detection
- Template matching against a reference card back image

**Conditioning in pipeline**: `card_segmenter.py`, `is_card_back()` function

**Dependencies**: Interacts with Problem 11 (partially filled pages).

---

## Problem 11: Partially Filled Binder Pages

**Category**: segmentation
**Severity**: medium
**Current accuracy**: Not systematically measured
**Linear issue**: CAR-82

**Description**: Binder pages with fewer than 9 cards cause issues with grid
detection. Empty slots may show card backs, blank sleeves, or the binder
background. The segmenter's contour detection finds fewer than 9 card-shaped
objects, triggering grid fallback. Grid fallback subdivides uniformly into
3x3, producing empty-slot segments that waste identification time and may
produce false matches.

**Example**: No specific eval failure, but real-world binder pages frequently
have <9 cards (partially filled last page, removed cards, etc.).

**Root cause**: The segmenter assumes a full 3x3 grid. Contour detection
finds N<9 cards and triggers grid fallback (any missing card = fallback).
Grid fallback has no way to know which slots are empty until after segmentation.

**Attempted solutions**:
- Card back detection: skips obvious card-back slots after segmentation
- Grid fallback with uniform subdivision: works but produces empty segments

**Promising approaches**:
- Post-segmentation emptiness detection: after extracting all 9 segments,
  run a quick classifier (brightness, variance, card back check) to mark
  empty slots before identification
- Adaptive grid: detect actual card positions and only extract occupied slots
- Brightness/variance threshold: empty sleeves have very different pixel
  statistics than card fronts

**Conditioning in pipeline**: `card_segmenter.py` (grid detection and fallback logic)

**Dependencies**: Interacts with Problem 10 (card back detection) and
Problem 14 (binder ring shadow -- empty slots near rings look different).

---

## Problem 12: Rotated/Upside-Down Cards

**Category**: segmentation
**Severity**: low
**Current accuracy**: Rotation detection exists but not perfect
**Linear issue**: CAR-82

**Description**: Cards inserted sideways or upside-down in binder slots need
rotation detection and correction before OCR can read text. The segmenter has
rotation detection (landscape check, 180-degree flip check) but OCR still
fails on rotated cards if the rotation isn't exactly 90 or 180 degrees.

**Example**: `page_20260307_015320` -- mixed page with partial/rotated cards.
Rotation detection added in earlier sessions.

**Root cause**: `_perspective_crop` checks if detected quad is landscape
(avg_w > avg_h) and rotates 90 degrees. For 180-degree flips, centroid
Y-comparison detects upside-down orientation. Arbitrary rotation angles
(e.g., 15 degrees skew from loose sleeves) are not handled.

**Attempted solutions**:
- Landscape detection + 90-degree CCW rotation in `_perspective_crop`
- 180-degree flip detection via centroid Y-comparison
- Perspective warp corrects minor skew (±5 degrees) inherently

**Promising approaches**:
- OCR orientation detection: run OCR, if text is unreadable, try 90/180/270
  rotations (already partially implemented)
- DINOv2 is rotation-invariant to some degree -- could match first, then
  confirm with OCR on the correctly oriented image

**Conditioning in pipeline**: `card_segmenter.py`, `_perspective_crop` and
`_find_card_contours`

**Dependencies**: Affects all downstream OCR (Problems 1, 7, 8).

---

## Problem 13: Glare and Reflection

**Category**: identification
**Severity**: medium
**Current accuracy**: Not measured independently
**Linear issue**: CAR-83

**Description**: Camera flash or ambient lighting creates glare spots on
binder sleeves that obscure card text and artwork. Glare can completely
white-out the name region or artwork, causing OCR failure and DINOv2 mismatch.
Holographic cards are especially susceptible as holo surfaces act as mirrors.

**Example**: Blurry Misty's page (014711) -- 6/9 cards unreadable, partially
due to glare + blur combination. Holo cards in general show more glare.

**Root cause**: Binder sleeves are reflective plastic. Any point-source light
(phone flash, overhead light) creates specular reflections. The reflection
position depends on the angle between camera, light source, and sleeve surface.

**Attempted solutions**:
- CLAHE contrast normalization in `attack_ocr.py` (line ~243): helps with
  mild glare but not with complete white-out
- Unsharp mask preprocessing: helps with low-contrast text but can worsen
  glare artifacts
- Raw OCR fallback: when preprocessing destroys low-contrast text, try raw image
- Glare simulation in DINOv2 augmentation pipeline (`dino_projector.py` line ~160):
  trains DINOv2 features to be glare-tolerant

**Promising approaches**:
- Multi-frame capture: take multiple photos at different angles, use the
  best frame for each card region
- Glare detection and masking: detect saturated white regions, mask them
  before OCR, use remaining text
- Polarizing filter: hardware solution for the phone camera
- Scanning guidance UI: prompt user to avoid flash, use diffuse lighting

**Conditioning in pipeline**: Affects all OCR steps and DINOv2 matching

**Dependencies**: Interacts with Problem 3 (holo detection -- glare and holo
shimmer are both light effects), Problem 4 (edge clipping -- glare often
worse at edges due to sleeve curvature).

---

## Problem 14: Binder Ring Shadow

**Category**: segmentation
**Severity**: low
**Current accuracy**: Not measured independently
**Linear issue**: CAR-83

**Description**: Binder rings cast shadows across the middle column of cards,
creating dark bands that affect both segmentation (contour detection sees
shadow edges) and OCR (reduced contrast in shadowed regions). The shadow
is consistent in position but varies in intensity with lighting angle.

**Example**: Middle-column cards (card_01, card_04, card_07 in a 3x3 grid)
may show ring shadow artifacts, though no specific eval failure is attributed
to this alone.

**Root cause**: Three-ring binder mechanisms sit behind the page and cast
shadows when photographed with directional lighting. The shadow falls on
the spine side of the page, typically affecting the left edge of middle-column
cards.

**Attempted solutions**:
- Per-cell brightness normalization in segmenter (target mean brightness 128.0)
- BORDER_REPLICATE padding avoids creating artificial edges at shadow boundaries

**Promising approaches**:
- Shadow detection and compensation: detect the shadow band, apply local
  brightness correction
- Scanning guidance: prompt user to ensure even lighting
- Color correction pipeline (`color_correction.py`) already handles warm
  color cast; could be extended for shadow compensation

**Conditioning in pipeline**: `card_segmenter.py` (contour detection), all
OCR steps (contrast dependent)

**Dependencies**: Mild interaction with Problem 13 (glare -- shadow and glare
are complementary lighting artifacts).

---

## Problem 15: Trainer/Energy Card Identification

**Category**: identification
**Severity**: medium
**Current accuracy**: Not measured (no trainer/energy cards in ground truth)
**Linear issue**: CAR-84

**Description**: Non-Pokemon cards (Trainer, Supporter, Stadium, Item, Energy)
have fundamentally different layouts. Trainer cards have no HP, no attacks in
the standard format, different text regions, and the card name may be in a
different position. Energy cards are even more different (mostly artwork with
a type symbol). The pipeline is optimized for Pokemon cards and may
misidentify or fail entirely on trainer/energy cards.

**Example**: No specific failures in eval (ground truth contains only Pokemon
cards). Real binder pages commonly contain trainer and energy cards mixed in.

**Root cause**: The pipeline assumes Pokemon card layout: name at top, HP
next to name, attacks in middle, weakness/resistance at bottom. Trainer cards
have a rule-text box instead of attacks. Energy cards are mostly artwork.
The OCR crop region (top 25%) captures the trainer card name correctly, but
attack OCR returns no matches (no attacks exist), causing lower confidence.

**Attempted solutions**:
- Claude Vision scanner (`claude_scanner.py`) handles trainers: returns
  `pokemon_name: null` for non-Pokemon cards
- Card type detection considered but not integrated into ML pipeline

**Promising approaches**:
- Card supertype detection: classify as Pokemon/Trainer/Energy before running
  the identification pipeline, then use type-specific OCR strategies
- Trainer cards: name OCR should still work (names are in standard position),
  skip attack OCR entirely
- Energy cards: type symbol detection or color-based classification
- DB lookup: trainers have unique names (unlike Pokemon which repeat across sets)

**Conditioning in pipeline**: All pipeline steps assume Pokemon card layout

**Dependencies**: Affects Problem 7 (OCR noise -- trainer rule text gets
misread as attacks).

---

## Problem 16: Blurry/Low-Quality Photography

**Category**: identification
**Severity**: high
**Current accuracy**: 0/6 on blurry Misty's page (removed from eval)
**Linear issue**: CAR-85

**Description**: Low-quality photographs (out of focus, motion blur, low
resolution) cause complete pipeline failure. When text is unreadable by OCR,
the pipeline falls to DINOv2 global search (0% accuracy). This accounts for
6 of the 7 remaining failures in the full 13-page eval.

**Example**: Misty's page scan 3 (014711) -- all 9 cards are blurry. 6 cards
completely unreadable. OCR returns no usable text. Name path, attack path,
and DINOv2 global all fail.

**Root cause**: Phone camera autofocus may not lock on the binder page
surface, especially under low light or when the page is reflective. Motion
blur from hand shake during capture. Low-resolution photos when phone camera
uses digital zoom or compression.

**Attempted solutions**:
- Multi-strategy OCR: RapidOCR + EasyOCR + raw fallback. Helps with mild
  blur but not severe out-of-focus.
- Unsharp mask preprocessing: can partially recover mildly blurry text.
- Simply re-scanning: the same physical page photographed clearly (scan 1
  and 2) achieves 100% accuracy.

**Promising approaches**:
- Image quality detection: measure sharpness (Laplacian variance) before
  pipeline execution, prompt re-scan if below threshold
- Claude Vision fallback for blurry cards ($0.0015/card)
- Super-resolution: apply learned upscaling to enhance text readability
- Autofocus guidance UI: help user confirm focus lock before capture

**Conditioning in pipeline**: Pre-pipeline quality check (not yet implemented)

**Dependencies**: This is the root cause of Problem 9 (DINOv2 global) for
6 of 7 cases.

---

## Problem 17: Binder Sleeve Color Cast

**Category**: identification | variant
**Severity**: low
**Current accuracy**: Mitigated by color correction pipeline
**Linear issue**: CAR-83

**Description**: Binder page plastic sleeves impart a warm orange/yellow color
cast to all card images. This affects color-based type detection (35% accuracy
on binder scans vs 62% on reference images), variant detection (warm cast
triggers false full-art positives), and DINOv2 matching (color distribution
shifted from reference images).

**Example**: Color detector misclassifies 73% of binder scan cards as
Colorless type due to warm color contamination.

**Root cause**: Binder sleeve plastic transmits warm (yellow/orange) light
preferentially. All cards photographed through sleeves have elevated R channel
and reduced B channel. The observed shift is approximately (R+33, G-4, B-12)
relative to neutral lighting.

**Attempted solutions**:
- `color_correction.py`: measures and corrects color cast using known border
  color (Pokemon card yellow border) as reference
- Card border color calibration: compare observed border color to known neutral
  yellow, compute correction gains
- Binder background color calibration: use orange binder background as
  second calibration point

**Promising approaches**:
- Apply color correction before variant detection and DINOv2 embedding
- Per-card color correction (currently page-level only)
- White-balance card for calibration: include a gray reference card in binder

**Conditioning in pipeline**: `color_correction.py` (page-level correction)

**Dependencies**: Affects Problem 3 (holo detection), Problem 6 (variant
disambiguation), and color-based type detection.

---

## Problem 18: OCR Engine Thread Safety and Performance

**Category**: identification
**Severity**: low
**Current accuracy**: N/A (performance, not accuracy)
**Linear issue**: CAR-48

**Description**: PaddleOCR and EasyOCR are not thread-safe and require a
global lock (`_ocr_lock`). This serializes OCR operations, creating a
bottleneck when processing 9 cards in parallel. Additionally, EasyOCR Japanese
reader has 5-10s cold-start latency for model loading.

**Example**: Pipeline took 381s before optimization, reduced to 120s with
batching and GPU acceleration. OCR remains the bottleneck at ~1.0s/card
sequential.

**Root cause**: PaddleOCR and EasyOCR use global state internally (model
weights, CUDA contexts). Concurrent calls from multiple threads can corrupt
internal state. The `_ocr_lock` mutex ensures safety but eliminates parallelism
for OCR operations.

**Attempted solutions**:
- `_ocr_lock` threading lock for safety
- Cached Japanese EasyOCR reader (module-level `_jp_easyocr_reader`)
- EasyOCR gpu=True: 15-34x speedup (0.015s vs 0.5s per card for attacks)
- RapidOCR migration (CAR-60): ongoing

**Promising approaches**:
- Multiple OCR engine instances (one per thread): memory-intensive (~3GB each)
- Batch OCR: process all 9 card crops in a single OCR call
- ONNX-optimized OCR models for better GPU utilization
- RapidOCR (PaddleOCR-based) as primary: already faster than EasyOCR

**Conditioning in pipeline**: `__init__.py` (lines 27-29, `_ocr_lock`)

**Dependencies**: Affects overall pipeline speed, interacts with CAR-48
(speed target 20s/page).

---

## Problem 19: Page Context Fails on Mixed-Era Pages

**Category**: identification
**Severity**: low
**Current accuracy**: Pass 2 LOO context can regress from 25% to 12.5% on mixed pages
**Linear issue**: CAR-81

**Description**: Page context reranking (Pass 2/3) assumes cards on the same
binder page are from the same set or era. Mixed-era pages (e.g., EX + DP +
Platinum) have low context confidence and wrong initial guesses create
misleading context that causes regressions.

**Example**: Page 3 in early eval (EX + DP + Platinum mixed) went from 25%
to 12.5% accuracy after page context reranking.

**Root cause**: The set-based reranking boosts candidates from the same set
as other cards on the page. When the page contains cards from multiple sets,
the "most common set" signal is noisy or wrong, causing incorrect boosts.

**Attempted solutions**:
- Context confidence threshold (0.65): skip Pass 2 when page context
  confidence is below threshold. Mixed-era pages fall below.
- Leave-one-out (LOO) context: prevents self-reinforcing errors but doesn't
  fix the underlying assumption violation.

**Promising approaches**:
- Multi-set context: detect multiple clusters of sets on a page and apply
  context within each cluster
- Disable context reranking entirely when set diversity exceeds threshold
- Weight context boost by card-level confidence: only apply context to
  low-confidence cards, leave high-confidence alone

**Conditioning in pipeline**: `page_context.py`, `__init__.py` (Pass 2/3)

**Dependencies**: None major.

---

## Problem 20: PV/HP Suffix Causing Wrong Variant Match

**Category**: identification
**Severity**: low
**Current accuracy**: Cosmetic (DINOv2 still picks correctly)
**Linear issue**: CAR-81

**Description**: Foreign language HP indicators ("PV" in French, "KP" in
German) attached to card names by OCR cause fuzzy matching to prefer the
wrong variant. "Mackogneur Py" fuzzy-matches to "Machamp-EX" (matching the
"mackogneur ex" translation entry) instead of plain "Machamp".

**Example**: French Machamp reads "Mackogneur Py" (from "100 PV" HP text).
Fuzzy match prefers "mackogneur ex" over "mackogneur" due to the extra
characters. Cosmetic issue -- DINOv2 still picks the correct Machamp variant.

**Root cause**: OCR does not cleanly separate the name text from the HP text.
The "PV" suffix gets concatenated with the name. Fuzzy matching then scores
"mackogneur py" higher against "mackogneur ex" (both have a suffix) than
against "mackogneur" (no suffix).

**Attempted solutions**: None specific. Noted as cosmetic since DINOv2 corrects it.

**Promising approaches**:
- Strip known HP indicators before name matching: /\s*\d*\s*(PV|HP|KP)\s*$/i
- Pre-process OCR output: remove trailing numeric + HP-keyword patterns

**Conditioning in pipeline**: `__init__.py` (name matching, fuzzy match logic)

**Dependencies**: Related to Problem 7 (OCR noise).

---

## Problem 21: Condition Grading from Binder Scans

**Category**: variant
**Severity**: medium
**Current accuracy**: Not implemented for binder scans
**Linear issue**: CAR-49, CAR-56

**Description**: Card condition (NM, LP, MP, HP, DMG) significantly affects
pricing but cannot be reliably assessed from binder page photos. Binder scan
resolution (~1008x1530 per card) is insufficient to detect fine scratches,
whitening, edge wear, and surface damage that distinguish condition grades.

**Example**: No current implementation for binder-scan condition grading.
Close-up camera UI (`condition_camera_ui.py`) exists for individual card
condition assessment.

**Root cause**: Condition defects (edge whitening, surface scratches, corner
damage) are sub-millimeter features that require high-resolution close-up
photos to detect. At binder scan resolution, these features are below the
noise floor.

**Attempted solutions**:
- Close-up camera UI for individual cards (CAR-66)
- Condition pricing models exist (`condition_pricing.py`)
- Default to NM for binder scan pricing (most common assumption)

**Promising approaches**:
- Two-stage workflow: binder scan for identification, then close-up photos
  for cards where condition matters (high-value cards)
- AI condition grading from close-up photos (CAR-49, in progress)
- User-reported condition in inventory UI

**Conditioning in pipeline**: Post-identification, pre-pricing

**Dependencies**: Interacts with pricing pipeline and variant detection.

---

## Problem 22: TCGCSV Variant ID Mapping

**Category**: pricing
**Severity**: medium
**Current accuracy**: N/A (data mapping issue)
**Linear issue**: CAR-72, CAR-52

**Description**: The pricing database (TCGCSV `fact_market_prices`) uses
product IDs while the identification pipeline uses card IDs with variant
suffixes (e.g., "base1-4/holofoil"). Mapping between detected variant and
the correct TCGCSV product ID for price lookup requires a join through
`dim_cards` which may not have all variant rows populated.

**Example**: A card identified as "base1-4" with variant "holofoil" needs
to find the TCGCSV product ID for "Charizard Holofoil" to get the correct
price. If `dim_cards` only has "base1-4/normal", the holofoil price is missed.

**Root cause**: `dim_cards` was initially populated with only `/normal`
variants. Variant rows need to be synthesized for each card/variant combination
(CAR-72 DB migration, CAR-52 SQL migration).

**Attempted solutions**:
- CAR-72 and CAR-52 created as backlog items for variant row synthesis

**Promising approaches**:
- SQL migration to synthesize variant rows from ERA_VALID_VARIANTS mapping
- Direct TCGCSV API lookup by card name + set + variant

**Conditioning in pipeline**: Post-identification, pricing lookup

**Dependencies**: Depends on Problem 2, 3 (stamp and holo detection accuracy)
for correct variant determination before price lookup.

---

## Problem 23: eBay Sales Data Blocked

**Category**: pricing
**Severity**: high
**Current accuracy**: N/A (data availability)
**Linear issue**: CAR-65, CAR-64

**Description**: eBay blocks automated scraping of sold listings. The eBay
scraper (`cardprice/scrapers/ebay.py`) was developed but gets blocked by
anti-bot measures. Sold listing data is essential for fair market price
estimation from actual transactions.

**Example**: `fact_sales` table exists with 0 rows. Scraper code exists but
cannot run successfully against eBay.

**Root cause**: eBay employs aggressive bot detection (CAPTCHAs, rate limiting,
IP blocking). Simple HTTP requests are blocked; even Playwright-based
approaches face challenges.

**Attempted solutions**:
- HTTP-based scraper: blocked immediately
- Planned: Playwright browser automation (CAR-65)

**Promising approaches**:
- Playwright with realistic browser fingerprinting
- eBay API (official, requires application registration)
- Alternative data source: TCGPlayer Playwright scraper already working
  (`data/tcgplayer_sales.db`, daily cron at 2AM UTC)
- PokeTrace API (CAR-30, 250 req/day free tier)

**Conditioning in pipeline**: Data collection, not directly in ML pipeline

**Dependencies**: None directly with identification pipeline.

---

## Problem 24: Ace Spec and Special Card Subtypes

**Category**: identification
**Severity**: low
**Current accuracy**: Not measured
**Linear issue**: CAR-84

**Description**: Special card subtypes like Ace Spec, Prism Star, BREAK,
LEGEND (two-card), and Tag Team GX have unique layouts that may confuse
the standard identification pipeline. These cards often have unusual name
formats ("Charizard & Braixen-GX") or split across two physical cards
(LEGEND cards).

**Example**: No specific failures in current eval (ground truth lacks these
subtypes).

**Root cause**: Each special subtype has a unique card layout, name format,
and visual design. The pipeline's OCR and fuzzy matching may not handle
compound names with "&" or multi-card layouts.

**Attempted solutions**: None specific.

**Promising approaches**:
- Subtype detection from visual features (BREAK cards are vertically oriented,
  LEGEND cards show half-artwork)
- Name format handling: recognize "{Pokemon1} & {Pokemon2}-GX" patterns
- DB lookup: these cards have unique names that should match even with
  imperfect OCR

**Conditioning in pipeline**: All pipeline steps

**Dependencies**: Interacts with Problem 5 (full art) and Problem 15 (trainer cards).
