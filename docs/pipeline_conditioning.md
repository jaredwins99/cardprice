# Pipeline Conditioning Logic

Last updated: 2026-03-19

This document specifies what checks, branching, and decision logic should occur
at each stage of the card identification + pricing pipeline. Each decision point
notes what information is needed, where in the code it lives (or should live),
its current implementation status, and priority.

Cross-references: `docs/open_problems.md` for root cause analysis;
`data/variant_tree.json` for era/set variant mapping;
`cardprice/ml/card_attributes.py` for O(1) attribute lookup.

---

## STAGE 1: SEGMENTATION

**Entry point**: `cardprice/ml/card_segmenter.py:segment_cards()`
**Called by**: `cardprice/server.py` line ~1781

```
Input: binder page photo (JPEG/HEIC from phone camera)
│
├── CHECK 1.1: Page fill level (full 3x3? partial page?)
│   ├── Information needed: contour count from card detection
│   ├── Current logic: if < 9 contours found, fall to grid fallback
│   │   - Fallback 1: contour-guided grid (6-8 contours -> compute
│   │     row/col boundaries from gutter midpoints)
│   │   - Fallback 2: page-outline + uniform grid subdivision
│   ├── Status: IMPLEMENTED (card_segmenter.py)
│   ├── Problem: partial pages (e.g., last page with 3-6 cards) trigger
│   │   grid fallback which creates empty card slots. No way to signal
│   │   "this slot is intentionally empty."
│   ├── Needed: card-back detection (Stage 2 Step 0) catches empty slots
│   │   post-segmentation. Good enough for now.
│   └── Priority: LOW — card-back detection covers this
│
├── CHECK 1.2: Card orientation (rotated? upside down?)
│   ├── Information needed: OCR result from rotated crop
│   ├── Current logic: rotation detection is deferred to Stage 2 (Step 1b
│   │   in identify_card_v2). If OCR fails at 0 degrees, tries 90CW and
│   │   90CCW. Validates rotation by checking DB for the OCR name + HP.
│   ├── Status: IMPLEMENTED (__init__.py lines 3949-4025)
│   ├── Problem: rotation detection runs per-card, wasting time if the
│   │   entire page is rotated. Could detect page-level rotation once.
│   ├── Needed: page-level EXIF orientation check before segmentation
│   └── Priority: LOW — per-card rotation works, just slower
│
├── CHECK 1.3: Edge card clipping (Open Problem 4)
│   ├── Information needed: card corner positions relative to image edges
│   ├── Current logic: asymmetric expansion — corners near image edges
│   │   get 14% expansion, interior corners get 4%. Uses BORDER_REPLICATE
│   │   to avoid black borders.
│   ├── Status: IMPLEMENTED (card_segmenter.py, `_perspective_crop`)
│   ├── Problem: cannot recover pixels physically outside the photograph.
│   │   User guidance (scanning overlay) would help.
│   └── Priority: MEDIUM — solved for fixable cases; remaining failures
│       are photography quality
│
└── Output: list of individual card image paths
```

---

## STAGE 2: NAME OCR

**Entry point**: `_run_name_and_hp()` in `cardprice/ml/__init__.py`
**OCR engine**: RapidOCR (ONNX Runtime) on top 25% of card, 3x upscale + unsharp mask

```
Input: individual card image
│
├── STEP 0: Card-back detection
│   ├── Information needed: average blue channel in center region
│   ├── Current logic: `_is_card_back()` checks blue hue dominance in
│   │   center 40% of card. Returns early with confidence 0.95.
│   ├── Status: IMPLEMENTED (__init__.py line 3890)
│   └── Priority: DONE
│
├── CHECK 2.1: Possessive prefix line detection (Open Problem 1)
│   ├── Information needed: multiple OCR text lines in top 30-35% crop
│   ├── Current logic: possessive fragment concatenation at __init__.py
│   │   ~line 2917. Detects fragments ending in `'s` and combines them
│   │   with non-possessive fragments on adjacent lines.
│   ├── Patterns to detect:
│   │   - "Team Aqua's" / "Team Magma's" -> prepend to Pokemon name
│   │   - "Misty's" / "Brock's" / "Lt. Surge's" / etc. -> prepend
│   │   - "Dark" (prefix, not possessive) -> prepend
│   │   - "Lillie's" / "Cynthia's" (SM/SWSH era trainers) -> prepend
│   ├── Status: PARTIALLY IMPLEMENTED — concatenation exists but fragile.
│   │   Relies on OCR detecting both lines. Fails when owner name line
│   │   is clipped by segmentation (Problem 4 interaction).
│   ├── Needed:
│   │   - Increase crop to top 30% when top-25% yields only one text line
│   │   - Multi-line grouping: sort OCR detections by Y-coordinate,
│   │     group lines within 15px vertical gap
│   │   - DB awareness: if "Seadra" matches 3 cards but "Misty's Seadra"
│   │     matches 1, prefer the owner-qualified name
│   └── Priority: MEDIUM — affects Gym/Rocket/Team Aqua/Magma sets
│
├── CHECK 2.2: Full art card layout (Open Problem 5)
│   ├── Information needed: whether artwork extends to card edges
│   ├── Current logic: none at OCR stage. `_check_full_art` exists in
│   │   variant_detector.py but runs AFTER identification, not before.
│   ├── Where to implement: in `_run_name_and_hp()`, before the OCR crop.
│   │   Run `_check_full_art` first; if True, use full-card OCR or
│   │   bottom-20% crop instead of top-25% crop.
│   ├── Status: NOT IMPLEMENTED at OCR stage
│   ├── Needed:
│   │   - Quick full-art check (edge strip saturation analysis, ~5ms)
│   │   - Alternative OCR crop for full art: bottom 20% (name is at bottom)
│   │     or full card with name-pattern filtering
│   │   - For illustration rares: name may be in a text box overlaid on art
│   └── Priority: MEDIUM — no full art cards in current ground truth, but
│       needed for SV/SWSH era cards
│
├── CHECK 2.3: Trainer/Energy card detection
│   ├── Information needed: supertype from layout heuristics or OCR text
│   ├── Current logic: no explicit trainer detection. Name OCR reads the
│   │   trainer name (e.g., "Professor's Research") and fuzzy matches
│   │   against all card names including trainers.
│   ├── Where to implement: `_run_name_and_hp()` or as a pre-filter
│   ├── Status: PARTIALLY IMPLEMENTED — works by accident since trainer
│   │   names are in the card_names list. HP detection returns None for
│   │   trainers (correct).
│   ├── Needed:
│   │   - Energy cards have no text name in the same position; need
│   │     type symbol detection or full-card OCR for "Basic Fire Energy" etc.
│   │   - Trainer layout differs by era (WotC trainers have text at top;
│   │     modern trainers have name at top + full art)
│   └── Priority: LOW — trainers generally OCR correctly already
│
├── CHECK 2.4: Foreign language card handling
│   ├── Information needed: OCR text that doesn't match English names
│   ├── Current logic: `_load_translation_names()` maps foreign names
│   │   (French, German, Spanish, Japanese, Chinese) to English equivalents.
│   │   OCR reads foreign text, fuzzy matches against English + all
│   │   translations. (See bug #10 in MEMORY.md: "Hoopa" from "hMochopa")
│   ├── Status: IMPLEMENTED (__init__.py lines 62-130)
│   ├── Problem: garbled foreign text can fuzzy-match to wrong English name.
│   │   E.g., "hMochopa" (garbled French "Machopeur") -> "Hoopa" (English).
│   │   Translation lookup correctly finds "Mackogneur" -> "Machamp" but
│   │   only when OCR reads the foreign name cleanly.
│   └── Priority: LOW — translations cover 5 languages, working well
│
├── CHECK 2.5: Short/garbage OCR name rejection
│   ├── Information needed: length and DB match of OCR name
│   ├── Current logic: reject names < 3 chars (__init__.py line 3939).
│   │   Prevents "tty" -> "Skitty", "ch" -> "Trapinch" mismatches.
│   ├── Status: IMPLEMENTED
│   └── Priority: DONE
│
├── CHECK 2.6: OCR fallback (raw vs preprocessed)
│   ├── Information needed: whether unsharp mask degrades low-contrast text
│   ├── Current logic: runs preprocessed (unsharp mask) first, then raw
│   │   OCR as fallback if preprocessing destroys text. Takes best result.
│   ├── Status: IMPLEMENTED (ocr_matcher.py)
│   └── Priority: DONE
│
└── Output: (ocr_name, ocr_confidence, ocr_raw, hp_value)
```

---

## STAGE 3: CANDIDATE GENERATION

**Entry point**: `_get_candidates_from_db()` in `cardprice/ml/__init__.py`
**Database**: PostgreSQL `dim_cards` table

```
Input: OCR name + HP value + color/type
│
├── CHECK 3.1: Name has possessive prefix?
│   ├── Information needed: possessive prefix detected in Stage 2
│   ├── Current logic: full name (with prefix) is used for DB query.
│   │   `_get_candidates_from_db(name="Misty's Seadra")` returns only
│   │   Misty's Seadra cards. Without prefix, returns all Seadra cards.
│   ├── Status: IMPLEMENTED (depends on Stage 2 concatenation quality)
│   └── Priority: depends on Stage 2 improvements
│
├── CHECK 3.2: Type/color used as scoring signal, not filter
│   ├── Information needed: color_type from color_detector, color_conf
│   ├── Current logic: type is passed to `_score_candidates_combined()` for
│   │   scoring bonus but NOT used as a DB filter. This avoids dropping
│   │   correct candidates when color detection is wrong (e.g., "Colorless"
│   │   cards with colored backgrounds).
│   ├── Status: IMPLEMENTED (__init__.py lines 4036-4039)
│   └── Priority: DONE — correct design decision
│
├── CHECK 3.3: HP as a hard filter
│   ├── Information needed: hp_value from OCR
│   ├── Current logic: HP IS used as a DB filter when present. Dramatically
│   │   reduces candidate count (e.g., "Pikachu" 50+ -> "Pikachu HP 60" 5).
│   ├── Status: IMPLEMENTED
│   ├── Risk: HP OCR error (e.g., reads "80" as "60") eliminates the
│   │   correct card. Could soften to HP +/- 10 range.
│   └── Priority: LOW — HP OCR is quite reliable
│
├── CHECK 3.4: Multiple cards with same name?
│   ├── Information needed: candidate count from DB query
│   ├── Current logic (branching):
│   │   - 0 candidates: name OCR failed or wrong name -> attack path (Stage 5)
│   │   - 1 candidate: DINOv2 sanity check (score >= 0.30) -> accept or reject
│   │   - 2-20 candidates: combined DINOv2 + attack scoring (Stage 4)
│   │   - 20+ candidates: combined scoring with higher threshold (0.50)
│   ├── Status: IMPLEMENTED (__init__.py lines 4059-4174)
│   └── Priority: DONE
│
├── CHECK 3.5: Name path failure -> attack fallback
│   ├── Information needed: whether name path produced an acceptable result
│   ├── Current logic: `name_path_failed` flag tracks when:
│   │   - Single candidate rejected by DINOv2 (score < 0.30)
│   │   - Combined scoring below threshold
│   │   - Wrong OCR name from rotation fallback
│   │   Falls through to Stage 5 (attack-based identification).
│   ├── Status: IMPLEMENTED (__init__.py lines 4058, 4089, 4174)
│   └── Priority: DONE
│
└── Output: list of candidate card_ids (0 to ~50)
```

---

## STAGE 4: VISUAL MATCHING (DINOv2 + Attack Scoring)

**Entry point**: `_score_candidates_combined()` in `cardprice/ml/__init__.py`
**Also**: `_dino_dot_product_against_refs()` for pure visual matching

```
Input: card image + candidate card_ids
│
├── CHECK 4.1: Candidate count determines scoring strategy
│   ├── 1 candidate: pure DINOv2 sanity check (>= 0.30)
│   ├── 2-3 candidates: DINOv2 dot product, threshold 0.35
│   ├── 4-10 candidates: DINOv2 + attack scoring, threshold 0.45
│   ├── 10+ candidates: higher threshold 0.50
│   ├── Status: IMPLEMENTED (__init__.py lines 4099-4105)
│   └── Priority: DONE
│
├── CHECK 4.2: OCR confidence modulates threshold
│   ├── Information needed: ocr_conf from Stage 2
│   ├── Current logic:
│   │   - ocr_conf >= 0.90: threshold capped at 0.35 (trust the name)
│   │   - ocr_conf >= 0.80: threshold capped at 0.40
│   │   - Clear gap (>= 0.04) between 1st and 2nd: threshold capped at 0.38
│   ├── Status: IMPLEMENTED (__init__.py lines 4114-4130)
│   └── Priority: DONE
│
├── CHECK 4.3: Low DINOv2 + low OCR -> also try attack path
│   ├── Information needed: dino_score and ocr_conf
│   ├── Current logic: if dino_score < 0.60 AND ocr_conf < 0.90, save
│   │   the ref_match result but don't return it yet. Also run attack
│   │   path and pick the better result.
│   ├── Rationale: low DINOv2 through sleeves may mean wrong OCR name.
│   │   Attack path provides independent signal.
│   ├── Status: IMPLEMENTED (__init__.py lines 4163-4170)
│   └── Priority: DONE
│
├── CHECK 4.4: Type scoring bonus
│   ├── Information needed: color_type, color_conf from Stage 2
│   ├── Current logic: matching type gives ~0.05 score bonus to candidates
│   │   of the right type. Not a hard filter.
│   ├── Status: IMPLEMENTED (inside _score_candidates_combined)
│   └── Priority: DONE
│
└── Output: best match card_id + confidence (or fall to Stage 5)
```

---

## STAGE 5: ATTACK-BASED IDENTIFICATION (Fallback)

**Entry point**: `identify_by_attacks()` in `cardprice/ml/attack_ocr.py`
**Called when**: name OCR failed OR name path produced low confidence

```
Input: card image (original, un-rotated if rotation was applied)
│
├── CHECK 5.1: Use original image, not rotated
│   ├── Information needed: whether rotation was applied in Step 1b
│   ├── Current logic: `original_image_path` preserved at __init__.py
│   │   line 3873. When name_path_failed, attack OCR uses the original
│   │   image because rotated images have unreadable attack text.
│   ├── Status: IMPLEMENTED (__init__.py line 4189)
│   └── Priority: DONE
│
├── CHECK 5.2: HP filtering on attack candidates
│   ├── Information needed: hp_value from Stage 2, structured_attacks.json
│   ├── Current logic: when HP detected and >= 5 attack candidates, filter
│   │   to cards with matching HP. Keeps unknowns to avoid false negatives.
│   ├── Status: IMPLEMENTED (__init__.py lines 4205-4223)
│   └── Priority: DONE
│
├── CHECK 5.3: Era filtering from page context
│   ├── Information needed: page_era from identify_page_v2 pass 2
│   ├── Current logic: filter attack candidates to same-era cards when
│   │   page_era is known and >= 3 era-matched candidates exist.
│   │   Uses `_eras_compatible()` for fuzzy era matching.
│   ├── Status: IMPLEMENTED (__init__.py lines 4228-4239)
│   └── Priority: DONE
│
├── CHECK 5.4: High candidate count penalty
│   ├── Information needed: candidate count and score gap
│   ├── Current logic: when >= 30 candidates share the same attacks AND
│   │   score gap between 1st and 2nd is < 0.05, apply 15% penalty.
│   │   Many common attacks (e.g., "Tackle", "Scratch") match 50+ cards
│   │   where DINOv2 picks essentially randomly.
│   ├── Status: IMPLEMENTED (__init__.py lines 4252-4257)
│   └── Priority: DONE
│
├── CHECK 5.5: Attack intersection vs union
│   ├── Information needed: multiple detected attacks
│   ├── Current logic: intersection of per-attack candidate sets (fall back
│   │   to union if intersection is empty). Bug #3 from MEMORY.md.
│   ├── Status: IMPLEMENTED (attack_ocr.py)
│   └── Priority: DONE
│
├── CHECK 5.6: PRERELEASE stamp text rejection
│   ├── Information needed: OCR text matching stamp patterns
│   ├── Current logic: fuzzy threshold 0.70 (raised from 0.60) rejects
│   │   "PRENELEMEE" -> "peerless edge" (score 0.636). Real attacks
│   │   score 0.875+. Bug #2 from MEMORY.md.
│   ├── Status: IMPLEMENTED (attack_ocr.py threshold)
│   └── Priority: DONE
│
└── Output: best match card_id + confidence (or fall to Stage 6 ensemble)
```

---

## STAGE 6: PAGE CONTEXT

**Entry point**: `identify_page_v2()` in `cardprice/ml/__init__.py`
**Context builder**: `cardprice/ml/page_context.py`

```
Input: all per-card results from Stages 2-5
│
├── CHECK 6.1: Build page context from high-confidence results
│   ├── Information needed: card_ids with confidence > threshold
│   ├── Current logic: `identify_page_context()` extracts set_ids from
│   │   confident results, finds majority set and era. Returns
│   │   {likely_sets, era, confidence}.
│   ├── Status: IMPLEMENTED (page_context.py)
│   └── Priority: DONE
│
├── CHECK 6.2: Context confidence gating
│   ├── Information needed: page context confidence score
│   ├── Current logic:
│   │   - confidence < 0.30: skip page context entirely
│   │   - confidence < 0.65: skip pass 2 re-runs (mixed-era pages)
│   │   - confidence >= 0.65: apply context to re-run low-confidence cards
│   ├── Rationale: mixed-era pages (e.g., EX + DP + Platinum) cause
│   │   context to steer cards toward the wrong dominant era.
│   ├── Status: IMPLEMENTED (__init__.py lines 648-656 in identify_page)
│   └── Priority: DONE
│
├── CHECK 6.3: Leave-one-out context for re-runs
│   ├── Information needed: page results excluding current card
│   ├── Current logic: when re-running card i, build context from
│   │   results[0:i] + results[i+1:]. Prevents self-reinforcing errors.
│   ├── Status: IMPLEMENTED (__init__.py lines 668-670)
│   └── Priority: DONE
│
├── CHECK 6.4: Set sequence detection (consecutive card numbers)
│   ├── Information needed: card numbers from identified cards
│   ├── Current logic: NOT IMPLEMENTED. Could detect runs like
│   │   "sv6-101, sv6-102, ?, sv6-104" and infer the missing card
│   │   is sv6-103.
│   ├── Where to implement: `page_context.py`, new function
│   │   `detect_set_sequence()`
│   ├── Status: NOT IMPLEMENTED
│   └── Priority: LOW — would help very few cases; most binders are
│       not perfectly sequential
│
├── CHECK 6.5: Era-based ensemble penalty
│   ├── Information needed: page_era, ensemble result's era
│   ├── Current logic: when ensemble fallback result is from a different
│   │   era than page_era, apply -0.10 confidence penalty. Helps
│   │   name/attack paths beat wrong-era ensemble picks.
│   ├── Status: IMPLEMENTED (__init__.py lines 4337-4342)
│   └── Priority: DONE
│
├── CHECK 6.6: OCR name mismatch penalty for ensemble
│   ├── Information needed: ocr_name, ensemble result's card_id
│   ├── Current logic: when OCR read a valid name (conf >= 0.70) and
│   │   ensemble picked a card with a DIFFERENT name, apply -0.30
│   │   penalty. Prevents ensemble from overriding a correct OCR read.
│   ├── Status: IMPLEMENTED (__init__.py lines 4326-4335)
│   └── Priority: DONE
│
└── Output: refined card_ids for all cards on page
```

---

## STAGE 7: VARIANT DETECTION (post-identification)

**Entry point**: `_apply_variant_detection()` in `cardprice/ml/__init__.py`
**Variant tree**: `data/variant_tree.json`
**Card attributes**: `cardprice/ml/card_attributes.py`
**Detector**: `cardprice/ml/variant_detector.py`

```
Input: identified card_id + card image
│
├── STEP 7.0: Lookup card attributes
│   ├── Call: card_attributes.get_card_attrs(card_id)
│   ├── Returns: era, possible_variants, variant_checks, is_1st_edition_eligible,
│   │   is_stamped_eligible, has_reverse_holo, rarity, supertype
│   ├── Status: IMPLEMENTED (card_attributes.py, O(1) lazy-loaded cache)
│   └── Priority: DONE
│
├── CHECK 7.1: Era-based variant gating
│   ├── Information needed: era number from card_attributes
│   ├── Current logic: `_ERA_VARIANT_ALLOWED` dict maps era -> valid variants.
│   │   If detector returns a variant not in the era's allowed set, override
│   │   to "normal". Also gated by variant_tree's per-set possible_variants.
│   ├── Era-specific rules:
│   │   - Era 1 (WotC): normal, holofoil, 1st_edition, 1st_edition_holofoil,
│   │     unlimited, unlimited_holofoil, shadowless, shadowless_holofoil,
│   │     reverse_holofoil (base6, ecard1-3 only)
│   │   - Era 2 (EX): normal, holofoil, reverse_holofoil (+ stamped for ex7-16)
│   │   - Eras 3-4 (DP/HGSS): normal, holofoil, reverse_holofoil
│   │   - Eras 5-6 (BW/XY): + full_art
│   │   - Eras 7-9 (SM/SWSH/SV): + gold, rainbow_rare
│   ├── Status: IMPLEMENTED (__init__.py lines 200-230, lines 329-341)
│   └── Priority: DONE
│
├── CHECK 7.2: 1st Edition stamp detection (Open Problem tangent)
│   ├── Information needed: card image, stamp region [0.02, 0.44, 0.24, 0.65]
│   ├── Detection method: `stamp_ocr` — PaddleOCR on stamp region looking for
│   │   "1st" and "EDITION" text + circular blob detection
│   ├── Eligible sets: base1, base2, base3, base5, gym1, gym2, neo1-4
│   │   (from _FIRST_EDITION_SETS in card_attributes.py)
│   ├── Current logic: handled inside detect_variant() for eligible sets.
│   │   Records "1st_edition_ocr" in checks_run.
│   ├── Status: IMPLEMENTED (variant_detector.py `_check_1st_edition`)
│   ├── Problem: 1st edition stamps are small (~15x15px at binder resolution).
│   │   Detection works on close-up photos but marginal on binder scans.
│   └── Priority: MEDIUM — 1st edition has 10-100x price premium
│
├── CHECK 7.3: Shadowless detection (Base Set only)
│   ├── Information needed: card image, set_id == "base1"
│   ├── Detection method: `edge_gradient` — right/bottom border gradient
│   │   analysis. Shadowless cards lack the dark gradient drop shadow.
│   ├── Current logic: inside detect_variant() for base1 cards
│   ├── Status: IMPLEMENTED (variant_detector.py)
│   ├── Problem: binder sleeves add their own edge artifacts, potentially
│   │   masking the shadow/no-shadow difference.
│   └── Priority: LOW — shadowless only applies to base1 (~100 cards)
│
├── CHECK 7.4: EX-era stamp detection (Open Problem 2)
│   ├── Information needed: card image, set_id in ex7-ex16
│   ├── Detection method: `stamped_detector` — DINOv2 features + logistic
│   │   regression classifier, stamp region [0.55, 0.35, 0.88, 0.55]
│   ├── Current logic:
│   │   1. Check is_stamped_set(set_id) via variant_tree
│   │   2. Run stamp_classifier.classify_stamp_region()
│   │   3. If stamped=True, override variant to "reverse_holofoil"
│   ├── Status: IMPLEMENTED but UNRELIABLE (68.8% LOO on binder scans)
│   ├── Known issues:
│   │   - Sleeve plastic mutes stamp texture
│   │   - Only 17 examples in ground truth (need more training data)
│   │   - Stamps are translucent overlays, DINOv2 features too coarse
│   ├── Needed:
│   │   - OCR on stamp region at higher resolution (stamps have readable text
│   │     like "DRAGON FRONTIERS", "TEAM ROCKET")
│   │   - More training data from real binder scans
│   │   - Close-up photo mode for variant confirmation
│   └── Priority: HIGH — stamped vs non-stamped affects price significantly
│
├── CHECK 7.5: Holographic detection (Open Problem 3)
│   ├── Information needed: card image HSV analysis
│   ├── Detection method: `holo_detector` — HSV hue spread + spatial noise
│   │   analysis. Compares art region vs border region signal.
│   ├── Current logic: inside detect_variant(), works on direct card photos
│   ├── Status: IMPLEMENTED but BROKEN for binder scans (~0% accuracy)
│   ├── Root cause: holographic shimmer is light-angle-dependent. Single
│   │   static photo through plastic sleeve captures none of it.
│   ├── Needed (from most to least promising):
│   │   - Rarity-based inference: if DB says this card_number is ALWAYS holo
│   │     (e.g., WotC rare holos), skip detection and assume holofoil
│   │   - DB variant existence check: if only holofoil variant exists in
│   │     fact_market_prices for this card_id, assume holofoil
│   │   - Multi-frame video capture (WebRTC tilt detection)
│   │   - Close-up individual card photos outside sleeves
│   ├── Where to implement rarity inference:
│   │   In `_apply_variant_detection()`, after era gating:
│   │   ```
│   │   attrs = get_card_attrs(card_id)
│   │   if attrs.rarity in ("Rare Holo", "Rare Holo EX") and
│   │      "holofoil" in attrs.possible_variants:
│   │       # Check if non-holo version exists for this card number
│   │       # If no non-holo variant exists, this card IS holofoil
│   │   ```
│   └── Priority: HIGH — holo vs non-holo is the most common variant axis,
│       affects price on many WotC/EX era cards
│
├── CHECK 7.6: Reverse holo detection
│   ├── Information needed: holo signal in border vs art region
│   ├── Current logic: holo_detector distinguishes holo (art-only shimmer)
│   │   from reverse holo (border shimmer) by comparing signal regions
│   ├── Status: IMPLEMENTED but same problem as 7.5 — shimmer invisible
│   │   through sleeves
│   ├── Needed: same approaches as 7.5. Additionally, reverse holos have
│   │   visible type-pattern overlays (SM era chevrons, BW type symbols)
│   │   that could be detected by texture analysis even without shimmer.
│   └── Priority: MEDIUM — reverse holo price is typically < 2x normal
│
├── CHECK 7.7: Full art detection (eras 5+)
│   ├── Information needed: edge strip saturation + hue variance
│   ├── Detection method: `full_art_detector` — analyzes outer 5% edge strips.
│   │   Full art cards have high saturation artwork extending to borders.
│   │   Requires 3/4 edge strips passing. Era-gated to era 5+ (BW onward).
│   ├── Current logic: inside detect_variant(), era-gated
│   ├── Status: IMPLEMENTED (variant_detector.py)
│   ├── Problem: works well on direct photos. Through sleeves, edge strip
│   │   analysis is somewhat degraded but still usable (artwork colors
│   │   are still visible through plastic).
│   └── Priority: MEDIUM — needed for modern era pricing
│
├── CHECK 7.8: Gold card detection (eras 7+)
│   ├── Information needed: dominant gold hue (H 15-45) covering >40% of card
│   ├── Detection method: `gold_detector` — HSV gold dominance analysis
│   ├── Current logic: inside detect_variant(), era-gated to era 7+
│   ├── Status: IMPLEMENTED (variant_detector.py)
│   ├── Note: gold cards have a distinctive overall color cast that IS
│   │   visible through sleeves, unlike shimmer effects.
│   └── Priority: LOW — gold cards are rare but detection works reasonably
│
├── CHECK 7.9: Rainbow rare detection (eras 7+)
│   ├── Information needed: high saturation across 4+ of 6 hue segments
│   ├── Detection method: `rainbow_detector` — multi-hue-peak analysis
│   ├── Current logic: inside detect_variant(), era-gated to era 7+
│   ├── Status: IMPLEMENTED (variant_detector.py)
│   ├── Note: rainbow rares have extreme color saturation that may be
│   │   partially visible through sleeves.
│   └── Priority: LOW — rainbow rares are rare but distinctive
│
├── CHECK 7.10: Variant-specific card_id remapping
│   ├── Information needed: detected variant, card_names.json
│   ├── Current logic: if variant != "normal", try remapping card_id from
│   │   "base1-4/normal" to "base1-4/holofoil". Only remap if the target
│   │   card_id exists in the database.
│   ├── Status: IMPLEMENTED (__init__.py lines 344-364)
│   ├── Problem: currently loads card_names.json from disk on every call.
│   │   Should use the card_attributes cache instead.
│   └── Priority: LOW — functional but inefficient
│
└── Output: result dict with detected_variant, variant_confidence,
    variant_checks_run, (optional) stamp_result
```

---

## STAGE 8: PRICING

**Entry point**: `cardprice/server.py` (after identification)
**Data sources**:
- TCGCSV -> PostgreSQL `fact_market_prices` (30.9M rows, NM aggregate, 2yr history)
- JustTCG -> SQLite `data/justtcg_prices.db` (per-condition aggregate, free API)
- TCGPlayer Playwright -> SQLite `data/tcgplayer_sales.db` (individual sales, daily cron)

```
Input: card_id + detected_variant
│
├── CHECK 8.1: Variant-specific price lookup
│   ├── Information needed: card_id with variant suffix (e.g., "base1-4/holofoil")
│   ├── Current logic: query fact_market_prices with full card_id including
│   │   variant. TCGCSV stores separate rows for normal/holofoil/reverse_holofoil.
│   ├── Status: IMPLEMENTED
│   ├── Problem: variant detection (Stage 7) is unreliable for holo/reverse
│   │   through sleeves, so the price lookup often uses "normal" when the
│   │   actual card is holofoil.
│   └── Priority: HIGH — price accuracy depends on variant accuracy
│
├── CHECK 8.2: Fallback to normal price
│   ├── Information needed: whether variant-specific price exists
│   ├── Current logic: if no price found for detected variant, fall back
│   │   to "normal" variant price.
│   ├── Status: IMPLEMENTED
│   └── Priority: DONE
│
├── CHECK 8.3: Condition-specific pricing
│   ├── Information needed: card condition (NM/LP/MP/HP/DMG)
│   ├── Current logic: TCGCSV provides NM aggregate prices. JustTCG provides
│   │   per-condition aggregates. TCGPlayer Playwright provides individual
│   │   sales with condition labels.
│   ├── Status: PARTIALLY IMPLEMENTED
│   │   - condition_assessor.py exists for condition grading from photos
│   │   - condition_report.py generates condition reports
│   │   - Not yet integrated end-to-end with pricing
│   ├── Needed: pipeline from condition grade -> price adjustment
│   └── Priority: MEDIUM — affects price accuracy, especially for older cards
│
├── CHECK 8.4: Multi-source price aggregation
│   ├── Information needed: prices from all 3 data sources
│   ├── Current logic: three separate data stores queried independently.
│   │   No unified aggregation or confidence scoring across sources.
│   ├── Needed: price confidence score based on:
│   │   - Recency of price data
│   │   - Agreement across sources
│   │   - Number of sales/listings
│   ├── Status: NOT IMPLEMENTED (each source queried independently)
│   └── Priority: LOW — single source is usually sufficient
│
└── Output: price per condition, price source, price confidence
```

---

## STAGE 9: CLAUDE VISION OVERRIDE (Optional Pass 3)

**Entry point**: `identify_page()` Pass 3, `identify_page_vision_first()`
**Module**: `cardprice/ml/claude_vision.py`
**Cost**: ~$0.01-0.03 per card (Sonnet API)

```
Input: card images + ML pipeline results from Stages 2-6
│
├── CHECK 9.1: ML + vision name agreement
│   ├── Information needed: ML card name, Claude vision card name
│   ├── Current logic: compare base names (stripping EX/V/GX suffixes).
│   │   If they agree, boost ML confidence by +0.10 and annotate
│   │   "(confirmed by Claude vision)".
│   ├── Status: IMPLEMENTED (__init__.py lines 982-995)
│   └── Priority: DONE
│
├── CHECK 9.2: Vision confidence gating
│   ├── Information needed: vision_conf from Claude response
│   ├── Current logic:
│   │   - vision_conf < 0.50: skip entirely
│   │   - vision_conf < 0.80: don't override ML result
│   │   - vision_conf >= 0.80: allowed to override ML
│   ├── Status: IMPLEMENTED (__init__.py lines 729-787)
│   └── Priority: DONE
│
├── CHECK 9.3: Attack-based DB matching (vision provides attacks)
│   ├── Information needed: attack names from Claude vision
│   ├── Current logic: when vision and ML disagree on name, use
│   │   vision's name + attacks to search DB. Cross-reference with
│   │   ML's visual candidate pool.
│   ├── Status: IMPLEMENTED (identify_page_vision_first, lines 1044-1066)
│   └── Priority: DONE
│
├── CHECK 9.4: Card back detection by vision
│   ├── Information needed: vision response identifying card back
│   ├── Current logic: if vision returns "card back", mark as
│   │   card_id=None with method="vision_cardback".
│   ├── Status: IMPLEMENTED (__init__.py lines 957-964)
│   └── Priority: DONE
│
└── Output: optionally overridden card_id + confidence
```

---

## Decision Summary: Highest Priority Gaps

| # | Gap | Stage | Open Problem | Impact |
|---|-----|-------|-------------|--------|
| 1 | Holo vs non-holo detection through sleeves | 7.5 | Problem 3 | HIGH: affects pricing on thousands of WotC/EX era cards. Rarity-based inference is implementable now. |
| 2 | EX-era stamp detection reliability | 7.4 | Problem 2 | HIGH: 68.8% accuracy is unacceptable for pricing. OCR on stamp text is the most promising path. |
| 3 | Possessive prefix concatenation robustness | 2.1 | Problem 1 | MEDIUM: affects Gym/Rocket/Team sets. Needs wider crop + multi-line OCR grouping. |
| 4 | Full art card OCR strategy | 2.2 | Problem 5 | MEDIUM: not measured yet, but will block SV/SWSH era scanning. |
| 5 | Variant-aware pricing integration | 8.1 | n/a | HIGH: price accuracy depends on variant accuracy. Rarity-based inference (gap 1) would unblock this. |
| 6 | Condition-to-price pipeline | 8.3 | n/a | MEDIUM: condition grading exists but not integrated with pricing lookup. |

### Implementation Order

1. **Rarity-based holo inference** (Gap 1) -- highest ROI. Uses existing DB
   data (`rarity` column in `dim_cards`, `possible_variants` from
   `card_attributes`). Many WotC rares are holo-only; checking whether
   a non-holo variant exists for the card number eliminates the need for
   visual detection. Implement in `_apply_variant_detection()`.

2. **Stamp OCR** (Gap 2) -- run PaddleOCR at 3x upscale on the stamp region
   `[0.55, 0.35, 0.88, 0.55]`, look for known stamp text strings from
   `variant_tree.json:stamped_sets`. More reliable than texture analysis.

3. **Possessive prefix improvement** (Gap 3) -- increase OCR crop to 30% on
   retry when only one text line found; add multi-line Y-coordinate grouping.

4. **Full art layout detection** (Gap 4) -- run `_check_full_art` before OCR
   crop; switch to bottom-20% or full-card OCR when full art detected.
