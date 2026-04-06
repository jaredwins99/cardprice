"""ML modules for card identification and price prediction.

Cascade card identification pipeline.

Tries identification methods in order of cost/speed:
1. Perceptual hash (free, instant) -- accept if distance < 5
2. DINOv2 + FAISS (free, ~1s) -- accept if similarity > 0.65
2.5. CLIP image-to-image (free, ~2s) -- accept if similarity > 0.75
2.7. OCR name reading (free, ~1s) -- accept if fuzzy score >= 90 and confidence > 0.70
2.8. DP-era level detection (free, ~1s) -- OCR "LV.XX" + level map matching
3. Claude Haiku vision API ($0.0015/card) -- accept if db-match confidence > 0.5
"""

import hashlib
import logging
import os
import pickle
import sys
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Thread lock for OCR engines (PaddleOCR/EasyOCR are not thread-safe)
_ocr_lock = threading.Lock()
_jp_easyocr_reader = None  # Cached Japanese EasyOCR reader (slow to init)

# Thread lock for GPU operations (DINOv2/CLIP forward passes)

# Persistent process pool for parallel card identification
_card_pool = None
_card_pool_lock = threading.Lock()

# In-memory LRU cache for identify_card results, keyed by md5 of file contents.
_scan_cache: OrderedDict = OrderedDict()
_SCAN_CACHE_MAX = 100
_ROBUST_CONFIDENCE_THRESHOLD = 0.65

# Known EX-era stamp text for fuzzy matching against OCR output.
# Maps set ID to the text that appears on stamped cards from that set.
EX_STAMP_NAMES = {
    "ex7": "TEAM ROCKET RETURNS", "ex8": "DEOXYS", "ex9": "EMERALD",
    "ex10": "UNSEEN FORCES", "ex11": "DELTA SPECIES", "ex12": "LEGEND MAKER",
    "ex13": "HOLON PHANTOMS", "ex14": "CRYSTAL GUARDIANS",
    "ex15": "DRAGON FRONTIERS", "ex16": "POWER KEEPERS",
}
# Reverse lookup: stamp name -> set ID
_STAMP_NAME_TO_SET = {v: k for k, v in EX_STAMP_NAMES.items()}
# Pre-built list of stamp names for rapidfuzz extractOne
_STAMP_NAME_CHOICES = list(EX_STAMP_NAMES.values())


def _fuzzy_match_stamp_text(texts, min_score=55):
    """Fuzzy-match OCR texts from the stamp region against known EX stamp names.

    Args:
        texts: list of OCR text strings from the stamp region.
        min_score: minimum fuzz score to accept (default 55). Pass 0 to get
            the raw best match regardless of threshold (caller can apply
            its own threshold based on context, e.g. lower the bar when the
            matched set matches the card's identified set).

    Returns:
        (set_id, score) if best match >= min_score, else (None, best_score).
        Note: when below min_score, set_id is None but score is still returned
        so callers can decide if a contextual lower threshold applies.
    """
    from rapidfuzz import fuzz, process

    # Concatenate all texts and also try each individually
    combined = " ".join(texts).upper().strip()
    candidates = [combined] + [t.upper().strip() for t in texts]

    best_set_id = None
    best_score = 0
    for candidate in candidates:
        if len(candidate) < 3:
            continue
        match = process.extractOne(
            candidate, _STAMP_NAME_CHOICES,
            scorer=fuzz.partial_ratio,
        )
        if match and match[1] > best_score:
            best_score = match[1]
            best_set_id = _STAMP_NAME_TO_SET[match[0]]

    if best_score >= min_score:
        return best_set_id, best_score
    # Return the candidate set_id even when below threshold so context-aware
    # callers can apply their own (possibly lower) threshold.
    return None, best_score


# Resolve data paths relative to the project root (two levels up from this file).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_HASH_DB_PATH = _PROJECT_ROOT / "data" / "hash_db.pkl"
_DINO_INDEX_PATH = _PROJECT_ROOT / "data" / "dino_index.faiss"
_DINO_CARD_IDS_PATH = _PROJECT_ROOT / "data" / "dino_card_ids.pkl"
_CLIP_IMAGE_INDEX_PATH = _PROJECT_ROOT / "data" / "clip_image_index.pkl"
_CLIP_AUGMENTED_INDEX_PATH = _PROJECT_ROOT / "data" / "clip_augmented_index.pkl"

# ---------------------------------------------------------------------------
# Lazy-loaded singletons for heavy ML resources
# ---------------------------------------------------------------------------
_hash_db = None
_dino_faiss_index = None
_dino_card_ids = None
_clip_image_index = None


_translation_names_cache = None


def _load_translation_names() -> dict[str, str]:
    """Load multilingual card translations and return {lower_name: english_name}.

    Maps translated Pokemon names (French, German, etc.) to their English
    equivalents so OCR-read foreign names can be resolved.

    Sources:
    1. card_translations.json — per-card translations from TCGdex (works well for
       European languages; JA/zh-tw have different set IDs so few match).
    2. pokemon_species_names.json — per-species translations from PokeAPI CSV data.
       Provides 1025 species names per language. Essential for Japanese/Chinese/Korean.
    3. Japanese TCG prefix mapping (わるい→Dark, ひかる→Shining, etc.) combined with
       species names to generate compound card names like "Dark Electrode".
    """
    global _translation_names_cache
    if _translation_names_cache is not None:
        return _translation_names_cache

    import json
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    trans_path = data_dir / "card_translations.json"
    names_path = data_dir / "card_names.json"
    species_path = data_dir / "pokemon_species_names.json"

    result: dict[str, str] = {}

    # --- Source 1: TCGdex per-card translations ---
    if trans_path.exists():
        # Build card_id -> english_name from card_names.json
        id_to_eng: dict[str, str] = {}
        if names_path.exists():
            with open(names_path) as f:
                for row in json.load(f):
                    id_to_eng[row[0]] = row[1]

        with open(trans_path) as f:
            trans_data = json.load(f)

        for lang, cards in trans_data.items():
            for cid, tname in cards.items():
                tname_lower = tname.lower().strip()
                if tname_lower in result:
                    continue  # already mapped
                # Find English name for this card ID
                eng = id_to_eng.get(cid)
                if not eng:
                    # Try with /normal suffix
                    eng = id_to_eng.get(f"{cid}/normal")
                if eng:
                    result[tname_lower] = eng

        logger.info("Loaded %d translation names from TCGdex", len(result))

    # --- Source 2: PokeAPI species names ---
    if species_path.exists():
        with open(species_path) as f:
            species_data = json.load(f)

        species_count = 0
        for lang, mapping in species_data.items():
            for foreign_lower, eng_name in mapping.items():
                if foreign_lower not in result:
                    result[foreign_lower] = eng_name
                    species_count += 1

        logger.info("Added %d species translation names from PokeAPI", species_count)

    # --- Source 3: Japanese TCG prefix combinations ---
    # Japanese TCG cards use prefixes like わるい (Dark), ひかる (Shining) etc.
    # Combined with species names, these produce card names like わるいマルマイン → Dark Electrode
    _JA_TCG_PREFIXES = {
        "わるい": "Dark",          # Team Rocket dark Pokemon
        "やさしい": "Light",       # Neo Destiny light Pokemon
        "ひかる": "Shining",       # Neo shining Pokemon
        "かがやく": "Radiant",     # SWSH/SV radiant Pokemon
        "かりんの": "Karen's",
        "マチスの": "Lt. Surge's",
        "カスミの": "Misty's",
        "タケシの": "Brock's",
        "エリカの": "Erika's",
        "ナツメの": "Sabrina's",
        "キョウの": "Koga's",
        "カツラの": "Blaine's",
        "サカキの": "Giovanni's",
    }
    # --- Source 4: Japanese/Korean suffix combinations ---
    # Modern TCG cards use suffixes: ex, EX, V, VMAX, VSTAR, GX etc.
    # Japanese cards append these directly: ギャラドスex, ピカチュウV
    # Korean cards do the same: 갸라도스ex, 피카츄V
    _TCG_SUFFIXES = {
        "ex": " ex",           # SV-era lowercase ex
        "EX": "-EX",           # XY/BW-era uppercase EX
        "V": " V",             # SWSH-era V
        "VMAX": " VMAX",       # SWSH-era VMAX
        "VSTAR": " VSTAR",     # SWSH-era VSTAR
        "GX": "-GX",           # SM-era GX
        "δ": " δ",             # EX-era delta species
    }

    if species_path.exists():
        ja_species = species_data.get("ja", {})
        ko_species = species_data.get("ko", {})
        zhtw_species = species_data.get("zh-tw", {})
        zhcn_species = species_data.get("zh-cn", {})

        # JA prefix combinations (Dark, Light, Gym Leaders, etc.)
        prefix_count = 0
        for ja_prefix, en_prefix in _JA_TCG_PREFIXES.items():
            for ja_name, en_name in ja_species.items():
                compound_ja = f"{ja_prefix}{ja_name}"
                compound_en = f"{en_prefix} {en_name}"
                if compound_ja not in result:
                    result[compound_ja] = compound_en
                    prefix_count += 1
        logger.info("Added %d JA prefix combinations", prefix_count)

        # JA, KO, and ZH suffix combinations (ex, V, VMAX, VSTAR, GX, EX)
        suffix_count = 0
        for lang_code, lang_species in [
            ("ja", ja_species), ("ko", ko_species),
            ("zh-tw", zhtw_species), ("zh-cn", zhcn_species),
        ]:
            for suffix_foreign, suffix_en in _TCG_SUFFIXES.items():
                for foreign_name, en_name in lang_species.items():
                    compound = f"{foreign_name}{suffix_foreign}"
                    compound_en = f"{en_name}{suffix_en}"
                    if compound not in result:
                        result[compound] = compound_en
                        suffix_count += 1
        logger.info("Added %d JA/KO/ZH suffix combinations", suffix_count)

        # KO prefix combinations (Korean Gym Leaders, Dark, Light, etc.)
        _KO_TCG_PREFIXES = {
            "다크 ": "Dark",           # Dark Pokemon
            "라이트 ": "Light",        # Light Pokemon
            "빛나는 ": "Shining",      # Shining Pokemon
        }
        ko_prefix_count = 0
        for ko_prefix, en_prefix in _KO_TCG_PREFIXES.items():
            for ko_name, en_name in ko_species.items():
                compound_ko = f"{ko_prefix}{ko_name}"
                compound_en = f"{en_prefix} {en_name}"
                if compound_ko not in result:
                    result[compound_ko] = compound_en
                    ko_prefix_count += 1
        logger.info("Added %d KO prefix combinations", ko_prefix_count)

    _translation_names_cache = result
    logger.info("Total translation names: %d", len(result))
    return result


_name_lookup_cache = None


def _get_name_lookup() -> tuple[list[str], dict[str, str]]:
    """Return (name_list_lower, lower_to_original) with English + translations merged."""
    global _name_lookup_cache
    if _name_lookup_cache is not None:
        return _name_lookup_cache

    from cardprice.ml.ocr_matcher import _load_unique_pokemon_names
    unique_names = _load_unique_pokemon_names()
    name_list_lower = [n.lower() for n in unique_names]
    lower_to_original = {n.lower(): n for n in unique_names}

    # Merge multilingual translations (foreign name -> English name)
    for tname_lower, eng_name in _load_translation_names().items():
        if tname_lower not in lower_to_original:
            lower_to_original[tname_lower] = eng_name
            name_list_lower.append(tname_lower)

    _name_lookup_cache = (name_list_lower, lower_to_original)
    return _name_lookup_cache


def _get_hash_db():
    """Lazy-load and cache the perceptual hash database."""
    global _hash_db
    if _hash_db is None:
        if not _HASH_DB_PATH.exists():
            return None
        logger.info("Loading hash DB from %s ...", _HASH_DB_PATH)
        with open(_HASH_DB_PATH, "rb") as f:
            _hash_db = pickle.load(f)
        logger.info("Hash DB loaded (%d entries).", len(_hash_db))
    return _hash_db


def _get_dino_index():
    """Lazy-load and cache the FAISS index and card-ID mapping for DINOv2."""
    global _dino_faiss_index, _dino_card_ids
    if _dino_faiss_index is None:
        if not _DINO_INDEX_PATH.exists() or not _DINO_CARD_IDS_PATH.exists():
            return None, None
        import faiss
        logger.info("Loading FAISS index from %s ...", _DINO_INDEX_PATH)
        _dino_faiss_index = faiss.read_index(str(_DINO_INDEX_PATH))
        with open(_DINO_CARD_IDS_PATH, "rb") as f:
            _dino_card_ids = pickle.load(f)
        logger.info("FAISS index loaded (%d vectors).", _dino_faiss_index.ntotal)
    return _dino_faiss_index, _dino_card_ids


def _get_clip_image_index():
    """Lazy-load and cache the CLIP image embedding index.

    Prefers the augmented index (clip_augmented_index.pkl) if it exists,
    as it bridges the domain gap between clean digital reference images
    and phone photos. Falls back to the standard clip_image_index.pkl.
    """
    global _clip_image_index
    if _clip_image_index is None:
        # Prefer augmented index (better for phone photos of binder pages)
        if _CLIP_AUGMENTED_INDEX_PATH.exists():
            idx_path = _CLIP_AUGMENTED_INDEX_PATH
            label = "augmented CLIP image index"
        elif _CLIP_IMAGE_INDEX_PATH.exists():
            idx_path = _CLIP_IMAGE_INDEX_PATH
            label = "CLIP image index"
        else:
            return None
        logger.info("Loading %s from %s ...", label, idx_path)
        with open(idx_path, "rb") as f:
            _clip_image_index = pickle.load(f)
        logger.info("%s loaded (%d entries).", label.capitalize(),
                    len(_clip_image_index["card_ids"]))
    return _clip_image_index


def _cache_store(file_hash, result):
    """Store a result in the LRU cache, evicting oldest if over capacity."""
    if file_hash is None:
        return
    _scan_cache[file_hash] = result
    _scan_cache.move_to_end(file_hash)
    while len(_scan_cache) > _SCAN_CACHE_MAX:
        _scan_cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Era-to-variant allowlists for variant gating
# ---------------------------------------------------------------------------
_ERA_VARIANT_ALLOWED = {
    # Era 1: WotC Classic (1999-2003) — Base Set through Skyridge
    # Broad union: 1st Edition (base1-neo4), reverse holo (base6, ecard1-3),
    # shadowless (base1 only), unlimited variants.  Set-specific gating is
    # handled by variant_detector.SET_SPECIAL_VARIANTS.
    1: {
        "normal", "holofoil", "reverse_holofoil",
        "1st_edition", "1st_edition_holofoil",
        "unlimited", "unlimited_holofoil",
        "shadowless", "shadowless_holofoil",
    },
    # Era 2: EX era (2003-2007) — No 1st Edition. Reverse holo + stamped.
    2: {"normal", "holofoil", "reverse_holofoil", "ex_set_stamp"},
    # Era 3: Diamond & Pearl / Platinum (2007-2010)
    3: {"normal", "holofoil", "reverse_holofoil"},
    # Era 4: HeartGold SoulSilver (2010-2011)
    4: {"normal", "holofoil", "reverse_holofoil"},
    # Era 5: Black & White (2011-2013) — First full art EX cards
    5: {"normal", "holofoil", "reverse_holofoil", "full_art"},
    # Era 6: XY (2014-2016)
    6: {"normal", "holofoil", "reverse_holofoil", "full_art"},
    # Era 7: Sun & Moon (2017-2019) — Rainbow rares, gold secret rares
    7: {"normal", "holofoil", "reverse_holofoil", "full_art",
        "gold", "rainbow_rare"},
    # Era 8: Sword & Shield (2020-2022) — Alt art V/VMAX/VSTAR
    8: {"normal", "holofoil", "reverse_holofoil", "full_art",
        "gold", "rainbow_rare"},
    # Era 9: Scarlet & Violet (2023+) — Illustration rares, special art rares
    9: {"normal", "holofoil", "reverse_holofoil", "full_art",
        "gold", "rainbow_rare"},
}


def _apply_variant_detection(result, image_path, detect_variants=True,
                             precomputed_stamp_set_id=None,
                             precomputed_stamp_match_score=0,
                             precomputed_stamp_texts=None):
    """Apply variant detection to a v2 result dict (in-place).

    Uses variant_tree to determine which checks are relevant for the card's
    set/era, then runs the appropriate detectors:
      - OpenCV heuristics (variant_detector.detect_variant) for holo/reverse/etc.
      - Stamp classifier (stamp_classifier.classify_stamp) for EX-era stamped cards
      - 1st Edition OCR detection (built into variant_detector)

    Falls back gracefully if any detector or model is unavailable.

    Safe to call on any result -- silently skips if card_id is None,
    detect_variants is False, or if detection fails for any reason.
    """
    card_id = result.get("card_id")
    if not card_id or not detect_variants:
        return

    try:
        from cardprice.ml.variant_detector import detect_variant
        from cardprice.ml.era_detector import get_card_era

        era = get_card_era(card_id)
        checks_run = []

        # --- Determine possible variants via variant_tree ---
        possible_variants = ["normal", "holofoil"]
        try:
            from cardprice.ml.variant_tree import get_possible_variants
            possible_variants = get_possible_variants(card_id)
        except Exception as e:
            logger.debug("variant_tree lookup failed, using defaults: %s", e)

        # If the only possibilities are "normal" and "holofoil" (most modern
        # cards), we can still run the base detector but skip specialized checks.
        has_specialized = any(
            v not in ("normal", "holofoil") for v in possible_variants
        )

        # --- Base variant detection (OpenCV heuristics) ---
        variant = detect_variant(image_path, era=era, card_id=card_id, fast=True)
        confidence = 1.0  # placeholder; detect_variant doesn't return confidence
        checks_run.append("variant_detector")

        # The base OpenCV holo detector is unreliable for reverse_holofoil on
        # binder scans (warm lighting, color casts → false positives).
        # Downgrade to "normal" here; the stamp_detection pipeline will set
        # reverse_holofoil only when DINOv2 evidence is strong (conf >= 0.90).
        if variant == "reverse_holofoil":
            logger.debug(
                "base detector returned reverse_holofoil for %s, "
                "downgrading to normal (unreliable on binder scans)",
                card_id,
            )
            variant = "normal"

        # --- Stamp classifier DISABLED ---
        # The stamp_classifier (edge_ratio based) is unreliable: it returns
        # inverted results (stamped cards = False, non-stamped = True).
        # Stamp detection needs a better approach (e.g. DINOv2 differential
        # comparison against reference, or direct stamp region OCR).
        # For now, variant detection for EX-era stamps is not automated.

        # --- 1st Edition detection ---
        # Already handled inside detect_variant() for eligible sets,
        # just record that the check was run.
        if "1st_edition" in possible_variants or "1st_edition_holofoil" in possible_variants:
            if "1st_edition_ocr" not in checks_run:
                checks_run.append("1st_edition_ocr")

        # --- Note reverse holo possibility (can't fully detect from binder scans) ---
        if "reverse_holofoil" in possible_variants:
            if "reverse_holo_check" not in checks_run:
                checks_run.append("reverse_holo_check")

        # --- Era gating: override to "normal" if variant is invalid for this era ---
        allowed = _ERA_VARIANT_ALLOWED.get(era, {"normal", "holofoil", "reverse_holofoil"})
        if variant not in allowed:
            logger.debug("variant %s not allowed for era %d, overriding to normal", variant, era)
            variant = "normal"

        # Also gate against variant_tree's possible variants for this set
        if variant not in possible_variants and variant != "normal":
            logger.debug(
                "variant %s not in possible_variants %s for %s, overriding to normal",
                variant, possible_variants, card_id,
            )
            variant = "normal"

        # --- Attempt card_id remapping (for future when variant rows exist) ---
        if variant != "normal":
            base_id = card_id.rsplit("/", 1)[0]
            remapped_id = f"{base_id}/{variant}"

            # Check if remapped card_id exists in card_names.json
            id_exists = False
            try:
                _names_path = _PROJECT_ROOT / "data" / "card_names.json"
                if _names_path.exists():
                    import json
                    with open(_names_path) as f:
                        card_names = json.load(f)
                    id_exists = remapped_id in card_names
            except Exception:
                pass

            if id_exists:
                result["card_id"] = remapped_id
                logger.info("variant remap: %s -> %s", card_id, remapped_id)
            else:
                logger.debug("variant remap %s not in DB, keeping %s", remapped_id, card_id)

        result["detected_variant"] = variant
        result["variant_confidence"] = confidence
        result["variant_checks_run"] = checks_run

        # --- Era-gated variant detection pipeline ---
        # After identification, run the complete conditional detection tree.
        # In fast mode (default), only cheap pixel checks run (shadowless,
        # world championship).  Full mode adds OCR + holo pattern analysis.
        try:
            from cardprice.ml.stamp_detection import detect_all_variants
            # Always use fast=True — expensive DINOv2 stamp detection for
            # EX-era sets should be pre-computed in batch, not run sequentially
            # after identification (adds 2-8s per EX card).
            stamp_pipeline_result = detect_all_variants(
                image_path, card_id, fast=True)
            flags = stamp_pipeline_result.get("variant_flags", {})

            if stamp_pipeline_result["stamps_detected"]:
                result["stamps_detected"] = stamp_pipeline_result["stamps_detected"]
                result["stamp_details"] = stamp_pipeline_result["stamp_details"]
                checks_run.append("variant_detection_pipeline")

                # If 1st Edition stamp detected and variant wasn't already
                # set to 1st_edition, update it (stamp pipeline is authoritative)
                if flags.get("1st_edition"):
                    first_ed_conf = stamp_pipeline_result["stamp_details"]["1st_edition"]["confidence"]
                    if variant != "1st_edition" and first_ed_conf >= 0.70:
                        logger.info(
                            "variant pipeline: overriding variant %s -> "
                            "1st_edition (conf=%.2f) for %s",
                            variant, first_ed_conf, card_id,
                        )
                        result["detected_variant"] = "1st_edition"
                        result["variant_confidence"] = first_ed_conf

                # If EX set stamp detected, set variant to ex_set_stamp
                # (triggers logo overlay in image_overlay.py)
                if flags.get("ex_stamped_reverse"):
                    ex_conf = stamp_pipeline_result["stamp_details"]["ex_set_stamp"]["confidence"]
                    if ex_conf >= 0.75:
                        logger.info(
                            "variant pipeline: overriding variant %s -> "
                            "ex_set_stamp (EX stamp, conf=%.2f) for %s",
                            variant, ex_conf, card_id,
                        )
                        result["detected_variant"] = "ex_set_stamp"
                        result["variant_confidence"] = ex_conf

                # If reverse_holo detected, set variant to reverse_holofoil
                # Require very high confidence (0.90) to avoid false positives
                # on binder scans from warm lighting / color casts.
                if flags.get("reverse_holofoil"):
                    rh_conf = stamp_pipeline_result["stamp_details"]["reverse_holo"]["confidence"]
                    if variant not in ("reverse_holofoil", "1st_edition") and result.get("detected_variant") != "ex_set_stamp" and rh_conf >= 0.90:
                        logger.info(
                            "variant pipeline: overriding variant %s -> "
                            "reverse_holofoil (reverse holo, conf=%.2f) for %s",
                            variant, rh_conf, card_id,
                        )
                        result["detected_variant"] = "reverse_holofoil"
                        result["variant_confidence"] = rh_conf

                # If holo_finish detected, set variant to holofoil
                if flags.get("holofoil"):
                    hf_conf = stamp_pipeline_result["stamp_details"]["holo_finish"]["confidence"]
                    if variant not in ("holofoil", "1st_edition", "reverse_holofoil") and hf_conf >= 0.60:
                        logger.info(
                            "variant pipeline: overriding variant %s -> "
                            "holofoil (holo finish, conf=%.2f) for %s",
                            variant, hf_conf, card_id,
                        )
                        result["detected_variant"] = "holofoil"
                        result["variant_confidence"] = hf_conf

                # If prerelease stamp detected, record it on the result
                if flags.get("prerelease"):
                    pr_conf = stamp_pipeline_result["stamp_details"]["prerelease"]["confidence"]
                    result["prerelease_detected"] = True
                    result["prerelease_confidence"] = pr_conf

                # If staff stamp detected, record it on the result
                if flags.get("staff"):
                    st_conf = stamp_pipeline_result["stamp_details"]["staff_stamp"]["confidence"]
                    result["staff_stamp_detected"] = True
                    result["staff_stamp_confidence"] = st_conf

                # If shadowless detected, record it on the result
                if flags.get("shadowless"):
                    sh_conf = stamp_pipeline_result["stamp_details"]["shadowless"]["confidence"]
                    result["shadowless_detected"] = True
                    result["shadowless_confidence"] = sh_conf

                # World Championship reproduction warning
                if flags.get("is_reproduction"):
                    result["is_reproduction"] = True

            result["stamps_checked"] = stamp_pipeline_result["stamps_checked"]
            result["variant_flags"] = flags

            # --- Fast EX stamp detection (ex7-ex16) ---
            # The full pipeline skips ex_set_stamp in fast mode (2-8s per card).
            # Use precomputed reference embeddings for ~100-200ms detection.
            _EX_STAMPED_SETS = frozenset({
                "ex7", "ex8", "ex9", "ex10", "ex11",
                "ex12", "ex13", "ex14", "ex15", "ex16",
            })
            set_id = card_id.split("-")[0] if "-" in card_id.split("/")[0] else card_id.split("/")[0]
            # Extract set_id properly: "ex15-26/normal" -> "ex15"
            card_portion = card_id.split("/")[0]
            last_dash = card_portion.rfind("-")
            if last_dash != -1:
                set_id = card_portion[:last_dash]
            else:
                set_id = card_portion

            if set_id in _EX_STAMPED_SETS:
                # PRIMARY signal: OCR fuzzy-matched the stamp text against
                # known EX set names. This is the most reliable signal — it
                # only fires when readable text matching "DELTA SPECIES",
                # "DRAGON FRONTIERS", etc. was found in the stamp region.
                # Holo cards with metallic glare can fool the DINOv2
                # differential (e.g. Metang ex11-49) but their OCR text is
                # the illustrator name, not the set name, so they correctly
                # score 0 against EX_STAMP_NAMES.
                #
                # Stamp data comes from the OCR worker via the precomputed
                # path (page scans) or directly on the result (single-card
                # path). Prefer the explicit precomputed values if passed.
                ocr_stamp_score = (
                    precomputed_stamp_match_score
                    if precomputed_stamp_match_score
                    else result.get("stamp_match_score", 0) or 0
                )
                ocr_stamp_set = (
                    precomputed_stamp_set_id
                    or result.get("stamp_set_id")
                )
                ocr_stamp_texts = (
                    precomputed_stamp_texts
                    if precomputed_stamp_texts
                    else (result.get("stamp_texts", []) or [])
                )

                # Context-aware threshold: when the OCR's best-guess set
                # matches the card's identified set, lower the bar from 60
                # to 45. This catches Nidoqueen (ex15-7) where the OCR sees
                # rules text + faint stamp text and scores 48 against
                # "DRAGON FRONTIERS" — below the 60 hard threshold but the
                # set agreement is itself a strong signal that the card
                # really is from that stamped set.
                set_matches = (ocr_stamp_set == set_id)
                ocr_threshold = 45 if set_matches else 60
                if ocr_stamp_score >= ocr_threshold and ocr_stamp_set:
                    logger.info(
                        "OCR stamp match: %s -> %s (score=%d, set_match=%s)",
                        card_id, ocr_stamp_set, ocr_stamp_score, set_matches,
                    )
                    flags["ex_stamped_reverse"] = True
                    result["variant_flags"] = flags
                    result.setdefault("stamps_detected", []).append("ex_set_stamp")
                    result.setdefault("stamp_details", {})["ex_set_stamp"] = {
                        "detected": True,
                        "confidence": min(0.70 + ocr_stamp_score / 200.0, 0.99),
                        "position": "artwork_bottom_right",
                        "evidence": "ocr_text_fuzzy_match",
                        "matched_set": ocr_stamp_set,
                        "match_score": ocr_stamp_score,
                    }
                    result["detected_variant"] = "ex_set_stamp"
                    result["variant_confidence"] = min(0.70 + ocr_stamp_score / 200.0, 0.99)
                else:
                    # SECONDARY signal: DINOv2 differential, but ONLY when
                    # the stamp region contained no readable text at all
                    # (otherwise the OCR found illustrator credits / rules
                    # text, which means the artwork bottom-right looks
                    # different from reference but NOT due to a stamp).
                    has_meaningful_ocr = any(len(t) >= 4 for t in ocr_stamp_texts)
                    if not has_meaningful_ocr:
                        try:
                            from cardprice.ml.stamp_detection import check_ex_stamp_fast
                            is_stamped, ex_conf = check_ex_stamp_fast(
                                str(image_path), card_id)
                            checks_run.append("ex_stamp_fast")
                            if "stamps_checked" in result:
                                result["stamps_checked"].append("ex_set_stamp_fast")

                            if is_stamped and ex_conf >= 0.70:
                                logger.info(
                                    "fast EX stamp: detected stamp on %s "
                                    "(conf=%.2f) [OCR fallback]",
                                    card_id, ex_conf,
                                )
                                flags["ex_stamped_reverse"] = True
                                result["variant_flags"] = flags
                                result.setdefault("stamps_detected", []).append("ex_set_stamp")
                                result.setdefault("stamp_details", {})["ex_set_stamp"] = {
                                    "detected": True,
                                    "confidence": ex_conf,
                                    "position": "artwork_bottom_right",
                                    "evidence": "dino_differential_fast",
                                }
                                result["detected_variant"] = "ex_set_stamp"
                                result["variant_confidence"] = ex_conf
                        except Exception as e:
                            logger.debug("fast EX stamp check failed: %s", e)

        except Exception as e:
            logger.debug("variant detection pipeline failed: %s", e)
    except Exception as e:
        logger.debug("variant detection failed: %s", e)


def identify_card(image_path, session=None, page_context=None):
    """Identify a card using the cascade pipeline.

    Returns dict with keys: card_id, confidence, method, explanation, raw_response.
    Results are cached by md5 of the image file contents (up to 100 entries).
    """
    # Check in-memory cache keyed by file content hash.
    try:
        file_hash = hashlib.md5(Path(image_path).read_bytes()).hexdigest()
        if file_hash in _scan_cache:
            logger.info("Cache HIT for %s (md5=%s)", image_path, file_hash)
            _scan_cache.move_to_end(file_hash)
            return _scan_cache[file_hash]
    except Exception as e:
        logger.warning("Could not hash image file for cache lookup: %s", e)
        file_hash = None

    # Convert HEIC/HEIF if needed; tolerate conversion failures.
    try:
        from cardprice.utils.image_convert import ensure_compatible
        image_path = ensure_compatible(str(image_path))
    except Exception as e:
        logger.warning("Image conversion failed, using original path: %s", e)
        image_path = str(image_path)

    result = {"card_id": None, "confidence": 0.0, "method": None, "explanation": None, "raw_response": {}}

    # Tier 1: Perceptual hash (fastest, cheapest)
    try:
        from cardprice.ml.hash_matcher import match_card, CONFIDENT_THRESHOLD
        hash_db = _get_hash_db()
        if hash_db is not None:
            logger.info("Tier 1 (hash): searching hash database ...")
            matches = match_card(image_path, str(_HASH_DB_PATH), hash_db=hash_db)
            if matches and matches[0][1] < CONFIDENT_THRESHOLD:
                # Hash DB stores card_ids with underscore (filename stem).
                # Convert last '_' to '/' to match dim_cards card_id format.
                raw_cid = matches[0][0]
                last_under = raw_cid.rfind("_")
                card_id = (raw_cid[:last_under] + "/" + raw_cid[last_under + 1:]) if last_under != -1 else raw_cid
                distance = matches[0][1]
                result["card_id"] = card_id
                result["confidence"] = float(max(0.0, 1.0 - distance / 15.0))
                result["method"] = "hash"
                result["explanation"] = f"Exact visual match (perceptual hash distance: {distance})"
                result["raw_response"] = {
                    "matches": [(cid, int(d)) for cid, d in matches[:5]],
                    "top_alternatives": [(cid, int(d)) for cid, d in matches[1:4]] if len(matches) > 1 else []
                }
                logger.info("Tier 1 (hash): MATCH %s (distance=%d)", card_id, distance)
                _cache_store(file_hash, result)
                return result
            elif matches:
                logger.info("Tier 1 (hash): best distance=%d >= threshold %d, falling through",
                            matches[0][1], CONFIDENT_THRESHOLD)
            else:
                logger.info("Tier 1 (hash): no matches within threshold")
        else:
            logger.info("Tier 1 (hash): SKIPPED -- hash DB not found at %s "
                        "(build with: python -m cardprice.cli build-hash-index)",
                        _HASH_DB_PATH)
    except ImportError as e:
        logger.info("Tier 1 (hash): SKIPPED -- missing dependency: %s", e)
    except Exception as e:
        logger.warning("Tier 1 (hash): ERROR -- %s", e)

    # Tier 2: DINOv2 + FAISS (good accuracy, no API cost)
    _preproc_tmp = None
    try:
        from cardprice.ml.dino_matcher import identify_card as dino_identify
        dino_idx, dino_cids = _get_dino_index()
        if dino_idx is not None:
            # Preprocess for DINOv2: CLAHE + glare removal + border crop
            # improves scores by ~+0.05 on phone photos with sleeves.
            dino_query_path = image_path
            try:
                from cardprice.ml.preprocess import preprocess_for_matching
                _preproc_tmp = preprocess_for_matching(image_path)
                dino_query_path = _preproc_tmp
                logger.info("Tier 2 (dino): using preprocessed image")
            except Exception as e:
                logger.debug("Tier 2 (dino): preprocessing skipped: %s", e)

            logger.info("Tier 2 (dino): searching FAISS index ...")
            matches = dino_identify(dino_query_path, faiss_index=dino_idx, card_ids_list=dino_cids)
            if matches:
                # DINOv2 index stores card_ids with set dir prefix: "bw5/bw5-107/normal"
                # Strip the first path segment to get "bw5-107/normal"
                raw_cid = matches[0][0]
                parts = raw_cid.split("/", 1)
                card_id = parts[1] if len(parts) > 1 else raw_cid
                similarity = float(matches[0][1])
                # Phone photos score ~0.4-0.6 against digital refs;
                # digital-to-digital scores ~0.8+. Use 0.65 as threshold.
                if similarity > 0.65:
                    # Build explanation with top alternatives
                    alt_list = []
                    for alt_raw, alt_score in matches[1:4]:
                        alt_parts = alt_raw.split("/", 1)
                        alt_id = alt_parts[1] if len(alt_parts) > 1 else alt_raw
                        alt_list.append((alt_id, float(alt_score)))

                    alt_str = ", ".join(f"{a[0]} ({a[1]:.0%})" for a in alt_list)
                    result["card_id"] = card_id
                    result["confidence"] = similarity
                    result["method"] = "dino"
                    result["explanation"] = f"Visual similarity match ({similarity:.0%}). Top alternatives: {alt_str}" if alt_str else f"Visual similarity match ({similarity:.0%})"
                    result["raw_response"] = {
                        "top_matches": matches[:5],
                        "top_alternatives": alt_list
                    }
                    logger.info("Tier 2 (dino): MATCH %s (similarity=%.4f)", card_id, similarity)
                    _cache_store(file_hash, result)
                    return result
                else:
                    logger.info("Tier 2 (dino): best similarity=%.4f < threshold 0.65, falling through",
                                similarity)
            else:
                logger.info("Tier 2 (dino): no matches found")
        else:
            logger.info("Tier 2 (dino): SKIPPED -- FAISS index not found at %s "
                        "(build with: python -m cardprice.cli build-dino-index)",
                        _DINO_INDEX_PATH)
    except ImportError as e:
        logger.info("Tier 2 (dino): SKIPPED -- missing dependency: %s", e)
    except Exception as e:
        logger.warning("Tier 2 (dino): ERROR -- %s", e)
    finally:
        if _preproc_tmp:
            try:
                os.unlink(_preproc_tmp)
            except OSError:
                pass

    # Tier 2.5: CLIP image-to-image — DISABLED
    # CLIP loading causes SIGSEGV when PaddlePaddle is resident (safetensors +
    # PaddlePaddle memory corruption).  CLIP contributes 0 unique correct IDs
    # in eval, so disabling has zero accuracy impact.
    logger.debug("Tier 2.5 (clip): DISABLED — PaddlePaddle SIGSEGV incompatibility")

    # Tier 2.7: OCR card name reading (free, fast, complements visual matchers)
    try:
        from cardprice.ml.ocr_matcher import identify_card_by_ocr
        logger.info("Tier 2.7 (ocr): reading card name via OCR ...")
        ocr_matches = identify_card_by_ocr(image_path, top_k=5, page_context=page_context)
        if ocr_matches:
            best_id, best_conf, best_details = ocr_matches[0]
            fuzzy_score = best_details["fuzzy_score"]
            # Accept if fuzzy score >= 90 (strong name match) and overall conf > 0.70
            if fuzzy_score >= 90 and best_conf > 0.70:
                # Build alternatives from remaining matches
                alt_list = [
                    (m[0], m[2]["matched_name"], m[1])
                    for m in ocr_matches[1:4]
                ]
                alt_str = ", ".join(
                    f"{a[1]} ({a[2]:.0%})" for a in alt_list
                )
                result["card_id"] = best_id
                result["confidence"] = best_conf
                result["method"] = "ocr"
                result["explanation"] = (
                    f"Card name read via OCR: {best_details['ocr_cleaned']!r} "
                    f"-> {best_details['matched_name']} "
                    f"(fuzzy={fuzzy_score:.0f})"
                )
                if alt_str:
                    result["explanation"] += f". Alternatives: {alt_str}"
                result["raw_response"] = {
                    "ocr_raw": best_details["ocr_raw"],
                    "ocr_cleaned": best_details["ocr_cleaned"],
                    "matched_name": best_details["matched_name"],
                    "fuzzy_score": fuzzy_score,
                    "ocr_confidence": best_details["ocr_confidence"],
                    "top_alternatives": alt_list,
                }
                logger.info(
                    "Tier 2.7 (ocr): MATCH %s (name=%s, fuzzy=%d, conf=%.4f)",
                    best_id, best_details["matched_name"], fuzzy_score, best_conf,
                )
                _cache_store(file_hash, result)
                return result
            else:
                logger.info(
                    "Tier 2.7 (ocr): best=%s fuzzy=%d conf=%.2f "
                    "(need fuzzy>=90 and conf>0.70), falling through",
                    best_details["matched_name"], fuzzy_score, best_conf,
                )
        else:
            logger.info("Tier 2.7 (ocr): no matches found")
    except ImportError as e:
        logger.info("Tier 2.7 (ocr): SKIPPED -- missing dependency: %s", e)
    except Exception as e:
        logger.warning("Tier 2.7 (ocr): ERROR -- %s", e)

    # Tier 3: Claude Haiku vision API (highest accuracy, costs money)
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.info("Tier 3 (claude): SKIPPED -- ANTHROPIC_API_KEY not set")
        else:
            from cardprice.ml.claude_scanner import scan_card, match_to_database
            logger.info("Tier 3 (claude): calling Claude Haiku vision API ...")
            scan_result = scan_card(image_path, model="claude-haiku-4-5")
            matched_id, match_conf = match_to_database(scan_result, session)
            if matched_id and match_conf > 0.5:
                result["card_id"] = matched_id
                result["confidence"] = float(match_conf)
                result["method"] = "claude"
                result["explanation"] = "Identified by AI vision"
                result["raw_response"] = {
                    "scan_result": scan_result,
                    "top_alternatives": []
                }
                logger.info("Tier 3 (claude): MATCH %s (confidence=%.2f)", matched_id, match_conf)
                _cache_store(file_hash, result)
                return result
            elif matched_id:
                logger.info("Tier 3 (claude): identified %s but low confidence=%.2f (threshold=0.5)",
                            matched_id, match_conf)
                result["raw_response"] = {
                    "scan_result": scan_result,
                    "top_alternatives": []
                }
            else:
                logger.info("Tier 3 (claude): API responded but no DB match found")
                result["raw_response"] = {
                    "scan_result": scan_result,
                    "top_alternatives": []
                }
    except ImportError as e:
        logger.info("Tier 3 (claude): SKIPPED -- missing dependency: %s", e)
    except Exception as e:
        logger.warning("Tier 3 (claude): ERROR -- %s", e)

    logger.info("No confident match found for %s", image_path)
    if result.get("raw_response") and "top_alternatives" not in result["raw_response"]:
        result["raw_response"]["top_alternatives"] = []
    if result["card_id"] and not result.get("explanation"):
        result["explanation"] = f"No confident match. Best guess: {result['card_id']} at {result['confidence']:.0%}"
    _cache_store(file_hash, result)
    return result


def identify_page(card_image_paths, session=None):
    """Identify all cards on a binder page with context-aware two-pass strategy.

    Pass 1: Run identify_card on all cards to get initial results.
    Pass 2: Build page context from high-confidence results, then re-run
            low-confidence cards with page_context for set disambiguation.

    Args:
        card_image_paths: List of paths to individual card segment images.
        session: Optional DB session.

    Returns:
        List of result dicts (same format as identify_card), one per card.
    """
    from cardprice.ml.page_context import identify_page_context

    RERUN_THRESHOLD = 0.70  # re-run cards below this with context

    # Pass 1: run hybrid (multisignal when strong, ensemble fallback) on all cards
    results = []
    for path in card_image_paths:
        r = identify_card_hybrid(str(path), session=session)
        results.append(r)

    # Build page context from confident results
    ctx = identify_page_context(results)
    logger.info("Page context: sets=%s era=%s confidence=%.2f",
                ctx.get("likely_sets", [])[:3], ctx.get("era"), ctx.get("confidence", 0))

    if not ctx.get("likely_sets") or ctx["confidence"] < 0.3:
        return results  # not enough context to help

    # Guard: only apply page context when the page is coherent (single set/era).
    # On mixed-era pages (e.g. EX + DP + Platinum), context hurts by steering
    # cards toward whatever wrong era dominates the initial guesses.
    skip_pass2 = ctx["confidence"] < 0.65
    if skip_pass2:
        logger.info("Page context too weak (%.2f < 0.65), skipping pass 2", ctx["confidence"])

    # Pass 2: re-run cards with ensemble + leave-one-out page context
    # Re-run when: low confidence, OR methods disagreed (non-"agree" result)
    if not skip_pass2:
        for i, (path, result) in enumerate(zip(card_image_paths, results)):
            needs_rerun = (
                result["confidence"] < RERUN_THRESHOLD
                or "(agree)" not in (result.get("method") or "")
            )
            if needs_rerun:
                # Build leave-one-out context (exclude current card to avoid self-reinforcing errors)
                loo_results = results[:i] + results[i+1:]
                loo_ctx = identify_page_context(loo_results)
                if not loo_ctx.get("likely_sets") or loo_ctx["confidence"] < 0.3:
                    continue

                logger.info("Pass 2: re-running card %d (%s, conf=%.2f, method=%s) with LOO context (sets=%s)",
                            i, result.get("card_id"), result["confidence"], result.get("method"),
                            loo_ctx["likely_sets"][:3])
                new_result = identify_card_hybrid(str(path), session=session, page_context=loo_ctx)
                # Accept context result if:
                # - It changed the card to one matching the page context, OR
                # - It has higher confidence
                old_set = (result.get("card_id") or "").split("-")[0].split("/")[0]
                new_set = (new_result.get("card_id") or "").split("-")[0].split("/")[0]
                ctx_sets = set(loo_ctx.get("likely_sets", []))
                context_match = new_set in ctx_sets and old_set not in ctx_sets
                if context_match or new_result["confidence"] > result["confidence"]:
                    new_result["explanation"] = (new_result.get("explanation") or "") + " (with page context)"
                    results[i] = new_result

    # Pass 3: Claude vision verification/override
    # Claude identifies the Pokemon name, HP, era with near-perfect accuracy.
    # We combine that with the ML pipeline's visual candidates:
    #   - If ML already has the right Pokemon, keep it (agreement)
    #   - If Claude disagrees on the Pokemon name, search DB for Claude's
    #     name+HP and cross-reference with ML's visual candidate pool
    #   - Fall back to pure DB lookup if no ML candidates match
    use_vision = os.environ.get("CARDPRICE_VISION", "1") != "0"
    if use_vision:
        try:
            from cardprice.ml.claude_vision import (
                identify_cards_vision_parallel,
                match_vision_to_db,
            )
            from sqlalchemy import text as sa_text

            logger.info("Pass 3: Running Claude vision on %d cards ...",
                        len(card_image_paths))
            vision_results = identify_cards_vision_parallel(
                card_image_paths, model="sonnet", max_workers=4,
            )

            # Need a DB session for candidate lookups
            from cardprice.db.session import SessionLocal
            own_sess = session is None
            sess = session or SessionLocal()

            try:
                for i, (vr, ml_result) in enumerate(zip(vision_results, results)):
                    if vr is None:
                        continue

                    vision_name = (vr.get("pokemon_name") or "").strip()
                    vision_conf = vr.get("confidence", 0)
                    vision_hp = vr.get("hp")
                    vision_num = vr.get("card_number")
                    # Skip low-confidence or unrecognized vision results
                    if not vision_name or len(vision_name) < 2:
                        continue
                    if vision_name.lower() in ("unknown", "none", "card back"):
                        continue
                    if vision_conf < 0.50:
                        logger.debug("Card %d: vision=%s conf=%.2f too low, skipping",
                                     i, vision_name, vision_conf)
                        continue

                    ml_card_id = ml_result.get("card_id") or ""

                    # Check if ML's pick already matches Claude's name
                    ml_name = ""
                    if ml_card_id:
                        row = sess.execute(
                            sa_text("SELECT name FROM dim_cards WHERE card_id = :cid"),
                            {"cid": ml_card_id.split("/")[0] + "/normal"},
                        ).fetchone()
                        if row:
                            ml_name = row[0]

                    # Check agreement: compare base Pokemon names
                    # Claude often drops suffixes (ex, δ, LV.X) so compare
                    # the base name portion
                    import re as _re
                    def _base_name(n):
                        return _re.sub(
                            r'\s*(ex|EX|δ|delta|V|VSTAR|VMAX|GX|LV\.\w+|Star)\s*$',
                            '', n,
                        ).strip().lower()

                    names_agree = (
                        _base_name(ml_name) == _base_name(vision_name)
                        or vision_name.lower() in ml_name.lower()
                        or ml_name.lower() in vision_name.lower()
                    )

                    if names_agree:
                        # ML already has the right Pokemon — confirm it
                        results[i]["confidence"] = min(
                            ml_result.get("confidence", 0) + 0.10, 1.0,
                        )
                        results[i]["explanation"] = (
                            (results[i].get("explanation") or "") +
                            " (confirmed by Claude vision)"
                        )
                        logger.info("Card %d: ML+vision agree on %s (ml=%s)",
                                    i, vision_name, ml_name)
                        continue

                    # Claude disagrees on the Pokemon name.
                    # Override when: Claude is confident AND the names are
                    # truly different (not just suffix differences).
                    # But be cautious: only override high-confidence ML when
                    # Claude also has high confidence AND a card number.
                    ml_conf = ml_result.get("confidence", 0)
                    # Only override ML when we're very confident in vision:
                    # - Vision needs conf >= 0.80 to override any ML result
                    # - Vision needs a card number to override high-confidence ML
                    if vision_conf < 0.80:
                        logger.debug("Card %d: vision=%s(%.2f) vs ML=%s(%.2f), "
                                     "vision conf too low to override, keeping ML",
                                     i, vision_name, vision_conf, ml_name, ml_conf)
                        continue

                    # Search DB for cards matching Claude's identification
                    num_only = None
                    if vision_num:
                        num_only = vision_num.split("/")[0].strip().lstrip("0") or None

                    # Build query: name + optional HP + optional number
                    q = """
                        SELECT c.card_id, c.name, c.hp, c.card_number,
                               s.name as set_name
                        FROM dim_cards c
                        JOIN dim_sets s ON c.set_id = s.set_id
                        WHERE LOWER(c.name) = LOWER(:name)
                    """
                    params = {"name": vision_name}
                    if vision_hp and isinstance(vision_hp, (int, float)) and vision_hp >= 30:
                        q += " AND c.hp = :hp"
                        params["hp"] = str(int(vision_hp))
                    if num_only:
                        q += " AND LTRIM(c.card_number, '0') = :num"
                        params["num"] = num_only

                    db_candidates = sess.execute(sa_text(q), params).fetchall()

                    if not db_candidates:
                        # Relax: try without HP/number filters
                        db_candidates = sess.execute(
                            sa_text("""
                                SELECT c.card_id, c.name, c.hp, c.card_number,
                                       s.name as set_name
                                FROM dim_cards c
                                JOIN dim_sets s ON c.set_id = s.set_id
                                WHERE LOWER(c.name) = LOWER(:name)
                            """),
                            {"name": vision_name},
                        ).fetchall()

                    if not db_candidates:
                        logger.debug("Card %d: vision=%s, no DB candidates", i, vision_name)
                        continue

                    # Cross-reference with ML's visual candidate pool
                    ml_candidates = set()
                    raw = ml_result.get("raw_response", {})
                    for key in ("scored_candidates", "top_matches", "top_alternatives"):
                        for item in raw.get(key, []):
                            if isinstance(item, dict):
                                cid = item.get("card_id", "")
                            elif isinstance(item, (list, tuple)):
                                cid = str(item[0])
                            elif isinstance(item, str):
                                cid = item
                            else:
                                continue
                            # Normalize: strip variant suffix for matching
                            base = cid.split("/")[0] if "/" in cid else cid
                            ml_candidates.add(base)

                    # Find best DB candidate that also appears in ML visual pool
                    best_id = None
                    for row in db_candidates:
                        cid_base = row[0].split("/")[0]
                        if cid_base in ml_candidates:
                            best_id = row[0]
                            break

                    # If no overlap, just take the first DB candidate
                    if not best_id and len(db_candidates) == 1:
                        best_id = db_candidates[0][0]
                    elif not best_id and num_only:
                        # If we had a card number match, trust it
                        best_id = db_candidates[0][0]
                    elif not best_id:
                        # Fall back to match_vision_to_db scoring
                        best_id, _ = match_vision_to_db(vr, session=sess)

                    if best_id:
                        results[i] = {
                            "card_id": best_id,
                            "confidence": 0.85,
                            "method": "claude_vision",
                            "explanation": (
                                f"Claude vision: {vision_name} "
                                f"(ML had {ml_name or ml_card_id})"
                            ),
                            "raw_response": {
                                "vision_result": vr,
                                "ml_result": ml_result,
                            },
                        }
                        logger.info("Card %d: vision OVERRIDE %s -> %s (%s)",
                                    i, ml_card_id, best_id, vision_name)
                    else:
                        logger.debug("Card %d: vision=%s, couldn't resolve DB match",
                                     i, vision_name)
            finally:
                if own_sess:
                    sess.close()
        except Exception as e:
            logger.warning("Pass 3 (Claude vision) failed: %s", e)

    return results


def identify_page_vision_first(card_image_paths, session=None):
    """Identify cards using ML pipeline + multi-step Claude vision.

    Runs ML (DINOv2/CLIP) and multi-step Claude vision (name, attacks,
    number, era, HP) in parallel.  Combines signals:

    1. If ML and vision agree on name → confirm ML's pick (boost confidence)
    2. If they disagree → search ML candidates for vision's name
    3. If no ML candidate matches → try attack-based DB matching
    4. If attack match works → use that (vision name + attacks = strong)
    5. Last resort → use vision's name + number for DB lookup

    Args:
        card_image_paths: List of paths to card segment images.
        session: Optional DB session.

    Returns:
        List of result dicts, one per card.
    """
    import re as _re
    from concurrent.futures import ThreadPoolExecutor
    from cardprice.ml.claude_vision import (
        identify_cards_multi_step_parallel,
        match_attacks_to_db,
        match_multi_step_to_db,
    )
    from cardprice.db.session import SessionLocal
    from sqlalchemy import text as sa_text

    own_sess = session is None
    sess = session or SessionLocal()

    try:
        # Run ML pipeline and multi-step Claude vision in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            ml_future = pool.submit(identify_page, card_image_paths, sess)
            vision_future = pool.submit(
                identify_cards_multi_step_parallel,
                card_image_paths, "sonnet", 45, 4,
            )
            ml_results = ml_future.result()
            vision_results = vision_future.result()

        def _base_name(name):
            return _re.sub(
                r'\s*(ex|EX|δ|delta|V|VSTAR|VMAX|GX|LV\.\w+|Star)\s*$',
                '', name,
            ).strip().lower()

        results = list(ml_results)  # start with ML results

        for i, vr in enumerate(vision_results):
            if vr is None:
                continue
            vision_name = (vr.get("pokemon_name") or "").strip()
            vision_conf = vr.get("confidence", 0)
            vision_attacks = vr.get("attacks", [])
            vision_number = vr.get("card_number")
            vision_hp = vr.get("hp")

            if not vision_name or len(vision_name) < 2:
                continue
            if vision_name.lower() in ("unknown", "none", "card back",
                                        "pokemon", "pokémon"):
                # Detected card back or unreadable
                if vision_name.lower() in ("card back",):
                    results[i] = {
                        "card_id": None,
                        "confidence": 0.90,
                        "method": "vision_cardback",
                        "explanation": "Claude vision detected card back",
                    }
                continue

            ml_card_id = results[i].get("card_id") or ""
            ml_conf = results[i].get("confidence", 0)

            # Look up ML's pick's name from DB
            ml_name = ""
            if ml_card_id:
                base_cid = ml_card_id.split("/")[0] + "/normal"
                row = sess.execute(
                    sa_text("SELECT name FROM dim_cards WHERE card_id = :cid"),
                    {"cid": base_cid},
                ).fetchone()
                if row:
                    ml_name = row[0]

            # --- Case 1: Names agree → confirm ML's pick ---
            if (ml_name and (
                _base_name(ml_name) == _base_name(vision_name)
                or vision_name.lower() in ml_name.lower()
                or ml_name.lower() in vision_name.lower()
            )):
                results[i]["confidence"] = min(
                    results[i].get("confidence", 0) + 0.10, 1.0,
                )
                results[i]["explanation"] = (
                    (results[i].get("explanation") or "") +
                    " (confirmed by Claude vision)"
                )
                logger.info("Card %d: ML+vision agree on %s", i, vision_name)
                continue

            # --- Case 2: Names disagree → search ML candidate pool ---
            raw = results[i].get("raw_response", {})
            ml_candidates = []
            for key in ("scored_candidates", "top_matches", "top_alternatives"):
                for item in raw.get(key, []):
                    if isinstance(item, dict):
                        ml_candidates.append(item.get("card_id", ""))
                    elif isinstance(item, (list, tuple)):
                        ml_candidates.append(str(item[0]))
                    elif isinstance(item, str):
                        ml_candidates.append(item)

            best_match = None
            for cid in ml_candidates:
                if "/" not in cid:
                    cid_norm = cid + "/normal"
                else:
                    cid_norm = cid
                row = sess.execute(
                    sa_text("SELECT name FROM dim_cards WHERE card_id = :cid"),
                    {"cid": cid_norm},
                ).fetchone()
                if row and (
                    _base_name(row[0]) == _base_name(vision_name)
                    or vision_name.lower() in row[0].lower()
                ):
                    best_match = cid_norm
                    break

            if best_match:
                results[i] = {
                    "card_id": best_match,
                    "confidence": 0.85,
                    "method": "vision+ml_rerank",
                    "explanation": (
                        f"Claude vision ({vision_name}) reranked ML candidates"
                    ),
                    "raw_response": {
                        "vision_result": vr,
                        "ml_result": ml_results[i],
                    },
                }
                logger.info("Card %d: vision+ML rerank: %s -> %s",
                            i, ml_card_id, best_match)
                continue

            # --- Case 3: Try attack-based matching ---
            if vision_attacks and len(vision_attacks) >= 1:
                atk_id, atk_conf = match_attacks_to_db(
                    vision_attacks, pokemon_name=vision_name,
                    hp=vision_hp, card_number=vision_number,
                    session=sess,
                )
                if atk_id and atk_conf >= 0.55:
                    results[i] = {
                        "card_id": atk_id,
                        "confidence": atk_conf,
                        "method": "vision_attacks",
                        "explanation": (
                            f"Claude vision ({vision_name}) + "
                            f"attack match [{', '.join(vision_attacks)}]"
                        ),
                        "raw_response": {
                            "vision_result": vr,
                            "ml_result": ml_results[i],
                        },
                    }
                    logger.info("Card %d: attack match: %s (%s) -> %s",
                                i, vision_name, vision_attacks, atk_id)
                    continue

            # --- Case 4: Fall back to vision name+number DB lookup ---
            db_id, db_conf = match_multi_step_to_db(vr, session=sess)
            if db_id and db_conf >= 0.60:
                results[i] = {
                    "card_id": db_id,
                    "confidence": db_conf,
                    "method": "vision_db",
                    "explanation": (
                        f"Claude vision ({vision_name}) DB match"
                    ),
                    "raw_response": {
                        "vision_result": vr,
                        "ml_result": ml_results[i],
                    },
                }
                logger.info("Card %d: vision DB match: %s -> %s (conf=%.2f)",
                            i, vision_name, db_id, db_conf)
            else:
                logger.debug("Card %d: vision=%s, no override found, "
                             "keeping ML=%s", i, vision_name, ml_card_id)

    finally:
        if own_sess:
            sess.close()

    return results


def identify_card_robust(image_path, session=None):
    """Identify a card, trying rotations if the initial attempt is low-confidence.

    Tries identify_card at 0 degrees first.  If confidence is below
    _ROBUST_CONFIDENCE_THRESHOLD, also tries 90, 180, and 270 degree
    rotations and returns whichever attempt produced the highest confidence.

    Temporary rotated images are cleaned up after use.
    """
    from PIL import Image

    best = identify_card(image_path, session=session)
    if best["confidence"] >= _ROBUST_CONFIDENCE_THRESHOLD:
        return best

    logger.info("Robust: 0deg confidence=%.4f < %.2f, trying rotations ...",
                best["confidence"], _ROBUST_CONFIDENCE_THRESHOLD)

    tmp_files = []
    try:
        for angle in (90, 180, 270):
            try:
                img = Image.open(image_path)
                rotated = img.rotate(-angle, expand=True)  # negative = clockwise
                suffix = Path(image_path).suffix or ".png"
                fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix=f"card_rot{angle}_")
                os.close(fd)
                tmp_files.append(tmp_path)
                rotated.save(tmp_path)
                rotated.close()
                img.close()

                candidate = identify_card(tmp_path, session=session)
                logger.info("Robust: %ddeg -> confidence=%.4f method=%s card=%s",
                            angle, candidate["confidence"], candidate.get("method"), candidate.get("card_id"))

                if candidate["confidence"] > best["confidence"]:
                    best = candidate
                    best["explanation"] = (best.get("explanation") or "") + f" (matched at {angle}deg rotation)"

                if best["confidence"] >= _ROBUST_CONFIDENCE_THRESHOLD:
                    break
            except Exception as e:
                logger.warning("Robust: rotation %ddeg failed: %s", angle, e)
    finally:
        for tmp_path in tmp_files:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    logger.info("Robust: best result confidence=%.4f method=%s card=%s",
                best["confidence"], best.get("method"), best.get("card_id"))
    return best


# ---------------------------------------------------------------------------
# Helper: normalize card_id from index format to DB format
# ---------------------------------------------------------------------------

def _normalize_card_id(raw_cid: str) -> str:
    """Normalize a raw card_id from DINOv2/CLIP index to DB format.

    Index format: "set/set-num/variant" (e.g. "bw5/bw5-107/normal")
    DB format:    "set-num/variant"     (e.g. "bw5-107/normal")

    If the raw_cid has three path segments (set/id/variant), strip the first.
    Otherwise return as-is.
    """
    parts = raw_cid.split("/")
    if len(parts) >= 3:
        # "bw5/bw5-107/normal" -> "bw5-107/normal"
        return "/".join(parts[1:])
    if len(parts) == 2:
        return raw_cid
    # Single segment -- unlikely but handle gracefully
    return raw_cid


# ---------------------------------------------------------------------------
# Ensemble: DINOv2 + CLIP parallel voting
# ---------------------------------------------------------------------------

# Thresholds for the ensemble voter
_ENSEMBLE_AGREEMENT_CONFIDENCE = 0.85   # both top-1 agree -> this confidence floor
_ENSEMBLE_BOOST_FACTOR = 0.10           # bonus for appearing in both top-10 lists
_ENSEMBLE_MIN_ACCEPT = 0.55             # minimum ensemble score to accept


def _run_dino(image_path: str, query_embedding=None) -> list[tuple[str, float]]:
    """Run DINOv2 identification, returning normalized (card_id, score) top-10.

    Applies CLAHE + glare removal preprocessing when available, matching
    the cascade's Tier 2 behaviour.
    """
    from cardprice.ml.dino_matcher import identify_card as dino_identify
    dino_idx, dino_cids = _get_dino_index()
    if dino_idx is None:
        return []
    query_path = image_path
    preproc_tmp = None
    if query_embedding is None:
        try:
            from cardprice.ml.preprocess import preprocess_for_matching
            preproc_tmp = preprocess_for_matching(image_path)
            query_path = preproc_tmp
        except Exception:
            pass
    try:
        matches = dino_identify(query_path, faiss_index=dino_idx, card_ids_list=dino_cids,
                                top_k=10, query_embedding=query_embedding)
        return [(_normalize_card_id(cid), score) for cid, score in matches]
    finally:
        if preproc_tmp:
            try:
                os.unlink(preproc_tmp)
            except OSError:
                pass


def _run_clip(image_path: str, query_embedding=None) -> list[tuple[str, float]]:
    """Run CLIP image-to-image identification, returning normalized (card_id, score) top-10.

    DISABLED: CLIP loading causes SIGSEGV when PaddlePaddle is resident in the
    same process (safetensors weight materializer + PaddlePaddle memory corruption).
    CLIP also contributes 0 unique correct identifications in eval, so disabling
    it has zero accuracy impact while eliminating a major crash vector.
    """
    return []


def identify_card_ensemble(image_path, session=None, page_context=None,
                           _dino_embedding=None, _clip_embedding=None,
                           _precomputed_ocr_name=None):
    """Identify a card using DINOv2 + CLIP ensemble voting.

    Runs both methods in parallel, then combines their results:
    1. Get top-10 from each method
    2. Apply page context reranking if available
    2.5. Run border color analysis to filter candidates by era
         (e.g. DP-era yellow border filters out EX/BW candidates)
    3. Cards appearing in both lists get a score boost
    4. If both top-1 agree, assign high confidence regardless of individual scores
    5. If they disagree, use the method with higher relative margin (top1 - top2)

    When _dino_embedding and _clip_embedding are provided, no GPU is needed.

    Returns dict with keys: card_id, confidence, method, explanation, raw_response.
    """
    image_path = str(image_path)

    # Convert HEIC/HEIF if needed
    try:
        from cardprice.utils.image_convert import ensure_compatible
        image_path = ensure_compatible(image_path)
    except Exception as e:
        logger.warning("Image conversion failed, using original path: %s", e)

    # --- Phase 1: Run DINOv2 and CLIP in parallel ---
    dino_results = []
    clip_results = []
    dino_error = None
    clip_error = None

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_run_dino, image_path, query_embedding=_dino_embedding): "dino",
            pool.submit(_run_clip, image_path, query_embedding=_clip_embedding): "clip",
        }
        for future in as_completed(futures):
            method = futures[future]
            try:
                results = future.result()
                if method == "dino":
                    dino_results = results
                else:
                    clip_results = results
            except Exception as e:
                logger.warning("Ensemble: %s failed: %s", method, e)
                if method == "dino":
                    dino_error = e
                else:
                    clip_error = e

    # Apply page context reranking if available
    if page_context and page_context.get("likely_sets"):
        try:
            from cardprice.ml.page_context import rerank_with_context
            if dino_results:
                dino_results = rerank_with_context(dino_results, page_context)
            if clip_results:
                clip_results = rerank_with_context(clip_results, page_context)
            logger.info("Ensemble: applied page context (sets=%s)", page_context["likely_sets"][:3])
        except Exception as e:
            logger.debug("Ensemble: page context reranking failed: %s", e)

    # --- Phase 1.5: Border analysis to filter by era ---
    # When DINOv2 and CLIP disagree, border color can eliminate wrong-era candidates.
    border_info = None
    _BORDER_FILTER_MIN_CONFIDENCE = 0.35
    try:
        from cardprice.ml.border_analyzer import analyze_border, SET_TO_ERA
        border_info = analyze_border(image_path=image_path)
        logger.info("Ensemble: border analysis: color=%s era=%s confidence=%.2f sets=%d",
                     border_info["border_color"], border_info["era"],
                     border_info["confidence"], len(border_info["era_sets"]))

        if border_info["confidence"] >= _BORDER_FILTER_MIN_CONFIDENCE:
            era_set_ids = set(border_info["era_sets"])

            def _card_id_to_set(card_id: str) -> str:
                """Extract set ID from card_id like 'dp1-4/normal' -> 'dp1'."""
                # card_id format: "set-num/variant" e.g. "dp1-4/normal"
                base = card_id.split("/")[0]  # "dp1-4"
                # Set ID is everything before the last hyphen-number segment
                # e.g. "dp1-4" -> "dp1", "ex6-112" -> "ex6", "bw5-105" -> "bw5"
                parts = base.rsplit("-", 1)
                return parts[0] if len(parts) == 2 else base

            def _filter_by_era(results, era_sets):
                """Filter results to only cards from matching era sets."""
                return [(cid, score) for cid, score in results
                        if _card_id_to_set(cid) in era_sets]

            dino_filtered = _filter_by_era(dino_results, era_set_ids) if dino_results else []
            clip_filtered = _filter_by_era(clip_results, era_set_ids) if clip_results else []

            # Safety: only apply filtering if it doesn't eliminate ALL candidates
            # from BOTH lists. If one list is fully eliminated, that's fine (the
            # other method was probably right).
            if dino_filtered or clip_filtered:
                d_removed = len(dino_results) - len(dino_filtered)
                c_removed = len(clip_results) - len(clip_filtered)
                if d_removed > 0 or c_removed > 0:
                    logger.info("Ensemble: border filter removed %d/%d dino, %d/%d clip candidates "
                                "(era=%s, confidence=%.2f)",
                                d_removed, len(dino_results),
                                c_removed, len(clip_results),
                                border_info["era"], border_info["confidence"])
                    dino_results = dino_filtered if dino_filtered else dino_results
                    clip_results = clip_filtered if clip_filtered else clip_results
            else:
                logger.info("Ensemble: border filter would remove ALL candidates, skipping "
                            "(era=%s, confidence=%.2f)",
                            border_info["era"], border_info["confidence"])
    except ImportError as e:
        logger.info("Ensemble: border analysis SKIPPED -- missing dependency: %s", e)
    except Exception as e:
        logger.warning("Ensemble: border analysis ERROR -- %s", e)

    logger.info("Ensemble: DINOv2 returned %d results, CLIP returned %d results",
                len(dino_results), len(clip_results))

    # If both failed, return empty result
    if not dino_results and not clip_results:
        logger.info("Ensemble: both methods returned no results")
        return {
            "card_id": None, "confidence": 0.0, "method": "ensemble",
            "explanation": "Both DINOv2 and CLIP failed to produce results",
            "raw_response": {"dino_error": str(dino_error), "clip_error": str(clip_error)},
        }

    # If only one succeeded, fall back to it directly
    if not dino_results:
        logger.info("Ensemble: only CLIP available, using CLIP result directly")
        return _single_method_result(clip_results, "clip (ensemble fallback)")
    if not clip_results:
        logger.info("Ensemble: only DINOv2 available, using DINOv2 result directly")
        return _single_method_result(
            dino_results, "dino (ensemble fallback)",
            ocr_name=_precomputed_ocr_name, image_path=image_path,
            page_context=page_context,
        )

    # --- Phase 2: Ensemble voting ---
    dino_top1_id, dino_top1_score = dino_results[0]
    clip_top1_id, clip_top1_score = clip_results[0]

    # Build lookup dicts for both sets (card_id -> score)
    dino_dict = {cid: score for cid, score in dino_results}
    clip_dict = {cid: score for cid, score in clip_results}

    # Find overlap: cards in both top-10 lists
    overlap_ids = set(dino_dict.keys()) & set(clip_dict.keys())
    logger.info("Ensemble: %d cards overlap in both top-10 lists", len(overlap_ids))

    # Compute relative margins (top1 - top2 gap)
    dino_margin = (dino_results[0][1] - dino_results[1][1]) if len(dino_results) >= 2 else dino_results[0][1]
    clip_margin = (clip_results[0][1] - clip_results[1][1]) if len(clip_results) >= 2 else clip_results[0][1]

    # --- Decision logic ---
    result = {
        "card_id": None, "confidence": 0.0, "method": "ensemble",
        "explanation": None,
        "raw_response": {
            "dino_top10": dino_results,
            "clip_top10": clip_results,
            "overlap_ids": list(overlap_ids),
            "dino_margin": dino_margin,
            "clip_margin": clip_margin,
            "border_analysis": {
                "color": border_info["border_color"],
                "era": border_info["era"],
                "confidence": border_info["confidence"],
            } if border_info else None,
        },
    }

    # Case 1: Both top-1 agree -- strong signal
    if dino_top1_id == clip_top1_id:
        # Average the scores and apply agreement bonus
        avg_score = (dino_top1_score + clip_top1_score) / 2.0
        ensemble_confidence = max(avg_score + _ENSEMBLE_BOOST_FACTOR,
                                  _ENSEMBLE_AGREEMENT_CONFIDENCE)
        result["card_id"] = dino_top1_id
        result["confidence"] = min(ensemble_confidence, 1.0)
        result["method"] = "ensemble (agree)"
        result["explanation"] = (
            f"DINOv2 and CLIP both agree on top match. "
            f"DINOv2={dino_top1_score:.3f}, CLIP={clip_top1_score:.3f}, "
            f"ensemble={result['confidence']:.3f}"
        )
        logger.info("Ensemble: AGREEMENT on %s (confidence=%.4f)",
                     result["card_id"], result["confidence"])
        return result

    # Case 2: Top-1 disagree -- check overlap and margins
    # Score each candidate by combining signals
    candidate_scores = {}

    # Score all overlap cards (appear in both top-10)
    for cid in overlap_ids:
        d_score = dino_dict[cid]
        c_score = clip_dict[cid]
        # Weighted average with overlap boost
        combined = (d_score + c_score) / 2.0 + _ENSEMBLE_BOOST_FACTOR
        candidate_scores[cid] = {
            "combined": combined,
            "dino": d_score,
            "clip": c_score,
            "in_both": True,
        }

    # Also score top-1 from each method if not already in candidates
    for cid, d_score, c_score, source in [
        (dino_top1_id, dino_top1_score, clip_dict.get(dino_top1_id), "dino"),
        (clip_top1_id, dino_dict.get(clip_top1_id), clip_top1_score, "clip"),
    ]:
        if cid not in candidate_scores:
            # Only in one list -- use that score alone (no boost)
            single_score = d_score if d_score is not None else c_score
            candidate_scores[cid] = {
                "combined": float(single_score) if single_score is not None else 0.0,
                "dino": d_score,
                "clip": c_score,
                "in_both": False,
            }

    # If any overlap exists, ALWAYS prefer the best overlap card.
    # A card appearing in both DINOv2 and CLIP top-10 is a much stronger
    # signal than a high single-method score (which may be noise).
    if overlap_ids:
        best_cid = max(
            overlap_ids,
            key=lambda k: candidate_scores[k]["combined"],
        )
        best_info = candidate_scores[best_cid]
        result["card_id"] = best_cid
        result["confidence"] = min(best_info["combined"], 1.0)
        result["method"] = "ensemble (overlap)"
        result["explanation"] = (
            f"Card found in both top-10 lists with boosted score. "
            f"DINOv2={best_info['dino']:.3f}, CLIP={best_info['clip']:.3f}, "
            f"combined={best_info['combined']:.3f}"
        )
        logger.info("Ensemble: OVERLAP winner %s (confidence=%.4f)",
                     result["card_id"], result["confidence"])
        return result

    # Neither top-1 in overlap -- try OCR tiebreaker before falling back to margin.
    # Uses pre-computed PaddleOCR name when available (zero cost), otherwise
    # falls back to EasyOCR extract_card_name (~5s).
    ocr_override = None
    try:
        ocr_cleaned = None
        if _precomputed_ocr_name:
            ocr_cleaned = _precomputed_ocr_name
        else:
            from cardprice.ml.ocr_matcher import extract_card_name, _clean_ocr_text
            ocr_text, ocr_conf = extract_card_name(image_path)
            ocr_cleaned = _clean_ocr_text(ocr_text)
        logger.info("Ensemble OCR tiebreaker: name=%r", ocr_cleaned)

        if ocr_cleaned and len(ocr_cleaned) >= 2:
            from cardprice.ml.ocr_matcher import _load_card_names
            card_names = _load_card_names()
            card_name_lookup = {cid: name for cid, name, _sid in card_names}

            dino_name = card_name_lookup.get(dino_top1_id, "")
            clip_name = card_name_lookup.get(clip_top1_id, "")

            from rapidfuzz import fuzz
            dino_fuzzy = fuzz.token_set_ratio(ocr_cleaned.lower(), dino_name.lower()) if dino_name else 0
            clip_fuzzy = fuzz.token_set_ratio(ocr_cleaned.lower(), clip_name.lower()) if clip_name else 0

            logger.info("Ensemble OCR tiebreaker: dino=%s (%r, fuzzy=%d), clip=%s (%r, fuzzy=%d)",
                         dino_top1_id, dino_name, dino_fuzzy,
                         clip_top1_id, clip_name, clip_fuzzy)

            OCR_MATCH_THRESH = 80
            OCR_MIN_GAP = 10

            dino_matches_ocr = dino_fuzzy >= OCR_MATCH_THRESH
            clip_matches_ocr = clip_fuzzy >= OCR_MATCH_THRESH
            gap = abs(dino_fuzzy - clip_fuzzy)

            if dino_matches_ocr and not clip_matches_ocr and gap >= OCR_MIN_GAP:
                ocr_override = "dino"
            elif clip_matches_ocr and not dino_matches_ocr and gap >= OCR_MIN_GAP:
                ocr_override = "clip"

            result["raw_response"]["ocr_text"] = ocr_cleaned
            result["raw_response"]["ocr_override"] = ocr_override
    except Exception as e:
        logger.warning("Ensemble OCR tiebreaker: ERROR -- %s", e)

    DINO_ACCEPT_THRESHOLD = 0.65
    CLIP_ACCEPT_THRESHOLD = 0.75
    if ocr_override is not None:
        use_dino = ocr_override == "dino"
        decision_reason = "ocr"
    else:
        dino_below = dino_top1_score < DINO_ACCEPT_THRESHOLD
        clip_below = clip_top1_score < CLIP_ACCEPT_THRESHOLD
        if dino_below and not clip_below:
            use_dino = True
            decision_reason = "confidence_gate"
        elif clip_below and not dino_below:
            use_dino = True
            decision_reason = "confidence_gate"
        elif dino_below and clip_below:
            use_dino = dino_top1_score >= clip_top1_score
            decision_reason = "both_low"
        else:
            margin_diff = abs(dino_margin - clip_margin)
            if margin_diff < 0.02:
                use_dino = dino_top1_score >= clip_top1_score
            else:
                use_dino = dino_margin > clip_margin
            decision_reason = "margin"

    if use_dino:
        winner_id, winner_score = dino_top1_id, dino_top1_score
        winner_method = "dino"
        winner_margin = dino_margin
        loser_method = "clip"
        loser_margin = clip_margin
    else:
        winner_id, winner_score = clip_top1_id, clip_top1_score
        winner_method = "clip"
        winner_margin = clip_margin
        loser_method = "dino"
        loser_margin = dino_margin

    result["card_id"] = winner_id
    result["confidence"] = float(winner_score)
    if decision_reason == "ocr":
        result["method"] = f"ensemble (ocr: {winner_method})"
        result["explanation"] = (
            f"DINOv2 and CLIP disagree. OCR read {ocr_cleaned!r} which matches "
            f"{winner_method}'s candidate. Using {winner_method} over {loser_method}. "
            f"DINOv2 top1: {dino_top1_id} ({dino_top1_score:.3f}), "
            f"CLIP top1: {clip_top1_id} ({clip_top1_score:.3f})"
        )
    elif decision_reason == "confidence_gate":
        result["method"] = f"ensemble (confidence_gate: {winner_method})"
        result["explanation"] = (
            f"DINOv2 and CLIP disagree. Preferring {winner_method} ({winner_score:.3f}) — "
            f"DINOv2 is more reliable for exact visual matching when they disagree. "
            f"DINOv2 top1: {dino_top1_id} ({dino_top1_score:.3f}), "
            f"CLIP top1: {clip_top1_id} ({clip_top1_score:.3f})"
        )
    elif decision_reason == "both_low":
        result["method"] = f"ensemble (both_low: {winner_method})"
        result["explanation"] = (
            f"DINOv2 and CLIP disagree, both below thresholds. Using higher score: "
            f"{winner_method} ({winner_score:.3f}). "
            f"DINOv2 top1: {dino_top1_id} ({dino_top1_score:.3f}), "
            f"CLIP top1: {clip_top1_id} ({clip_top1_score:.3f})"
        )
    else:
        result["method"] = f"ensemble (margin: {winner_method})"
        result["explanation"] = (
            f"DINOv2 and CLIP disagree. Using {winner_method} (margin={winner_margin:.3f}) "
            f"over {loser_method} (margin={loser_margin:.3f}). "
            f"DINOv2 top1: {dino_top1_id} ({dino_top1_score:.3f}), "
            f"CLIP top1: {clip_top1_id} ({clip_top1_score:.3f})"
        )
    logger.info("Ensemble: %s winner %s via %s (confidence=%.4f, margin=%.4f)",
                 decision_reason.upper(), result["card_id"], winner_method,
                 result["confidence"], winner_margin)
    return result


# ---------------------------------------------------------------------------
# Multi-signal identification: combines ALL available signals
# ---------------------------------------------------------------------------

# DB metadata cache for multi-signal scoring: {card_id: {name, hp, set_id, supertype, subtypes}}
_card_metadata_cache: dict[str, dict] | None = None


def _get_card_metadata() -> dict[str, dict]:
    """Lazy-load card metadata from dim_cards for multi-signal filtering.

    Returns dict keyed by card_id with values containing name, hp, set_id,
    supertype, and subtypes.
    """
    global _card_metadata_cache
    if _card_metadata_cache is not None:
        return _card_metadata_cache

    from cardprice.db.session import engine
    from sqlalchemy import text as sql_text

    with engine.connect() as conn:
        rows = conn.execute(sql_text(
            "SELECT card_id, name, hp, set_id, supertype, subtypes FROM dim_cards"
        )).fetchall()

    _card_metadata_cache = {}
    for r in rows:
        _card_metadata_cache[r[0]] = {
            "name": r[1],
            "hp": r[2],
            "set_id": r[3],
            "supertype": r[4],
            "subtypes": r[5] if r[5] else [],
        }
    logger.info("Loaded %d card metadata entries for multi-signal scoring.", len(_card_metadata_cache))
    return _card_metadata_cache


def identify_card_multisignal(image_path, session=None, page_context=None):
    """Identify a card by combining ALL available signals.

    This is the ultimate identification method, sitting above the ensemble.
    It runs all signal extractors in parallel, then scores candidates from
    the DINOv2+CLIP top-10 lists against every extracted signal.

    Signals used:
        1. DINOv2 top-10 (visual similarity)
        2. CLIP top-10 (visual + semantic similarity)
        3. OCR card name (fuzzy text match)
        4. HP value (from hp_detector)
        5. Card type (from type_detector, color-based)
        6. Border/era analysis (from border_analyzer)
        7. Page context (set/era prior from neighboring cards)
        8. DP-era level (OCR "LV.XX" + dp_level_map.json matching)

    Scoring approach:
        - Start with visual similarity score (avg of DINO+CLIP if both present)
        - Apply bonuses/penalties for each signal match:
            +0.15 for exact name match (fuzzy >= 90)
            +0.05 for partial name match (fuzzy >= 70)
            +0.12 for DP-era level match (name + LV.XX from OCR)
            +0.10 for HP match (only when candidate has visual overlap)
            +0.00 for HP match on single-visual candidates (skipped)
            +0.05 for type match (top-1 detected type)
            +0.08 for era/set match from border analysis
            +0.10 for page context set match
            +0.05 for page context era match
        - Candidates with signal contradictions get penalties:
            -0.15 for name mismatch when OCR is confident
            -0.05 for HP mismatch when candidate has visual overlap (mild)
            +0.00 for HP mismatch on single-visual candidates (skipped)
            -0.05 for era mismatch when border analysis is confident
        - HP discount rationale: HP OCR is noisy on binder-sleeve photos
          (glare, partial occlusion, angle). When a candidate only appears
          in one visual model's top-10, HP match/mismatch is skipped to
          prevent a misread HP from overriding visual similarity scores.

    Args:
        image_path: Path to the card image.
        session: Optional DB session.
        page_context: Optional page context dict from identify_page_context.

    Returns:
        Dict with keys: card_id, confidence, method, explanation, raw_response.
        raw_response includes all signal details for debugging.
    """
    image_path = str(image_path)

    # Convert HEIC/HEIF if needed
    try:
        from cardprice.utils.image_convert import ensure_compatible
        image_path = ensure_compatible(image_path)
    except Exception as e:
        logger.warning("Image conversion failed, using original path: %s", e)

    # --- Phase 1: Run ALL signal extractors in parallel ---
    dino_results = []
    clip_results = []
    ocr_name = None
    ocr_confidence = 0.0
    ocr_raw = None
    hp_value = None
    type_predictions = []
    border_info = None
    dp_level_name = None
    dp_level_value = None
    dp_level_candidates = []

    signal_errors = {}

    def _run_ocr_name_signal():
        from cardprice.ml.ocr_matcher import extract_card_name, _clean_ocr_text
        raw_text, conf = extract_card_name(image_path)
        cleaned = _clean_ocr_text(raw_text)
        return cleaned, conf, raw_text

    def _run_hp_signal():
        from cardprice.ml.hp_detector import detect_hp
        return detect_hp(image_path)

    def _run_type_signal():
        from cardprice.ml.type_detector import detect_type
        return detect_type(image_path, top_n=3)

    def _run_border_signal():
        from cardprice.ml.border_analyzer import analyze_border
        return analyze_border(image_path)

    def _run_dp_level_signal():
        from cardprice.ml.ocr_matcher import extract_card_name_all_fragments, _extract_level_from_ocr
        fragments = extract_card_name_all_fragments(image_path)
        if not fragments:
            return None, None, []
        ocr_texts = [t for t, c in fragments]
        name, level = _extract_level_from_ocr(ocr_texts)
        if name and level:
            from cardprice.ml.ocr_matcher import match_by_dp_level
            candidates = match_by_dp_level(name, level)
            return name, level, candidates
        return name, level, []

    with ThreadPoolExecutor(max_workers=7) as pool:
        futures = {
            pool.submit(_run_dino, image_path): "dino",
            pool.submit(_run_clip, image_path): "clip",
            pool.submit(_run_ocr_name_signal): "ocr",
            pool.submit(_run_hp_signal): "hp",
            pool.submit(_run_type_signal): "type",
            pool.submit(_run_border_signal): "border",
            pool.submit(_run_dp_level_signal): "dp_level",
        }
        for future in as_completed(futures):
            signal = futures[future]
            try:
                res = future.result()
                if signal == "dino":
                    dino_results = res
                elif signal == "clip":
                    clip_results = res
                elif signal == "ocr":
                    ocr_name, ocr_confidence, ocr_raw = res
                elif signal == "hp":
                    hp_value = res
                elif signal == "type":
                    type_predictions = res
                elif signal == "border":
                    border_info = res
                elif signal == "dp_level":
                    dp_level_name, dp_level_value, dp_level_candidates = res
            except Exception as e:
                logger.warning("Multisignal: %s extractor failed: %s", signal, e)
                signal_errors[signal] = str(e)

    logger.info(
        "Multisignal signals: dino=%d clip=%d ocr=%r hp=%s type=%s border=%s level=%s(%s)",
        len(dino_results), len(clip_results),
        ocr_name, hp_value,
        type_predictions[0] if type_predictions else None,
        border_info.get("era") if border_info else None,
        dp_level_value, dp_level_name,
    )

    # --- Phase 2: Pool all candidate card_ids ---
    candidate_visual_scores: dict[str, dict] = {}

    for cid, score in dino_results:
        if cid not in candidate_visual_scores:
            candidate_visual_scores[cid] = {"dino": 0.0, "clip": 0.0}
        candidate_visual_scores[cid]["dino"] = score

    for cid, score in clip_results:
        if cid not in candidate_visual_scores:
            candidate_visual_scores[cid] = {"dino": 0.0, "clip": 0.0}
        candidate_visual_scores[cid]["clip"] = score

    if not candidate_visual_scores:
        logger.info("Multisignal: no visual candidates, returning empty result")
        return {
            "card_id": None, "confidence": 0.0, "method": "multisignal",
            "explanation": "No visual candidates from DINOv2 or CLIP",
            "raw_response": {"signal_errors": signal_errors},
        }

    # --- Phase 2.5: Add OCR-matched candidates to the pool ---
    card_meta = _get_card_metadata()
    ocr_matches = []
    if ocr_name and len(ocr_name) >= 2:
        try:
            from cardprice.ml.ocr_matcher import fuzzy_match_card_name
            ocr_matches = fuzzy_match_card_name(ocr_name, top_k=10, score_cutoff=70.0)
            for cid, _name, _sid, _score in ocr_matches:
                if cid not in candidate_visual_scores:
                    candidate_visual_scores[cid] = {"dino": 0.0, "clip": 0.0}
        except Exception as e:
            logger.warning("Multisignal: OCR fuzzy match failed: %s", e)

    # --- Phase 2.6: Add DP-level-matched candidates to the pool ---
    dp_level_match_ids = set()
    if dp_level_candidates:
        for cid, _name, _sid, _score, _details in dp_level_candidates:
            dp_level_match_ids.add(cid)
            if cid not in candidate_visual_scores:
                candidate_visual_scores[cid] = {"dino": 0.0, "clip": 0.0}

    # Build OCR lookup for fast scoring
    ocr_match_by_id = {}
    if ocr_matches:
        for cid, name, sid, fuzzy_score in ocr_matches:
            ocr_match_by_id[cid] = {"name": name, "set_id": sid, "fuzzy_score": fuzzy_score}

    # Extract signal parameters
    detected_type = type_predictions[0][0] if type_predictions else None
    detected_type_conf = type_predictions[0][1] if type_predictions else 0.0

    era_sets = set(border_info.get("era_sets", [])) if border_info else set()
    border_conf = border_info.get("confidence", 0.0) if border_info else 0.0

    page_sets = set(page_context.get("likely_sets", [])) if page_context else set()
    page_era = page_context.get("era") if page_context else None
    page_conf = page_context.get("confidence", 0.0) if page_context else 0.0

    # --- Phase 3: Score each candidate against all signals ---
    scored_candidates = []

    for cid, vis_scores in candidate_visual_scores.items():
        meta = card_meta.get(cid)
        if meta is None:
            continue

        # Base visual score: average of available visual scores
        vis_parts = []
        if vis_scores["dino"] > 0:
            vis_parts.append(vis_scores["dino"])
        if vis_scores["clip"] > 0:
            vis_parts.append(vis_scores["clip"])
        base_score = sum(vis_parts) / len(vis_parts) if vis_parts else 0.0

        bonuses = []
        penalties = []
        total_adjustment = 0.0

        # --- OCR name matching ---
        if ocr_name and len(ocr_name) >= 2:
            cand_name = meta.get("name", "")
            if cid in ocr_match_by_id:
                fuzzy = ocr_match_by_id[cid]["fuzzy_score"]
            else:
                try:
                    from rapidfuzz import fuzz
                    fuzzy = fuzz.token_set_ratio(ocr_name.lower(), cand_name.lower())
                except ImportError:
                    fuzzy = 0

            if fuzzy >= 90:
                total_adjustment += 0.15
                bonuses.append(f"name={cand_name}(fuzzy={fuzzy:.0f})")
            elif fuzzy >= 70:
                total_adjustment += 0.05
                bonuses.append(f"name~={cand_name}(fuzzy={fuzzy:.0f})")
            elif ocr_confidence > 0.5 and fuzzy < 50:
                total_adjustment -= 0.15
                penalties.append(f"name_mismatch(ocr={ocr_name!r},card={cand_name!r},fuzzy={fuzzy:.0f})")

        # --- HP matching ---
        # Only trust HP if the detected value is plausible (>= 30).
        # OCR often misreads HP as "10" or single digits from card numbers/damage.
        #
        # Discount HP when the candidate lacks visual overlap (only appears in
        # one of DINOv2/CLIP, not both).  A single-model candidate boosted by
        # noisy HP can override a visually stronger candidate that both models
        # agree on.  Without corroboration from a second visual model, the HP
        # signal should carry less weight.  Similarly, don't heavily penalize
        # HP mismatch on candidates that DO have visual overlap -- the HP OCR
        # is often wrong on binder-sleeve photos.
        if hp_value is not None and hp_value >= 30:
            cand_hp = meta.get("hp")
            if cand_hp is not None:
                has_visual_overlap = (vis_scores["dino"] > 0 and vis_scores["clip"] > 0)

                if has_visual_overlap:
                    # Both visual models found this candidate: HP is a useful
                    # disambiguator between visually similar cards.
                    if cand_hp == hp_value:
                        total_adjustment += 0.10
                        bonuses.append(f"hp={hp_value}")
                    else:
                        # Mild penalty -- HP OCR is unreliable on binder photos
                        # so don't punish too hard when visual evidence is strong.
                        total_adjustment -= 0.05
                        penalties.append(f"hp_mismatch(detected={hp_value},card={cand_hp},mild)")
                else:
                    # Only one visual model found this candidate: HP signal is
                    # not trustworthy enough to override visual similarity scores.
                    # A misread HP can boost a wrong candidate from one model while
                    # penalizing the correct candidate from the other model,
                    # flipping the result.  Skip HP adjustment entirely for
                    # single-visual candidates to let visual scores decide.
                    if cand_hp == hp_value:
                        bonuses.append(f"hp={hp_value}(skipped,single_visual)")
                    else:
                        penalties.append(f"hp_mismatch(detected={hp_value},card={cand_hp},skipped)")

        # --- DP-era level matching ---
        if dp_level_value is not None and cid in dp_level_match_ids:
            total_adjustment += 0.12
            bonuses.append(f"dp_level={dp_level_value}(name={dp_level_name!r})")

        # --- Border/era matching ---
        if era_sets and border_conf > 0.3:
            cand_set = meta.get("set_id", "")
            if cand_set in era_sets:
                total_adjustment += 0.08
                bonuses.append(f"era_match(set={cand_set})")
            elif border_conf > 0.6:
                total_adjustment -= 0.05
                penalties.append(f"era_mismatch(set={cand_set})")

        # --- Page context matching ---
        if page_sets and page_conf > 0.3:
            cand_set = meta.get("set_id", "")
            if cand_set in page_sets:
                total_adjustment += 0.10
                bonuses.append(f"page_set={cand_set}")
            else:
                try:
                    from cardprice.ml.page_context import _era_for_set
                    cand_era = _era_for_set(cand_set)
                    if cand_era and cand_era == page_era:
                        total_adjustment += 0.05
                        bonuses.append(f"page_era={cand_era}")
                except Exception:
                    pass

        # --- Visual overlap bonus: in both DINO and CLIP top-10 ---
        if vis_scores["dino"] > 0 and vis_scores["clip"] > 0:
            total_adjustment += 0.05
            bonuses.append("visual_overlap")

        final_score = max(base_score + total_adjustment, 0.0)  # no upper cap — let bonuses differentiate

        scored_candidates.append({
            "card_id": cid,
            "final_score": final_score,
            "base_visual": base_score,
            "adjustment": total_adjustment,
            "dino_score": vis_scores["dino"],
            "clip_score": vis_scores["clip"],
            "bonuses": bonuses,
            "penalties": penalties,
            "name": meta.get("name", ""),
            "hp": meta.get("hp"),
            "set_id": meta.get("set_id", ""),
        })

    if not scored_candidates:
        return {
            "card_id": None, "confidence": 0.0, "method": "multisignal",
            "explanation": "No candidates with valid DB metadata",
            "raw_response": {"signal_errors": signal_errors},
        }

    # Sort by final score descending
    scored_candidates.sort(key=lambda c: c["final_score"], reverse=True)

    best = scored_candidates[0]
    runner_up = scored_candidates[1] if len(scored_candidates) > 1 else None

    # Build detailed explanation
    signals_used = []
    if dino_results:
        signals_used.append(f"DINOv2({best['dino_score']:.3f})")
    if clip_results:
        signals_used.append(f"CLIP({best['clip_score']:.3f})")
    if ocr_name:
        signals_used.append(f"OCR({ocr_name!r})")
    if hp_value is not None:
        signals_used.append(f"HP({hp_value})")
    if dp_level_value is not None:
        signals_used.append(f"Level(LV.{dp_level_value},{dp_level_name!r})")
    if detected_type:
        signals_used.append(f"Type({detected_type}:{detected_type_conf:.0%})")
    if border_info:
        signals_used.append(f"Era({border_info.get('era')})")
    if page_context and page_sets:
        signals_used.append(f"PageCtx({list(page_sets)[:2]})")

    explanation_parts = [
        f"Multi-signal: {best['name']} ({best['card_id']}) score={best['final_score']:.3f}",
        f"Visual={best['base_visual']:.3f} + adjustments={best['adjustment']:+.3f}",
        f"Signals: {', '.join(signals_used)}",
    ]
    if best["bonuses"]:
        explanation_parts.append(f"Bonuses: {', '.join(best['bonuses'])}")
    if best["penalties"]:
        explanation_parts.append(f"Penalties: {', '.join(best['penalties'])}")
    if runner_up:
        explanation_parts.append(
            f"Runner-up: {runner_up['name']} ({runner_up['card_id']}) "
            f"score={runner_up['final_score']:.3f}"
        )

    result = {
        "card_id": best["card_id"],
        "confidence": min(best["final_score"], 1.0),  # cap for display, raw score may exceed 1.0
        "method": "multisignal",
        "explanation": ". ".join(explanation_parts),
        "raw_response": {
            "dino_top10": dino_results,
            "clip_top10": clip_results,
            "ocr_name": ocr_name,
            "ocr_raw": ocr_raw,
            "ocr_confidence": ocr_confidence,
            "hp_detected": hp_value,
            "dp_level_name": dp_level_name,
            "dp_level_value": dp_level_value,
            "dp_level_candidates": [(c[0], c[1], c[3]) for c in dp_level_candidates[:5]] if dp_level_candidates else [],
            "type_detected": type_predictions[:3] if type_predictions else [],
            "border_info": border_info,
            "page_context": page_context,
            "scored_candidates": scored_candidates[:10],
            "signal_errors": signal_errors,
        },
    }

    logger.info(
        "Multisignal: BEST %s (%s) score=%.4f (visual=%.3f adj=%+.3f) "
        "bonuses=%s penalties=%s",
        best["card_id"], best["name"], best["final_score"],
        best["base_visual"], best["adjustment"],
        best["bonuses"], best["penalties"],
    )

    return result


def identify_card_hybrid(image_path, session=None, page_context=None):
    """Best-of-both identification: multi-signal when strong, ensemble fallback.

    Multi-signal excels when OCR/HP/level signals are available (e.g. DP-era
    cards with readable names). Ensemble is more robust when those signals are
    absent or noisy (e.g. holo glare, poor scan quality).

    Strategy:
        1. Run multi-signal (all 6 extractors).
        2. Check if the winner has meaningful non-visual bonuses
           (name match, HP match, level match — NOT just era_match or visual_overlap).
        3. If yes → use multi-signal result.
        4. If no → fall back to ensemble (confidence-gated).
    """
    ms_result = identify_card_multisignal(image_path, session=session, page_context=page_context)

    # Check if multisignal has strong non-generic bonuses
    scored = ms_result.get("raw_response", {}).get("scored_candidates", [])
    has_strong_signal = False
    if scored:
        top = scored[0]
        bonuses = top.get("bonuses", [])
        # "Strong" means at least one bonus that isn't just era_match, visual_overlap,
        # or a discounted/skipped HP bonus (which signals low-confidence HP match).
        strong_bonuses = [
            b for b in bonuses
            if not b.startswith("era_match") and not b.startswith("visual_overlap")
            and not b.startswith("page_")
            and "discounted" not in b and "skipped" not in b
        ]
        has_strong_signal = len(strong_bonuses) > 0

    if has_strong_signal:
        logger.info("Hybrid: using multi-signal (strong bonuses: %s)",
                     [b for b in scored[0].get("bonuses", []) if not b.startswith("era_match")])
        return ms_result

    # Fall back to ensemble
    logger.info("Hybrid: multi-signal has no strong bonuses, falling back to ensemble")
    ens_result = identify_card_ensemble(image_path, session=session, page_context=page_context)
    return ens_result


def _single_method_result(results: list[tuple[str, float]], method_label: str,
                          *, ocr_name: str | None = None,
                          image_path: str | None = None,
                          page_context: dict | None = None) -> dict:
    """Build a result dict from a single method's top-10 list.

    When *ocr_name* is provided (or can be extracted from *image_path*),
    verifies the top-1 card name against OCR.  If the top-1 name doesn't
    match but a lower-ranked candidate does, that candidate is promoted.

    When *page_context* is provided and contains ``likely_sets``, results
    are reranked before selection.
    """
    if not results:
        return {
            "card_id": None, "confidence": 0.0, "method": method_label,
            "explanation": "No results from single method fallback",
            "raw_response": {},
        }

    # --- Page context reranking (if not already applied upstream) ----------
    if page_context and page_context.get("likely_sets"):
        try:
            from cardprice.ml.page_context import rerank_with_context
            results = rerank_with_context(results, page_context)
            logger.info("_single_method_result: applied page context reranking")
        except Exception as e:
            logger.debug("_single_method_result: page context reranking failed: %s", e)

    # --- OCR name verification --------------------------------------------
    ocr_cleaned = ocr_name
    ocr_override_idx = None  # index into results if OCR promotes a candidate

    try:
        # If no precomputed name, try to extract one from the image
        if not ocr_cleaned and image_path:
            from cardprice.ml.ocr_matcher import extract_card_name, _clean_ocr_text
            ocr_text, _ocr_conf = extract_card_name(image_path)
            ocr_cleaned = _clean_ocr_text(ocr_text)

        if ocr_cleaned and len(ocr_cleaned) >= 3:
            from cardprice.ml.ocr_matcher import _load_card_names
            from rapidfuzz import fuzz

            card_names = _load_card_names()
            card_name_lookup = {cid: name for cid, name, _sid in card_names}

            OCR_MATCH_THRESH = 75

            # Score each candidate against the OCR name
            fuzzy_scores = []
            for i, (cid, _score) in enumerate(results[:10]):
                cname = card_name_lookup.get(cid, "")
                fs = fuzz.token_set_ratio(ocr_cleaned.lower(), cname.lower()) if cname else 0
                fuzzy_scores.append((i, cid, cname, fs))

            top1_fuzzy = fuzzy_scores[0][3] if fuzzy_scores else 0
            logger.info("_single_method_result OCR verify: ocr=%r, top1=%s (%r, fuzzy=%d)",
                        ocr_cleaned, results[0][0],
                        fuzzy_scores[0][2] if fuzzy_scores else "?", top1_fuzzy)

            if top1_fuzzy < OCR_MATCH_THRESH:
                # Top-1 doesn't match OCR — scan remaining candidates
                for i, cid, cname, fs in fuzzy_scores[1:]:
                    if fs >= OCR_MATCH_THRESH:
                        ocr_override_idx = i
                        logger.info("_single_method_result OCR override: promoting #%d %s "
                                    "(%r, fuzzy=%d) over top-1 %s (fuzzy=%d)",
                                    i + 1, cid, cname, fs, results[0][0], top1_fuzzy)
                        break
    except Exception as e:
        logger.warning("_single_method_result OCR verification failed: %s", e)

    # --- Build result -----------------------------------------------------
    if ocr_override_idx is not None:
        card_id, score = results[ocr_override_idx]
        orig_top1_id, orig_top1_score = results[0]
        alt_list = [(cid, s) for cid, s in results[:4] if cid != card_id][:3]
        alt_str = ", ".join(f"{a[0]} ({a[1]:.0%})" for a in alt_list)
        explanation = (f"OCR-verified fallback: promoted #{ocr_override_idx + 1} "
                       f"({score:.0%}) over top-1 {orig_top1_id} ({orig_top1_score:.0%}). "
                       f"OCR name={ocr_cleaned!r}")
        if alt_str:
            explanation += f". Alternatives: {alt_str}"
        return {
            "card_id": card_id,
            "confidence": float(score),
            "method": method_label,
            "explanation": explanation,
            "raw_response": {
                "top_matches": results[:5],
                "ocr_name": ocr_cleaned,
                "ocr_override_from": orig_top1_id,
                "ocr_override_to": card_id,
            },
        }

    card_id, score = results[0]
    alt_list = [(cid, s) for cid, s in results[1:4]]
    alt_str = ", ".join(f"{a[0]} ({a[1]:.0%})" for a in alt_list)
    explanation = f"Single method fallback ({score:.0%})"
    if ocr_cleaned:
        explanation += f", OCR confirmed ({ocr_cleaned!r})"
    if alt_str:
        explanation += f". Alternatives: {alt_str}"
    return {
        "card_id": card_id,
        "confidence": float(score),
        "method": method_label,
        "explanation": explanation,
        "raw_response": {"top_matches": results[:5], "top_alternatives": alt_list},
    }


# ---------------------------------------------------------------------------
# Reference-matching page identification pipeline
# ---------------------------------------------------------------------------

_REF_MATCH_CONFIDENCE_THRESHOLD = 0.45  # below this, fall back to cascade


def _extract_signals_for_ref(image_path: str) -> dict:
    """Run cheap classifiers in parallel for a single card image.

    Returns a dict with keys: ocr_name, ocr_confidence, hp, type_predictions,
    dino_top10, dino_name_vote.
    """
    ocr_name = None
    ocr_confidence = 0.0
    hp_value = None
    type_predictions = []
    dino_results = []

    def _do_ocr():
        from cardprice.ml.ocr_matcher import extract_card_name, _clean_ocr_text
        raw_text, conf = extract_card_name(image_path)
        cleaned = _clean_ocr_text(raw_text)
        return cleaned, conf

    def _do_hp():
        from cardprice.ml.hp_detector import detect_hp
        return detect_hp(image_path)

    def _do_type():
        from cardprice.ml.type_detector import detect_type
        return detect_type(image_path, top_n=3)

    def _do_dino():
        return _run_dino(image_path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_do_ocr): "ocr",
            pool.submit(_do_hp): "hp",
            pool.submit(_do_type): "type",
            pool.submit(_do_dino): "dino",
        }
        for future in as_completed(futures):
            signal = futures[future]
            try:
                res = future.result()
                if signal == "ocr":
                    ocr_name, ocr_confidence = res
                elif signal == "hp":
                    hp_value = res
                elif signal == "type":
                    type_predictions = res
                elif signal == "dino":
                    dino_results = res
            except Exception as e:
                logger.warning("Signal %s failed for %s: %s", signal, image_path, e)

    # DINOv2 name voting: extract Pokemon name from each top-5 result's card_id,
    # then take the plurality name.
    dino_name_vote = None
    if dino_results:
        from collections import Counter
        name_counts = Counter()
        for card_id, _score in dino_results[:5]:
            # card_id format: "set-num/variant" e.g. "base1-4/normal"
            # We need the Pokemon name from the DB.  As a fast heuristic,
            # extract the card portion and look up the name from dim_cards.
            # But DB lookups per card are slow.  Instead, group by the
            # card portion minus the variant (cards with same set-num are
            # the same Pokemon).
            card_portion = card_id.split("/")[0] if "/" in card_id else card_id
            name_counts[card_portion] += 1

        if name_counts:
            # The most common card portion in top-5 -> look up its name
            top_card_portion = name_counts.most_common(1)[0][0]
            # Query the DB for this card's name (fast single lookup)
            try:
                from sqlalchemy import text as sa_text
                from cardprice.db.session import SessionLocal
                sess = SessionLocal()
                try:
                    row = sess.execute(
                        sa_text("SELECT name FROM dim_cards WHERE card_id LIKE :pattern LIMIT 1"),
                        {"pattern": f"{top_card_portion}/%"},
                    ).fetchone()
                    if row:
                        dino_name_vote = row[0]
                finally:
                    sess.close()
            except Exception as e:
                logger.warning("DINOv2 name vote DB lookup failed: %s", e)

    return {
        "ocr_name": ocr_name,
        "ocr_confidence": ocr_confidence,
        "hp": hp_value,
        "type_predictions": type_predictions,
        "dino_top10": dino_results,
        "dino_name_vote": dino_name_vote,
    }


def _choose_best_name(signals: dict) -> tuple:
    """Pick the best Pokemon name from OCR and DINOv2 name voting.

    Returns (name, source) where source is "ocr" or "dino_vote" or None.
    Prefers OCR when confidence is high (>= 0.5 and name length >= 3).
    Falls back to DINOv2 plurality name vote.
    """
    ocr_name = signals.get("ocr_name")
    ocr_conf = signals.get("ocr_confidence", 0.0)
    dino_name = signals.get("dino_name_vote")

    # OCR is preferred when reasonably confident
    if ocr_name and len(ocr_name) >= 3 and ocr_conf >= 0.5:
        return ocr_name, "ocr"

    # Fall back to DINOv2 name voting
    if dino_name:
        return dino_name, "dino_vote"

    # Last resort: use OCR even if low confidence
    if ocr_name and len(ocr_name) >= 3:
        return ocr_name, "ocr_low"

    return None, None


def identify_card_ref_matching(image_path, session=None, page_context=None):
    """Identify a card using reference-image matching with attribute narrowing.

    Pipeline:
        1. Run cheap classifiers in parallel: OCR name, HP, type, DINOv2 top-10.
        2. Determine best name signal (OCR preferred, DINOv2 name vote fallback).
        3. Call ref_matcher.match_by_reference() with narrowed attributes.
        4. If ref_matcher confidence < 0.45, fall back to identify_card() cascade.

    Args:
        image_path: Path to the card image.
        session: Optional DB session.
        page_context: Optional page context dict (unused here, kept for API compat).

    Returns:
        Dict with keys: card_id, confidence, method, explanation, raw_response.
    """
    image_path = str(image_path)

    # Convert HEIC/HEIF if needed
    try:
        from cardprice.utils.image_convert import ensure_compatible
        image_path = ensure_compatible(image_path)
    except Exception as e:
        logger.warning("Image conversion failed, using original path: %s", e)

    # Step 1: Extract all signals in parallel
    signals = _extract_signals_for_ref(image_path)

    logger.info(
        "Ref-match signals: ocr=%r (%.2f), hp=%s, type=%s, dino_vote=%r, dino_top10=%d",
        signals["ocr_name"], signals["ocr_confidence"],
        signals["hp"],
        signals["type_predictions"][0][0] if signals["type_predictions"] else None,
        signals["dino_name_vote"],
        len(signals["dino_top10"]),
    )

    # Step 2: Choose best name
    best_name, name_source = _choose_best_name(signals)

    if best_name is None:
        logger.info("Ref-match: no name signal available, falling back to cascade")
        return identify_card(image_path, session=session, page_context=page_context)

    # Step 3: Extract HP and type for narrowing
    hp_value = signals["hp"]
    card_type = None
    if signals["type_predictions"]:
        top_type, top_type_conf = signals["type_predictions"][0]
        # Only use type if reasonably confident (>40% pixel vote share)
        if top_type_conf >= 0.40 and top_type != "Colorless":
            card_type = top_type

    logger.info(
        "Ref-match: querying candidates with name=%r (source=%s), hp=%s, type=%s",
        best_name, name_source, hp_value, card_type,
    )

    # Step 4: Run reference matching
    from cardprice.ml.ref_matcher import match_by_reference
    best_card_id, best_score = match_by_reference(
        query_image_path=image_path,
        pokemon_name=best_name,
        hp=hp_value,
        card_type=card_type,
        session=session,
    )

    # If no match found with HP+type narrowing, try relaxing constraints
    if best_card_id is None or best_score < _REF_MATCH_CONFIDENCE_THRESHOLD:
        # Try without type constraint
        if card_type is not None:
            logger.info("Ref-match: relaxing type constraint (was %s)", card_type)
            cid2, score2 = match_by_reference(
                query_image_path=image_path,
                pokemon_name=best_name,
                hp=hp_value,
                card_type=None,
                session=session,
            )
            if score2 > best_score:
                best_card_id, best_score = cid2, score2

    if best_card_id is None or best_score < _REF_MATCH_CONFIDENCE_THRESHOLD:
        # Try without HP constraint
        if hp_value is not None:
            logger.info("Ref-match: relaxing HP constraint (was %s)", hp_value)
            cid3, score3 = match_by_reference(
                query_image_path=image_path,
                pokemon_name=best_name,
                hp=None,
                card_type=None,
                session=session,
            )
            if score3 > best_score:
                best_card_id, best_score = cid3, score3

    # Step 5: Check confidence threshold
    if best_card_id is None or best_score < _REF_MATCH_CONFIDENCE_THRESHOLD:
        logger.info(
            "Ref-match: low confidence (%.3f < %.2f), falling back to cascade",
            best_score, _REF_MATCH_CONFIDENCE_THRESHOLD,
        )
        return identify_card(image_path, session=session, page_context=page_context)

    # Build result
    alt_info = ""
    if signals["dino_top10"]:
        alts = signals["dino_top10"][:3]
        alt_info = " | DINOv2 top-3: " + ", ".join(
            f"{cid} ({s:.0%})" for cid, s in alts
        )

    explanation = (
        f"Reference match via {name_source} name={best_name!r}, "
        f"hp={hp_value}, type={card_type}, "
        f"similarity={best_score:.3f}{alt_info}"
    )

    result = {
        "card_id": best_card_id,
        "confidence": float(best_score),
        "method": f"ref_match({name_source})",
        "explanation": explanation,
        "raw_response": {
            "signals": {
                "ocr_name": signals["ocr_name"],
                "ocr_confidence": signals["ocr_confidence"],
                "hp": signals["hp"],
                "type_top1": signals["type_predictions"][0] if signals["type_predictions"] else None,
                "dino_name_vote": signals["dino_name_vote"],
                "name_used": best_name,
                "name_source": name_source,
                "type_used": card_type,
            },
            "dino_top10": signals["dino_top10"],
            "ref_match_score": best_score,
            "ref_match_card_id": best_card_id,
        },
    }

    return result


def identify_page_ref_matching(card_image_paths, session=None):
    """Identify all cards on a binder page using reference-image matching.

    This pipeline runs cheap classifiers (OCR, HP, type, DINOv2) in parallel
    for each card, determines the best name signal, then does targeted
    reference-image comparison against narrowed DB candidates.

    Pipeline per card:
        1. Parallel signal extraction: OCR name, HP, type, DINOv2 FAISS top-10.
        2. Name determination: prefer OCR if confident, else DINOv2 name voting
           (plurality name from top-5 FAISS hits).
        3. ref_matcher.match_by_reference(image, name, hp, type) narrows to
           2-20 DB candidates and does DINOv2 embedding comparison vs reference
           images.
        4. If ref_matcher confidence < 0.45, fall back to identify_card() cascade.

    All cards are processed in parallel (one thread per card).

    Args:
        card_image_paths: List of paths to individual card segment images.
        session: Optional DB session.

    Returns:
        List of result dicts (same format as identify_card), one per card.
    """
    if not card_image_paths:
        return []

    n_cards = len(card_image_paths)
    logger.info("identify_page_ref_matching: processing %d cards", n_cards)

    # Process all cards in parallel
    results = [None] * n_cards

    def _process_card(idx, path):
        """Process a single card through the ref-matching pipeline."""
        try:
            return idx, identify_card_ref_matching(str(path), session=session)
        except Exception as e:
            logger.warning("Card %d ref-match failed: %s", idx, e, exc_info=True)
            # Fall back to cascade on any error
            try:
                return idx, identify_card(str(path), session=session)
            except Exception as e2:
                logger.error("Card %d cascade fallback also failed: %s", idx, e2)
                return idx, {
                    "card_id": None,
                    "confidence": 0.0,
                    "method": "ref_match_error",
                    "explanation": f"Both ref-match and cascade failed: {e}",
                    "raw_response": {},
                }

    with ThreadPoolExecutor(max_workers=min(n_cards, 6)) as pool:
        futures = [
            pool.submit(_process_card, i, path)
            for i, path in enumerate(card_image_paths)
        ]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    # Summary logging
    methods = [r.get("method", "?") for r in results if r]
    confidences = [r.get("confidence", 0) for r in results if r]
    ref_count = sum(1 for m in methods if m and m.startswith("ref_match"))
    fallback_count = len(methods) - ref_count
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    logger.info(
        "identify_page_ref_matching: %d/%d ref-matched, %d fallback, avg confidence=%.3f",
        ref_count, n_cards, fallback_count, avg_conf,
    )

    return results


# ---------------------------------------------------------------------------
# V2 Pipeline: color + name OCR + HP -> DB filter -> DINOv2 dot product
# ---------------------------------------------------------------------------

# Confidence thresholds for the v2 pipeline
_V2_DINO_ACCEPT_THRESHOLD = 0.40      # DINOv2 dot product vs filtered refs
_V2_CANDIDATE_DISAMBIGUATION_LIMIT = 3  # above this, also run attack OCR
_V2_FALLBACK_CONFIDENCE = 0.40         # below this, fall back to ensemble
_V2_FALLBACK_MIN_ACCEPT = 0.70         # reject v2_fallback results below this


def _run_color_detect(image_path: str) -> tuple:
    """Run color/type detection on a card image.

    Returns (type_name, confidence) or (None, 0.0) on failure.
    """
    try:
        from cardprice.ml.color_detector import detect_color_type
        predictions = detect_color_type(image_path, top_n=3)
        if predictions:
            return predictions[0][0], predictions[0][1]
    except Exception as e:
        logger.warning("v2 color_detect failed: %s", e)
    return None, 0.0


def _run_name_and_hp(image_path: str, _hold_lock: bool = True) -> tuple:
    """Run a SINGLE RapidOCR pass to extract both name and HP from a card.

    Crops the top 25% of the card, upscales 3x with unsharp mask, and runs
    RapidOCR (ONNX Runtime) detection + recognition once.  Then splits the
    detected text regions into name candidates (left/large) and HP candidates
    (right side, numeric pattern).

    Returns (cleaned_name, name_conf, raw_text, hp_value) or
            (None, 0.0, None, None).

    Args:
        _hold_lock: If True (default), acquire _ocr_lock for thread safety.
            Set to False when the caller has already ensured singletons are
            initialized and ONNX Runtime is handling concurrency (e.g. the
            batched OCR path in identify_page_v2).
    """
    def _inner():
        try:
            name, conf, raw, hp = _paddle_ocr_name_and_hp(image_path)
            if name and len(name) >= 2:
                return name, conf, raw, hp
            # PaddleOCR found HP but no name -- still return HP
            if hp is not None:
                # Try Japanese OCR fallback for the name
                try:
                    jp_name = _try_japanese_ocr(image_path)
                    if jp_name:
                        return jp_name, 0.70, f"[JP]{jp_name}", hp
                except Exception as e:
                    logger.debug("v2 japanese_ocr failed: %s", e)
                return None, 0.0, None, hp
        except Exception as e:
            logger.warning("v2 name_and_hp failed: %s", e)

        # Japanese OCR fallback: if English OCR found nothing,
        # try reading Japanese text and mapping to English name.
        try:
            jp_name = _try_japanese_ocr(image_path)
            if jp_name:
                return jp_name, 0.70, f"[JP]{jp_name}", None
        except Exception as e:
            logger.debug("v2 japanese_ocr failed: %s", e)

        return None, 0.0, None, None

    if _hold_lock:
        with _ocr_lock:
            return _inner()
    else:
        return _inner()


def _ocr_card_number(image_path: str) -> tuple:
    """Read the card number from the bottom of a card image.

    Pokemon cards print their collector number (e.g. "205/165") in small
    text at the bottom-left.  At binder segment resolution (~1008x1530),
    this text is only ~8-12px tall — very challenging for OCR.

    Strategy: try multiple upscale levels (4x, 6x) and preprocessing
    variants (plain, CLAHE, sharpen) to maximize detection probability.
    Also applies OCR character substitutions (O->0, l->1, etc.) as a
    fallback pass.

    Returns:
        (card_number, set_total) — e.g. ("205", "165"), or (None, None).
    """
    import cv2
    import re
    import numpy as np

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None, None

        h, w = img.shape[:2]
        # Crop bottom 13% (87-100%) — card number is at very bottom
        bottom = img[int(h * 0.87):int(h * 0.97), int(w * 0.03):int(w * 0.97)]

        from cardprice.ml.ocr_matcher import get_rapid_engine
        rapid_engine = get_rapid_engine()

        # Try multiple preprocessing variants for robustness
        bh, bw = bottom.shape[:2]
        variants = []
        for scale in (4, 6):
            big = cv2.resize(bottom, (bw * scale, bh * scale),
                             interpolation=cv2.INTER_CUBIC)
            # Plain upscale
            variants.append(big)
            # CLAHE enhanced
            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enh = clahe.apply(gray)
            variants.append(cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR))
            # Sharpen
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharp = cv2.filter2D(big, -1, kernel)
            variants.append(sharp)

        # Pass 1: look for N/M pattern in raw OCR text
        for variant in variants:
            result, _ = rapid_engine(variant)
            if not result:
                continue
            for box, text, conf in result:
                m = re.search(r'(\d{1,4})\s*/\s*(\d{1,4})', text)
                if m:
                    card_num = m.group(1).lstrip('0') or '0'
                    set_total = m.group(2).lstrip('0') or '0'
                    logger.info("card_number OCR: found %s/%s (raw=%r, conf=%.2f)",
                                card_num, set_total, text, conf)
                    return card_num, set_total

        # Pass 2: OCR character substitutions (O->0, l->1, etc.)
        for variant in variants:
            result, _ = rapid_engine(variant)
            if not result:
                continue
            for box, text, conf in result:
                fixed = text
                for old, new in [('O', '0'), ('o', '0'), ('l', '1'),
                                 ('I', '1'), ('S', '5'), ('B', '8')]:
                    fixed = fixed.replace(old, new)
                m = re.search(r'(\d{1,4})\s*[/|\\]\s*(\d{1,4})', fixed)
                if m:
                    card_num = m.group(1).lstrip('0') or '0'
                    set_total = m.group(2).lstrip('0') or '0'
                    logger.info("card_number OCR: found %s/%s via substitution (raw=%r, conf=%.2f)",
                                card_num, set_total, text, conf)
                    return card_num, set_total

    except Exception as e:
        logger.warning("card_number OCR failed: %s", e)

    return None, None


def _paddle_ocr_name_and_hp(image_path: str):
    """Single PaddleOCR pass on top 25% of card to extract name + HP.

    Reuses the same PaddleOCR detection/recognition singletons and
    preprocessing as _paddle_ocr_name(), but also extracts HP from the
    right-side text regions instead of requiring a separate EasyOCR call.

    Returns (matched_name, confidence, raw_ocr_text, hp_value).
    Any field may be None if not detected.
    """
    import cv2
    import re
    import numpy as np
    from pathlib import Path
    from rapidfuzz import fuzz, process
    from cardprice.ml.ocr_matcher import (
        get_rapid_engine, _paddle_ocr_name,
        _unsharp_mask_ocr, _clean_name_ocr,
        _load_unique_pokemon_names,
    )
    from cardprice.ml.hp_detector import _parse_hp_from_texts, _is_valid_hp
    from cardprice.ml.preprocess import upscale_for_ocr

    rapid_engine = get_rapid_engine()

    img = cv2.imread(str(image_path))
    if img is None:
        return None, 0.0, None, None

    h, w = img.shape[:2]

    # ------------------------------------------------------------------
    # Run PaddleOCR on multiple crop regions of top 25%
    # Collect ALL detected text regions with their bounding box positions
    # ------------------------------------------------------------------
    _NON_NAME_WORDS = {"stage", "basic", "hp", "stage i", "stage ii",
                       "stage 1", "stage 2", "stage1", "stage2",
                       "stagei", "stageii", "trainer", "supporter",
                       "pokemon", "item", "energy", "stage i pokemon",
                       "stage ii pokemon", "stadium", "tool",
                       "troingo", "trainor",
                       # OCR misspellings of TRAINER (garbled text that
                       # fuzzy-matches translations like German "Turner"→Clay)
                       "traner", "tralner", "traiher", "tralher",
                       "tpaner", "tpainer", "traher", "traner",
                       "pokémon", "pokemon tool"}

    # Each detected text: (text, confidence, x_center_frac, y_center_frac, width_frac, height_frac, method)
    all_detections = []

    # Crop specs ordered by likelihood of finding the name quickly.
    # top25 is the most complete region and works for ~90% of cards.
    # Additional crops are tried only if needed (no good name found).
    _OCR_UPSCALE = 2  # 2x is faster than 3x with equal or better accuracy
    crop_specs = [
        (0.00, 0.25, 0.03, 0.97, "top25"),
        (0.00, 0.15, 0.05, 0.95, "top15"),
        (0.03, 0.18, 0.05, 0.95, "skip3"),
        # Deeper crop: catches card names when segment bleeds from card above
        # (e.g., actual name at 20-30% because top 15% is adjacent card text)
        (0.10, 0.35, 0.03, 0.97, "deep10_35"),
    ]

    found_good_name = False

    for top_frac, bot_frac, left_frac, right_frac, label in crop_specs:
        y1 = int(h * top_frac)
        y2 = int(h * bot_frac)
        x1 = int(w * left_frac)
        x2 = int(w * right_frac)
        crop = img[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]

        # Pad so text isn't at the very edge
        pad = 30
        crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        crop = _unsharp_mask_ocr(crop)
        crop_up = upscale_for_ocr(crop, scale=_OCR_UPSCALE)

        result, _ = rapid_engine(crop_up)
        if not result:
            continue

        up_h, up_w = crop_up.shape[:2]
        pad_scaled = pad * _OCR_UPSCALE

        for box, text, conf in result:
            conf = float(conf)
            if not text or len(text.strip()) < 2 or conf < 0.3:
                continue
            text = text.strip()

            # Compute bounding rect from polygon box
            pts = np.array(box, dtype=np.float32)
            x, y, bw, bh = cv2.boundingRect(pts)

            # Compute position as fraction of the crop region
            # (account for the padding we added)
            cx = (x + bw / 2 - pad_scaled) / (crop_w * _OCR_UPSCALE)
            cy = (y + bh / 2 - pad_scaled) / (crop_h * _OCR_UPSCALE)
            text_w_frac = bw / (crop_w * _OCR_UPSCALE)
            text_h_frac = bh / (crop_h * _OCR_UPSCALE)

            # Map x position back to full card width fraction
            card_x_frac = left_frac + cx * (right_frac - left_frac)

            all_detections.append((text, conf, card_x_frac, cy, text_w_frac, text_h_frac, label))

        # Early exit if we found a high-confidence, SHORT name (not a long
        # sentence like "Evolves from Tentacool Put Tentacruel on the Basic Pokemon").
        # Long fragments may contain the name but we need a tighter crop to isolate it.
        if any(c > 0.8 and 3 <= sum(1 for ch in t if ch.isalpha()) and len(t.strip()) <= 25
               and t.strip().lower() not in _NON_NAME_WORDS
               and not re.match(r"(?i)evolves?\s+from\s+", t.strip())
               for t, c, _, _, _, _, _ in all_detections):
            found_good_name = True
            break

    # ------------------------------------------------------------------
    # Raw OCR fallback (no unsharp mask): the preprocessing pipeline
    # can destroy low-contrast text on some cards. If we found no
    # name-like text (only HP digits), retry top25 without unsharp mask.
    # ------------------------------------------------------------------
    has_name_text = any(
        sum(1 for ch in t if ch.isalpha()) >= 3 and c > 0.5
        for t, c, _, _, _, _, _ in all_detections
    )
    if not has_name_text:
        y2_raw = int(h * 0.25)
        x1_raw = int(w * 0.03)
        x2_raw = int(w * 0.97)
        crop_raw = img[0:y2_raw, x1_raw:x2_raw]
        crop_h_raw, crop_w_raw = crop_raw.shape[:2]
        # Run raw without unsharp mask or upscale
        result_raw, _ = rapid_engine(crop_raw)
        if result_raw:
            for box, text, conf in result_raw:
                conf = float(conf)
                if not text or len(text.strip()) < 2 or conf < 0.3:
                    continue
                text = text.strip()
                pts = np.array(box, dtype=np.float32)
                x, y, bw, bh = cv2.boundingRect(pts)
                cx = (x + bw / 2) / crop_w_raw
                card_x_frac = 0.03 + cx * 0.94
                text_w_frac = bw / crop_w_raw
                text_h_frac = bh / crop_h_raw
                all_detections.append((text, conf, card_x_frac, 0.5, text_w_frac, text_h_frac, "raw25"))

    # ------------------------------------------------------------------
    # CLAHE grayscale fallback on top 25% if no good detections
    # ------------------------------------------------------------------
    if not any(c > 0.5 for _, c, _, _, _, _, _ in all_detections):
        y2 = int(h * 0.25)
        x1_clahe = int(w * 0.03)
        x2_clahe = int(w * 0.97)
        crop = img[0:y2, x1_clahe:x2_clahe]
        crop_h, crop_w = crop.shape[:2]
        pad = 30
        crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        crop_up = upscale_for_ocr(crop, scale=_OCR_UPSCALE)
        gray = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        result, _ = rapid_engine(enhanced_bgr)
        if result:
            pad_scaled = pad * _OCR_UPSCALE
            for box, text, conf in result:
                conf = float(conf)
                if not text or len(text.strip()) < 2 or conf < 0.3:
                    continue
                text = text.strip()
                pts = np.array(box, dtype=np.float32)
                x, y, bw, bh = cv2.boundingRect(pts)
                cx = (x + bw / 2 - pad_scaled) / (crop_w * _OCR_UPSCALE)
                card_x_frac = 0.03 + cx * 0.94
                text_w_frac = bw / (crop_w * _OCR_UPSCALE)
                text_h_frac = bh / (crop_h * _OCR_UPSCALE)
                all_detections.append((text, conf, card_x_frac, 0.5, text_w_frac, text_h_frac, "clahe25"))

    if not all_detections:
        return None, 0.0, None, None

    # ------------------------------------------------------------------
    # Split detections into NAME candidates and HP candidates
    # ------------------------------------------------------------------
    # HP candidates: text on the right side (card_x > 0.50) matching
    #   digits + "HP" or just digits in HP range
    hp_value = None
    hp_texts = []
    name_raw_candidates = []

    # Find the tallest text fragment — this is the card name font size
    max_text_h = max((th for _, _, _, _, _, th, _ in all_detections), default=0.1)

    for text, conf, card_x, cy, text_w, text_h, label in all_detections:
        text_upper = text.upper().strip()

        # Check if this looks like an HP value
        is_hp_candidate = False

        # Explicit HP pattern anywhere in text
        hp_match = re.search(r'HP\s*(\d{2,3})', text_upper)
        if not hp_match:
            hp_match = re.search(r'(\d{2,3})\s*HP', text_upper)
        if hp_match:
            is_hp_candidate = True
            hp_texts.append((text, conf))

        # Pure numeric on right side of card (x > 0.50)
        # Also handle trailing non-digit garbage like "130@" (energy symbol)
        digits_only = re.fullmatch(r'\d{2,3}', text.strip())
        if not digits_only:
            digits_only = re.match(r'^(\d{2,3})[^0-9]', text.strip())
        if digits_only and card_x > 0.50:
            # Extract just the digits (handles "130@" etc.)
            digit_str = re.match(r'(\d+)', text.strip()).group(1)
            val = int(digit_str)
            if _is_valid_hp(val):
                is_hp_candidate = True
                hp_texts.append((digit_str, conf))

        # Also check for "HP" near a number with OCR noise
        from cardprice.ml.hp_detector import _normalize_ocr_digits
        norm = _normalize_ocr_digits(text_upper)
        hp_match_norm = re.search(r'HP\s*(\d{2,3})', norm)
        if not hp_match_norm:
            hp_match_norm = re.search(r'(\d{2,3})\s*HP', norm)
        if hp_match_norm and not is_hp_candidate:
            is_hp_candidate = True
            hp_texts.append((text, conf))

        # Name candidates: text that is NOT purely HP, on left/center,
        # or wider text regions (the name is the largest text)
        # Include even HP-region text as name candidate if it has alpha chars
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count >= 2:
            # text_size_ratio: 1.0 = tallest text on card, 0.5 = half as tall
            text_size_ratio = text_h / max_text_h if max_text_h > 0 else 1.0
            name_raw_candidates.append((text, conf, label, text_size_ratio))

    # Parse HP from collected HP-region texts
    if hp_texts:
        hp_value = _parse_hp_from_texts(hp_texts)

    # ------------------------------------------------------------------
    # Raw OCR HP supplement: if preprocessing garbled the HP text (e.g.
    # "70HP" → "TOHPO"), run raw (unprocessed) OCR on the full top 25%
    # to recover the HP value.  Only runs when HP was NOT found from
    # preprocessed passes.  This is cheap (~30ms) and fixes misidentification
    # when all Ariados/Golem/etc variants compete without an HP filter.
    # ------------------------------------------------------------------
    if hp_value is None:
        from cardprice.ml.hp_detector import _normalize_ocr_digits as _norm_digits
        y2_hp = int(h * 0.25)
        x1_hp = int(w * 0.03)
        x2_hp = int(w * 0.97)
        crop_hp = img[0:y2_hp, x1_hp:x2_hp]
        crop_hp_h, crop_hp_w = crop_hp.shape[:2]
        result_hp_raw, _ = rapid_engine(crop_hp)
        if result_hp_raw:
            hp_raw_texts = []
            for box, text, conf in result_hp_raw:
                conf = float(conf)
                if not text or conf < 0.3:
                    continue
                text = text.strip()
                text_upper = text.upper()
                hp_match = re.search(r'HP\s*(\d{2,3})', text_upper)
                if not hp_match:
                    hp_match = re.search(r'(\d{2,3})\s*HP', text_upper)
                if hp_match:
                    hp_raw_texts.append((text, conf))
                else:
                    # Check with OCR digit normalization
                    norm = _norm_digits(text_upper)
                    hp_match_n = re.search(r'HP\s*(\d{2,3})', norm)
                    if not hp_match_n:
                        hp_match_n = re.search(r'(\d{2,3})\s*HP', norm)
                    if hp_match_n:
                        hp_raw_texts.append((text, conf))
                    else:
                        # Pure digits on right side in HP range
                        digits_only = re.fullmatch(r'\d{2,3}', text.strip())
                        if digits_only:
                            # Compute position to check if right-side
                            pts_hp = np.array(box, dtype=np.float32)
                            x_hp, _, bw_hp, _ = cv2.boundingRect(pts_hp)
                            cx_hp = (x_hp + bw_hp / 2) / crop_hp_w
                            card_x_hp = 0.03 + cx_hp * 0.94
                            if card_x_hp > 0.50:
                                val = int(text.strip())
                                if _is_valid_hp(val):
                                    hp_raw_texts.append((text.strip(), conf))
            if hp_raw_texts:
                hp_value = _parse_hp_from_texts(hp_raw_texts)
                if hp_value is not None:
                    logger.info("Raw OCR HP supplement found HP=%d", hp_value)

    # ------------------------------------------------------------------
    # Fuzzy-match name candidates (same logic as detect_pokemon_name)
    # ------------------------------------------------------------------
    # Clean and filter
    name_candidates = []
    for text, conf, method, size_ratio in name_raw_candidates:
        cleaned = _clean_name_ocr(text)
        if not cleaned or len(cleaned) < 3:
            continue
        alpha_count = sum(1 for c in cleaned if c.isalpha())
        # Allow possessive fragments like "N's" through even if alpha ratio
        # is low — the apostrophe is a legitimate part of the owner name.
        is_possessive_frag = bool(re.search(r"[''\u2019][sS]$", cleaned))
        if alpha_count / len(cleaned) < 0.7 and not is_possessive_frag:
            continue
        # Skip non-name words (exact match + fuzzy for OCR garbling)
        cl = cleaned.lower()
        if cl in _NON_NAME_WORDS:
            continue
        # Fuzzy check: "TRANER" → "trainer" at 83%, catch all OCR garbling
        if any(fuzz.ratio(cl, nw) >= 75 for nw in ("trainer", "supporter", "stadium", "pokemon", "energy")):
            continue
        # Skip "Evolves from X" text — always names the pre-evolution, not the card
        if re.match(r"(?i)evolves?\s+from\s+", cleaned):
            continue
        name_candidates.append((cleaned, conf, method, size_ratio))

    if not name_candidates:
        # Relaxed filter
        for text, conf, method, size_ratio in name_raw_candidates:
            cleaned = _clean_name_ocr(text)
            if cleaned and len(cleaned) >= 2:
                if cleaned.lower() not in _NON_NAME_WORDS:
                    name_candidates.append((cleaned, conf, method, size_ratio))

    if not name_candidates:
        return None, 0.0, None, hp_value

    # For long OCR fragments (>20 chars), extract individual words that
    # exactly match known Pokemon names. This handles instruction text like
    # "Put Seadra on the Basic Pokemon" → extracts "Seadra".
    _known_names = _load_unique_pokemon_names()
    _known_lower = {n.lower() for n in _known_names}
    word_candidates = []
    for cleaned, conf, method, size_ratio in name_candidates:
        if len(cleaned) > 20:
            for word in cleaned.split():
                word = word.strip(".,;:!?()[]")
                if len(word) >= 4 and word.lower() in _known_lower:
                    word_candidates.append((word, conf * 0.9, method + "_word", size_ratio))
    name_candidates.extend(word_candidates)

    # ------------------------------------------------------------------
    # Possessive name concatenation: cards like "Lillie's Clefairy ex"
    # often have the owner ("Lillie's") on a separate line from the
    # Pokemon name ("Clefairy"). Combine possessive fragments with
    # name-like fragments to form the full card name.
    #
    # OCR may read the possessive prefix in several garbled forms:
    #   "TEAMAQUA'S" — correct but uppercase/no-space
    #   "TEHAQUA'S"  — missing letters
    #   "TEAMAQUA"   — apostrophe-s dropped entirely
    #   "Misty"      — owner name without 's
    # ------------------------------------------------------------------
    # Build set of known owner prefixes (without "'s") for fallback matching
    _owner_prefixes_lower = set()
    for _kn in _known_names:
        _poss_m = re.match(r"^(.+?)[''\u2019]s\s", _kn)
        if _poss_m:
            _owner_prefixes_lower.add(_poss_m.group(1).lower())
            # Also add no-space version: "team aqua" -> "teamaqua"
            _owner_prefixes_lower.add(_poss_m.group(1).lower().replace(" ", ""))

    possessive_frags = []
    non_possessive = []
    for cleaned, conf, method, size_ratio in name_candidates:
        txt = cleaned.strip()
        is_possessive = False
        # Match "X's" or "X'S" pattern (case-insensitive), including
        # curly apostrophes and OCR-swapped s' ordering
        if re.search(r"[''\u2019][sS]$", txt):
            is_possessive = True
        else:
            # Fallback: check if the fragment matches a known owner prefix
            # without the apostrophe-s (OCR dropped it entirely).
            # e.g., "TEAMAQUA" or "Misty" -> known owner prefix
            txt_lower = txt.lower().replace("'", "").replace("\u2019", "")
            txt_nospace = txt_lower.replace(" ", "")
            if txt_lower in _owner_prefixes_lower or txt_nospace in _owner_prefixes_lower:
                is_possessive = True
                # Restore the possessive suffix for combining
                txt = txt + "'s"
            elif len(txt) >= 5:
                # Fuzzy match against known owners for heavily garbled OCR
                # e.g., "TEAAQUA" (missing M) -> "teamaqua"
                for _ow in _owner_prefixes_lower:
                    if len(_ow) >= 5 and fuzz.ratio(txt_nospace, _ow) >= 78.0:
                        is_possessive = True
                        txt = txt + "'s"
                        break

        if is_possessive:
            possessive_frags.append((txt, conf, method, size_ratio))
        else:
            non_possessive.append((cleaned, conf, method, size_ratio))
    if possessive_frags and non_possessive:
        for poss, pconf, pmethod, psize in possessive_frags:
            for nptext, npconf, npmethod, npsize in non_possessive:
                # Skip combining with non-name words
                if nptext.lower() in _NON_NAME_WORDS:
                    continue
                # Skip combining with text that already contains a possessive
                if re.search(r"[''\u2019][sS]\s", nptext):
                    continue
                combined_name = f"{poss} {nptext}"
                combined_conf = min(pconf, npconf)
                combined_size = max(psize, npsize)
                name_candidates.append((combined_name, combined_conf, pmethod + "+poss", combined_size))

    # Deduplicate
    seen = set()
    deduped = []
    for cleaned, conf, method, size_ratio in name_candidates:
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            deduped.append((cleaned, conf, method, size_ratio))

    # Fuzzy match against Pokemon names DB (English + translations)
    name_list_lower, lower_to_original = _get_name_lookup()

    _OCR_CONFUSIONS = {
        'y': 'x', 'x': 'y', 'l': 'i', 'i': 'l',
        'u': 'v', 'v': 'u', 'o': 'c', 'c': 'o',
        'n': 'h', 'h': 'n', 'rn': 'm', 'm': 'rn', 'd': 'cl',
    }

    best_match = None
    best_score = 0.0
    best_ocr_text = ""
    # Track possessive-combined matches separately so we can prefer them
    # when they are more specific than the base name match.
    best_poss_match = None
    best_poss_score = 0.0
    best_poss_ocr_text = ""
    best_poss_fuzzy_score = 0.0

    for cleaned, ocr_conf, method, size_ratio in deduped:
        query = cleaned.lower()
        if ocr_conf < 0.15 and len(cleaned) < 5:
            continue

        queries = [query]
        for old_char, new_char in _OCR_CONFUSIONS.items():
            if old_char in query:
                alt = query.replace(old_char, new_char, 1)
                if alt != query:
                    queries.append(alt)

        best_for_this = None
        for q in queries:
            if q in lower_to_original:
                match_name = lower_to_original[q]
                score = 100.0
                if best_for_this is None or score > best_for_this[1]:
                    best_for_this = (match_name, score)
                continue
            # Higher cutoff for confusion-substituted queries (speculative)
            cutoff = 85.0 if q != query else 60.0
            m = process.extractOne(q, name_list_lower, scorer=fuzz.ratio, score_cutoff=cutoff)
            if m is not None:
                matched_lower, score, _idx = m
                mn = lower_to_original[matched_lower]
                if len(q) == len(matched_lower):
                    score = min(100.0, score + 3.0)
                if best_for_this is None or score > best_for_this[1]:
                    best_for_this = (mn, score)

        if best_for_this is None and len(query) >= 6:
            matches = process.extractOne(query, name_list_lower, scorer=fuzz.partial_ratio, score_cutoff=85.0)
            if matches is not None:
                matched_lower, score, _idx = matches
                # Reject if matched name is much shorter than query (pathological partial match)
                if len(matched_lower) >= len(query) * 0.5:
                    score = score * 0.85
                    best_for_this = (lower_to_original[matched_lower], score)

        if best_for_this is None:
            continue

        match_name, score = best_for_this
        if len(cleaned) <= 3 and score < 90:
            continue
        if len(cleaned) <= 4 and score < 75 and ocr_conf < 0.4:
            continue
        # Reject weak fuzzy matches — these are likely OCR hallucinations.
        # A score < 75 means the raw text is far from any known name.
        if score < 75 and score != 100.0:
            continue

        # Boost text that is physically larger on the card — the card name
        # is always the biggest text in the name region. "Evolves from X"
        # and "Stage 1" are in smaller fonts.
        # size_ratio: 1.0 = tallest text, <1.0 = smaller text
        size_bonus = size_ratio * 15.0  # up to +15 points for largest text
        combined = score * 0.7 + min(ocr_conf, 1.0) * 100.0 * 0.15 + size_bonus

        # Track possessive-combined matches separately
        if "+poss" in method and score >= 85.0:
            if combined > best_poss_score:
                best_poss_score = combined
                best_poss_match = match_name
                best_poss_ocr_text = cleaned
                best_poss_fuzzy_score = score

        if combined > best_score:
            best_score = combined
            best_match = match_name
            best_ocr_text = cleaned

    # ------------------------------------------------------------------
    # Possessive specificity override: if a possessive-combined match
    # (e.g., "Team Aqua's Poochyena") matched a known name well, prefer
    # it over the base name alone (e.g., "Poochyena"). The combined name
    # is more specific and narrows to the correct card/set.
    # ------------------------------------------------------------------
    if (best_poss_match and best_match and best_poss_fuzzy_score >= 85.0
            and best_poss_match != best_match
            and best_match.lower() in best_poss_match.lower()):
        logger.debug("Possessive override: %r -> %r (poss_score=%.1f, base_score=%.1f)",
                     best_match, best_poss_match, best_poss_score, best_score)
        best_match = best_poss_match
        best_score = best_poss_score
        best_ocr_text = best_poss_ocr_text

    if best_match is None or best_score / 100.0 < 0.65:
        return None, 0.0, None, hp_value

    confidence = min(1.0, best_score / 100.0)
    return best_match, confidence, best_ocr_text, hp_value


def _run_name_ocr(image_path: str) -> tuple:
    """Run OCR to extract the Pokemon name from a card image.

    Returns (cleaned_name, confidence, raw_text) or (None, 0.0, None).
    Uses thread lock since PaddleOCR/EasyOCR models are not thread-safe.

    DEPRECATED: Use _run_name_and_hp() instead for combined name+HP in one pass.
    """
    with _ocr_lock:
        try:
            from cardprice.ml.ocr_matcher import detect_pokemon_name
            name, conf = detect_pokemon_name(image_path)
            if name and len(name) >= 2:
                return name, conf, name
        except Exception as e:
            logger.warning("v2 name_ocr failed: %s", e)

        # Japanese OCR fallback: if English OCR found nothing,
        # try reading Japanese text and mapping to English name.
        try:
            jp_name = _try_japanese_ocr(image_path)
            if jp_name:
                return jp_name, 0.70, f"[JP]{jp_name}"
        except Exception as e:
            logger.debug("v2 japanese_ocr failed: %s", e)

    return None, 0.0, None


# Cached reverse index: {japanese_lower: english_name} built from translation names
_ja_reverse_index: dict[str, str] | None = None


def _get_ja_reverse_index() -> dict[str, str]:
    """Build/return a reverse index mapping Japanese text -> English card name.

    Priority order (highest wins, overrides lower sources):
    1. jp_en_pokemon_names.json -- authoritative PokeAPI species names,
       LOADED FIRST to override any cross-language ID collisions from
       card_translations.json.  This fixes the bug where Japanese set IDs
       in card_translations.json (e.g. "neo2-1" = キャタピー) collided with
       different English cards (neo2-1 = Espeon), corrupting 22+ base
       species names (Pikachu -> Gloom, Lugia -> Shuckle, etc.).
    2. _load_translation_names() -- compound names (e.g. わるいマルマイン
       -> Dark Electrode) from the JP prefix mapping.  Filtered to entries
       containing Japanese characters.  Only adds entries NOT already
       present from Source 1.
    """
    global _ja_reverse_index
    if _ja_reverse_index is not None:
        return _ja_reverse_index

    import json
    import re
    JP_CHAR_RE = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')

    result: dict[str, str] = {}

    # Source 1 (authoritative species): jp_en_pokemon_names.json.
    # Loaded FIRST so Source 3 cannot overwrite correct Pokemon species
    # with cross-language ID collisions.
    jp_map_path = Path(__file__).resolve().parent.parent.parent / "data" / "jp_en_pokemon_names.json"
    if jp_map_path.exists():
        with open(jp_map_path) as f:
            for ja_name, en_name in json.load(f).items():
                result[ja_name.lower()] = en_name

    # Source 2 (authoritative trainers/energy): curated JP→EN bridge file.
    # Loaded BEFORE Source 3 so corrupted trainer/energy translations from
    # card_translations.json (caused by cross-language ID collisions) cannot
    # overwrite hand-verified mappings.  Format: {jp_name: english_name}.
    try:
        bridge_path = Path(__file__).resolve().parent.parent.parent / "data" / "jp_en_trainer_energy.json"
        if bridge_path.exists():
            with open(bridge_path) as f:
                bridge = json.load(f)
            added_bridge = 0
            for jp_name, eng_name in bridge.items():
                jp_lower = jp_name.lower().strip()
                if jp_lower:
                    # Force override — curated bridge is authoritative for
                    # trainer/energy names.  Pokemon species in Source 1 are
                    # still protected because bridge file only has T/E names.
                    result[jp_lower] = eng_name
                    added_bridge += 1
            logger.info("Source 2 (curated trainer/energy bridge): added %d entries", added_bridge)
    except Exception as e:
        logger.warning("Failed to load curated trainer/energy bridge: %s", e)

    # Source 3: translation names (compound species names like わるいマルマイン).
    # TCGdex JA uses different set IDs than our ptcgio DB, so ID-based joins
    # produce cross-language collisions.  We only add entries NOT already
    # covered by Sources 1-2 (authoritative) to avoid importing corrupted
    # mappings.  This means JP trainer/energy coverage is limited to the
    # curated bridge file, but Pokemon species are comprehensive.
    for foreign_lower, eng_name in _load_translation_names().items():
        if JP_CHAR_RE.search(foreign_lower) and foreign_lower not in result:
            result[foreign_lower] = eng_name

    logger.info("Japanese reverse index: %d entries", len(result))
    _ja_reverse_index = result
    return result


# ---------------------------------------------------------------------------
# Japanese OCR: in-process RapidOCR with Japanese ONNX model
# ---------------------------------------------------------------------------
# Replaces the previous PaddleOCR subprocess approach.  RapidOCR uses
# ONNX Runtime (already in dependencies for English OCR), runs in-process
# (no subprocess overhead), and the Japanese CRNN v2 model is only 3.4MB.
# Benchmarks: ~0.3s per card vs 18s for PaddleOCR subprocess.

_ja_ocr_engine = None
_ja_ocr_engine_lock = threading.Lock()


def _get_ja_ocr_engine():
    """Lazy singleton for the Japanese RapidOCR engine.

    Uses japan_rec_crnn_v2.onnx (3.4MB) from data/models/.  The CRNN v2
    architecture needs rec_img_shape=[3,32,320] (height=32, not default 48).
    """
    global _ja_ocr_engine
    if _ja_ocr_engine is not None:
        return _ja_ocr_engine

    with _ja_ocr_engine_lock:
        if _ja_ocr_engine is not None:
            return _ja_ocr_engine
        try:
            from rapidocr_onnxruntime import RapidOCR
            model_path = _PROJECT_ROOT / "data" / "models" / "japan_rec_crnn_v2.onnx"
            if not model_path.exists():
                logger.warning(
                    "Japanese OCR model not found at %s — JP OCR disabled. "
                    "Download from: https://huggingface.co/spaces/RapidAI/RapidOCR/"
                    "resolve/main/models/text_rec/japan_rec_crnn_v2.onnx",
                    model_path,
                )
                return None
            _ja_ocr_engine = RapidOCR(
                rec_model_path=str(model_path),
                rec_img_shape=[3, 32, 320],
            )
            logger.info("Japanese RapidOCR engine loaded from %s", model_path.name)
            return _ja_ocr_engine
        except Exception as e:
            logger.warning("Failed to load Japanese RapidOCR engine: %s", e)
            return None


def _rapid_ja_ocr(image_path: str) -> list[tuple[str, float]]:
    """Run RapidOCR Japanese on a card image.

    Tries multiple crops of the name region with early exit on good results.
    Runs in-process (no subprocess), ~0.2-0.3s per card after warmup.

    Returns list of (text, confidence) pairs.
    """
    import cv2
    engine = _get_ja_ocr_engine()
    if engine is None:
        return []

    img = cv2.imread(str(image_path))
    if img is None:
        return []

    h, w = img.shape[:2]
    all_texts: list[tuple[str, float]] = []

    # Crop specs for the name region (top portion of card)
    crop_specs = [
        (0.00, 0.25),  # top25 — broadest
        (0.00, 0.15),  # top15 — narrower
        (0.02, 0.20),  # skip2 — skip border
    ]

    # RapidOCR ONNX is not guaranteed thread-safe; serialize engine calls
    with _ja_ocr_engine_lock:
        for top_frac, bot_frac in crop_specs:
            crop = img[int(h * top_frac):int(h * bot_frac), :]
            # Pad + upscale for better OCR on small text
            pad = 20
            crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            crop = cv2.resize(crop, None, fx=3, fy=3)

            try:
                result, _ = engine(crop)
            except Exception as e:
                logger.debug("RapidOCR Japanese call failed: %s", e)
                continue

            if not result:
                continue

            for item in result:
                # RapidOCR returns (box, text, score) tuples
                if len(item) >= 3:
                    text = item[1]
                    try:
                        score = float(item[2])
                    except (TypeError, ValueError):
                        continue
                    if score > 0.5 and len(text.strip()) >= 2:
                        all_texts.append((text, score))

            # Early exit if we found good text on the broadest crop
            if all_texts:
                break

    return all_texts


# Backwards-compatible alias — old call sites use _paddle_ja_ocr_subprocess.
# The new implementation is RapidOCR-based but keeps the same signature.
_paddle_ja_ocr_subprocess = _rapid_ja_ocr


def _try_jp_dino_match(image_path: str, threshold: float = 0.78) -> str | None:
    """Global DINOv2 search against the JP embeddings.

    Returns a "jp_<tcg_product_id>" card_id if a confident match is found,
    otherwise None.  Used as the primary JP identification path: when a card
    looks Japanese, we directly visual-match it to the 3,273 JP card images.

    Threshold 0.78 is conservative — true matches usually score 0.85+, and
    we want to avoid false positives that route an English card to JP pricing.
    """
    try:
        from cardprice.ml.ref_matcher import search_jp_embeddings
        results = search_jp_embeddings(image_path, top_k=1)
        if not results:
            return None
        jp_id, score = results[0]
        if score >= threshold:
            logger.info("JP DINOv2 match: %s (sim=%.3f)", jp_id, score)
            return jp_id
        logger.debug("JP DINOv2 best match below threshold: %s (sim=%.3f)", jp_id, score)
        return None
    except Exception as e:
        logger.debug("JP DINOv2 search failed: %s", e)
        return None


def _try_japanese_ocr(image_path: str) -> str | None:
    """Try Japanese OCR on the name region, return English name if found.

    Returns either:
      - "jp_<tcg_product_id>" — if JP DINOv2 search found a confident match
        in the JP card embeddings.  Caller treats this as a direct card_id.
      - English Pokemon name (str) — fallback when DINOv2 isn't confident
        but RapidOCR Japanese + reverse index found a name.
      - None — no Japanese signal at all.

    Strategy:
    1. Run JP DINOv2 global search FIRST.  If confident, return jp_<id>.
    2. Otherwise, run RapidOCR Japanese on the name region for a heuristic
       English name (fuzzy-matched via the JA reverse index).
    """
    # --- Step 1: JP DINOv2 global search (preferred — gives specific card) ---
    jp_id = _try_jp_dino_match(image_path)
    if jp_id:
        return jp_id

    # --- Step 2: Fall back to OCR + reverse-index lookup ---
    import re
    from rapidfuzz import fuzz, process

    JP_CHAR_RE = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+')

    # Quick check: does the image contain CJK-looking text?
    # Skip the expensive PaddleOCR subprocess (~50s first time) if the
    # card image has no Japanese characters visible.  Check by running
    # RapidOCR on the name region — if it returns ANY text at all,
    # the card likely has Latin text (not Japanese), so skip.
    # Only proceed if RapidOCR returned literally nothing.
    import cv2 as _cv2
    _img_check = _cv2.imread(str(image_path))
    if _img_check is None:
        return None
    _h, _w = _img_check.shape[:2]
    # Check if the top portion has any high-saturation colorful content
    # (Japanese cards tend to have colorful artwork filling the name area)
    # This is a heuristic — not perfect but avoids 50s subprocess on blanks
    _top_crop = _img_check[:int(_h * 0.25), :]
    _hsv = _cv2.cvtColor(_top_crop, _cv2.COLOR_BGR2HSV)
    _mean_sat = float(_hsv[:, :, 1].mean())
    _mean_val = float(_hsv[:, :, 2].mean())
    # Very dark or very low saturation = likely card back or empty slot
    if _mean_val < 40 or (_mean_sat < 20 and _mean_val < 100):
        logger.debug("Japanese OCR skip: dark/low-sat top region (sat=%.0f val=%.0f)", _mean_sat, _mean_val)
        return None

    ja_index = _get_ja_reverse_index()
    if not ja_index:
        return None

    ja_names = list(ja_index.keys())

    # Run PaddleOCR in subprocess to avoid OOM when other models are loaded
    ocr_results = _paddle_ja_ocr_subprocess(image_path)
    if not ocr_results:
        return None

    best_en_name = None
    best_score = 0.0
    best_ja_text = ""

    for text, conf in ocr_results:
        # Extract Japanese character runs
        jp_matches = JP_CHAR_RE.findall(text)
        for jp_text in jp_matches:
            # Clean common OCR artifacts
            jp_clean = jp_text.rstrip('・。、')
            if len(jp_clean) < 2:
                continue

            jp_lower = jp_clean.lower()

            # Exact match first
            if jp_lower in ja_index:
                en_name = ja_index[jp_lower]
                logger.info("Japanese PaddleOCR exact: '%s' -> '%s' (conf=%.3f)",
                            jp_clean, en_name, conf)
                return en_name

            # Fuzzy match — handles partial reads like るいマルマイン
            match = process.extractOne(
                jp_lower, ja_names,
                scorer=fuzz.ratio,
                score_cutoff=70,
            )
            if match and match[1] > best_score:
                best_score = match[1]
                best_en_name = ja_index[match[0]]
                best_ja_text = jp_clean
                logger.debug("Japanese PaddleOCR fuzzy candidate: '%s' ~ '%s' -> '%s' (score=%.1f, conf=%.3f)",
                             jp_clean, match[0], best_en_name, match[1], conf)

    if best_en_name and best_score >= 70:
        logger.info("Japanese PaddleOCR fuzzy: '%s' -> '%s' (score=%.1f)",
                    best_ja_text, best_en_name, best_score)
        return best_en_name

    return None


def _run_hp_detect(image_path: str):
    """Run HP detection on a card image.

    Returns int HP value or None.
    """
    with _ocr_lock:
        try:
            from cardprice.ml.hp_detector import detect_hp
            return detect_hp(image_path)
        except Exception as e:
            logger.warning("v2 hp_detect failed: %s", e)
    return None


def _run_attack_ocr(image_path: str) -> list:
    """Run OCR to extract attack/move names from a card image.

    Returns list of attack name strings, or empty list on failure.
    Uses RapidOCR (not EasyOCR) to avoid loading ~500MB of EasyOCR models.
    """
    try:
        from cardprice.ml.attack_ocr import extract_attack_names_paddle
        candidates = extract_attack_names_paddle(image_path)
        # extract_attack_names_paddle returns [(text, confidence), ...]
        # Return just the text strings
        return [text for text, _conf in candidates if text]
    except Exception as e:
        logger.warning("v2 attack_ocr failed: %s", e)
    return []


_CARD_NAME_SUFFIXES = [
    " LV.X", " LV. X", " Lv.X",
    " VMAX", " VSTAR", " V-UNION",
    " V", " GX", " EX", " ex",
    "-GX", "-EX", "-ex",
]


def _strip_card_suffix(name: str) -> str | None:
    """Strip common Pokemon card suffixes (V, EX, LV.X etc.) from OCR name.

    Returns the base name without suffix, or None if no suffix found.
    """
    for suffix in _CARD_NAME_SUFFIXES:
        if name.endswith(suffix):
            base = name[: -len(suffix)].strip()
            if len(base) >= 2:
                return base
    return None


def _get_candidates_from_db(
    name: str,
    hp=None,
    card_type=None,
    session=None,
) -> list:
    """Query DB for candidate card_ids matching name/hp/type.

    Thin wrapper around ref_matcher.get_candidate_card_ids that also handles
    fuzzy name matching when exact match returns nothing.

    Returns list of card_id strings.
    """
    from cardprice.ml.ref_matcher import get_candidate_card_ids

    # First try exact name match
    candidates = get_candidate_card_ids(
        pokemon_name=name, hp=hp, card_type=card_type, session=session,
    )

    if candidates:
        return candidates

    # Try stripping card suffix (V, EX, LV.X, etc.) BEFORE relaxing HP.
    # "Flygon" + hp=120 is better than "Flygon LV.X" + hp=None.
    base_name = _strip_card_suffix(name)
    if base_name:
        logger.info("v2 DB lookup: stripping suffix %r -> %r", name, base_name)
        candidates = get_candidate_card_ids(
            pokemon_name=base_name, hp=hp, card_type=card_type, session=session,
        )
        if candidates:
            logger.info("v2 DB lookup: suffix-stripped -> %d candidates for %r",
                        len(candidates), base_name)
            return candidates

    # Exact match failed -- try without HP/type constraints
    if hp is not None or card_type is not None:
        candidates = get_candidate_card_ids(
            pokemon_name=name, hp=None, card_type=None, session=session,
        )
        if candidates:
            logger.info("v2 DB lookup: relaxed HP/type -> %d candidates for %r",
                        len(candidates), name)
            return candidates

    # Also try suffix-stripped base name without HP/type
    if base_name and (hp is not None or card_type is not None):
        candidates = get_candidate_card_ids(
            pokemon_name=base_name, hp=None, card_type=None, session=session,
        )
        if candidates:
            logger.info("v2 DB lookup: suffix-stripped + relaxed -> %d for %r",
                        len(candidates), base_name)
            return candidates

    # Still nothing -- try fuzzy name matching via ocr_matcher
    try:
        from cardprice.ml.ocr_matcher import fuzzy_match_card_name
        fuzzy_hits = fuzzy_match_card_name(name, top_k=20, score_cutoff=75.0)
        if fuzzy_hits:
            # fuzzy_match_card_name returns (card_id, name, set_id, score)
            # Get unique card_ids
            seen = set()
            fuzzy_cids = []
            for cid, _name, _sid, _score in fuzzy_hits:
                if cid not in seen:
                    seen.add(cid)
                    fuzzy_cids.append(cid)
            logger.info("v2 DB lookup: fuzzy match -> %d candidates for %r",
                        len(fuzzy_cids), name)
            return fuzzy_cids
    except Exception as e:
        logger.warning("v2 DB fuzzy lookup failed: %s", e)

    return []


def _get_candidates_from_sets_broad(
    name: str,
    set_ids: set,
    session=None,
) -> list:
    """Query DB for card_ids whose name CONTAINS `name` within specific sets.

    This is used by page context reranking to find cards like
    "Team Aqua's Poochyena" when OCR only read "Poochyena", within
    the page's likely sets.

    Returns list of card_id strings.
    """
    if not name or not set_ids:
        return []
    from sqlalchemy import text as _text
    own_session = session is None
    if own_session:
        from cardprice.db.session import SessionLocal
        session = SessionLocal()
    try:
        # Use ILIKE '%name%' to find cards containing the name
        # Also search by exact/prefix match (normal _get_candidates_from_db behavior)
        rows = session.execute(
            _text(
                "SELECT card_id FROM dim_cards "
                "WHERE LOWER(name) LIKE LOWER(:pattern) "
                "AND set_id = ANY(:sets) "
                "ORDER BY card_id"
            ),
            {"pattern": f"%{name}%", "sets": list(set_ids)},
        ).fetchall()
        result = [r[0] for r in rows]
        if result:
            logger.info(
                "v2 broad set query: %d candidates for %r in sets %s",
                len(result), name, list(set_ids)[:5],
            )
        return result
    except Exception as e:
        logger.warning("v2 broad set query failed: %s", e)
        return []
    finally:
        if own_session and session:
            session.close()


def _filter_candidates_by_attacks(
    candidates: list,
    attack_names: list,
    session=None,
) -> list:
    """Filter candidate card_ids by attack name matching.

    Uses the attack index (data/attack_index.pkl) to check which candidates
    have attacks matching the OCR-detected attack names.

    Returns filtered list of card_ids (subset of input). If filtering would
    eliminate all candidates, returns the original list unchanged.
    """
    if not attack_names or not candidates:
        return candidates

    try:
        from cardprice.ml.attack_ocr import _load_attack_index, _load_attack_db
        idx = _load_attack_index()
        card_to_attacks = idx.get("card_to_attacks", {})
        atk_to_cards = idx.get("attack_to_cards", {})
        attack_db = _load_attack_db()

        if not card_to_attacks and not atk_to_cards and not attack_db:
            logger.info("v2 attack filter: no attack index available")
            return candidates

        # Strategy: find candidates whose attacks overlap with detected attacks
        detected_lower = {a.lower().strip() for a in attack_names if a}

        scored = []
        for cid in candidates:
            # Try full card_id first (attack index keys include variant),
            # then fall back to base card_id (without variant) for compat
            card_attacks = card_to_attacks.get(cid, [])
            if not card_attacks:
                base_cid = cid.split("/")[0] if "/" in cid else cid
                card_attacks = card_to_attacks.get(base_cid, [])
            # Fallback to precomputed OCR attack DB
            if not card_attacks and attack_db:
                base_cid = cid.split("/")[0] if "/" in cid else cid
                card_attacks = attack_db.get(base_cid, [])
            card_attacks_lower = {a.lower() for a in card_attacks}

            # Count how many detected attacks match this card's attacks
            # Use fuzzy matching for OCR noise tolerance
            matches = 0
            try:
                from rapidfuzz import fuzz
                for det_atk in detected_lower:
                    for card_atk in card_attacks_lower:
                        if fuzz.ratio(det_atk, card_atk) >= 75:
                            matches += 1
                            break
            except ImportError:
                # Fall back to exact matching
                matches = len(detected_lower & card_attacks_lower)

            if matches > 0:
                scored.append((cid, matches))

        if scored:
            # Sort by number of matching attacks (descending)
            scored.sort(key=lambda x: x[1], reverse=True)
            filtered = [cid for cid, _score in scored]
            logger.info(
                "v2 attack filter: %d/%d candidates have matching attacks "
                "(detected: %s)",
                len(filtered), len(candidates), list(detected_lower),
            )
            return filtered

        logger.info("v2 attack filter: no candidates matched attacks, keeping all %d",
                     len(candidates))
    except Exception as e:
        logger.warning("v2 attack filter failed: %s", e)

    return candidates


def _dino_dot_product_against_refs(
    image_path: str,
    candidate_card_ids: list,
    query_embedding=None,
) -> list:
    """Compute DINOv2 dot product between query image and reference images.

    For each candidate card_id, looks up the reference image, computes the
    DINOv2 embedding similarity (cosine = dot product of L2-normalized vectors),
    and returns results sorted by similarity.

    Returns list of (card_id, similarity_score) sorted descending.
    """
    from cardprice.ml.ref_matcher import (
        get_reference_image_path,
        compute_embedding_similarity,
    )

    # Resolve reference images for each candidate
    ref_paths = []
    ref_card_ids = []
    for cid in candidate_card_ids:
        ref_path = get_reference_image_path(cid)
        if ref_path is not None:
            ref_paths.append(ref_path)
            ref_card_ids.append(cid)

    if not ref_paths:
        logger.info("v2 DINOv2: no reference images found for %d candidates",
                     len(candidate_card_ids))
        return []

    # Preprocess query image for DINOv2 (CLAHE + glare removal)
    # Skip if we already have a pre-computed embedding
    query_path = image_path
    preproc_tmp = None
    if query_embedding is None:
        try:
            from cardprice.ml.preprocess import preprocess_for_matching
            preproc_tmp = preprocess_for_matching(image_path)
            query_path = preproc_tmp
        except Exception:
            pass

    try:
        similarities = compute_embedding_similarity(
            query_path, ref_paths, ref_card_ids,
            query_embedding=query_embedding,
        )

        # Pair up and sort
        results = list(zip(ref_card_ids, similarities))
        results.sort(key=lambda x: x[1], reverse=True)

        if results:
            logger.info(
                "v2 DINOv2: top match %s (%.4f) out of %d refs",
                results[0][0], results[0][1], len(results),
            )
            for i, (cid, sim) in enumerate(results[:3]):
                logger.debug("  #%d: %s (%.4f)", i + 1, cid, sim)

        return results
    finally:
        if preproc_tmp:
            try:
                os.unlink(preproc_tmp)
            except OSError:
                pass


@lru_cache(maxsize=2000)
def _get_card_type_cached(card_id: str) -> str | None:
    """Look up the primary Pokemon type for a card_id.

    Uses DB (dim_pokemon.types via dim_cards.pokemon_id) with JSON fallback.
    Returns the first type string (e.g. "Fire") or None.
    """
    # Try DB first
    try:
        from cardprice.ml.ref_matcher import _get_session
        session = _get_session()
        if session is not None:
            try:
                from sqlalchemy import text
                row = session.execute(text("""
                    SELECT p.types
                    FROM dim_cards c
                    JOIN dim_pokemon p ON c.pokemon_id = p.pokemon_id
                    WHERE c.card_id = :cid
                    LIMIT 1
                """), {"cid": card_id}).fetchone()
                if row and row[0]:
                    return row[0][0]  # First type
            finally:
                session.close()
    except Exception:
        pass

    # JSON fallback
    try:
        from cardprice.ml.ref_matcher import _load_card_names_fallback
        entries = _load_card_names_fallback()
        for entry in entries:
            if entry[0] == card_id and len(entry) > 4 and entry[4]:
                return entry[4][0]  # First type
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Structured attacks lazy singleton
# ---------------------------------------------------------------------------
_structured_attacks: dict | None = None
_STRUCTURED_ATTACKS_PATH = _PROJECT_ROOT / "data" / "structured_attacks.json"


def _load_structured_attacks() -> dict:
    global _structured_attacks
    if _structured_attacks is not None:
        return _structured_attacks
    if not _STRUCTURED_ATTACKS_PATH.exists():
        _structured_attacks = {}
        return _structured_attacks
    import json
    with open(_STRUCTURED_ATTACKS_PATH) as f:
        _structured_attacks = json.load(f)
    logger.info("Loaded structured attacks: %d cards", len(_structured_attacks))
    return _structured_attacks


def _build_text_fingerprint(data: dict) -> str:
    """Build a text fingerprint from structured attack/ability data for fuzzy matching."""
    parts: list[str] = []
    for atk in data.get("attacks", []):
        if atk.get("name"):
            parts.append(atk["name"])
        if atk.get("text"):
            parts.append(atk["text"])
        if atk.get("damage"):
            parts.append(atk["damage"])
    for ab in data.get("abilities", []):
        if ab.get("name"):
            parts.append(ab["name"])
        if ab.get("text"):
            parts.append(ab["text"])
    return " ".join(parts)


def _get_fulltext_from_image(image_path: str) -> str:
    """Run RapidOCR on the full card image and return all detected text concatenated."""
    try:
        from cardprice.ml.ocr_matcher import get_rapid_engine
        import cv2
        engine = get_rapid_engine()
        img = cv2.imread(image_path)
        if img is None:
            return ""
        result = engine(img)
        if not result or not result[0]:
            return ""
        # result[0] is list of (bbox, text, confidence)
        texts = [item[1] for item in result[0] if item[1]]
        return " ".join(texts)
    except Exception as e:
        logger.warning("v2 combined: fulltext OCR failed: %s", e)
        return ""


def _score_candidates_combined(
    image_path: str,
    candidate_card_ids: list,
    query_embedding=None,
    precomputed_attacks=None,
    type_detected: str | None = None,
    type_confidence: float = 0.0,
    precomputed_fulltext: str | None = None,
    ocr_card_num: str | None = None,
    ocr_set_total: str | None = None,
) -> list[tuple[str, float, dict]]:
    """Score candidates using DINOv2 visual similarity, attack OCR overlap, and full-text matching.

    Combined score = w_dino * dino_score + w_attack * attack_score + w_fulltext * fulltext_score
    When attack OCR finds nothing, falls back to pure DINOv2.
    Full-text matching uses structured attack/ability data against the scanned card's OCR text.
    """
    from cardprice.ml.attack_ocr import extract_attack_names_paddle, _load_attack_index, _load_attack_db

    dino_results = _dino_dot_product_against_refs(image_path, candidate_card_ids, query_embedding=query_embedding)
    if not dino_results:
        return []

    # For Trainer/Supporter/Item cards: re-score using artwork-only crop.
    # Full-card DINOv2 is unreliable for Trainers — text dominates (60% of card)
    # and orange binder tint confuses embeddings. Artwork crop eliminates wrong
    # matches (e.g. blue pl4 Buffer Piece vs green ex15 Buffer Piece).
    # Detect Trainer cards: they have no attacks in structured_attacks.json
    _structured = _load_structured_attacks()
    _all_trainer = len(candidate_card_ids) >= 2 and all(
        not _structured.get(c.split("/")[0], {}).get("attacks")
        for c in candidate_card_ids
    )
    if _all_trainer:
        try:
            import cv2, tempfile
            import numpy as np
            from cardprice.ml.dino_matcher import extract_embedding as _dino_emb
            from cardprice.ml.ref_matcher import get_reference_image_path

            img = cv2.imread(str(image_path))
            if img is not None:
                h, w = img.shape[:2]
                art = img[int(h*0.15):int(h*0.55), int(w*0.10):int(w*0.90)]
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                cv2.imwrite(tmp.name, art)
                q_art = _dino_emb(tmp.name)
                os.unlink(tmp.name)

                art_scores = {}
                for cid in candidate_card_ids:
                    ref = get_reference_image_path(cid)
                    if ref:
                        ref_img = cv2.imread(str(ref))
                        if ref_img is not None:
                            rh, rw = ref_img.shape[:2]
                            ref_art = ref_img[int(rh*0.15):int(rh*0.55), int(rw*0.10):int(rw*0.90)]
                            tmp2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                            cv2.imwrite(tmp2.name, ref_art)
                            r_art = _dino_emb(tmp2.name)
                            os.unlink(tmp2.name)
                            art_scores[cid] = float(np.dot(q_art, r_art))

                if art_scores:
                    dino_results = sorted(art_scores.items(), key=lambda x: -x[1])
                    logger.info("v2 combined: Trainer artwork DINOv2: %s",
                                {c: f"{s:.3f}" for c, s in dino_results})
        except Exception as e:
            logger.warning("v2 combined: Trainer artwork crop failed: %s", e)
    dino_scores = {cid: score for cid, score in dino_results}

    # Trainer artwork DINOv2 re-scoring removed — added DB query + N DINOv2
    # inferences per trainer card for marginal benefit. Full-card DINOv2 with
    # attack OCR scoring is sufficient.

    # Attack OCR — use pre-computed if available, else run RapidOCR (not EasyOCR,
    # which loads ~500MB of models and pushes RSS over 4GB)
    if precomputed_attacks is not None:
        ocr_candidates = precomputed_attacks
    else:
        ocr_candidates = []
        try:
            with _ocr_lock:
                ocr_candidates = extract_attack_names_paddle(image_path)
        except Exception as e:
            logger.warning("v2 combined: attack OCR failed: %s", e)

    detected_attacks = [text.lower().strip() for text, _conf in ocr_candidates if text]

    idx = _load_attack_index()
    card_to_attacks = idx.get("card_to_attacks", {})
    attack_db = _load_attack_db()
    structured = _load_structured_attacks()

    try:
        from rapidfuzz import fuzz
        use_rapidfuzz = True
    except ImportError:
        use_rapidfuzz = False

    attack_scores = {}
    attack_details = {}
    for cid in candidate_card_ids:
        base_cid = cid.split("/")[0] if "/" in cid else cid

        # --- Attack names: prefer structured_attacks (19,895 cards) over attack_index (16,978) ---
        card_attacks = []
        use_ocr_ocr = False
        struct_data = structured.get(base_cid)
        if struct_data:
            # Extract attack + ability names from structured data
            card_attacks = [a["name"] for a in struct_data.get("attacks", []) if a.get("name")]
            card_attacks += [a["name"] for a in struct_data.get("abilities", []) if a.get("name")]

        # Fallback 1: attack_index.pkl (curated attack names)
        if not card_attacks:
            card_attacks = card_to_attacks.get(cid, [])
            if not card_attacks:
                card_attacks = card_to_attacks.get(base_cid, [])

        # Fallback 2: attack_db.json (noisy OCR text)
        if not card_attacks and attack_db:
            db_attacks = attack_db.get(base_cid, [])
            if db_attacks:
                card_attacks = db_attacks
                use_ocr_ocr = True

        if not card_attacks or not detected_attacks:
            attack_scores[cid] = 0.0
            attack_details[cid] = []
            continue

        card_attacks_lower = [a.lower() for a in card_attacks]
        matched = []

        if use_ocr_ocr:
            # OCR-to-OCR: both sides are noisy, use token_set_ratio
            for card_atk in card_attacks_lower:
                if len(card_atk) < 3:
                    continue
                for det_atk in detected_attacks:
                    if len(det_atk) < 3:
                        continue
                    if use_rapidfuzz and fuzz.token_set_ratio(det_atk, card_atk) >= 80:
                        matched.append(card_atk)
                        break
        else:
            for card_atk in card_attacks_lower:
                for det_atk in detected_attacks:
                    min_len = min(len(det_atk), len(card_atk))
                    threshold = 80 if min_len <= 5 else 70
                    if use_rapidfuzz:
                        if fuzz.ratio(det_atk, card_atk) >= threshold:
                            matched.append(card_atk)
                            break
                    else:
                        from difflib import SequenceMatcher
                        t = 0.75 if min_len <= 5 else 0.65
                        if SequenceMatcher(None, det_atk, card_atk).ratio() >= t:
                            matched.append(card_atk)
                            break

        # Score: proportion of detected attacks matched, plus bonus if attack count matches
        proportion = len(matched) / len(card_attacks_lower) if card_attacks_lower else 0.0
        # Bonus when card's attack count matches what the scan detected
        count_match_bonus = 0.15 if len(card_attacks_lower) == len(detected_attacks) else 0.0
        count_bonus = 0.1 * min(len(matched), 3)
        raw_score = proportion + count_match_bonus + count_bonus
        # Discount OCR-to-OCR matches (noisier than curated attack names)
        if use_ocr_ocr:
            raw_score *= 0.7
        attack_scores[cid] = raw_score
        attack_details[cid] = matched

    # --- Full-text matching: structured attack/ability text vs scanned card OCR ---
    fulltext_scores = {}
    if structured and use_rapidfuzz:
        # Get the full OCR text from the scanned card
        if precomputed_fulltext is not None:
            scan_fulltext = precomputed_fulltext
        else:
            scan_fulltext = _get_fulltext_from_image(image_path)

        if scan_fulltext:
            scan_fulltext_lower = scan_fulltext.lower()
            for cid in candidate_card_ids:
                base_cid = cid.split("/")[0] if "/" in cid else cid
                struct_data = structured.get(base_cid)
                if not struct_data:
                    fulltext_scores[cid] = 0.0
                    continue
                fingerprint = _build_text_fingerprint(struct_data).lower()
                if not fingerprint:
                    fulltext_scores[cid] = 0.0
                    continue
                # token_set_ratio handles subset matching well (OCR may miss some text)
                ratio = fuzz.token_set_ratio(scan_fulltext_lower, fingerprint)
                fulltext_scores[cid] = ratio / 100.0  # Normalize to 0-1
        else:
            for cid in candidate_card_ids:
                fulltext_scores[cid] = 0.0
    else:
        for cid in candidate_card_ids:
            fulltext_scores[cid] = 0.0

    # Dynamic weights — lean toward DINOv2 when visual gap is clear
    any_attacks = any(s > 0 for s in attack_scores.values())
    any_fulltext = any(s > 0 for s in fulltext_scores.values())
    if not any_attacks and not any_fulltext:
        w_dino, w_attack, w_fulltext = 1.0, 0.0, 0.0
    elif not any_attacks:
        # Only fulltext signal available
        w_dino, w_attack, w_fulltext = 1.0, 0.0, 0.0
    elif not any_fulltext:
        # Only attack signal available (original behavior)
        dino_vals = sorted(dino_scores.values(), reverse=True)
        dino_gap = (dino_vals[0] - dino_vals[1]) if len(dino_vals) >= 2 else 0.0
        if dino_gap >= 0.10:
            w_dino, w_attack, w_fulltext = 0.7, 0.3, 0.0
        else:
            w_dino, w_attack, w_fulltext = 0.5, 0.5, 0.0
    else:
        # All three signals available
        dino_vals = sorted(dino_scores.values(), reverse=True)
        dino_gap = (dino_vals[0] - dino_vals[1]) if len(dino_vals) >= 2 else 0.0
        if dino_gap >= 0.10:
            w_dino, w_attack, w_fulltext = 0.6, 0.4, 0.0
        else:
            w_dino, w_attack, w_fulltext = 0.5, 0.5, 0.0

    # Type bonus/penalty — small tie-breaker signal
    use_type_signal = (
        type_detected is not None
        and type_confidence >= 0.40
        and type_detected != "Colorless"
    )

    results = []
    for cid in candidate_card_ids:
        d = dino_scores.get(cid, 0.0)
        a = attack_scores.get(cid, 0.0)
        ft = fulltext_scores.get(cid, 0.0)
        combined = w_dino * d + w_attack * a + w_fulltext * ft

        type_bonus = 0.0
        if use_type_signal:
            card_type = _get_card_type_cached(cid)
            if card_type is not None:
                if card_type == type_detected:
                    type_bonus = 0.05
                elif type_confidence >= 0.60:
                    type_bonus = -0.03
            # else: unknown type, no adjustment
        combined += type_bonus

        # Card number tiebreaker: when OCR read a collector number from the
        # bottom of the card, boost candidates whose DB card_number matches.
        # This resolves same-artwork reprints across sets (e.g. Buffer Piece
        # in ex3 vs ex15 vs pl4) where DINOv2 scores are within 0.01.
        card_num_bonus = 0.0
        if ocr_card_num:
            base_cid = cid.split("/")[0] if "/" in cid else cid
            # Extract card number from the card_id (e.g. "ex15-72" -> "72")
            cid_num = base_cid.rsplit("-", 1)[-1] if "-" in base_cid else None
            if cid_num and cid_num == ocr_card_num:
                card_num_bonus = 0.08
                # Additional bonus if set total also matches
                if ocr_set_total:
                    try:
                        from sqlalchemy import text as sa_text
                        from cardprice.ml.ref_matcher import _get_session
                        _sess = _get_session()
                        row = _sess.execute(
                            sa_text(
                                "SELECT s.total_cards FROM dim_sets s "
                                "JOIN dim_cards c ON c.set_id = s.set_id "
                                "WHERE c.card_id = :cid"
                            ),
                            {"cid": cid},
                        ).fetchone()
                        _sess.close()
                        if row and str(row[0]) == ocr_set_total:
                            card_num_bonus = 0.15  # Strong match: both number and set total
                            logger.info(
                                "v2 combined: card_number %s/%s matches %s (set total %s)",
                                ocr_card_num, ocr_set_total, cid, row[0],
                            )
                    except Exception:
                        pass  # DB error, keep the card_number-only bonus
        combined += card_num_bonus

        results.append((cid, combined, {
            "dino_score": round(d, 4),
            "attack_score": round(a, 4),
            "fulltext_score": round(ft, 4),
            "matched_attacks": attack_details.get(cid, []),
            "type_bonus": round(type_bonus, 4),
            "card_num_bonus": round(card_num_bonus, 4),
        }))

    results.sort(key=lambda x: x[1], reverse=True)
    if results:
        t = results[0]
        logger.info("v2 combined: top=%s score=%.4f (dino=%.4f, atk=%.4f, ft=%.4f, type=%.4f, cardnum=%.4f) %d candidates",
                     t[0], t[1], t[2]["dino_score"], t[2]["attack_score"], t[2]["fulltext_score"], t[2]["type_bonus"], t[2].get("card_num_bonus", 0.0), len(results))
    return results


def _is_card_back(image_path: str) -> bool:
    """Detect Pokemon card backs via HSV analysis.

    Two detection strategies (either triggers detection):

    1. **Blue-dominance** (original): Card backs are uniformly blue
       (HSV hue ~90-130, medium+ saturation) across center AND edges.
       Requires: >40% overall blue AND >70% blue in outer edge strips.

    2. **Color uniformity** (handles color-cast sleeves): Card backs seen
       through tinted binder sleeves lose their blue hue but remain
       *extremely* uniform -- a single 5-degree hue bin covers 90%+ of
       all pixels, with high saturation everywhere.  No card front is
       this uniform because artwork/text/borders create hue diversity.
       Requires: top hue bin >= 85% of pixels AND mean saturation > 150.
    """
    try:
        import cv2
        import numpy as np
        img = cv2.imread(image_path)
        if img is None:
            return False
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # --- Strategy 1: Blue-dominance ---
        blue = (hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 130) & (hsv[:, :, 1] > 50)
        overall = float(np.mean(blue))

        if overall >= 0.40:
            eh, ew = int(h * 0.10), int(w * 0.10)
            edge_pixels = np.concatenate([
                blue[:eh, :].ravel(), blue[-eh:, :].ravel(),
                blue[:, :ew].ravel(), blue[:, -ew:].ravel(),
            ])
            edge_ratio = float(np.mean(edge_pixels))
            logger.debug("card_back: blue overall=%.3f edge=%.3f for %s",
                          overall, edge_ratio, image_path)
            if edge_ratio > 0.70:
                return True

        # --- Strategy 2: Color uniformity (handles tinted sleeves) ---
        # Card backs through any color sleeve are a single solid hue.
        # Bin hues into 5-degree buckets (36 bins over 0-179 range).
        hue_flat = hsv[:, :, 0].ravel()
        sat_flat = hsv[:, :, 1].ravel()
        bins = np.bincount(hue_flat // 5, minlength=36)
        top_bin_ratio = float(bins.max()) / len(hue_flat)
        sat_mean = float(np.mean(sat_flat))
        logger.debug("card_back: top_bin=%.3f sat_mean=%.1f for %s",
                      top_bin_ratio, sat_mean, image_path)
        if top_bin_ratio >= 0.85 and sat_mean > 150:
            return True

        return False
    except Exception as e:
        logger.warning("card_back check failed: %s", e)
        return False


def identify_card_v2(image_path, session=None, page_era=None, _precomputed_ocr=None,
                     _precomputed_dino_embedding=None, _precomputed_attacks=None,
                     _precomputed_clip_embedding=None, _precomputed_easyocr_name=None,
                     detect_variants=True, use_claude_vision_fallback=False,
                     correct_perspective=False, enable_card_number_ocr=False):
    """V2 card identification: color + name OCR + HP -> DB filter -> DINOv2.

    This pipeline is fundamentally different from v1 (cascade/ensemble):
    instead of searching the entire 20k-card FAISS index, it uses cheap
    classifiers to narrow candidates to 2-20, then does precise DINOv2
    dot product against only those reference images.

    Pipeline:
        1. Run in parallel: color_detect + name_ocr + hp_detect
        2. Query DB: get candidates matching (name, hp, type)
        3. If candidates <= 3: DINOv2 dot product, return best
        4. If candidates > 3: also run attack_ocr, filter by attack match,
           then DINOv2
        5. If no candidates (OCR failed): fall back to ensemble method
        6. If still unidentified and use_claude_vision_fallback=True,
           send the card image to Claude vision API

    Args:
        image_path: Path to the card image.
        session: Optional SQLAlchemy DB session.
        page_era: Optional era string (e.g. "ex", "e-card") from page context.
            Used to filter attack fallback candidates.
        _precomputed_ocr: Optional dict with pre-computed OCR results from
            batch processing. Keys: ocr_name, ocr_conf, ocr_raw, hp_value,
            color_type, color_conf.  When provided, Step 1 is skipped entirely.
        _precomputed_dino_embedding: Optional pre-computed DINOv2 query embedding.
            When provided, skips GPU DINOv2 extraction (used for batch processing).
        _precomputed_attacks: Optional pre-computed attack OCR results.
            When provided, skips EasyOCR attack extraction.
        detect_variants: Whether to run variant detection after identification
            (default True).  Set False to skip variant classification entirely.
        use_claude_vision_fallback: If True and the card remains unidentified
            after all ML steps, send the image to Claude vision API as a last
            resort. Costs API credits. Default False.
        correct_perspective: If True, apply perspective correction to the card
            image before identification. Saves a corrected copy alongside the
            original. Default False.

    Returns:
        Dict with keys: card_id, confidence, method, explanation, raw_response.
        When detect_variants is True, may also include:
            detected_variant, variant_confidence, variant_checks_run.
    """
    image_path = str(image_path)

    # Optional perspective correction
    if correct_perspective:
        try:
            import cv2 as _cv2_corr
            from cardprice.ml.card_corrector import correct_card_image
            _corr_img = _cv2_corr.imread(image_path)
            if _corr_img is not None:
                _corr_out = correct_card_image(_corr_img)
                _corr_path = image_path + '_corrected.png'
                _cv2_corr.imwrite(_corr_path, _corr_out)
                logger.info("v2: perspective-corrected image saved: %s", _corr_path)
                image_path = _corr_path
        except Exception as e:
            logger.warning("v2: perspective correction failed, using original: %s", e)

    # Convert HEIC/HEIF if needed
    try:
        from cardprice.utils.image_convert import ensure_compatible
        image_path = ensure_compatible(image_path)
    except Exception as e:
        logger.warning("v2: image conversion failed, using original: %s", e)

    # Preserve original path before potential rotation fix (Step 1b).
    # If rotation is a false positive, the attack fallback (Step 5) needs
    # the original un-rotated image to read attacks correctly.
    original_image_path = image_path

    # Check cache
    try:
        file_hash = hashlib.md5(Path(image_path).read_bytes()).hexdigest()
        cache_key = f"v2_{file_hash}"
        if cache_key in _scan_cache:
            logger.info("v2: cache HIT for %s", image_path)
            _scan_cache.move_to_end(cache_key)
            return _scan_cache[cache_key]
    except Exception:
        file_hash = None
        cache_key = None

    # -----------------------------------------------------------------------
    # Step 0: Card-back detection (reject before any expensive processing)
    # -----------------------------------------------------------------------
    if _is_card_back(image_path):
        result = {
            "card_id": None,
            "confidence": 0.95,
            "method": "v2_card_back",
            "explanation": "Detected Pokemon card back (blue background with Pokeball)",
            "raw_response": {},
        }
        _cache_store(cache_key, result)
        logger.info("v2: card back detected for %s", image_path)
        return result

    # -----------------------------------------------------------------------
    # Step 1: Run cheap classifiers in parallel
    # -----------------------------------------------------------------------
    # Name OCR + HP detection share a single PaddleOCR pass on the top 25%
    # of the card (~3s saved vs separate PaddleOCR + EasyOCR calls).
    color_type = None
    color_conf = 0.0
    ocr_name = None
    ocr_conf = 0.0
    ocr_raw = None
    hp_value = None

    # Stamp text from precomputed worker (set even when name OCR fails).
    # Used to narrow candidates when the card name is unreadable but the
    # stamp text in the artwork region was OCR'd successfully.
    precomputed_stamp_set_id = None
    precomputed_stamp_match_score = 0
    precomputed_stamp_texts: list[str] = []
    if _precomputed_ocr:
        # Use pre-computed OCR results from batch processing
        ocr_name = _precomputed_ocr.get("ocr_name")
        ocr_conf = _precomputed_ocr.get("ocr_conf", 0.0)
        ocr_raw = _precomputed_ocr.get("ocr_raw")
        hp_value = _precomputed_ocr.get("hp_value")
        color_type = _precomputed_ocr.get("color_type")
        color_conf = _precomputed_ocr.get("color_conf", 0.0)
        precomputed_stamp_set_id = _precomputed_ocr.get("stamp_set_id")
        precomputed_stamp_match_score = _precomputed_ocr.get("stamp_match_score", 0) or 0
        precomputed_stamp_texts = _precomputed_ocr.get("stamp_texts", []) or []
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            color_future = pool.submit(_run_color_detect, image_path)
            name_hp_future = pool.submit(_run_name_and_hp, image_path)

            try:
                color_type, color_conf = color_future.result(timeout=10)
            except Exception as e:
                logger.warning("v2 step1: color_detect error: %s", e)

            try:
                ocr_name, ocr_conf, ocr_raw, hp_value = name_hp_future.result(timeout=30)
            except Exception as e:
                logger.warning("v2 step1: name_and_hp error: %s", e)

    # Reject partial OCR names (< 3 chars) — they create bad candidate sets.
    # e.g. "tty" for Skitty, "ch" for Trapinch match wrong cards.
    if ocr_name and len(ocr_name) < 3:
        logger.info("v2: rejecting short OCR name %r (len=%d)", ocr_name, len(ocr_name))
        ocr_name = None
        ocr_conf = 0.0

    # -------------------------------------------------------------------
    # Step 1b: Rotation detection — if OCR failed, the card content may be
    # rotated 90° within the frame.  Try both 90° CW and CCW rotations;
    # keep the one that yields the best OCR name confidence.
    # -------------------------------------------------------------------
    if not ocr_name and not _precomputed_ocr:
        import cv2
        logger.info("v2: OCR failed — trying 90° rotations for %s", image_path)
        img_orig = cv2.imread(image_path)
        if img_orig is not None:
            best_rot_score = 0.0
            best_rot_result = None
            best_rot_path = None
            # Try 90° CW and 90° CCW
            for rot_code, rot_label in [
                (cv2.ROTATE_90_CLOCKWISE, "90CW"),
                (cv2.ROTATE_90_COUNTERCLOCKWISE, "90CCW"),
            ]:
                rotated = cv2.rotate(img_orig, rot_code)
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="rot_")
                os.close(tmp_fd)
                try:
                    cv2.imwrite(tmp_path, rotated)
                    rot_name, rot_conf, rot_raw, rot_hp = _run_name_and_hp(tmp_path)
                    # Composite score: name confidence + large bonus for HP.
                    # A correct rotation typically yields both name AND HP,
                    # while an upside-down card may get a spurious name but
                    # no HP (HP is in the top-right, only readable upright).
                    # HP detection is highly reliable (requires "\d+ HP/PV"
                    # pattern), so weight it heavily to prefer the rotation
                    # that finds HP even if it can't read a foreign name.
                    # Validate name against DB — spurious OCR from upside-down
                    # cards won't match any Pokemon name.
                    has_valid_name = False
                    if rot_name and len(rot_name) >= 3:
                        db_check = _get_candidates_from_db(
                            name=rot_name, hp=rot_hp, session=session,
                        )
                        has_valid_name = len(db_check) > 0
                        if not has_valid_name:
                            logger.info(
                                "v2: rotation %s: name %r not in DB, ignoring",
                                rot_label, rot_name,
                            )
                    rot_score = rot_conf if has_valid_name else 0.0
                    if rot_hp is not None:
                        rot_score += 0.80
                    logger.info(
                        "v2: rotation %s: name=%r conf=%.2f hp=%s score=%.2f",
                        rot_label, rot_name, rot_conf, rot_hp, rot_score,
                    )
                    if rot_score > best_rot_score:
                        best_rot_score = rot_score
                        best_rot_result = (rot_name, rot_conf, rot_raw, rot_hp)
                        # Clean up previous best if any
                        if best_rot_path and os.path.exists(best_rot_path):
                            os.unlink(best_rot_path)
                        best_rot_path = tmp_path
                    else:
                        os.unlink(tmp_path)
                except Exception as e:
                    logger.warning("v2: rotation %s failed: %s", rot_label, e)
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            if best_rot_result and best_rot_score >= 0.30:
                ocr_name, ocr_conf, ocr_raw, hp_value = best_rot_result
                image_path = best_rot_path
                # Re-run color detection on the corrected image
                try:
                    color_type, color_conf = _run_color_detect(image_path)
                except Exception:
                    pass
                logger.info(
                    "v2: ROTATION FIX applied — using rotated image, "
                    "name=%r conf=%.2f hp=%s (rot_score=%.2f)",
                    ocr_name, ocr_conf, hp_value, best_rot_score,
                )
            else:
                # No rotation helped — clean up
                if best_rot_path and os.path.exists(best_rot_path):
                    os.unlink(best_rot_path)

    # -----------------------------------------------------------------------
    # Step 1c: Card number OCR — read collector number from bottom of card
    # OFF BY DEFAULT. Binder page segments are too low-resolution
    # (~1008x1530) for the bottom number text (~8-12px tall) to OCR reliably,
    # and the 12-attempt loop blows the 10s/page budget. Caller can opt-in
    # via enable_card_number_ocr=True for single-card scans where the extra
    # latency is acceptable.
    # -----------------------------------------------------------------------
    ocr_card_num = None
    ocr_set_total = None
    if _precomputed_ocr is not None and "ocr_card_num" in _precomputed_ocr:
        # Caller (page scanner) precomputed it — accept whatever they pass
        ocr_card_num = _precomputed_ocr.get("ocr_card_num")
        ocr_set_total = _precomputed_ocr.get("ocr_set_total")
    elif enable_card_number_ocr:
        # Explicit opt-in (single-card scans only)
        try:
            ocr_card_num, ocr_set_total = _ocr_card_number(image_path)
        except Exception as e:
            logger.warning("v2 step1c: card_number OCR error: %s", e)

    logger.info(
        "v2 step1: name=%r (conf=%.2f), hp=%s, color=%s (conf=%.2f), card_num=%s/%s",
        ocr_name, ocr_conf, hp_value, color_type, color_conf,
        ocr_card_num, ocr_set_total,
    )

    # -----------------------------------------------------------------------
    # Japanese fast path: when JP DINOv2 found a confident match, the OCR
    # function returns the JP card_id directly (jp_<tcg_product_id>) instead
    # of an English Pokemon name.  Short-circuit the candidate lookup and
    # return the JP card_id straight through.  Downstream price lookup
    # auto-routes to fact_market_prices_jp via the UNION SQL.
    # -----------------------------------------------------------------------
    if ocr_name and isinstance(ocr_name, str) and ocr_name.startswith("jp_"):
        logger.info("v2: Japanese fast path — JP DINOv2 returned %s", ocr_name)
        result = {
            "card_id": ocr_name,
            "confidence": ocr_conf,
            "method": "v2_jp_dinov2",
            "raw_response": {
                "ocr_name": ocr_name,
                "ocr_raw": ocr_raw or f"[JP]{ocr_name}",
                "language": "ja",
            },
        }
        _apply_variant_detection(result, image_path, detect_variants=detect_variants,
                                 precomputed_stamp_set_id=precomputed_stamp_set_id,
                                 precomputed_stamp_match_score=precomputed_stamp_match_score,
                                 precomputed_stamp_texts=precomputed_stamp_texts)
        return result

    # -----------------------------------------------------------------------
    # Step 2: Query DB for candidates
    # -----------------------------------------------------------------------
    # Type detection: used as a SCORING signal (not a hard DB filter).
    # This avoids dropping correct candidates when color detection is wrong.
    use_type = None
    if color_type and color_conf >= 0.40 and color_type != "Colorless":
        use_type = color_type

    candidates = []
    if ocr_name:
        candidates = _get_candidates_from_db(
            name=ocr_name,
            hp=hp_value,
            card_type=None,  # Type used for scoring, not filtering
            session=session,
        )
        logger.info(
            "v2 step2: %d candidates for name=%r, hp=%s (type=%s for scoring only)",
            len(candidates), ocr_name, hp_value, use_type,
        )

    # -----------------------------------------------------------------------
    # Step 2b: Card number disambiguation — if we read a collector number,
    # filter candidates to those matching the card_number in the DB.
    # This directly resolves variant confusion (e.g., Mew ex 205/165 vs
    # 151/165 vs 193/165).
    # -----------------------------------------------------------------------
    card_num_match = None
    if ocr_card_num and candidates and len(candidates) >= 2:
        try:
            from sqlalchemy import text as sa_text
            from cardprice.ml.ref_matcher import _get_session
            _sess = session or _get_session()
            rows = _sess.execute(
                sa_text(
                    "SELECT card_id FROM dim_cards "
                    "WHERE LTRIM(card_number, '0') = :num"
                ),
                {"num": ocr_card_num},
            ).fetchall()
            num_card_ids = {r[0] for r in rows}
            if not session:
                _sess.close()

            # Intersect with current candidates
            num_matched = [c for c in candidates if c in num_card_ids]

            # If set_total was also read, further narrow by matching the
            # set's total_cards (e.g. "72/101" -> only sets with 101 cards).
            if ocr_set_total and len(num_matched) >= 2:
                try:
                    total_rows = _sess.execute(
                        sa_text(
                            "SELECT c.card_id FROM dim_cards c "
                            "JOIN dim_sets s ON s.set_id = c.set_id "
                            "WHERE c.card_id = ANY(:cids) "
                            "AND CAST(s.total_cards AS TEXT) = :total"
                        ),
                        {"cids": num_matched, "total": ocr_set_total},
                    ).fetchall()
                    total_matched = [r[0] for r in total_rows]
                    if len(total_matched) >= 1 and len(total_matched) < len(num_matched):
                        logger.info(
                            "v2 step2b: set_total %s narrowed %d -> %d candidates",
                            ocr_set_total, len(num_matched), len(total_matched),
                        )
                        num_matched = total_matched
                except Exception as e2:
                    logger.debug("v2 step2b: set_total filtering failed: %s", e2)

            if len(num_matched) == 1:
                # Unique match — card number + name uniquely identifies
                card_num_match = num_matched[0]
                logger.info(
                    "v2 step2b: card_number %s/%s uniquely matches %s among %d candidates",
                    ocr_card_num, ocr_set_total, card_num_match, len(candidates),
                )
            elif len(num_matched) >= 2:
                # Multiple matches (rare) — narrow candidates
                logger.info(
                    "v2 step2b: card_number %s matches %d candidates: %s",
                    ocr_card_num, len(num_matched), num_matched,
                )
                candidates = num_matched
            else:
                logger.info(
                    "v2 step2b: card_number %s matched no current candidates "
                    "(DB has %d cards with that number)",
                    ocr_card_num, len(num_card_ids),
                )
        except Exception as e:
            logger.warning("v2 step2b: card_number DB lookup failed: %s", e)

    # If card number uniquely identified the card, verify with DINOv2 sanity
    # check.  Card number OCR can misread (e.g. "4/102" -> "15/102") so
    # we confirm the visual match is reasonable before trusting it.
    if card_num_match:
        # DINOv2 against the card_num match AND all other candidates
        all_check_ids = [card_num_match] + [c for c in candidates if c != card_num_match]
        dino_all = _dino_dot_product_against_refs(
            image_path, all_check_ids,
            query_embedding=_precomputed_dino_embedding,
        )
        dino_score = 0.0
        best_other_score = 0.0
        for cid, score in dino_all:
            if cid == card_num_match:
                dino_score = score
            elif score > best_other_score:
                best_other_score = score

        # Accept card number match if DINOv2 confirms the visual match:
        # - Score is reasonable (>= 0.40), AND
        # - No other candidate scores much higher (within 0.15 of top).
        # When another candidate has a much better DINOv2 match, the OCR
        # likely misread the number (e.g. "35/102" -> "15/102").
        trust_card_num = (
            dino_score >= 0.40
            and (best_other_score - dino_score) < 0.15
        )
        if trust_card_num:
            explanation = (
                f"v2: card_number OCR={ocr_card_num}/{ocr_set_total} -> "
                f"{card_num_match} (name={ocr_name!r}, hp={hp_value}, "
                f"dino={dino_score:.3f})"
            )
            result = {
                "card_id": card_num_match,
                "confidence": max(ocr_conf, 0.85),
                "method": "v2_card_number",
                "explanation": explanation,
                "raw_response": {
                    "ocr_name": ocr_name, "ocr_confidence": ocr_conf,
                    "hp": hp_value, "color_type": color_type,
                    "color_confidence": color_conf,
                    "card_number": ocr_card_num, "set_total": ocr_set_total,
                    "n_candidates_before": len(candidates),
                    "dino_sanity": dino_score,
                },
            }
            _apply_variant_detection(result, image_path, detect_variants=detect_variants,
                                 precomputed_stamp_set_id=precomputed_stamp_set_id,
                                 precomputed_stamp_match_score=precomputed_stamp_match_score,
                                 precomputed_stamp_texts=precomputed_stamp_texts)
            _cache_store(cache_key, result)
            return result
        else:
            logger.info(
                "v2 step2b: card_number match %s rejected — dino=%.3f, "
                "best_other=%.3f (likely OCR misread)",
                card_num_match, dino_score, best_other_score,
            )
            card_num_match = None

    # -----------------------------------------------------------------------
    # Step 3/4: Combined DINOv2 + attack scoring for candidate disambiguation
    # -----------------------------------------------------------------------
    ref_match_result = None  # Low-confidence ref match saved for comparison
    name_path_failed = False  # Set when name-based path produces no acceptable result
    if candidates:
        # Single candidate: quick DINOv2 sanity check
        if len(candidates) == 1:
            only_cid = candidates[0]
            dino_check = _dino_dot_product_against_refs(image_path, [only_cid], query_embedding=_precomputed_dino_embedding)
            dino_score = dino_check[0][1] if dino_check else 0.0

            if dino_score >= 0.30:
                explanation = (
                    f"v2: single candidate match: name OCR={ocr_name!r}, hp={hp_value}, "
                    f"type={color_type} -> {only_cid} (dino={dino_score:.3f})"
                )
                result = {
                    "card_id": only_cid,
                    "confidence": max(ocr_conf, 0.70),
                    "method": "v2_single_candidate",
                    "explanation": explanation,
                    "raw_response": {
                        "ocr_name": ocr_name, "ocr_confidence": ocr_conf,
                        "hp": hp_value, "color_type": color_type,
                        "color_confidence": color_conf, "n_candidates": 1,
                        "dino_sanity": dino_score,
                    },
                }
                _apply_variant_detection(result, image_path, detect_variants=detect_variants,
                                 precomputed_stamp_set_id=precomputed_stamp_set_id,
                                 precomputed_stamp_match_score=precomputed_stamp_match_score,
                                 precomputed_stamp_texts=precomputed_stamp_texts)
                _cache_store(cache_key, result)
                return result
            else:
                logger.warning("v2: REJECTED single candidate %s — DINOv2 %.3f too low",
                               only_cid, dino_score)
                name_path_failed = True

        # Multiple candidates: combined DINOv2 + attack scoring
        elif len(candidates) >= 2:
            # Always use attack OCR for disambiguation between same-name candidates.
            # DINOv2 alone can't reliably distinguish printings of the same card
            # (scores differ by <0.05). Attacks uniquely identify the set printing.
            effective_attacks = _precomputed_attacks
            combined_results = _score_candidates_combined(image_path, candidates, query_embedding=_precomputed_dino_embedding, precomputed_attacks=effective_attacks, type_detected=use_type, type_confidence=color_conf, ocr_card_num=ocr_card_num, ocr_set_total=ocr_set_total)
            if combined_results:
                best_cid, best_score, best_detail = combined_results[0]
                alt_list = [(cid, score) for cid, score, _ in combined_results[1:4]]
                alt_str = ", ".join(f"{cid} ({s:.0%})" for cid, s in alt_list)

                n_cand = len(candidates)
                if n_cand <= 3:
                    effective_threshold = 0.35
                elif n_cand <= 10:
                    effective_threshold = 0.45
                else:
                    effective_threshold = 0.50

                # When OCR name confidence is high, the candidates are already
                # well-constrained by name — lower the threshold since we trust
                # the candidate set.  Also lower when the top match has clear
                # separation from 2nd place (the ranking is reliable).
                gap = 0.0
                if len(combined_results) >= 2:
                    gap = best_score - combined_results[1][1]
                if ocr_conf >= 0.80:
                    # High-confidence OCR name: candidates are trustworthy,
                    # the combined score just needs to pick the best one.
                    # Very high OCR conf (>=0.90) lowers further — the name
                    # is almost certainly correct so even low DINOv2 scores
                    # (WotC cards in orange sleeves) should be accepted.
                    if ocr_conf >= 0.90:
                        # Very high OCR conf — name is almost certainly correct.
                        # Accept the best candidate regardless of DINOv2 score.
                        # Stamped/sleeved cards can have very low DINOv2 similarity
                        # to clean reference images (e.g. 0.31 for Buffer Piece).
                        effective_threshold = 0.0
                    else:
                        effective_threshold = min(effective_threshold, 0.35)
                    logger.info("v2: high OCR conf %.2f -> lowered threshold to %.2f",
                                ocr_conf, effective_threshold)
                if gap >= 0.04:
                    # Clear separation between 1st and 2nd: ranking is reliable
                    effective_threshold = min(effective_threshold, 0.38)
                    logger.info("v2: clear gap %.3f between top two -> threshold %.2f",
                                gap, effective_threshold)

                if best_score >= effective_threshold:
                    attack_names = best_detail.get("matched_attacks", [])
                    explanation = (
                        f"v2: name OCR={ocr_name!r}, hp={hp_value}, "
                        f"type={color_type}, {n_cand} candidates, "
                        f"combined={best_score:.3f} (dino={best_detail['dino_score']:.3f}, "
                        f"atk={best_detail['attack_score']:.3f})"
                    )
                    if attack_names:
                        explanation += f", attacks={attack_names}"
                    if alt_str:
                        explanation += f". Alts: {alt_str}"

                    ref_match_result = {
                        "card_id": best_cid,
                        "confidence": float(best_score),
                        "method": "v2_ref_match",
                        "explanation": explanation,
                        "raw_response": {
                            "ocr_name": ocr_name, "ocr_confidence": ocr_conf,
                            "ocr_raw": ocr_raw, "hp": hp_value,
                            "color_type": color_type, "color_confidence": color_conf,
                            "attack_names": attack_names, "n_candidates": n_cand,
                            "combined_results": [(c, round(s, 4), d) for c, s, d in combined_results[:5]],
                        },
                    }
                    # If DINOv2 score is low (< 0.60), OCR name might be wrong.
                    # Save result but don't return yet — also try attack path.
                    # EXCEPTION: if OCR confidence is very high (>0.90), trust the
                    # name path. The attack path can pick the wrong card when multiple
                    # cards share the same attack (e.g., "Psychic Pulse" on 4 cards).
                    # Japanese OCR: accept low DINOv2 scores since domain gap is expected
                    _ja_ocr = ocr_raw and isinstance(ocr_raw, str) and ocr_raw.startswith("[JP]")
                    if _ja_ocr and len(candidates) <= 3:
                        logger.info("v2: Japanese OCR with %d candidates, accepting dino=%.3f",
                                    len(candidates), best_detail['dino_score'])
                        _apply_variant_detection(ref_match_result, image_path, detect_variants=detect_variants,
                                 precomputed_stamp_set_id=precomputed_stamp_set_id,
                                 precomputed_stamp_match_score=precomputed_stamp_match_score,
                                 precomputed_stamp_texts=precomputed_stamp_texts)
                        _cache_store(cache_key, ref_match_result)
                        return ref_match_result
                    elif best_detail['dino_score'] < 0.60 and ocr_conf < 0.90:
                        logger.info("v2: ref_match dino=%.3f < 0.60 and ocr_conf=%.2f < 0.90, "
                                    "will also try attack path",
                                    best_detail['dino_score'], ocr_conf)
                    else:
                        _apply_variant_detection(ref_match_result, image_path, detect_variants=detect_variants,
                                 precomputed_stamp_set_id=precomputed_stamp_set_id,
                                 precomputed_stamp_match_score=precomputed_stamp_match_score,
                                 precomputed_stamp_texts=precomputed_stamp_texts)
                        _cache_store(cache_key, ref_match_result)
                        return ref_match_result
                else:
                    logger.info("v2: best combined %.4f < %.2f, falling to ensemble",
                                best_score, effective_threshold)
                    name_path_failed = True

    # -----------------------------------------------------------------------
    # Step 5: Attack-based identification
    # Try attack OCR when: (a) name OCR failed entirely, (b) the OCR-based
    # candidate match scored low (< 0.60), suggesting the OCR name may be wrong,
    # or (c) the name-based path had candidates but couldn't produce an
    # acceptable result (e.g. wrong OCR name from rotation fallback).
    # Attack OCR has 92% recall — much more reliable than DINOv2 global search.
    # -----------------------------------------------------------------------
    attack_result = None
    if not ocr_name or ref_match_result is not None or name_path_failed:
        # When the name path failed (e.g. wrong name from rotation fallback),
        # use the original un-rotated image for attack OCR — the rotated image
        # won't have readable attack text.
        attack_image = original_image_path if name_path_failed else image_path
        logger.info("v2 step5: trying attack-based identification (name_path_failed=%s, image=%s)",
                     name_path_failed, Path(attack_image).name)
        try:
            from cardprice.ml.attack_ocr import identify_by_attacks
            if _precomputed_attacks is not None:
                atk_results = identify_by_attacks(
                    attack_image, precomputed_ocr_candidates=_precomputed_attacks,
                    type_detected=use_type, type_confidence=color_conf,
                    hp_value=str(hp_value) if hp_value else None,
                )
            else:
                with _ocr_lock:
                    atk_results = identify_by_attacks(
                        attack_image,
                        type_detected=use_type, type_confidence=color_conf,
                        hp_value=str(hp_value) if hp_value else None,
                    )
            if atk_results:
                atk_candidate_ids = [cid for cid, _s in atk_results[:50]]

                # HP filtering: when HP was detected, prune attack candidates
                # that have a different HP. This dramatically reduces the
                # candidate set (e.g., 50 "Triple Smash" cards → 5 with HP=60).
                if hp_value and len(atk_candidate_ids) >= 5:
                    _structured = _load_structured_attacks()
                    hp_matched = []
                    for cid in atk_candidate_ids:
                        base_cid = cid.split("/")[0] if "/" in cid else cid
                        s_data = _structured.get(base_cid) if _structured else None
                        if s_data:
                            card_hp = s_data.get("hp")
                            if card_hp and str(card_hp) == str(hp_value):
                                hp_matched.append(cid)
                            # Also keep if no HP data (don't exclude unknowns)
                            elif not card_hp:
                                hp_matched.append(cid)
                        else:
                            hp_matched.append(cid)  # keep unknowns
                    if len(hp_matched) >= 2:
                        logger.info("v2 step5: HP filter (hp=%s): %d/%d candidates",
                                    hp_value, len(hp_matched), len(atk_candidate_ids))
                        atk_candidate_ids = hp_matched

                # Era filtering: if page_era is known, prefer candidates
                # from the same era. Keep era-matched candidates first,
                # but fall back to all candidates if too few match.
                if page_era:
                    from cardprice.ml.page_context import _era_for_set, _extract_set_id, _eras_compatible
                    era_matched = [
                        cid for cid in atk_candidate_ids
                        if _eras_compatible(_era_for_set(_extract_set_id(cid)) or "", page_era)
                    ]
                    # Use era filter when: enough candidates OR many total
                    # candidates (indistinguishable by DINOv2).
                    if era_matched and (len(era_matched) >= 3 or len(atk_candidate_ids) >= 20):
                        logger.info("v2 step5: era filter %s: %d/%d candidates",
                                    page_era, len(era_matched), len(atk_candidate_ids))
                        atk_candidate_ids = era_matched

                # Track whether era filtering actually reduced the set
                era_filtered = page_era and len(atk_candidate_ids) < len(atk_results[:50])
                combined_results = _score_candidates_combined(attack_image, atk_candidate_ids, query_embedding=_precomputed_dino_embedding, precomputed_attacks=_precomputed_attacks, type_detected=use_type, type_confidence=color_conf, ocr_card_num=ocr_card_num, ocr_set_total=ocr_set_total)
                if combined_results:
                    best_cid, best_score, best_detail = combined_results[0]
                    # Boost confidence when era filtering significantly reduced
                    # the candidate set (strong prior from page context)
                    if era_filtered:
                        best_score = min(best_score + 0.10, 1.0)
                    # Penalize when many candidates share the same attacks
                    # (low discrimination — DINOv2 picks randomly among 50 Rattatas)
                    if len(atk_candidate_ids) >= 30 and len(combined_results) >= 2:
                        score_gap = combined_results[0][1] - combined_results[1][1]
                        if score_gap < 0.05:
                            best_score *= 0.85  # moderate penalty for low discrimination
                            logger.info("v2 step5: %d candidates, gap=%.3f -> penalty to %.3f",
                                        len(atk_candidate_ids), score_gap, best_score)
                    if best_score >= 0.35:
                        alt_list = [(cid, score) for cid, score, _ in combined_results[1:4]]
                        alt_str = ", ".join(f"{cid} ({s:.0%})" for cid, s in alt_list)
                        attack_names = best_detail.get("matched_attacks", [])
                        explanation = (
                            f"v2: attack OCR -> {len(atk_candidate_ids)} candidates, "
                            f"combined={best_score:.3f} (dino={best_detail['dino_score']:.3f}, "
                            f"atk={best_detail['attack_score']:.3f})"
                        )
                        if attack_names:
                            explanation += f", attacks={attack_names}"
                        if alt_str:
                            explanation += f". Alts: {alt_str}"

                        # When name path failed (bad rotation OCR), the attack
                        # path is recovering from a known-bad state. Boost
                        # confidence so it beats the unreliable ensemble.
                        if name_path_failed and best_detail.get("attack_score", 0) > 0:
                            best_score = min(best_score + 0.15, 1.0)
                            logger.info("v2 step5: name_path_failed boost -> %.3f", best_score)

                        attack_result = {
                            "card_id": best_cid,
                            "confidence": float(best_score),
                            "method": "v2_attack_fallback",
                            "explanation": explanation,
                            "raw_response": {
                                "ocr_name": ocr_name, "hp": hp_value,
                                "color_type": color_type,
                                "attack_candidates": atk_results[:5],
                                "combined_results": [(c, round(s, 4), d)
                                                     for c, s, d in combined_results[:5]],
                            },
                        }
                        logger.info("v2: attack fallback -> %s (combined=%.3f)",
                                    best_cid, best_score)
        except Exception as e:
            logger.warning("v2 step5: attack fallback failed: %s", e)

    # -----------------------------------------------------------------------
    # Step 5b: Stamp-set candidate rescue
    # -----------------------------------------------------------------------
    # When name OCR failed AND attack result is suspect (no result OR low
    # DINOv2 score), try a DINOv2-only search constrained to the set whose
    # stamp text we DID read. This recovers cards like Metagross δ where the
    # name "Metagross" is unreadable on the foiled stamped art but the stamp
    # text "DELTA SPECIES" was OCR'd successfully (-> ex11).
    stamp_rescue_result = None
    attack_dino_too_low = (
        attack_result is None
        or (
            attack_result.get("raw_response", {})
            .get("combined_results", [(None, 0, {"dino_score": 0})])[0][2]
            .get("dino_score", 0) < 0.50
        )
    )
    if (
        not ocr_name
        and precomputed_stamp_set_id
        and attack_dino_too_low
    ):
        try:
            from cardprice.ml.ref_matcher import _load_ref_embeddings
            import numpy as np

            # Compute scan embedding (use precomputed if available)
            if _precomputed_dino_embedding is not None:
                scan_emb = _precomputed_dino_embedding
            else:
                from cardprice.ml.dino_matcher import extract_embedding
                scan_emb = extract_embedding(image_path)
            scan_emb = scan_emb / (np.linalg.norm(scan_emb) or 1.0)

            ref = _load_ref_embeddings()
            set_keys = [k for k in ref if k.startswith(f"{precomputed_stamp_set_id}-")]
            if set_keys:
                mats = np.array([ref[k] for k in set_keys])
                mats = mats / np.linalg.norm(mats, axis=1, keepdims=True)
                sims = mats @ scan_emb
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                best_cid = set_keys[best_idx]
                if best_sim >= 0.40:
                    stamp_rescue_result = {
                        "card_id": best_cid,
                        "confidence": min(0.60 + (best_sim - 0.40) * 1.5, 0.92),
                        "method": "v2_stamp_set_rescue",
                        "explanation": (
                            f"v2: name OCR failed; stamp text -> {precomputed_stamp_set_id}; "
                            f"DINOv2 best in set = {best_cid} (sim={best_sim:.3f})"
                        ),
                        "raw_response": {
                            "ocr_name": None,
                            "stamp_set_id": precomputed_stamp_set_id,
                            "dino_sim": best_sim,
                        },
                    }
                    logger.info(
                        "v2 step5b: stamp-set rescue -> %s (sim=%.3f, set=%s)",
                        best_cid, best_sim, precomputed_stamp_set_id,
                    )
        except Exception as e:
            logger.warning("v2 step5b: stamp-set rescue failed: %s", e)

    # -----------------------------------------------------------------------
    # Step 6: Ensemble fallback (last resort)
    # -----------------------------------------------------------------------
    # When name path failed (bad rotation OCR), use original image and
    # discard the bogus OCR name so ensemble isn't penalized for mismatch.
    ensemble_image = original_image_path if name_path_failed else image_path
    if name_path_failed:
        logger.info("v2: name_path_failed — discarding bogus OCR name %r for ensemble/comparison",
                     ocr_name)
        ocr_name = None
        ocr_conf = 0.0
    logger.info(
        "v2 step6: running ensemble (ocr_name=%r, candidates=%d)",
        ocr_name, len(candidates),
    )
    fallback = identify_card_ensemble(ensemble_image, session=session,
                                      _dino_embedding=_precomputed_dino_embedding,
                                      _clip_embedding=_precomputed_clip_embedding)
    fallback_conf = fallback.get("confidence", 0.0)

    # Pick best among ref_match_result (if pending), attack_result, and ensemble.
    # When OCR name is valid, STRONGLY prefer name-matched results over
    # unconstrained ensemble (which ignores the name entirely).
    # When page_era is known, give era-matched results a 0.10 bonus so they
    # beat ensemble results from wrong eras.
    best_alt = None
    best_alt_conf = fallback_conf

    # If OCR read a valid name, penalize ensemble results that don't match it
    if ocr_name and ocr_conf >= 0.70:
        fallback_cid = fallback.get("card_id", "")
        if fallback_cid:
            from cardprice.ml.ref_matcher import get_candidate_card_ids
            name_cids = set(get_candidate_card_ids(ocr_name))
            if fallback_cid not in name_cids:
                # Ensemble picked a card with a different name — heavily penalize
                logger.info("v2: ensemble picked %s which doesn't match OCR name %r, penalizing",
                            fallback_cid, ocr_name)
                best_alt_conf -= 0.30

    if page_era:
        from cardprice.ml.page_context import _era_for_set, _extract_set_id, _eras_compatible
        fallback_cid = fallback.get("card_id", "")
        fallback_era = _era_for_set(_extract_set_id(fallback_cid)) if fallback_cid else None
        if fallback_era and not _eras_compatible(fallback_era, page_era):
            best_alt_conf -= 0.10  # penalize wrong-era ensemble result
    # When the name path produced a result (ref_match_result) AND OCR name
    # confidence is reasonable, penalize attack results that picked a card
    # with a DIFFERENT name.  Common attacks like "Tackle" appear on hundreds
    # of cards; single-attack matches mislead the attack path into picking
    # wrong species (e.g., Voltorb instead of Misty's Horsea).
    if attack_result and ref_match_result and ocr_name and ocr_conf >= 0.70:
        atk_cid = attack_result.get("card_id", "")
        if atk_cid:
            from cardprice.ml.ref_matcher import get_candidate_card_ids as _get_cids
            name_cids = set(_get_cids(ocr_name))
            if atk_cid not in name_cids:
                logger.info(
                    "v2: attack result %s doesn't match OCR name %r (conf=%.2f), "
                    "penalizing by 0.30",
                    atk_cid, ocr_name, ocr_conf,
                )
                attack_result["confidence"] -= 0.30

    # Stamp-set rescue is highly trusted when triggered (it only fires when
    # name OCR failed AND attack DINOv2 was weak, so it has the cleanest
    # signal we have for that card). Boost its weight against ensemble.
    if stamp_rescue_result:
        # Penalize ensemble harder than the regular case if it picked a card
        # from a DIFFERENT set than the stamp said.
        rescue_set = stamp_rescue_result["raw_response"]["stamp_set_id"]
        fallback_cid = fallback.get("card_id", "")
        if fallback_cid and not fallback_cid.startswith(f"{rescue_set}-"):
            best_alt_conf -= 0.25

    for candidate in [ref_match_result, attack_result, stamp_rescue_result]:
        if candidate and candidate["confidence"] > best_alt_conf:
            best_alt = candidate
            best_alt_conf = candidate["confidence"]
    if best_alt:
        logger.info("v2: %s (%.3f) > ensemble (%.3f)",
                     best_alt["method"], best_alt_conf, fallback_conf)
        _apply_variant_detection(best_alt, image_path, detect_variants=detect_variants,
                                 precomputed_stamp_set_id=precomputed_stamp_set_id,
                                 precomputed_stamp_match_score=precomputed_stamp_match_score,
                                 precomputed_stamp_texts=precomputed_stamp_texts)
        _cache_store(cache_key, best_alt)
        return best_alt

    fallback["method"] = f"v2_fallback({fallback.get('method', 'ensemble')})"

    # Reject low-confidence fallback results to avoid false positives.
    # Ensemble fallback is the least reliable path — wrong guesses typically
    # land in the 0.39-0.69 range while correct ones are >= 0.70.
    # Exception: Japanese OCR results have lower DINOv2 scores due to domain gap
    # between Japanese card photos and English reference images.
    min_accept = _V2_FALLBACK_MIN_ACCEPT
    # Japanese OCR results have lower DINOv2 scores due to domain gap
    # between Japanese card photos and English reference images.
    # Check multiple signals for Japanese identification.
    _is_ja = False
    if ocr_raw and isinstance(ocr_raw, str) and "[JP]" in ocr_raw:
        _is_ja = True
    if ocr_name and isinstance(ocr_name, str) and ocr_name.startswith("[JP]"):
        _is_ja = True
    # Also check the fallback raw_response for Japanese OCR markers
    _fb_raw = fallback.get("raw_response", {})
    if isinstance(_fb_raw, dict) and _fb_raw.get("ocr_raw", "").startswith("[JP]"):
        _is_ja = True
    if _is_ja:
        min_accept = 0.35
        logger.info("v2: Japanese OCR detected, lowering acceptance threshold to 0.35")
    # Preserve the rejected card_id so page_context can restore it if era matches.
    if fallback_conf < min_accept:
        logger.info(
            "v2: rejecting fallback result %s (confidence=%.3f < %.2f threshold)",
            fallback.get("card_id"), fallback_conf, _V2_FALLBACK_MIN_ACCEPT,
        )
        raw = fallback.get("raw_response", {})
        raw["rejected_card_id"] = fallback.get("card_id")
        raw["rejected_confidence"] = fallback_conf
        fallback["raw_response"] = raw
        fallback["card_id"] = None
        fallback["method"] = "unidentified"
        fallback["confidence"] = fallback_conf  # preserve original score for diagnostics

    fallback["explanation"] = (
        f"v2 fallback: OCR name={ocr_name!r} yielded {len(candidates)} candidates "
        f"but DINOv2 ref-match was insufficient. "
        + (fallback.get("explanation") or "")
    )
    # Preserve v2 signal info in raw_response
    raw = fallback.get("raw_response", {})
    raw["v2_signals"] = {
        "ocr_name": ocr_name,
        "ocr_confidence": ocr_conf,
        "hp": hp_value,
        "color_type": color_type,
        "color_confidence": color_conf,
        "n_candidates": len(candidates),
    }
    fallback["raw_response"] = raw

    _apply_variant_detection(fallback, image_path, detect_variants=detect_variants,
                                 precomputed_stamp_set_id=precomputed_stamp_set_id,
                                 precomputed_stamp_match_score=precomputed_stamp_match_score,
                                 precomputed_stamp_texts=precomputed_stamp_texts)

    # -----------------------------------------------------------------------
    # Step 7: Claude vision fallback (optional, last resort)
    # -----------------------------------------------------------------------
    if (use_claude_vision_fallback
            and fallback.get("method") == "unidentified"):
        logger.info("v2 step7: trying Claude vision fallback for %s", image_path)
        try:
            from cardprice.ml.claude_vision import identify_card_vision_fallback
            vision_result = identify_card_vision_fallback(
                image_path, session=session,
            )
            if vision_result.get("card_id"):
                logger.info(
                    "v2 step7: Claude vision identified %s -> %s (conf=%.2f)",
                    image_path, vision_result["card_id"],
                    vision_result["confidence"],
                )
                _apply_variant_detection(
                    vision_result, image_path,
                    detect_variants=detect_variants,
                    precomputed_stamp_set_id=precomputed_stamp_set_id,
                    precomputed_stamp_match_score=precomputed_stamp_match_score,
                    precomputed_stamp_texts=precomputed_stamp_texts,
                )
                _cache_store(cache_key, vision_result)
                return vision_result
            logger.info("v2 step7: Claude vision did not identify %s", image_path)
        except Exception as e:
            logger.error("v2 step7: Claude vision fallback failed: %s", e)

    _cache_store(cache_key, fallback)
    return fallback


def _name_ocr_worker(image_path: str) -> dict:
    """Run name+HP OCR for one card. Used by identify_page_v2 batch pipeline.

    Skips the Japanese OCR fallback (18s/card) — it almost never helps and
    destroys batch throughput. Japanese cards are handled in pass 3 reranking
    when page context is available.
    """
    result = {}

    # Skip card backs
    try:
        from cardprice.ml.card_segmenter import is_card_back as _is_cb
        if _is_cb(image_path):
            return {"ocr_name": None, "ocr_conf": 0.0, "ocr_raw": None,
                    "hp_value": None, "is_card_back": True}
    except Exception:
        pass

    try:
        # Run English OCR only — no Japanese fallback.
        name, conf, raw, hp = _paddle_ocr_name_and_hp(image_path)
        if name and len(name) >= 2:
            result["ocr_name"] = name
            result["ocr_conf"] = conf
            result["ocr_raw"] = raw
            result["hp_value"] = hp
        else:
            result["ocr_name"] = None
            result["ocr_conf"] = 0.0
            result["ocr_raw"] = raw
            result["hp_value"] = hp
    except Exception:
        result["ocr_name"] = None
        result["ocr_conf"] = 0.0
        result["ocr_raw"] = None
        result["hp_value"] = None

    # Stamp text OCR: crop bottom-right of artwork where stamps appear.
    # Binary signal: if OCR finds readable text there, the card is stamped.
    # Regular cards have artwork (no text) in this region.
    # Stamped cards have the set name (e.g. "POWER KEEPERS") as text.
    # After OCR, fuzzy match against known EX-era stamp names for set ID.
    # This runs on the same ONNX engine, same thread — negligible overhead.
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            # Stamp region: bottom-right of artwork (y: 35-55%, x: 55-90%)
            stamp_crop = img[int(h*0.35):int(h*0.55), int(w*0.55):int(w*0.90)]
            # Upscale 3x for better OCR on small text
            stamp_up = cv2.resize(stamp_crop, (stamp_crop.shape[1]*3, stamp_crop.shape[0]*3))
            from cardprice.ml.ocr_matcher import get_rapid_engine as _get_re
            stamp_result, _ = _get_re()(stamp_up)
            if stamp_result:
                # Filter out artist credits and other non-stamp text
                _NON_STAMP = {"illus", "arita", "sugimori", "nishida", "imakuni",
                              "komiya", "tokiya", "mitsuhiro", "atsuko", "ken",
                              "kagemaru", "himeno", "masakazu", "ryo", "kouki",
                              "saya", "planeta", "cr.", "5ban", "graphics"}
                texts = []
                for _, t, c in stamp_result:
                    if float(c) < 0.4 or len(t.strip()) < 3:
                        continue
                    # Skip if any word matches artist name
                    words = t.lower().split()
                    if any(w.strip(".,") in _NON_STAMP for w in words):
                        continue
                    # Skip if looks like artist credit (contains "illus" anywhere)
                    if "illus" in t.lower() or "ilus" in t.lower():
                        continue
                    texts.append(t)
                if texts:
                    result["has_stamp_text"] = True
                    result["stamp_texts"] = texts
                    # Fuzzy match against known EX stamp names. Use min_score=0
                    # to capture the raw best score; the caller in
                    # _apply_variant_detection applies a context-aware
                    # threshold (lower bar when matched_set == card's set).
                    matched_set, match_score = _fuzzy_match_stamp_text(texts, min_score=0)
                    # Always store the candidate set + score so context-aware
                    # logic can run later (Nidoqueen scores 48 against
                    # DRAGON FRONTIERS — below the 60 hard threshold but
                    # acceptable when card is identified as ex15-7).
                    if matched_set is None and match_score > 0:
                        # extractOne returned a candidate but below min_score=0
                        # threshold logic. Recompute to get the candidate set.
                        from rapidfuzz import fuzz, process
                        combined = " ".join(texts).upper().strip()
                        m = process.extractOne(combined, _STAMP_NAME_CHOICES,
                                                scorer=fuzz.partial_ratio)
                        if m:
                            matched_set = _STAMP_NAME_TO_SET[m[0]]
                    result["stamp_set_id"] = matched_set
                    result["stamp_set_name"] = EX_STAMP_NAMES.get(matched_set) if matched_set else None
                    result["stamp_match_score"] = match_score
    except Exception:
        pass

    return result


def _identify_card_worker(image_path, precomputed_ocr, dino_embedding_list=None):
    """Worker function for ProcessPoolExecutor — runs in a separate process.

    Each process loads its own OCR models (PaddleOCR, EasyOCR).
    DINOv2 embedding is pre-computed and passed in as a list (for pickling),
    so no GPU is needed in workers.
    """
    import numpy as np
    try:
        dino_emb = np.array(dino_embedding_list, dtype=np.float32) if dino_embedding_list else None
        return identify_card_v2(
            image_path, session=None,
            _precomputed_ocr=precomputed_ocr,
            _precomputed_dino_embedding=dino_emb,
        )
    except Exception as e:
        return {
            "card_id": None, "confidence": 0.0,
            "method": "v2_error",
            "explanation": f"Worker error: {e}",
            "raw_response": {},
        }


def identify_page_v2(card_image_paths, session=None,
                     detect_variants=False,
                     use_claude_vision_fallback=False,
                     correct_perspective=False):
    """V2 page identification: runs identify_card_v2 on each card, then
    applies page context reranking for low-confidence results.

    Pipeline:
        1. Run identify_card_v2 for each card in parallel.
        2. Build page context from high-confidence results (set/era inference).
        3. For low-confidence cards, re-run with page context boosting
           candidates from the inferred set.
        4. (Optional) For still-unidentified cards, use Claude vision API
           as a last resort (e.g. for Japanese cards that RapidOCR can't read).

    Args:
        card_image_paths: List of paths to individual card segment images.
        session: Optional SQLAlchemy DB session.
        detect_variants: Whether to run variant detection (holo, reverse holo,
            stamp checks) after identification. Default False for speed —
            variant detection can be triggered separately on-demand.
        use_claude_vision_fallback: If True, send unidentified cards to Claude
            vision API for identification. Costs API credits. Default False.
        correct_perspective: If True, apply perspective correction to each
            card image before identification. Default False.

    Returns:
        List of result dicts (same format as identify_card_v2), one per card.
    """
    from cardprice.ml.page_context import identify_page_context

    if not card_image_paths:
        return []

    # -------------------------------------------------------------------
    # Optional: apply perspective correction to each card image
    # -------------------------------------------------------------------
    if correct_perspective:
        try:
            import cv2 as _cv2_corr
            from cardprice.ml.card_corrector import correct_card_image
            corrected_paths = []
            for p in card_image_paths:
                p_str = str(p)
                try:
                    _img = _cv2_corr.imread(p_str)
                    if _img is not None:
                        _out = correct_card_image(_img)
                        _cp = p_str + '_corrected.png'
                        _cv2_corr.imwrite(_cp, _out)
                        corrected_paths.append(_cp)
                    else:
                        corrected_paths.append(p_str)
                except Exception as e:
                    logger.warning("identify_page_v2: perspective correction failed for %s: %s", p_str, e)
                    corrected_paths.append(p_str)
            card_image_paths = corrected_paths
            logger.info("identify_page_v2: perspective correction applied to %d cards", len(corrected_paths))
        except Exception as e:
            logger.warning("identify_page_v2: perspective correction import failed: %s", e)

    # -------------------------------------------------------------------
    # Limit OpenMP/MKL threads to reduce CPU over-subscription.
    # With 3 parallel threads each running ML inference (PaddleOCR,
    # EasyOCR, DINOv2/CLIP), uncapped OpenMP defaults (8-16 threads)
    # cause 3×16=48 threads fighting over ~16 cores.  Capping at 4
    # gives 3×4=12 threads, well within core count.
    # setdefault() respects any user-set override.
    # -------------------------------------------------------------------
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

    n_cards = len(card_image_paths)
    logger.info("identify_page_v2: processing %d cards", n_cards)

    # -----------------------------------------------------------------------
    # Pass 1a: Batch pre-compute ALL expensive operations in parallel.
    #
    # Three concurrent thread pools, each on non-competing resources:
    #   Pool 1: RapidOCR name + HP     (3 parallel threads, ONNX thread-safe)
    #   Pool 2: DINOv2 batch embed     (GPU, single forward pass)
    #   Pool 3: Color detection        (CPU, no lock, pure OpenCV)
    #   Pool 4: Attack OCR             (CPU, dispatched as name OCR completes)
    #
    # RapidOCR (ONNX Runtime) is thread-safe and benefits from parallel
    # execution.  Name OCR runs 3 cards concurrently for ~2x throughput.
    # -----------------------------------------------------------------------
    import time as _time

    # Eagerly import modules that threads will use to avoid circular import
    # issues when multiple threads try to import torchvision simultaneously.
    from cardprice.ml.attack_ocr import extract_attack_names_paddle as _extract_attacks_paddle
    from cardprice.ml.dino_matcher import extract_embedding_batch as _extract_batch
    from cardprice.ml.preprocess import preprocess_for_matching as _preprocess

    t_precomp_start = _time.time()

    precomputed = [None] * n_cards
    attack_results = [None] * n_cards  # None = not computed, [] = computed but empty
    dino_embeddings = [None] * n_cards
    clip_embeddings = [None] * n_cards
    stamp_texts = {}  # card_idx -> list of OCR text strings found in stamp region

    # Shared pool for attack OCR — tasks are submitted by the name OCR
    # thread as soon as each card's name confidence is known.
    _attack_pool = ThreadPoolExecutor(max_workers=4)
    _attack_futures = []

    def _submit_attack_ocr(card_idx, path):
        """Submit attack OCR for a single card to the shared pool."""
        def _run():
            try:
                attack_results[card_idx] = _extract_attacks_paddle(str(path))
            except Exception as e:
                logger.warning("identify_page_v2: attack OCR card %d failed: %s",
                               card_idx, e)
        fut = _attack_pool.submit(_run)
        _attack_futures.append(fut)

    def _batch_name_ocr():
        """Thread 1: RapidOCR name + HP for all cards.

        Runs name OCR for all cards in parallel using a thread pool
        (RapidOCR/ONNX Runtime is thread-safe).  After each card's name
        OCR completes, dispatches attack OCR if needed.

        The _hold_lock=False argument tells _run_name_and_hp to skip
        acquiring _ocr_lock since RapidOCR handles concurrency internally.
        """
        t0 = _time.time()

        # Ensure RapidOCR engine is initialized before spawning threads
        from cardprice.ml.ocr_matcher import get_rapid_engine as _ensure_rapid
        _ensure_rapid()

        def _name_ocr_one_postprocess(i, path, ocr_data):
            """Postprocess OCR result from worker process: store + dispatch attacks."""
            if not ocr_data:
                ocr_data = {"ocr_name": None, "ocr_conf": 0.0, "ocr_raw": None, "hp_value": None}
            precomputed[i] = ocr_data

            if ocr_data.get("is_card_back"):
                attack_results[i] = []
                logger.info("identify_page_v2: skipping OCR for card %d (card back)", i)
                return

            # Always dispatch attack OCR. Earlier we skipped this when name
            # conf >= 0.85, but that broke disambiguation for common Pokemon
            # with many printings (e.g. Metang has 23 printings; without
            # attacks, DINOv2 picked tk2a-5 over the correct ex11-49 Metang δ
            # because the visual scores were within 0.02). The cost is ~1-2s
            # extra per page, well within the 10s/page budget.
            _submit_attack_ocr(i, path)

        # Run name OCR in parallel threads. ONNX Runtime releases the GIL
        # during inference, giving ~30% speedup. The big win is skipping the
        # 18s/card Japanese OCR fallback in _name_ocr_worker.
        with ThreadPoolExecutor(max_workers=min(n_cards, 9)) as name_pool:
            futures = {
                name_pool.submit(_name_ocr_worker, str(path)): i
                for i, path in enumerate(card_image_paths)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    ocr_data = fut.result()
                    _name_ocr_one_postprocess(i, card_image_paths[i], ocr_data)
                except Exception as e:
                    logger.warning("identify_page_v2: name OCR thread %d failed: %s", i, e)

        logger.info("identify_page_v2: name OCR thread done in %.1fs",
                     _time.time() - t0)

    def _batch_color_detect():
        """Thread 3: Color/type detection for all cards (no lock, pure OpenCV).

        Runs all cards in parallel using a small thread pool.  Results are
        merged into the precomputed dicts after completion.
        """
        t0 = _time.time()

        def _color_one(idx, path):
            try:
                ctype, cconf = _run_color_detect(str(path))
                return idx, ctype, cconf
            except Exception as e:
                logger.warning("identify_page_v2: color card %d failed: %s", idx, e)
                return idx, None, 0.0

        # Color detection is fast (~20ms/card) — 4 threads is plenty
        with ThreadPoolExecutor(max_workers=4) as color_pool:
            futures = [
                color_pool.submit(_color_one, i, p)
                for i, p in enumerate(card_image_paths)
            ]
            for fut in as_completed(futures):
                idx, ctype, cconf = fut.result()
                if precomputed[idx] is None:
                    precomputed[idx] = {}
                precomputed[idx]["color_type"] = ctype
                precomputed[idx]["color_conf"] = cconf

        logger.info("identify_page_v2: color detect thread done in %.1fs",
                     _time.time() - t0)

    def _batch_stamp_ocr():
        """Thread 4: Quick stamp text OCR on artwork region for all cards.

        Crops the bottom-right of the artwork area where EX-era set stamps
        appear (~55-90% x, 35-55% y) and runs RapidOCR to detect any text.
        A non-empty result is a strong signal that a stamp is present.

        This runs in parallel with name OCR and DINOv2 — adds zero latency.
        Results are consumed after identification to set variant=ex_set_stamp
        for cards from stamped sets (ex7-ex16).
        """
        import cv2 as _cv2_stamp
        t0 = _time.time()
        try:
            from cardprice.ml.ocr_matcher import get_rapid_engine
            _stamp_engine = get_rapid_engine()
        except Exception as e:
            logger.warning("identify_page_v2: stamp OCR engine init failed: %s", e)
            return

        for i, path in enumerate(card_image_paths):
            try:
                img = _cv2_stamp.imread(str(path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                # Crop artwork bottom-right where EX stamps appear
                stamp_crop = img[int(h * 0.35):int(h * 0.55),
                                 int(w * 0.55):int(w * 0.90)]
                # Upscale 3x for better OCR on small stamp text
                stamp_up = _cv2_stamp.resize(
                    stamp_crop,
                    (stamp_crop.shape[1] * 3, stamp_crop.shape[0] * 3),
                    interpolation=_cv2_stamp.INTER_CUBIC,
                )
                result, _ = _stamp_engine(stamp_up)
                if result:
                    texts = [text for _, text, conf in result
                             if float(conf) > 0.3]
                    if texts:
                        stamp_texts[i] = texts
            except Exception as e:
                logger.debug("identify_page_v2: stamp OCR card %d failed: %s",
                             i, e)

        logger.info("identify_page_v2: stamp OCR thread done in %.1fs (%d/%d with text)",
                     _time.time() - t0, len(stamp_texts), n_cards)

    def _batch_embeddings():
        """Thread 2: DINOv2 batch embeddings (GPU, single forward pass)."""
        t0 = _time.time()
        preproc_paths = []
        preproc_temps = []
        for path in card_image_paths:
            try:
                tmp = _preprocess(str(path))
                preproc_paths.append(tmp)
                preproc_temps.append(tmp)
            except Exception:
                preproc_paths.append(str(path))
                preproc_temps.append(None)

        # DINOv2 batch (GPU)
        d_embs = _extract_batch(preproc_paths)
        for i, emb in enumerate(d_embs):
            dino_embeddings[i] = emb
        t_dino = _time.time() - t0

        # CLIP batch skipped — lazy-loaded only if ensemble tiebreaker fires
        # (CLIP contributes 0 unique correct IDs; loading it here causes heap corruption)

        for tmp in preproc_temps:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        logger.info("identify_page_v2: embeddings thread done in %.1fs (DINOv2=%.1fs)", _time.time()-t0, t_dino)

    # Run four main threads in parallel:
    #   - Name OCR (CPU, 3 parallel threads via internal pool)
    #   - DINOv2 batch (GPU, single forward pass)
    #   - Color detection (CPU, parallel across cards)
    #   - Stamp text OCR (CPU, RapidOCR on artwork region)
    # Attack OCR tasks are dispatched by name OCR threads into _attack_pool.
    # Stamp OCR is now piggybacked in _name_ocr_worker (same thread),
    # so only 3 parallel threads needed.
    with ThreadPoolExecutor(max_workers=3) as precomp_pool:
        f_name = precomp_pool.submit(_batch_name_ocr)
        f_dino = precomp_pool.submit(_batch_embeddings)
        f_color = precomp_pool.submit(_batch_color_detect)
        for f in [f_name, f_dino, f_color]:
            f.result()

    # Wait for any outstanding attack OCR tasks dispatched by name OCR thread
    for fut in _attack_futures:
        fut.result()
    _attack_pool.shutdown(wait=False)

    t_precomp_total = _time.time() - t_precomp_start
    logger.info("identify_page_v2: all pre-computation done in %.1fs", t_precomp_total)

    # -----------------------------------------------------------------------
    # Pass 1b: Run identify_card_v2 for ALL cards in parallel THREADS.
    # With OCR, attacks, DINOv2 and CLIP embeddings pre-computed, each call
    # is 100% CPU: DB queries, numpy dot products, fuzzy string matching.
    # -----------------------------------------------------------------------
    t_id_start = _time.time()
    results = [None] * n_cards

    def _thread_worker(i):
        try:
            return identify_card_v2(
                str(card_image_paths[i]),
                session=None,
                _precomputed_ocr=precomputed[i],
                _precomputed_dino_embedding=dino_embeddings[i],
                _precomputed_attacks=attack_results[i],
                _precomputed_clip_embedding=clip_embeddings[i],
                detect_variants=detect_variants,
            )
        except Exception as e:
            logger.warning("identify_page_v2: card %d failed: %s", i, e)
            return {
                "card_id": None, "confidence": 0.0,
                "method": "v2_error",
                "explanation": f"identify_card_v2 failed: {e}",
                "raw_response": {},
            }

    with ThreadPoolExecutor(max_workers=n_cards) as pool:
        futures = {pool.submit(_thread_worker, i): i for i in range(n_cards)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()

    t_id_elapsed = _time.time() - t_id_start
    logger.info("identify_page_v2: parallel identification done in %.1fs (%.1fs/card)",
                t_id_elapsed, t_id_elapsed / n_cards)

    # -----------------------------------------------------------------------
    # Japanese OCR retry DISABLED in batch pipeline.
    # _try_japanese_ocr takes 18s/card and only helps for actual Japanese
    # cards. English cards that fail OCR (blurry Lileep, Buffer Piece) get
    # no benefit — they waste 40+ seconds. Japanese cards should be handled
    # via single-card scan (identify_card_v2 with full _run_name_and_hp).
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Page context reranking DISABLED.
    # Cards on a binder page are independent — any card from any era/set can
    # appear. Page context actively caused wrong matches (e.g. Buffer Piece
    # from ex15 pushed to pl4, Golem from ex12 misidentified because ex12
    # wasn't in detected page sets). Each card must be identified on its own
    # signals: OCR name, DINOv2 visual similarity, attack text.
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Stamp detection via OCR: if identify_card_v2 found text in the
    # stamp region (bottom-right of artwork), and the card is from an
    # EX stamped set (ex7-ex16), mark it as ex_set_stamp.
    # Two signals: (1) text found + card from EX set, (2) fuzzy match
    # of stamp text against known set names (can detect stamp even if
    # card_id set doesn't match, and can correct set identification).
    # -----------------------------------------------------------------------
    _STAMPED_EX_SETS = {"ex7", "ex8", "ex9", "ex10", "ex11",
                        "ex12", "ex13", "ex14", "ex15", "ex16"}
    for i, result in enumerate(results):
        if not result.get("has_stamp_text"):
            continue
        card_id = result.get("card_id")
        stamp_texts_list = result.get("stamp_texts", [])

        # Path 1: Fuzzy match identifies the stamp set name directly
        matched_set = result.get("stamp_set_id")
        match_score = result.get("stamp_match_score", 0)
        if not matched_set and stamp_texts_list:
            matched_set, match_score = _fuzzy_match_stamp_text(stamp_texts_list)

        if matched_set:
            result["detected_variant"] = "ex_set_stamp"
            result["variant_confidence"] = min(0.95, match_score / 100.0)
            result["stamp_ocr_texts"] = stamp_texts_list
            result["stamp_set_id"] = matched_set
            result["stamp_set_name"] = EX_STAMP_NAMES[matched_set]
            result["stamp_match_score"] = match_score
            logger.debug("stamp fuzzy match: card %s -> %s (%s, score=%d)",
                         card_id, matched_set, EX_STAMP_NAMES[matched_set],
                         match_score)
            continue

        # Path 2: Text found in stamp region + card is from a known EX set
        if not card_id:
            continue
        set_id = card_id.split("-")[0]
        if set_id in _STAMPED_EX_SETS:
            result["detected_variant"] = "ex_set_stamp"
            result["variant_confidence"] = 0.85
            result["stamp_ocr_texts"] = stamp_texts_list

    n_identified = sum(1 for r in results if r.get("confidence", 0) >= 0.5)
    avg_conf = sum(r.get("confidence", 0) for r in results) / max(n_cards, 1)
    logger.info("identify_page_v2: %d/%d identified, avg confidence=%.3f",
                n_identified, n_cards, avg_conf)
    return results

    ctx_sets = set(ctx.get("likely_sets", []))

    for i, (path, result) in enumerate(zip(card_image_paths, results)):
        if result["confidence"] >= RERUN_THRESHOLD:
            continue

        # Build leave-one-out context (exclude current card)
        loo_results = results[:i] + results[i + 1:]
        loo_ctx = identify_page_context(loo_results)
        if not loo_ctx.get("likely_sets") or loo_ctx.get("confidence", 0) < 0.40:
            continue

        loo_sets = set(loo_ctx.get("likely_sets", []))

        logger.info(
            "identify_page_v2 pass2: re-examining card %d (conf=%.2f, method=%s) "
            "with page context sets=%s",
            i, result["confidence"], result.get("method"), list(loo_sets)[:3],
        )

        # Strategy: if v2 found candidates via OCR, check if any are in the
        # page context set and re-score with a set bonus.
        raw = result.get("raw_response", {})
        combined_results = raw.get("combined_results", [])
        v2_signals = raw.get("v2_signals", {})

        # Check if the current best is already from the page's set.
        # Even when the current set matches, still try the broad search
        # in case a "Team X's Pokemon" variant scores higher on DINOv2.
        # Only skip broad search for very high confidence (>= 0.80).
        current_set = _extract_set_from_card_id(result.get("card_id"))
        if current_set in loo_sets and result["confidence"] >= 0.80:
            result["confidence"] = min(result["confidence"] + 0.05, 1.0)
            result["explanation"] = (
                (result.get("explanation") or "") + " (page context confirms set)"
            )
            continue
        # If current set matches but confidence is moderate (< 0.80),
        # fall through to broad search. The card might be from the right
        # era but wrong variant (e.g. "Mightyena" from ex8 when the
        # correct answer is "Team Aqua's Mightyena" from ex4).

        # Collect best candidate from combined_results that's in a page set
        best_combined_cid = None
        best_combined_score = 0.0
        for entry in combined_results:
            cid, score = entry[0], entry[1]
            cand_set = _extract_set_from_card_id(cid)
            if cand_set in loo_sets and score >= _V2_FALLBACK_CONFIDENCE:
                best_combined_cid = cid
                best_combined_score = score
                break

        # Also do a broad name search within page context sets.
        # This catches cards like "Team Aqua's Poochyena" when OCR
        # only read "Poochyena" (OCR often misses name prefixes).
        best_broad_cid = None
        best_broad_score = 0.0
        ocr_name = v2_signals.get("ocr_name") or raw.get("ocr_name")
        if ocr_name:
            broad_candidates = _get_candidates_from_sets_broad(
                ocr_name, loo_sets, session=session,
            )
            # Remove candidates already in combined_results (avoid dup work)
            combined_cids = {e[0] for e in combined_results}
            broad_only = [c for c in broad_candidates if c not in combined_cids]
            if broad_only:
                dino_broad = _dino_dot_product_against_refs(
                    str(path), broad_only,
                    query_embedding=dino_embeddings[i] if dino_embeddings[i] is not None else None,
                )
                if dino_broad and dino_broad[0][1] >= _V2_FALLBACK_CONFIDENCE:
                    best_broad_cid = dino_broad[0][0]
                    best_broad_score = dino_broad[0][1]

        # Pick the best between combined-results set match and broad search
        old_cid = result.get("card_id")
        if best_broad_cid and best_broad_score > best_combined_score:
            # Broad search found a better match (e.g. "Team Aqua's Mightyena"
            # beats plain "Mightyena" from a different set)
            result["card_id"] = best_broad_cid
            result["confidence"] = float(best_broad_score) + 0.15
            result["method"] = "v2_page_context_requery"
            result["explanation"] = (
                f"v2 page context requery: {old_cid} -> {best_broad_cid} "
                f"(broad search in sets {list(loo_sets)[:3]}, "
                f"dino={best_broad_score:.3f}+0.15)"
            )
            logger.info(
                "identify_page_v2 pass2: card %d requeried %s -> %s "
                "(broad search, dino=%.3f)",
                i, old_cid, best_broad_cid, best_broad_score,
            )
        elif best_combined_cid:
            # Use the best from existing combined results in the page set
            result["card_id"] = best_combined_cid
            result["confidence"] = float(best_combined_score) + 0.10
            result["method"] = "v2_page_context"
            result["explanation"] = (
                f"v2 page context rerank: {old_cid} -> {best_combined_cid} "
                f"(set matches page context, score={best_combined_score:.3f}+0.10)"
            )
            logger.info(
                "identify_page_v2 pass2: card %d reranked %s -> %s (page context)",
                i, old_cid, best_combined_cid,
            )
        elif best_broad_cid:
            # Broad search found something, even if not great
            result["card_id"] = best_broad_cid
            result["confidence"] = float(best_broad_score) + 0.15
            result["method"] = "v2_page_context_requery"
            result["explanation"] = (
                f"v2 page context requery: {old_cid} -> {best_broad_cid} "
                f"(broad search in sets {list(loo_sets)[:3]}, "
                f"dino={best_broad_score:.3f}+0.15)"
            )
            logger.info(
                "identify_page_v2 pass2: card %d requeried %s -> %s "
                "(broad search, dino=%.3f)",
                i, old_cid, best_broad_cid, best_broad_score,
            )

    # -----------------------------------------------------------------------
    # Pass 3: Re-run fallback cards with era context
    # Only re-run cards that used attack_fallback, ensemble_fallback, or
    # were unidentified AND have low confidence. Don't touch high-confidence results.
    # -----------------------------------------------------------------------
    from cardprice.ml.page_context import _era_for_set, _extract_set_id
    page_era = ctx.get("era")
    if page_era and ctx.get("confidence", 0) >= 0.50:
        for i, (path, result) in enumerate(zip(card_image_paths, results)):
            method = result.get("method", "")
            conf = result["confidence"]
            # Only re-run fallback/low-quality/unidentified results with low confidence
            if "fallback" not in method and "page_context" not in method and method != "unidentified":
                continue
            if conf >= 0.80:
                continue

            # Build leave-one-out era context
            loo_results = results[:i] + results[i + 1:]
            loo_ctx = identify_page_context(loo_results)
            loo_era = loo_ctx.get("era")
            if not loo_era or loo_ctx.get("confidence", 0) < 0.40:
                continue

            logger.info(
                "identify_page_v2 pass3: re-running card %d (method=%s, conf=%.2f) "
                "with page_era=%s",
                i, method, result["confidence"], loo_era,
            )
            # Evict only this card's cache entry so identify_card_v2 re-runs it
            try:
                _path_hash = hashlib.md5(Path(path).read_bytes()).hexdigest()
                _scan_cache.pop(f"v2_{_path_hash}", None)
            except Exception:
                pass
            rerun = identify_card_v2(
                str(path), session=session, page_era=loo_era,
                _precomputed_ocr=precomputed[i],
                _precomputed_dino_embedding=dino_embeddings[i],
                _precomputed_attacks=attack_results[i],
                _precomputed_clip_embedding=clip_embeddings[i],
                detect_variants=detect_variants,
            )

            # Accept re-run if: (a) confidence improved, OR (b) the re-run
            # result is from the correct era and original wasn't.
            from cardprice.ml.page_context import _eras_compatible
            rerun_era = _era_for_set(_extract_set_id(rerun.get("card_id", ""))) if rerun.get("card_id") else None
            orig_era = _era_for_set(_extract_set_id(result.get("card_id", ""))) if result.get("card_id") else None
            era_improved = _eras_compatible(rerun_era or "", loo_era) and not _eras_compatible(orig_era or "", loo_era)
            conf_improved = rerun.get("confidence", 0) > result["confidence"]
            # Don't accept re-run from wrong era — attack fallback can pick
            # wrong-era cards via garbled OCR fuzzy matches.
            # Use compatible eras (adjacent eras like e-card/ex are OK).
            rerun_wrong_era = rerun_era is not None and not _eras_compatible(rerun_era, loo_era)
            if rerun_wrong_era:
                logger.info("identify_page_v2 pass3: card %d rejecting rerun %s (era %s != page %s)",
                            i, rerun.get("card_id"), rerun_era, loo_era)
                continue
            if conf_improved or (era_improved and rerun.get("confidence", 0) >= 0.40):
                old_cid = result.get("card_id")
                results[i] = rerun
                rerun["explanation"] = (
                    (rerun.get("explanation") or "")
                    + f" (pass3: era={loo_era}, was {old_cid})"
                )
                logger.info(
                    "identify_page_v2 pass3: card %d improved %s -> %s (era=%s)",
                    i, old_cid, rerun.get("card_id"), loo_era,
                )

    # -----------------------------------------------------------------------
    # Pass 4: Claude vision fallback for unidentified cards
    # Only runs when use_claude_vision_fallback=True. Sends card images to
    # Claude API for visual identification — handles Japanese, foreign text,
    # and other cases where OCR fails completely.
    # -----------------------------------------------------------------------
    # Also check env var so it can be enabled without code changes
    _vision_fallback_enabled = (
        use_claude_vision_fallback
        or os.environ.get("CARDPRICE_CLAUDE_VISION_FALLBACK", "").lower()
        in ("1", "true", "yes")
    )
    if _vision_fallback_enabled:
        unidentified_indices = [
            i for i, r in enumerate(results)
            if r and r.get("method") == "unidentified"
        ]
        if unidentified_indices:
            logger.info(
                "identify_page_v2 pass4: running Claude vision fallback for "
                "%d unidentified card(s): %s",
                len(unidentified_indices), unidentified_indices,
            )
            from cardprice.ml.claude_vision import identify_card_vision_fallback

            for i in unidentified_indices:
                try:
                    vision_result = identify_card_vision_fallback(
                        str(card_image_paths[i]),
                        session=session,
                    )
                    if vision_result.get("card_id"):
                        old_method = results[i].get("method")
                        old_conf = results[i].get("confidence", 0)
                        results[i] = vision_result
                        logger.info(
                            "identify_page_v2 pass4: card %d identified via "
                            "Claude vision: %s (conf=%.2f, was %s/%.2f)",
                            i, vision_result["card_id"],
                            vision_result["confidence"],
                            old_method, old_conf,
                        )
                    else:
                        logger.info(
                            "identify_page_v2 pass4: card %d still unidentified "
                            "after Claude vision (%s)",
                            i, vision_result.get("explanation", ""),
                        )
                except Exception as e:
                    logger.error(
                        "identify_page_v2 pass4: Claude vision failed for "
                        "card %d: %s", i, e,
                    )

    # Summary logging
    methods = [r.get("method", "?") for r in results if r]
    confidences = [r.get("confidence", 0) for r in results if r]
    v2_count = sum(1 for m in methods if m and m.startswith("v2"))
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    logger.info(
        "identify_page_v2: %d/%d v2-matched, avg confidence=%.3f",
        v2_count, n_cards, avg_conf,
    )

    # Reclaim memory after processing all cards.  Page scans allocate many
    # large temporary arrays (upscaled OCR crops, DINOv2 tensors, etc.)
    # that accumulate across 9 cards.  gc.collect() ensures they're freed
    # before control returns to the server, preventing OOM kills.
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return results


def _extract_set_from_card_id(card_id) -> str:
    """Extract set ID from a card_id like 'base1-4/normal' -> 'base1'.

    Returns empty string if card_id is None or malformed.
    """
    if not card_id:
        return ""
    base = card_id.split("/")[0]  # "base1-4"
    parts = base.rsplit("-", 1)
    return parts[0] if len(parts) == 2 else base
