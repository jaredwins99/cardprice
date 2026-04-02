"""OCR-based card identification by reading card name and number text.

Pokemon cards have:
- Card NAME printed in large text at the top
- Card NUMBER printed in tiny text at the bottom (e.g. "16/132")

This module reads both and uses them together for identification:
1. Crops the top portion for the name, bottom portion for the number
2. Preprocesses crops for OCR (upscaling, contrast, thresholding)
3. Runs OCR to extract text
4. Fuzzy-matches the name against dim_cards, uses number for disambiguation
5. Returns the best match with confidence

Card number OCR requires at least ~800px card height for reliable results.
Binder page scans at 630x880 are marginal; direct card photos work best.
The number is a strong disambiguation signal: name + number = unique card.

This sits in the cascade between CLIP (tier 2.5) and Claude (tier 3) as
tier 2.7 -- it's free, fast, and complements visual matchers by reading
the actual printed text that visual embeddings sometimes miss.

OCR backend priority:
    1. EasyOCR (preferred -- much better accuracy on card photos)
    2. Tesseract via pytesseract (fallback -- faster but poor on card text)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded card name lookup cache: {lowercase_name: [(card_id, name, set_id)]}
# ---------------------------------------------------------------------------
_card_names_cache: list[tuple[str, str, str]] | None = None

# Card number lookup cache: {(card_number, set_id): card_id}
# and set size cache: {set_id: max_card_number}
_card_numbers_cache: dict[tuple[str, str], str] | None = None
_set_sizes_cache: dict[str, int] | None = None

# OCR backend: resolved on first use
_ocr_backend: str | None = None

# ---------------------------------------------------------------------------
# Diamond & Pearl / Platinum era level mapping
# ---------------------------------------------------------------------------
# DP/Platinum era cards (2007-2010) have a unique "LV.XX" level indicator
# printed next to the card name. This is a strong disambiguation signal.
# The mapping file maps "name_lower|level_int" -> [card_id, ...].
_DP_LEVEL_MAP: dict[str, list[str]] | None = None
_DP_LEVEL_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "dp_level_map.json"

# Set IDs for the DP/Platinum era
DP_ERA_SETS = frozenset([
    "dp1", "dp2", "dp3", "dp4", "dp5", "dp6", "dp7", "dpp",
    "pl1", "pl2", "pl3", "pl4",
])


def _load_dp_level_map() -> dict[str, list[str]]:
    """Load the DP-era name+level -> card_id mapping from JSON.

    The mapping file is built from the pokemontcg.io GitHub mirror data
    and maps keys like "venusaur|51" to ["dp3-20"].
    """
    global _DP_LEVEL_MAP
    if _DP_LEVEL_MAP is not None:
        return _DP_LEVEL_MAP

    if not _DP_LEVEL_MAP_PATH.exists():
        logger.warning(
            "DP level map not found at %s. Level-based matching disabled.",
            _DP_LEVEL_MAP_PATH,
        )
        _DP_LEVEL_MAP = {}
        return _DP_LEVEL_MAP

    with open(_DP_LEVEL_MAP_PATH) as f:
        _DP_LEVEL_MAP = json.load(f)
    logger.info("Loaded DP level map (%d entries) from %s", len(_DP_LEVEL_MAP), _DP_LEVEL_MAP_PATH)
    return _DP_LEVEL_MAP


def _extract_level_from_ocr(ocr_texts: list[str]) -> tuple[str | None, int | None]:
    """Parse a "LV.XX" pattern from OCR text fragments.

    DP-era cards print "Name LV.XX" at the top. EasyOCR may return this as:
    - Separate fragments: ["Flygon", "Lv.65"]
    - Concatenated: ["Raikoun42"] or ["Duicunelv44"] (level stuck to name)
    - Garbled: ["tv.55"] (OCR misreads 'L' as 't')
    - Digits only: ["454"] (garbled "Lv.54")

    We check each fragment individually AND the combined text.

    Parameters
    ----------
    ocr_texts : list of str
        All OCR text fragments from the name region.

    Returns
    -------
    tuple of (card_name, level)
        card_name: the name part with LV removed, or None if no level found.
        level: the integer level value, or None if no level found.
    """
    # Join all fragments for combined analysis
    combined = " ".join(ocr_texts)

    # Pattern 1: explicit "LV.XX", "Lv.XX", or garbled "tv.XX" (OCR misreads L->t)
    # Also handles concatenated forms like "Duicunelv44"
    lv_pattern = re.compile(r"[LlTt][Vv]\s*\.?\s*(\d{1,3})", re.IGNORECASE)

    # Check combined text first
    lv_match = lv_pattern.search(combined)
    if lv_match:
        level = int(lv_match.group(1))
        if 1 <= level <= 120:  # valid Pokemon level range
            # Extract name: everything before the LV pattern in combined text
            name_part = combined[:lv_match.start()].strip()
            # Clean up trailing punctuation, commas, spaces, digits (HP value)
            name_part = re.sub(r"[\s,.:;0-9=]+$", "", name_part)
            # Remove non-alpha noise
            name_part = re.sub(r"[^A-Za-z\s\-']", "", name_part).strip()
            if name_part and len(name_part) >= 2:
                return name_part, level
            # If name_part is empty, the LV might be in a separate fragment
            # Try each fragment that doesn't contain the LV pattern as the name
            for t in ocr_texts:
                if lv_pattern.search(t):
                    # This fragment has the LV; extract name from it
                    clean_t = lv_pattern.sub("", t).strip()
                    clean_t = re.sub(r"[^A-Za-z\s\-']", "", clean_t).strip()
                    if clean_t and len(clean_t) >= 2:
                        return clean_t, level
                else:
                    # Fragment without LV -- could be the name
                    clean_t = re.sub(r"[^A-Za-z\s\-']", "", t).strip()
                    if clean_t and len(clean_t) >= 3:
                        return clean_t, level
            return None, level

    # Pattern 2: level digits concatenated with name, no "LV" prefix
    # e.g. "Raikoun42" -> name=Raikou, level=42
    # Check each fragment individually (not just combined text)
    for t in ocr_texts:
        t_stripped = t.strip()
        concat_match = re.search(
            r"([A-Za-z]{3,})(\d{2,3})$",
            t_stripped,
        )
        if concat_match:
            name_part = concat_match.group(1)
            candidate_level = int(concat_match.group(2))
            if 1 <= candidate_level <= 120:
                # Clean the name (remove trailing noise chars like 'n' from 'Raikoun')
                # The last char might be OCR noise if it doesn't match any card
                return name_part, candidate_level

    # Pattern 3: standalone 3-digit number that could be garbled "Lv.XX"
    # e.g. "454" might be "Lv.54" where the "L" was read as "4"
    # Only use this if we have a strong name fragment from another OCR result
    for t in ocr_texts:
        t_stripped = t.strip()
        if re.fullmatch(r"\d{3}", t_stripped):
            # 3-digit number: could be garbled "Lv.XX"
            # Try last 2 digits as level
            candidate_level = int(t_stripped[1:])
            if 10 <= candidate_level <= 99:
                # Find the strongest name fragment
                best_name = None
                for i, t2 in enumerate(ocr_texts):
                    if t2 == t:
                        continue
                    clean_t2 = re.sub(r"[^A-Za-z]", "", t2).strip()
                    if clean_t2 and len(clean_t2) >= 3:
                        if best_name is None or len(clean_t2) > len(best_name):
                            best_name = clean_t2
                if best_name:
                    return best_name, candidate_level

    return None, None


def match_by_dp_level(
    card_name: str,
    level: int,
    fuzzy_threshold: float = 70.0,
) -> list[tuple[str, str, str, float, dict]]:
    """Match a card using name + level against the DP-era level map.

    First tries exact name + exact level. If no match, tries:
    1. Fuzzy name + exact level
    2. Exact name + nearby levels (+-5)
    3. Fuzzy name + nearby levels

    Parameters
    ----------
    card_name : str
        The OCR-extracted card name (may be noisy).
    level : int
        The detected level number.
    fuzzy_threshold : float
        Minimum fuzzy score for name matching (0-100).

    Returns
    -------
    list of (card_id, name, set_id, score, details)
        Matches sorted by score descending. ``details`` contains level match info.
    """
    from rapidfuzz import fuzz

    level_map = _load_dp_level_map()
    if not level_map:
        return []

    card_names_db = _load_card_names()
    # Build lookup: card_id -> (name, set_id)
    card_info = {cid: (name, sid) for cid, name, sid in card_names_db}

    query_name = card_name.lower().strip()
    # Remove common OCR noise from the name
    query_name = re.sub(r"[^a-z\s\-']", "", query_name).strip()

    results = []

    # Collect all unique names in the level map
    level_map_names = set()
    for key in level_map:
        name_part = key.split("|")[0]
        level_map_names.add(name_part)

    # Strategy 1: Exact level, try name matching
    for delta in [0]:  # exact level first
        test_level = level + delta
        for map_name in level_map_names:
            key = f"{map_name}|{test_level}"
            if key not in level_map:
                continue
            # Check name similarity
            name_score = fuzz.ratio(query_name, map_name)
            token_score = fuzz.token_set_ratio(query_name, map_name)
            best_score = max(name_score, token_score)
            if best_score >= fuzzy_threshold:
                for cid in level_map[key]:
                    # Look up card info from DB
                    variant_cid = f"{cid}/normal"
                    info = card_info.get(variant_cid)
                    if info:
                        db_name, set_id = info
                    else:
                        db_name = map_name.title()
                        set_id = cid.split("-")[0] if "-" in cid else "unknown"
                    # Score: combine name match with level exactness bonus
                    combined_score = best_score + 10  # bonus for exact level match
                    combined_score = min(combined_score, 100.0)
                    results.append((
                        variant_cid, db_name, set_id, combined_score,
                        {
                            "level_detected": level,
                            "level_matched": test_level,
                            "level_exact": True,
                            "name_score": best_score,
                            "dp_era": True,
                        },
                    ))

    # If we got exact level matches, return them
    if results:
        results.sort(key=lambda r: -r[3])
        return results

    # Strategy 2: Nearby levels (+-1 to +-5), try name matching
    # OCR may misread a digit (e.g. 51 -> 55, 60 -> 65)
    for delta_range in [range(-2, 3), range(-5, 6)]:
        for delta in delta_range:
            if delta == 0:
                continue  # already tried
            test_level = level + delta
            if test_level < 1 or test_level > 120:
                continue
            for map_name in level_map_names:
                key = f"{map_name}|{test_level}"
                if key not in level_map:
                    continue
                name_score = fuzz.ratio(query_name, map_name)
                token_score = fuzz.token_set_ratio(query_name, map_name)
                best_score = max(name_score, token_score)
                if best_score >= fuzzy_threshold:
                    for cid in level_map[key]:
                        variant_cid = f"{cid}/normal"
                        info = card_info.get(variant_cid)
                        if info:
                            db_name, set_id = info
                        else:
                            db_name = map_name.title()
                            set_id = cid.split("-")[0] if "-" in cid else "unknown"
                        # Penalize for level mismatch
                        level_penalty = abs(delta) * 2
                        combined_score = best_score - level_penalty
                        combined_score = max(0.0, min(combined_score, 100.0))
                        results.append((
                            variant_cid, db_name, set_id, combined_score,
                            {
                                "level_detected": level,
                                "level_matched": test_level,
                                "level_exact": False,
                                "level_delta": delta,
                                "name_score": best_score,
                                "dp_era": True,
                            },
                        ))
        if results:
            break  # found matches in +-2, don't need +-5

    results.sort(key=lambda r: -r[3])
    # Deduplicate by card_id
    seen = set()
    deduped = []
    for r in results:
        if r[0] not in seen:
            seen.add(r[0])
            deduped.append(r)
    return deduped


def _detect_ocr_backend() -> str:
    """Detect which OCR backend is available."""
    global _ocr_backend
    if _ocr_backend is not None:
        return _ocr_backend

    # Prefer RapidOCR (fast ONNX Runtime, low memory)
    try:
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401
        _ocr_backend = "rapidocr"
        logger.info("OCR backend: rapidocr")
        return _ocr_backend
    except ImportError:
        pass

    # Fallback to tesseract
    try:
        import pytesseract  # noqa: F401
        pytesseract.get_tesseract_version()
        _ocr_backend = "tesseract"
        logger.info("OCR backend: tesseract (pytesseract)")
        return _ocr_backend
    except Exception:
        pass

    _ocr_backend = "none"
    logger.warning("No OCR backend available. Install rapidocr-onnxruntime or pytesseract+tesseract.")
    return _ocr_backend


# ---------------------------------------------------------------------------
# Card name database
# ---------------------------------------------------------------------------

def _load_card_names() -> list[tuple[str, str, str]]:
    """Load all (card_id, name, set_id) tuples from dim_cards.

    Results are cached in module-level _card_names_cache for reuse.
    Falls back to data/card_names.json if DB is unavailable.
    """
    global _card_names_cache
    if _card_names_cache is not None:
        return _card_names_cache

    # Try DB first
    try:
        from cardprice.db.session import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT card_id, name, set_id FROM dim_cards ORDER BY name")
            ).fetchall()

        _card_names_cache = [(r[0], r[1], r[2]) for r in rows]
        logger.info("Loaded %d card names from dim_cards for OCR matching.", len(_card_names_cache))
        return _card_names_cache
    except Exception as e:
        logger.warning("DB unavailable for card names: %s — trying JSON fallback", e)

    # Fallback: load from data/card_names.json
    import json
    from pathlib import Path
    fallback = Path(__file__).resolve().parent.parent.parent / "data" / "card_names.json"
    if fallback.exists():
        with open(fallback) as f:
            entries = json.load(f)
        _card_names_cache = [(e[0], e[1], e[2]) for e in entries]
        logger.info("Loaded %d card names from JSON fallback.", len(_card_names_cache))
        return _card_names_cache

    logger.error("No card name source available (DB down, no JSON fallback)")
    _card_names_cache = []
    return _card_names_cache


def reload_card_names():
    """Force reload of the card names cache (e.g. after DB update)."""
    global _card_names_cache
    _card_names_cache = None
    _load_card_names()


# ---------------------------------------------------------------------------
# Card number database
# ---------------------------------------------------------------------------

def _load_card_numbers() -> tuple[dict[tuple[str, str], str], dict[str, int]]:
    """Load card_number -> card_id mapping and set sizes from dim_cards.

    Returns
    -------
    tuple of (numbers_map, set_sizes)
        numbers_map: {(card_number_str, set_id): card_id}
        set_sizes: {set_id: max_numeric_card_number}
    """
    global _card_numbers_cache, _set_sizes_cache
    if _card_numbers_cache is not None and _set_sizes_cache is not None:
        return _card_numbers_cache, _set_sizes_cache

    from cardprice.db.session import engine
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT card_id, card_number, set_id FROM dim_cards "
                 "WHERE card_number IS NOT NULL ORDER BY set_id, card_number")
        ).fetchall()

    numbers_map: dict[tuple[str, str], str] = {}
    set_max: dict[str, int] = {}

    for card_id, card_number, set_id in rows:
        numbers_map[(str(card_number), set_id)] = card_id
        # Track max numeric card number per set (for set size estimation)
        try:
            num = int(card_number)
            if set_id not in set_max or num > set_max[set_id]:
                set_max[set_id] = num
        except (ValueError, TypeError):
            pass  # Non-numeric card numbers (e.g. "HGSS19", "SM150")

    _card_numbers_cache = numbers_map
    _set_sizes_cache = set_max
    logger.info(
        "Loaded %d card numbers from dim_cards (%d sets with numeric numbers).",
        len(numbers_map), len(set_max),
    )
    return _card_numbers_cache, _set_sizes_cache


# ---------------------------------------------------------------------------
# Image preprocessing for OCR
# ---------------------------------------------------------------------------

def _crop_name_region(image_path: str | Path) -> Image.Image:
    """Crop the top portion of a card image where the name is printed.

    Pokemon card names appear in the top ~12-18% of the card. We crop
    a generous region (top 20%) and then do additional processing.
    For images that might include binder sleeve edges, we also trim
    a small margin from the left/right sides.
    """
    img = Image.open(image_path)
    w, h = img.size

    # Trim 5% from each side to avoid binder sleeve edges
    left_margin = int(w * 0.05)
    right_margin = int(w * 0.95)

    # Crop top 20% of the card
    top = 0
    bottom = int(h * 0.20)

    crop = img.crop((left_margin, top, right_margin, bottom))
    return crop


def _preprocess_for_ocr(crop: Image.Image) -> Image.Image:
    """Preprocess a name-region crop for optimal OCR accuracy.

    Steps:
    1. Upscale 2x with FSRCNN super-resolution (falls back to INTER_CUBIC)
    2. Convert to grayscale
    3. Increase contrast
    4. Sharpen
    5. Binarize with adaptive-like thresholding
    """
    import numpy as np

    # FSRCNN super-resolution upscale (2x) for sharper text edges
    try:
        from cardprice.ml.preprocess import upscale_for_ocr
        import cv2

        # PIL -> OpenCV BGR
        cv_img = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
        cv_upscaled = upscale_for_ocr(cv_img, scale=2)
        # OpenCV BGR -> PIL RGB
        crop = Image.fromarray(cv2.cvtColor(cv_upscaled, cv2.COLOR_BGR2RGB))
    except Exception:
        # Fallback to PIL LANCZOS if FSRCNN/OpenCV unavailable
        w, h = crop.size
        if h < 100:
            scale = 300 / h
            crop = crop.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        elif h < 200:
            scale = 2.0
            crop = crop.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Grayscale
    gray = crop.convert("L")

    # Increase contrast
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)

    # Sharpen
    gray = gray.filter(ImageFilter.SHARPEN)

    # Simple threshold binarization - Pokemon card names are dark text on
    # lighter backgrounds (yellow/white name bar)
    # Use a moderate threshold to keep text visible
    gray = gray.point(lambda p: 255 if p > 140 else 0)

    return gray


# ---------------------------------------------------------------------------
# Card number region (bottom of card)
# ---------------------------------------------------------------------------

def _crop_number_region(image_path: str | Path) -> list[Image.Image]:
    """Crop the bottom portion of a card image where the set number is printed.

    Pokemon cards print "X/Y" (card number / set total) in very small text
    at the bottom of the card. The exact position varies by era:

    - Modern (BW+): bottom ~88-93% height, right side
    - DP/Platinum: bottom ~87-92% height
    - e-Card era: bottom ~88-93% height
    - WOTC era: bottom ~88-93% height

    Returns multiple crop regions to increase the chance of capturing the text.
    """
    img = Image.open(image_path)
    w, h = img.size

    crops = []
    # Trim sides to avoid binder sleeve edges
    l_margin = int(w * 0.05)
    r_margin = int(w * 0.95)

    # Multiple vertical ranges to cover era variations
    for top_pct, bot_pct in [(0.87, 0.93), (0.88, 0.94), (0.86, 0.92)]:
        top = int(h * top_pct)
        bottom = int(h * bot_pct)
        if bottom - top < 5:
            continue
        crop = img.crop((l_margin, top, r_margin, bottom))
        crops.append(crop)

    return crops


def _preprocess_number_region(crop: Image.Image) -> list[Image.Image]:
    """Preprocess a bottom-region crop for card number OCR.

    The card number text is extremely small (~5-8px in a typical card image).
    We need aggressive upscaling and contrast enhancement to make it readable.

    Returns multiple preprocessing variants to maximize OCR chances.
    """
    import numpy as np

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    variants = []
    w, h = crop.size

    # Skip if crop is too small
    if h < 3 or w < 10:
        return variants

    # Target: make the text at least 30-40px tall for OCR readability
    # Original text is ~5-8px, so we need 5-8x upscale minimum
    scale = max(8, 300 // max(h, 1))

    if has_cv2:
        arr = np.array(crop)
        # Upscale with cubic interpolation
        big = cv2.resize(arr, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

        # Variant 1: Sharpened color
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharp = cv2.filter2D(big, -1, kernel)
        variants.append(Image.fromarray(sharp))

        # Variant 2: Grayscale + CLAHE (local contrast enhancement)
        gray = cv2.cvtColor(big, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        variants.append(Image.fromarray(enhanced))

        # Variant 3: Color with LAB CLAHE (preserves color, enhances luminance)
        lab = cv2.cvtColor(big, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        color_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        variants.append(Image.fromarray(color_enhanced))

        # Variant 4: Unsharp mask + CLAHE on grayscale
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        unsharp = cv2.addWeighted(gray, 2.0, blurred, -1.0, 0)
        unsharp_enhanced = clahe.apply(unsharp)
        variants.append(Image.fromarray(unsharp_enhanced))
    else:
        # PIL-only fallback
        upscaled = crop.resize((w * scale, h * scale), Image.LANCZOS)

        # Variant 1: Sharpened color
        sharp = upscaled.filter(ImageFilter.SHARPEN)
        sharp = sharp.filter(ImageFilter.SHARPEN)
        variants.append(sharp)

        # Variant 2: High contrast grayscale
        gray = upscaled.convert("L")
        gray = ImageEnhance.Contrast(gray).enhance(3.0)
        gray = ImageEnhance.Sharpness(gray).enhance(2.0)
        variants.append(gray)

    return variants


def extract_card_number(image_path: str | Path) -> tuple[str | None, str | None, float]:
    """Extract the card set number from the bottom of a card image.

    Looks for the "X/Y" pattern where X is the card number within the set
    and Y is the total number of cards in the set.

    Parameters
    ----------
    image_path : str or Path
        Path to the card image.

    Returns
    -------
    tuple of (card_number, set_total, confidence)
        card_number: e.g. "16" (str to handle non-numeric like "H26")
        set_total: e.g. "132" (the denominator in X/Y)
        confidence: OCR confidence (0-1)
        All None/0 if no number found.
    """
    backend = _detect_ocr_backend()
    if backend == "none":
        return None, None, 0.0

    image_path = Path(image_path)
    crops = _crop_number_region(image_path)

    all_results: list[tuple[str, float]] = []

    for crop in crops:
        preprocessed_variants = _preprocess_number_region(crop)

        for variant in preprocessed_variants:
            if backend == "rapidocr":
                texts = _ocr_rapidocr_all_from_image(variant)
                for text, conf in texts:
                    all_results.append((text, conf))
            elif backend == "tesseract":
                import pytesseract
                # PSM 6: uniform block; PSM 7: single line
                for psm in [6, 7]:
                    try:
                        text = pytesseract.image_to_string(
                            variant,
                            config=f"--psm {psm}"
                        ).strip()
                        if text:
                            all_results.append((text, 0.5))
                    except Exception:
                        pass

    # Search all results for X/Y card number pattern
    return _parse_card_number(all_results)


def _ocr_rapidocr_all_from_image(image: Image.Image) -> list[tuple[str, float]]:
    """Run RapidOCR on a PIL Image and return all text fragments.

    Replacement for _ocr_easyocr_all_from_image. Uses RapidOCR (ONNX Runtime)
    which loads in ~1s and uses ~100MB RAM vs EasyOCR's ~16s and ~800MB.
    """
    import numpy as np
    import cv2

    engine = get_rapid_engine()
    img_array = np.array(image)

    # RapidOCR expects BGR 3-channel input
    if len(img_array.shape) == 2:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    try:
        result, _ = engine(img_bgr)
    except Exception:
        return []

    if not result:
        return []

    return [(text.strip(), float(conf)) for _box, text, conf in result if text.strip()]


def _ocr_easyocr_all_from_image(image: Image.Image) -> list[tuple[str, float]]:
    """DEPRECATED: Use _ocr_rapidocr_all_from_image instead.

    Kept for backward compatibility with external callers.
    """
    return _ocr_rapidocr_all_from_image(image)


def _parse_card_number(
    ocr_results: list[tuple[str, float]],
) -> tuple[str | None, str | None, float]:
    """Parse card number (X/Y) from OCR results.

    Tries progressively relaxed patterns to handle OCR noise.

    Parameters
    ----------
    ocr_results : list of (text, confidence)
        All OCR text fragments from the bottom region.

    Returns
    -------
    tuple of (card_number, set_total, confidence)
    """
    # Pass 1: Strict pattern -- digits/digits
    for text, conf in ocr_results:
        m = re.search(r'(\d{1,3})\s*/\s*(\d{2,3})', text)
        if m:
            num, total = int(m.group(1)), int(m.group(2))
            if 1 <= num <= total <= 999:
                return str(num), str(total), conf

    # Pass 2: Allow common OCR character confusions
    # O->0, l/I/T->1, S->5, B->8
    for text, conf in ocr_results:
        fixed = text
        for old, new in [('O', '0'), ('o', '0'), ('l', '1'), ('I', '1'),
                         ('T', '1'), ('S', '5'), ('B', '8'), ('D', '0')]:
            fixed = fixed.replace(old, new)
        m = re.search(r'(\d{1,3})\s*/\s*(\d{2,3})', fixed)
        if m:
            num, total = int(m.group(1)), int(m.group(2))
            if 1 <= num <= total <= 999:
                logger.debug(
                    "Card number found after OCR fix: %s/%s (from %r -> %r)",
                    m.group(1), m.group(2), text, fixed,
                )
                return str(num), str(total), conf * 0.8  # lower confidence for fixed

    # Pass 3: Allow slash-like characters (|, \, l, I)
    for text, conf in ocr_results:
        m = re.search(r'(\d{1,3})\s*[/|\\lI]\s*(\d{2,3})', text)
        if m:
            num, total = int(m.group(1)), int(m.group(2))
            if 1 <= num <= total <= 999:
                return str(num), str(total), conf * 0.7

    return None, None, 0.0


def disambiguate_by_card_number(
    candidates: list[tuple[str, str, str, float]],
    card_number: str,
    set_total: str | None = None,
) -> list[tuple[str, str, str, float]]:
    """Re-rank fuzzy name match candidates using the detected card number.

    Given a list of candidate cards (from fuzzy name matching) and a detected
    card number, boost candidates whose card_number matches.

    Parameters
    ----------
    candidates : list of (card_id, name, set_id, score)
        Candidate cards from fuzzy name matching.
    card_number : str
        The detected card number (e.g. "16").
    set_total : str or None
        The detected set total (e.g. "132"). Used to narrow down the set.

    Returns
    -------
    list of (card_id, name, set_id, score)
        Re-ranked candidates with matching card numbers boosted.
    """
    numbers_map, set_sizes = _load_card_numbers()

    boosted = []
    for card_id, name, set_id, score in candidates:
        # The card_id is like "dp3-16/normal". The numbers_map stores
        # card_ids as they appear in dim_cards (with variant suffix).
        # Look up: does (card_number, set_id) -> this card_id?
        matched_card = numbers_map.get((card_number, set_id))

        bonus = 0.0

        if matched_card is not None and matched_card == card_id:
            # Card number + set both match: strong signal
            bonus += 15.0

            # If set_total also matches the set size, extra bonus
            if set_total is not None:
                set_max = set_sizes.get(set_id)
                if set_max is not None:
                    try:
                        total_int = int(set_total)
                        # Set total printed on card should match set size
                        if abs(set_max - total_int) <= 5:
                            bonus += 10.0
                    except (ValueError, TypeError):
                        pass
        elif matched_card is not None:
            # The number exists in this set but maps to a different card
            # This is a negative signal: reduce score slightly
            bonus -= 5.0

        new_score = max(0.0, min(100.0, score + bonus))
        boosted.append((card_id, name, set_id, new_score))

    boosted.sort(key=lambda r: (-r[3], r[0]))
    return boosted


# ---------------------------------------------------------------------------
# OCR text extraction
# ---------------------------------------------------------------------------

_easyocr_reader = None


def get_easyocr_reader():
    """Get or create the shared English EasyOCR reader singleton.

    Other modules (attack_ocr, hp_detector) should import this instead
    of creating their own readers (~500MB RAM each).
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=True)
    return _easyocr_reader


def _ocr_tesseract(processed_image: Image.Image) -> str:
    """Run Tesseract OCR on a preprocessed image."""
    import pytesseract
    # PSM 7 = treat as a single text line (card names are one line)
    # PSM 6 = assume a single uniform block of text
    # Try PSM 7 first for single-line names
    text = pytesseract.image_to_string(
        processed_image,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.' -"
    ).strip()

    if not text or len(text) < 2:
        # Fall back to PSM 6 (block mode)
        text = pytesseract.image_to_string(
            processed_image,
            config="--psm 6"
        ).strip()

    return text


def _ocr_easyocr(image: Image.Image, use_raw: bool = True) -> tuple[str, float]:
    """Run EasyOCR on a card name crop.

    Args:
        image: The cropped name region (raw color preferred for EasyOCR).
        use_raw: If True, use the raw color image instead of preprocessed.
                 Research showed raw color works best with EasyOCR.

    Returns:
        Tuple of (text, confidence).
    """
    global _easyocr_reader
    import numpy as np

    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=True)

    img_array = np.array(image)
    results = _easyocr_reader.readtext(img_array, detail=1, batch_size=8)

    if not results:
        return "", 0.0

    # Return the text with highest confidence
    results.sort(key=lambda r: r[2], reverse=True)
    return results[0][1].strip(), float(results[0][2])


def _ocr_easyocr_all(image: Image.Image) -> list[tuple[str, float]]:
    """Run EasyOCR and return ALL detected text fragments with confidence.

    Unlike _ocr_easyocr which returns only the best fragment, this returns
    all fragments sorted by position (left-to-right, top-to-bottom) so that
    level detection can see "Flygon" + "Lv.65" as separate fragments.

    Returns:
        List of (text, confidence) tuples in reading order.
    """
    global _easyocr_reader
    import numpy as np

    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=True)

    img_array = np.array(image)
    results = _easyocr_reader.readtext(img_array, detail=1, batch_size=8)

    if not results:
        return []

    # Sort by position: top-left corner y, then x (reading order)
    # EasyOCR result format: [bbox, text, confidence]
    # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
    return [(r[1].strip(), float(r[2])) for r in results if r[1].strip()]


def extract_card_name(image_path: str | Path) -> tuple[str, float]:
    """Extract the card name from an image using OCR.

    Parameters
    ----------
    image_path : str or Path
        Path to the card image.

    Returns
    -------
    tuple of (str, float)
        Extracted text and OCR confidence (0-1).
    """
    backend = _detect_ocr_backend()
    if backend == "none":
        raise RuntimeError(
            "No OCR backend available. Install rapidocr-onnxruntime or tesseract + pytesseract."
        )

    crop = _crop_name_region(image_path)

    if backend == "rapidocr":
        # RapidOCR on raw color image
        texts = _ocr_rapidocr_all_from_image(crop)
        if texts:
            # Return highest-confidence fragment
            texts.sort(key=lambda t: t[1], reverse=True)
            text, conf = texts[0]
            if text and len(text) >= 2:
                return text, conf
        # Retry with a wider crop (top 25%) in case name is cut off
        img = Image.open(image_path)
        w, h = img.size
        wider_crop = img.crop((int(w * 0.03), 0, int(w * 0.97), int(h * 0.25)))
        texts = _ocr_rapidocr_all_from_image(wider_crop)
        if texts:
            texts.sort(key=lambda t: t[1], reverse=True)
            return texts[0]
        return "", 0.0
    elif backend == "tesseract":
        import pytesseract
        processed = _preprocess_for_ocr(crop)
        text = _ocr_tesseract(processed)
        try:
            data = pytesseract.image_to_data(
                processed, config="--psm 7", output_type=pytesseract.Output.DICT
            )
            confs = [int(c) for c in data["conf"] if int(c) > 0]
            avg_conf = sum(confs) / len(confs) / 100.0 if confs else 0.5
        except Exception:
            avg_conf = 0.5
        return text, avg_conf

    return "", 0.0


def extract_card_name_all_fragments(image_path: str | Path) -> list[tuple[str, float]]:
    """Extract ALL OCR text fragments from the card name region.

    Unlike extract_card_name which returns only the best fragment,
    this returns all fragments in reading order. This is important
    for DP-era level detection where "Flygon" and "Lv.65" may appear
    as separate OCR fragments.

    Parameters
    ----------
    image_path : str or Path
        Path to the card image.

    Returns
    -------
    list of (str, float)
        All text fragments with confidence, in reading order.
    """
    backend = _detect_ocr_backend()
    if backend not in ("rapidocr", "easyocr"):
        # Fallback: wrap single result from extract_card_name
        text, conf = extract_card_name(image_path)
        return [(text, conf)] if text else []

    crop = _crop_name_region(image_path)
    fragments = _ocr_rapidocr_all_from_image(crop)

    if not fragments or all(len(t) < 2 for t, _ in fragments):
        # Retry with wider crop
        img = Image.open(image_path)
        w, h = img.size
        wider_crop = img.crop((int(w * 0.03), 0, int(w * 0.97), int(h * 0.25)))
        fragments = _ocr_rapidocr_all_from_image(wider_crop)

    return fragments


# ---------------------------------------------------------------------------
# Fuzzy matching against the card database
# ---------------------------------------------------------------------------

def _clean_ocr_text(text: str) -> str:
    """Clean up OCR output for better fuzzy matching.

    Removes common OCR artifacts, extra whitespace, and non-name characters.
    Also strips DP-era "LV.XX" level indicators since those are handled
    separately by the level detection path.
    """
    if not text:
        return ""

    # Remove DP-era level indicators (LV.XX, Lv.65, lv42, etc.)
    # These are handled by _extract_level_from_ocr separately
    text = re.sub(r"[Ll][Vv]\s*\.?\s*\d{1,3}", "", text)

    # Remove common OCR artifacts (HP numbers, energy symbols text, etc.)
    # Pokemon card top region often shows: "Name  60 HP  [type]"
    # Strip HP and everything after it
    text = re.sub(r"\s+\d+\s*[Hh][Pp].*$", "", text)

    # Remove stray digits (HP values, etc.)
    text = re.sub(r"\b\d+\b", "", text)

    # Remove common OCR garbage characters
    text = re.sub(r"[|_~@#$%^&*(){}\[\]<>+=\\/:;,!?]", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove very short tokens that are likely noise (single chars except valid ones)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1 or t.upper() in ("X",)]
    text = " ".join(tokens)

    return text


def fuzzy_match_card_name(
    ocr_text: str,
    top_k: int = 5,
    score_cutoff: float = 60.0,
) -> list[tuple[str, str, str, float]]:
    """Fuzzy-match OCR text against all card names in the database.

    Parameters
    ----------
    ocr_text : str
        The OCR-extracted card name text.
    top_k : int
        Number of top matches to return.
    score_cutoff : float
        Minimum fuzzy match score (0-100) to include.

    Returns
    -------
    list of (card_id, name, set_id, score)
        Top matches sorted by score descending.
    """
    from rapidfuzz import fuzz, process

    cleaned = _clean_ocr_text(ocr_text)
    if not cleaned or len(cleaned) < 2:
        logger.warning("OCR text too short after cleaning: %r -> %r", ocr_text, cleaned)
        return []

    card_names = _load_card_names()

    # Build a list of unique names for matching (many cards share names across sets)
    # We'll match against unique names first, then return all card_ids for the best name
    unique_names: dict[str, list[tuple[str, str]]] = {}
    for card_id, name, set_id in card_names:
        lower_name = name.lower()
        if lower_name not in unique_names:
            unique_names[lower_name] = []
        unique_names[lower_name].append((card_id, set_id))

    # Use rapidfuzz process.extract for efficient batch fuzzy matching
    # token_set_ratio handles word order differences and partial matches well
    name_list = list(unique_names.keys())
    query = cleaned.lower()

    matches = process.extract(
        query,
        name_list,
        scorer=fuzz.token_set_ratio,
        limit=top_k * 4,  # Get extra candidates for re-ranking
        score_cutoff=score_cutoff,
    )

    if not matches:
        # Try with WRatio (weighted ratio) as fallback
        matches = process.extract(
            query,
            name_list,
            scorer=fuzz.WRatio,
            limit=top_k * 4,
            score_cutoff=score_cutoff,
        )

    # -----------------------------------------------------------------------
    # Re-rank to fix possessive name problem.
    #
    # token_set_ratio decomposes strings into token sets and scores based on
    # the intersection.  This means "Misty's Horsea" vs "Horsea" scores 100
    # because "horsea" is a perfect subset match — the extra "misty's" token
    # is ignored.  When the query CONTAINS a possessive prefix, we must
    # prefer matches that also include it.
    #
    # Strategy: blend token_set_ratio (good for partial OCR) with fuzz.ratio
    # (penalises length mismatches).  For multi-word queries, also add a
    # token coverage bonus when the match contains most of the query tokens.
    # -----------------------------------------------------------------------
    query_tokens = set(query.split())
    query_len = len(query)

    reranked = []
    for matched_name, tsr_score, _idx in matches:
        # fuzz.ratio penalises length differences, so "Misty's Horsea" vs
        # "Horsea" scores ~63 while vs "Misty's Horsea" scores ~96.
        ratio_score = fuzz.ratio(query, matched_name)

        # Token coverage: what fraction of query tokens appear in the match?
        match_tokens = set(matched_name.split())
        if query_tokens:
            # Count tokens that fuzzy-match (handles OCR typos like "mistys"→"misty's")
            covered = 0
            for qt in query_tokens:
                for mt in match_tokens:
                    if fuzz.ratio(qt, mt) >= 75:
                        covered += 1
                        break
            coverage = covered / len(query_tokens)
        else:
            coverage = 1.0

        # Blend: token_set_ratio (60%) + ratio (25%) + coverage bonus (15%)
        # For single-token queries, coverage is always 1.0 so this is harmless.
        blended = tsr_score * 0.60 + ratio_score * 0.25 + coverage * 100.0 * 0.15

        reranked.append((matched_name, blended, tsr_score))

    # Sort by blended score descending
    reranked.sort(key=lambda r: -r[1])

    # Flatten: for each matched name, include all card_ids
    results: list[tuple[str, str, str, float]] = []
    seen_names = set()
    for matched_name, blended_score, _tsr_score in reranked:
        if matched_name in seen_names:
            continue
        seen_names.add(matched_name)
        for card_id, set_id in unique_names[matched_name]:
            # Use the original case name from the first entry
            original_name = next(
                n for _, n, _ in card_names if n.lower() == matched_name
            )
            results.append((card_id, original_name, set_id, blended_score))

    # Sort by score desc, then by card_id for stability
    results.sort(key=lambda r: (-r[3], r[0]))
    return results[:top_k]


# ---------------------------------------------------------------------------
# Main entry point: identify a card by OCR
# ---------------------------------------------------------------------------

def identify_card_by_ocr(
    image_path: str | Path,
    top_k: int = 5,
    page_context: dict | None = None,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Identify a card by reading its name via OCR and fuzzy-matching.

    For DP/Platinum era cards (2007-2010), detects the "LV.XX" level
    indicator and uses it as a strong disambiguation signal before
    falling back to generic fuzzy name matching.

    Parameters
    ----------
    image_path : str or Path
        Path to the card image.
    top_k : int
        Number of top matches to return.
    page_context : dict, optional
        Page-level context for set disambiguation.

    Returns
    -------
    list of (card_id, confidence, details)
        Matches sorted by confidence descending. ``details`` contains:
        - ``ocr_raw``: raw OCR text
        - ``ocr_cleaned``: cleaned text used for matching
        - ``matched_name``: the card name from the database
        - ``set_id``: the set the card belongs to
        - ``fuzzy_score``: the fuzzy match score (0-100)
        - ``ocr_confidence``: OCR engine's confidence in the text extraction
        - ``level_detected``: (if present) the LV.XX value detected
        - ``dp_era``: (if present) True if matched via DP-era level lookup
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Step 1: Extract ALL text fragments via OCR (needed for level detection)
    all_fragments = extract_card_name_all_fragments(image_path)
    all_texts = [t for t, _ in all_fragments]
    ocr_conf = max((c for _, c in all_fragments), default=0.0)
    ocr_text = " ".join(all_texts)

    logger.info("OCR extracted fragments: %s", all_fragments)

    if not ocr_text or len(ocr_text.strip()) < 2:
        logger.info("OCR returned no usable text for %s", image_path)
        return []

    # Step 1.5: Detect DP-era level pattern ("LV.XX")
    level_name, level_val = _extract_level_from_ocr(all_texts)
    if level_val is not None:
        logger.info(
            "DP-era level detected: name=%r level=%d (from OCR: %s)",
            level_name, level_val, all_texts,
        )
        # Try level-based matching (high precision for DP/Platinum era)
        level_matches = match_by_dp_level(
            level_name or _clean_ocr_text(ocr_text),
            level_val,
        )
        if level_matches:
            results = []
            for card_id, name, set_id, score, details in level_matches[:top_k]:
                # High confidence: level + name match is very strong
                fuzzy_conf = score / 100.0
                overall_conf = (ocr_conf * 0.2 + fuzzy_conf * 0.8)
                # Bonus for exact level match
                if details.get("level_exact"):
                    overall_conf = min(1.0, overall_conf + 0.15)
                else:
                    overall_conf = min(1.0, overall_conf + 0.05)

                result_details = {
                    "ocr_raw": ocr_text,
                    "ocr_cleaned": level_name or _clean_ocr_text(ocr_text),
                    "matched_name": name,
                    "set_id": set_id,
                    "fuzzy_score": score,
                    "ocr_confidence": ocr_conf,
                    "level_detected": level_val,
                    "level_matched": details.get("level_matched"),
                    "level_exact": details.get("level_exact", False),
                    "dp_era": True,
                }
                results.append((card_id, overall_conf, result_details))

            results.sort(key=lambda r: -r[1])
            logger.info(
                "OCR+Level match for %s: top=%s (%.2f), level=%d, %d total",
                image_path.name,
                results[0][0] if results else "none",
                results[0][1] if results else 0,
                level_val,
                len(results),
            )
            return results
        else:
            logger.info(
                "Level %d detected but no level map match; falling back to fuzzy",
                level_val,
            )

    # Step 2: Standard fuzzy matching (no level detected, or level match failed)
    cleaned = _clean_ocr_text(ocr_text)
    logger.info("OCR cleaned: %r", cleaned)

    if not cleaned or len(cleaned) < 2:
        logger.info("OCR text too short after cleaning for %s", image_path)
        return []

    # Fuzzy match against database (get extra candidates for reranking)
    matches = fuzzy_match_card_name(cleaned, top_k=max(top_k, 20))

    if not matches:
        logger.info("No fuzzy matches found for OCR text %r", cleaned)
        return []

    # Step 2.5: If level was detected but level-map failed, prefer DP-era sets
    if level_val is not None:
        # Boost DP-era cards in the fuzzy results
        boosted = []
        for card_id, name, set_id, score in matches:
            if set_id in DP_ERA_SETS:
                boosted.append((card_id, name, set_id, min(score + 10, 100.0)))
            else:
                boosted.append((card_id, name, set_id, score))
        matches = boosted
        matches.sort(key=lambda r: -r[3])
        logger.info("OCR: boosted DP-era cards (level=%d detected)", level_val)

    # Step 2.6: Try card number OCR for disambiguation
    card_num_str, set_total_str, num_conf = None, None, 0.0
    try:
        card_num_str, set_total_str, num_conf = extract_card_number(image_path)
        if card_num_str:
            logger.info(
                "Card number detected: %s/%s (conf=%.2f)",
                card_num_str, set_total_str, num_conf,
            )
            matches = disambiguate_by_card_number(
                matches, card_num_str, set_total_str,
            )
    except Exception as e:
        logger.debug("Card number extraction failed: %s", e)

    if page_context and page_context.get("likely_sets"):
        try:
            from cardprice.ml.page_context import rerank_with_context
            # Convert matches to (card_id, score) tuples for reranking
            as_tuples = [(cid, score / 100.0) for cid, _name, _sid, score in matches]
            reranked = rerank_with_context(as_tuples, page_context)
            # Rebuild matches list in new order
            reranked_ids = [cid for cid, _ in reranked]
            match_by_id = {cid: (cid, name, sid, score) for cid, name, sid, score in matches}
            matches = [match_by_id[cid] for cid in reranked_ids if cid in match_by_id]
            logger.info("OCR: reranked with page context (sets=%s)", page_context["likely_sets"][:3])
        except Exception as e:
            logger.debug("OCR: page context reranking failed: %s", e)

    matches = matches[:top_k]

    # Step 3: Convert to output format
    # Combine OCR confidence with fuzzy score for overall confidence
    results = []
    for card_id, name, set_id, fuzzy_score in matches:
        fuzzy_conf = fuzzy_score / 100.0
        overall_conf = (ocr_conf * 0.3 + fuzzy_conf * 0.7)
        if fuzzy_score >= 95:
            overall_conf = min(1.0, overall_conf + 0.1)

        result_details = {
            "ocr_raw": ocr_text,
            "ocr_cleaned": cleaned,
            "matched_name": name,
            "set_id": set_id,
            "fuzzy_score": fuzzy_score,
            "ocr_confidence": ocr_conf,
        }
        # Include level info if detected (even if level-map match failed)
        if level_val is not None:
            result_details["level_detected"] = level_val
            result_details["dp_era"] = set_id in DP_ERA_SETS
        # Include card number info if detected
        if card_num_str:
            result_details["card_number_detected"] = card_num_str
            result_details["set_total_detected"] = set_total_str
            result_details["number_confidence"] = num_conf

        results.append((card_id, overall_conf, result_details))

    results.sort(key=lambda r: -r[1])
    logger.info(
        "OCR match results for %s: top=%s (%.2f), %d total",
        image_path.name,
        results[0][0] if results else "none",
        results[0][1] if results else 0,
        len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Targeted Pokemon name detection (high-accuracy name-only OCR)
# ---------------------------------------------------------------------------

# Unique Pokemon names cache for fuzzy matching (built from dim_cards)
_unique_pokemon_names: list[str] | None = None


def _load_unique_pokemon_names() -> list[str]:
    """Load deduplicated Pokemon names from dim_cards for fuzzy matching.

    Returns a sorted list of unique card names (original case).
    """
    global _unique_pokemon_names
    if _unique_pokemon_names is not None:
        return _unique_pokemon_names

    card_names = _load_card_names()
    seen = set()
    unique = []
    for _card_id, name, _set_id in card_names:
        lower = name.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(name)
    unique.sort()
    _unique_pokemon_names = unique
    return _unique_pokemon_names


def _clean_name_ocr(text: str) -> str:
    """Clean OCR text specifically for Pokemon name matching.

    Strips trailing level numbers, HP values, stage text, and noise
    while preserving the core Pokemon name including special characters
    like periods (Mr. Mime) and hyphens (Ho-Oh).
    """
    if not text:
        return ""

    # Remove DP-era level indicators (Lv.55, lv44, etc.)
    text = re.sub(r"[LlTt][Vv]\s*\.?\s*\d{1,3}", "", text)
    # Remove trailing digits that got stuck to the name (e.g. "Venusaur55", "Raikoun42")
    text = re.sub(r"(\D)\d{2,4}$", r"\1", text)
    # Remove HP values
    text = re.sub(r"\s*\d+\s*[Hh][Pp].*$", "", text)
    text = re.sub(r"\s*[Hh][Pp]\s*\d+.*$", "", text)
    # Remove "STAGE", "BASIC", and "TRAINER" text
    text = re.sub(r"\b(?:STAGE|stage|Stage|BASIC|basic|Basic|TRAINER|trainer|Trainer|SUPPORTER|supporter|Supporter)\s*\d?\b", "", text)
    # Remove stray digits
    text = re.sub(r"\b\d+\b", "", text)
    # Remove noise characters (keep letters, spaces, periods, hyphens, apostrophes)
    text = re.sub(r"[^A-Za-z\s.\-'éδ]", "", text)
    # Remove trailing single chars that are OCR noise (e.g. "Venusaur_" -> "Venusaur")
    text = re.sub(r"\s+[a-z]\s*$", "", text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _unsharp_mask_ocr(img):
    """Apply unsharp mask to boost text sharpness for OCR.

    Only for OCR path — DO NOT use before DINOv2/CLIP embeddings.
    """
    import cv2
    blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 2.5, blurred, -1.5, 0)


# RapidOCR lazy singleton (replaces PaddleOCR — ONNX Runtime, 5x faster, no crashes)
_rapid_engine = None

# Backward-compat aliases — other modules import these directly
_paddle_det = None
_paddle_rec = None


def get_rapid_engine():
    """Return the shared RapidOCR engine singleton.

    Uses rapidocr_onnxruntime which bundles PP-OCRv4 models via ONNX Runtime.
    Much faster and more stable than PaddleOCR (no MKL crashes, no SIGSEGV).
    """
    global _rapid_engine, _paddle_det, _paddle_rec
    if _rapid_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _rapid_engine = RapidOCR()
        # Set compat aliases so modules importing _paddle_det/_paddle_rec get
        # a non-None value (they should migrate to get_rapid_engine() later)
        _paddle_det = _rapid_engine
        _paddle_rec = _rapid_engine
    return _rapid_engine


def get_paddle_engines():
    """Backward-compatible wrapper — returns (engine, engine).

    Other modules (hp_detector, attack_ocr, variant_detector) import this and
    unpack into (det, rec).  They still call the PaddleOCR det/rec API on the
    returned objects, so they will need their own migration.  For now this
    ensures the import doesn't fail and the RapidOCR engine is initialized.
    """
    engine = get_rapid_engine()
    return engine, engine


def _paddle_ocr_name(img, h, w, *, debug=False):
    """Run RapidOCR on the name region of a card image.

    Uses RapidOCR (ONNX Runtime) as a drop-in replacement for PaddleOCR.
    RapidOCR does detection + recognition in a single call.

    Tries multiple crop regions and preprocessing variants:
    1. Color crops at top 15%/20%/25% with 3x upscale + unsharp mask
    2. CLAHE grayscale fallback on top 25% if color crops fail

    The 3x upscale is critical -- OCR detection models struggle
    with the small text in binder-scan card segments (~630x880px).

    Returns list of (text, confidence, method) tuples, or empty list.
    """
    import cv2

    engine = get_rapid_engine()

    results = []

    def _detect_and_recognize(crop_img, label):
        """Run RapidOCR detection + recognition on a preprocessed crop."""
        result, _elapse = engine(crop_img)
        if not result:
            return
        for box, text, conf in result:
            text = text.strip()
            conf = float(conf)
            if text and len(text) >= 2 and conf > 0.3:
                results.append((text, conf, 'rapid'))
                if debug:
                    print(f"  RapidOCR [{label}]: '{text}' conf={conf:.3f}")

    # --- Strategy 1: Color crops with 3x upscale + unsharp mask ---
    crop_specs = [
        (0.00, 0.15, 0.05, 0.95, "top15"),
        (0.00, 0.20, 0.05, 0.95, "top20"),
        (0.00, 0.25, 0.03, 0.97, "top25"),
        (0.03, 0.18, 0.05, 0.95, "skip3"),
    ]

    # Common non-name texts to exclude from early-exit check
    _NON_NAME_WORDS = {"stage", "basic", "hp", "stage i", "stage ii",
                       "stage 1", "stage 2", "trainer", "supporter",
                       "pokemon", "item", "energy", "stage i pokemon",
                       "stage ii pokemon"}

    from cardprice.ml.preprocess import upscale_for_ocr

    for top_frac, bot_frac, left_frac, right_frac, label in crop_specs:
        y1 = int(h * top_frac)
        y2 = int(h * bot_frac)
        x1 = int(w * left_frac)
        x2 = int(w * right_frac)
        crop = img[y1:y2, x1:x2]
        # Pad so text isn't at the very edge (detection needs margin)
        crop = cv2.copyMakeBorder(crop, 30, 30, 30, 30, cv2.BORDER_REPLICATE)
        crop = _unsharp_mask_ocr(crop)
        # 3x upscale -- FSRCNN 2x then cubic 1.5x for sharper text edges
        crop_up = upscale_for_ocr(crop, scale=3)
        _detect_and_recognize(crop_up, label)

        # Early exit if we found a high-confidence name (>0.8, 3+ alpha chars)
        if any(c > 0.8 and sum(1 for ch in t if ch.isalpha()) >= 3
               and t.strip().lower() not in _NON_NAME_WORDS
               for t, c, _ in results):
            return results

    # --- Strategy 2: CLAHE grayscale on top 25% (helps with holo/metallic) ---
    if not any(c > 0.5 for _, c, _ in results):
        y2 = int(h * 0.25)
        crop = img[0:y2, int(w * 0.03):int(w * 0.97)]
        crop = cv2.copyMakeBorder(crop, 30, 30, 30, 30, cv2.BORDER_REPLICATE)
        crop_up = upscale_for_ocr(crop, scale=3)
        gray = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # RapidOCR handles both grayscale and BGR input
        _detect_and_recognize(enhanced, "clahe25")

    if debug and not results:
        print("  RapidOCR: no text detected in any crop")

    return results


def detect_pokemon_name(
    image_path: str | Path,
    *,
    debug: bool = False,
) -> tuple[str | None, float]:
    """Detect the Pokemon name from a card image using targeted OCR.

    This function focuses specifically on reading the card name, which is the
    largest text on every Pokemon card, always printed at the top. It uses
    multiple crop strategies and preprocessing variants to maximize accuracy.

    Strategy:
    1. Try multiple top-region crops (different percentages for different eras)
    2. Upscale each crop 3x for better OCR
    3. Run EasyOCR with lowered thresholds on color images
    4. Clean OCR output and fuzzy-match against all Pokemon names in DB
    5. Return the best match and confidence

    Parameters
    ----------
    image_path : str or Path
        Path to the card segment image.
    debug : bool
        If True, print debug info about all OCR candidates.

    Returns
    -------
    tuple of (name, confidence)
        name: The matched Pokemon name from the database, or None if no match.
        confidence: Combined OCR + fuzzy match confidence (0.0 - 1.0).
    """
    import cv2
    import numpy as np
    from rapidfuzz import fuzz, process
    from cardprice.ml.preprocess import upscale_for_ocr

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]

    # -- Step 1: Extract OCR text from multiple crop regions --
    # Different eras/conditions need different crops:
    #   - Standard cards: top 15% works well
    #   - e-Card era: name sits lower, need top 25%
    #   - Binder sleeves: may need to skip top 3-5%
    crop_specs = [
        # (top%, bottom%, left%, right%, label)
        (0, 15, 5, 95, "top15"),
        (0, 20, 5, 95, "top20"),
        (0, 25, 3, 97, "top25"),
        (3, 18, 5, 95, "skip3"),
    ]

    raw_candidates: list[tuple[str, float, str]] = []  # (text, conf, method)

    # --- Try RapidOCR first (fast ONNX Runtime, better on non-holo text) ---
    paddle_tried = False
    try:
        paddle_candidates = _paddle_ocr_name(img, h, w, debug=debug)
        paddle_tried = True
        if paddle_candidates:
            raw_candidates.extend(paddle_candidates)
    except Exception as _paddle_err:
        if debug:
            print(f"  RapidOCR failed: {_paddle_err}")

    # --- RapidOCR retry with additional crops if first pass found nothing ---
    # Previously fell back to EasyOCR (~500MB GPU RAM). Now retry with RapidOCR
    # using CLAHE preprocessing and different crop regions to avoid OOM.
    if not any(c > 0.5 for _, c, _ in raw_candidates):
        rapid_engine = get_rapid_engine()

        for top_pct, bot_pct, left_pct, right_pct, label in crop_specs:
            y1 = int(h * top_pct / 100)
            y2 = int(h * bot_pct / 100)
            x1 = int(w * left_pct / 100)
            x2 = int(w * right_pct / 100)
            crop = img[y1:y2, x1:x2]

            # Apply unsharp mask for OCR sharpness boost
            crop = _unsharp_mask_ocr(crop)

            # Upscale 3x -- FSRCNN 2x then cubic 1.5x for sharper text edges
            crop_up = upscale_for_ocr(crop, scale=3)

            # Run RapidOCR on color image (replaces EasyOCR to save ~500MB)
            try:
                result, _ = rapid_engine(crop_up)
                if result:
                    for _bbox, text, conf in result:
                        text = text.strip()
                        if len(text) >= 2:
                            raw_candidates.append((text, float(conf), f"rapid_{label}"))
            except Exception:
                pass

            # Early exit: if we found a high-confidence text (>0.95) with
            # 5+ alpha chars, we likely have the full name and don't need more crops
            if any(c > 0.95 and sum(1 for ch in t if ch.isalpha()) >= 5
                   for t, c, _ in raw_candidates):
                break

    # If no good candidates yet, try CLAHE-enhanced grayscale on the widest crop
    # This helps with low-contrast or holographic card backgrounds
    if not any(c > 0.4 for _, c, _ in raw_candidates):
        y1 = 0
        y2 = int(h * 0.25)
        x1 = int(w * 0.03)
        x2 = int(w * 0.97)
        crop = img[y1:y2, x1:x2]
        crop = _unsharp_mask_ocr(crop)
        crop_up = upscale_for_ocr(crop, scale=3)
        gray = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Run RapidOCR on CLAHE-enhanced grayscale (replaces EasyOCR)
        try:
            rapid_engine = get_rapid_engine()
            # RapidOCR expects 3-channel input
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            result, _ = rapid_engine(enhanced_bgr)
            if result:
                for _bbox, text, conf in result:
                    text = text.strip()
                    if len(text) >= 2:
                        raw_candidates.append((text, float(conf), "rapid_clahe25"))
        except Exception:
            pass

    if debug:
        print(f"  Raw OCR candidates: {raw_candidates}")

    if not raw_candidates:
        return None, 0.0

    # -- Step 2: Clean and filter candidates --
    # Keep only texts that look like Pokemon names (mostly alphabetic, 3+ chars)
    name_candidates: list[tuple[str, float, str]] = []
    for text, conf, method in raw_candidates:
        cleaned = _clean_name_ocr(text)
        if not cleaned or len(cleaned) < 3:
            continue
        # Check alphabetic ratio — but allow possessive fragments like "N's"
        alpha_count = sum(1 for c in cleaned if c.isalpha())
        if alpha_count / len(cleaned) < 0.7:
            if not re.search(r"[''\u2019][sS]$", cleaned):
                continue
        name_candidates.append((cleaned, conf, method))

    if debug:
        print(f"  Cleaned candidates: {name_candidates}")

    if not name_candidates:
        # Fallback: try the raw candidates with less strict filtering
        for text, conf, method in raw_candidates:
            cleaned = _clean_name_ocr(text)
            if cleaned and len(cleaned) >= 2:
                name_candidates.append((cleaned, conf, method))

    if not name_candidates:
        return None, 0.0

    # -- Step 3: Fuzzy match each candidate against all Pokemon names --
    unique_names = _load_unique_pokemon_names()
    name_list_lower = [n.lower() for n in unique_names]
    # Build reverse lookup
    lower_to_original = {n.lower(): n for n in unique_names}

    best_match: str | None = None
    best_score: float = 0.0
    best_ocr_conf: float = 0.0
    best_ocr_text: str = ""

    # Deduplicate candidates (same cleaned text)
    seen_cleaned = set()
    deduped: list[tuple[str, float, str]] = []
    for cleaned, conf, method in name_candidates:
        key = cleaned.lower()
        if key not in seen_cleaned:
            seen_cleaned.add(key)
            deduped.append((cleaned, conf, method))

    # Common OCR character confusions for Pokemon names
    # Maps of chars that look alike: used to generate alternate queries
    _OCR_CONFUSIONS = {
        'y': 'x', 'x': 'y',  # Y/X confusion common in handwriting/OCR
        'l': 'i', 'i': 'l',  # l/I confusion
        'u': 'v', 'v': 'u',  # u/v confusion
        'o': 'c', 'c': 'o',  # o/c confusion
        'n': 'h', 'h': 'n',  # n/h confusion
        'rn': 'm', 'm': 'rn',  # rn/m confusion
        'd': 'cl',            # d/cl confusion
    }

    for cleaned, ocr_conf, method in deduped:
        query = cleaned.lower()

        # Skip very low confidence OCR results (likely garbage)
        if ocr_conf < 0.15 and len(cleaned) < 5:
            if debug:
                print(f"    SKIP '{cleaned}' (conf={ocr_conf:.2f}, too short+low)")
            continue

        # Build query variants: original + OCR confusion alternatives
        queries = [query]
        for old_char, new_char in _OCR_CONFUSIONS.items():
            if old_char in query:
                alt = query.replace(old_char, new_char, 1)
                if alt != query:
                    queries.append(alt)

        best_for_this: tuple[str, float] | None = None

        for q in queries:
            # Try exact match first (fast path)
            if q in lower_to_original:
                match_name = lower_to_original[q]
                score = 100.0
                if best_for_this is None or score > best_for_this[1]:
                    best_for_this = (match_name, score)
                continue

            # Primary: fuzz.ratio (strict character-level similarity)
            m = process.extractOne(
                q,
                name_list_lower,
                scorer=fuzz.ratio,
                score_cutoff=60.0,
            )
            if m is not None:
                matched_lower, score, _idx = m
                mn = lower_to_original[matched_lower]
                # Bonus for exact-length match (same # chars = more likely correct)
                if len(q) == len(matched_lower):
                    score = min(100.0, score + 3.0)
                if best_for_this is None or score > best_for_this[1]:
                    best_for_this = (mn, score)

        if best_for_this is None:
            # Fallback: partial_ratio for substring matches
            # e.g. "tias" -> "latias" (high partial score)
            matches = process.extractOne(
                query,
                name_list_lower,
                scorer=fuzz.partial_ratio,
                score_cutoff=85.0,
            )
            if matches is not None:
                matched_lower, score, _idx = matches
                score = score * 0.85  # discount partial matches
                match_name = lower_to_original[matched_lower]
                best_for_this = (match_name, score)

        if best_for_this is None:
            continue

        match_name, score = best_for_this

        # Reject weak matches: if OCR text is short and score is mediocre,
        # it's likely matching random garbage to a short Pokemon name
        if len(cleaned) <= 3 and score < 90:
            if debug:
                print(f"    REJECT '{cleaned}' -> {match_name} "
                      f"(short text + low score {score:.0f})")
            continue
        if len(cleaned) <= 4 and score < 75 and ocr_conf < 0.4:
            if debug:
                print(f"    REJECT '{cleaned}' -> {match_name} "
                      f"(short text + low score + low conf)")
            continue

        # Combined score: weight fuzzy match heavily, OCR conf as tiebreaker
        combined = score * 0.8 + min(ocr_conf, 1.0) * 100.0 * 0.2
        if combined > best_score:
            best_score = combined
            best_match = match_name
            best_ocr_conf = ocr_conf
            best_ocr_text = cleaned

        if debug:
            print(f"    '{cleaned}' (conf={ocr_conf:.2f}) -> {match_name} "
                  f"(fuzzy={score:.0f}, combined={combined:.1f})")

    if best_match is None:
        return None, 0.0

    # Compute final confidence (0-1 scale)
    fuzzy_conf = best_score / 100.0
    confidence = min(1.0, fuzzy_conf)

    # Reject matches with very low combined score -- likely false positives
    # from garbage OCR on blurry images matching short Pokemon names
    if confidence < 0.65:
        if debug:
            print(f"  REJECT FINAL: '{best_ocr_text}' -> {best_match} "
                  f"(combined conf={confidence:.3f} < 0.65)")
        return None, 0.0

    if debug:
        print(f"  RESULT: '{best_ocr_text}' -> {best_match} (conf={confidence:.3f})")

    return best_match, confidence


# ---------------------------------------------------------------------------
# CLI / standalone testing
# ---------------------------------------------------------------------------

def test_on_directory(image_dir: str | Path) -> list[dict[str, Any]]:
    """Test OCR matching on all images in a directory.

    Parameters
    ----------
    image_dir : str or Path
        Directory containing card images.

    Returns
    -------
    list of dict
        Results for each image.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    valid_ext = {".png", ".jpg", ".jpeg", ".webp"}
    images = sorted(
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_ext
    )

    results = []
    for img_path in images:
        try:
            matches = identify_card_by_ocr(img_path, top_k=3)
            if matches:
                best_id, best_conf, best_details = matches[0]
                results.append({
                    "image": img_path.name,
                    "ocr_raw": best_details["ocr_raw"],
                    "ocr_cleaned": best_details["ocr_cleaned"],
                    "best_match": best_id,
                    "best_name": best_details["matched_name"],
                    "confidence": best_conf,
                    "fuzzy_score": best_details["fuzzy_score"],
                    "all_matches": [
                        (m[0], m[2]["matched_name"], m[1]) for m in matches
                    ],
                })
                print(
                    f"  {img_path.name}: OCR={best_details['ocr_cleaned']!r} "
                    f"-> {best_details['matched_name']} ({best_id}) "
                    f"[fuzzy={best_details['fuzzy_score']:.0f}, conf={best_conf:.2f}]"
                )
            else:
                results.append({
                    "image": img_path.name,
                    "ocr_raw": "",
                    "ocr_cleaned": "",
                    "best_match": None,
                    "best_name": None,
                    "confidence": 0,
                    "fuzzy_score": 0,
                    "all_matches": [],
                })
                print(f"  {img_path.name}: NO MATCH")
        except Exception as e:
            logger.warning("Error processing %s: %s", img_path, e)
            results.append({
                "image": img_path.name,
                "error": str(e),
                "best_match": None,
                "confidence": 0,
            })
            print(f"  {img_path.name}: ERROR - {e}")

    # Summary
    matched = sum(1 for r in results if r.get("best_match"))
    print(f"\nSummary: {matched}/{len(results)} cards identified by OCR")
    return results


def test_card_number_on_directory(image_dir: str | Path) -> list[dict[str, Any]]:
    """Test card number OCR on all images in a directory.

    Parameters
    ----------
    image_dir : str or Path
        Directory containing card images.

    Returns
    -------
    list of dict
        Results for each image.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    valid_ext = {".png", ".jpg", ".jpeg", ".webp"}
    images = sorted(
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_ext
    )

    results = []
    found = 0
    for img_path in images:
        try:
            card_num, set_total, conf = extract_card_number(img_path)
            if card_num:
                found += 1
                print(
                    f"  {img_path.name}: {card_num}/{set_total} "
                    f"(conf={conf:.2f})"
                )
            else:
                print(f"  {img_path.name}: no number found")
            results.append({
                "image": img_path.name,
                "card_number": card_num,
                "set_total": set_total,
                "confidence": conf,
            })
        except Exception as e:
            logger.warning("Error processing %s: %s", img_path, e)
            print(f"  {img_path.name}: ERROR - {e}")
            results.append({
                "image": img_path.name,
                "error": str(e),
            })

    print(f"\nSummary: {found}/{len(results)} card numbers found")
    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m cardprice.ml.ocr_matcher <image_or_directory>")
        print("       python -m cardprice.ml.ocr_matcher --number <image_or_directory>")
        print("       python -m cardprice.ml.ocr_matcher data/inbox/page_*/")
        sys.exit(1)

    # Check for --number flag
    args = sys.argv[1:]
    number_mode = False
    if "--number" in args:
        number_mode = True
        args.remove("--number")

    if not args:
        print("No target specified.")
        sys.exit(1)

    target = Path(args[0])

    if number_mode:
        # Card number OCR mode
        if target.is_dir():
            test_card_number_on_directory(target)
        elif target.is_file():
            card_num, set_total, conf = extract_card_number(target)
            if card_num:
                print(f"Card number: {card_num}/{set_total} (conf={conf:.2f})")
            else:
                print("No card number found.")
        else:
            print(f"Not found: {target}")
            sys.exit(1)
    else:
        # Standard name OCR mode
        if target.is_dir():
            test_on_directory(target)
        elif target.is_file():
            matches = identify_card_by_ocr(target, top_k=5)
            for card_id, conf, details in matches:
                print(
                    f"  {card_id}: {details['matched_name']} "
                    f"(set={details['set_id']}, fuzzy={details['fuzzy_score']:.0f}, "
                    f"conf={conf:.2f})"
                )
                print(f"    OCR raw: {details['ocr_raw']!r}")
                print(f"    OCR cleaned: {details['ocr_cleaned']!r}")
                if details.get("card_number_detected"):
                    print(
                        f"    Card number: {details['card_number_detected']}"
                        f"/{details['set_total_detected']}"
                    )
        else:
            print(f"Not found: {target}")
            sys.exit(1)
