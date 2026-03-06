"""OCR-based feature extraction from Pokemon card images.

Extracts text-based features that help narrow card identification:
  - HP value (top-right area, e.g. "60 HP" or "HP 60")
  - Attack damage values (right side of attack text area)
  - Weakness/resistance info (bottom strip)
  - Card name (top-left area)

Uses EasyOCR (neural network) as primary OCR engine -- much better on
binder-sleeve photos than tesseract.  Falls back to pytesseract if
easyocr is not installed.

Preprocessing: Otsu thresholding + 4x upscale gives the best results
on both clean reference images and binder-sleeve segments.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Region definitions (fraction of card dimensions)
# ---------------------------------------------------------------------------
# HP is in the top-right corner.  We crop generously so different card
# eras (classic, EX, GX, V, ex, VMAX, VSTAR) all fall within the box.
_HP_REGION = {"x1": 0.35, "y1": 0.0, "x2": 1.0, "y2": 0.12}

# Card name is in the top-left: roughly left 65%, top 10%
_NAME_REGION = {"x1": 0.02, "y1": 0.0, "x2": 0.55, "y2": 0.10}

# Damage numbers on the right side of the attack description area
_DAMAGE_REGION = {"x1": 0.70, "y1": 0.45, "x2": 1.0, "y2": 0.85}

# Full attack text area (for reading attack names + damage together)
_ATTACK_REGION = {"x1": 0.0, "y1": 0.45, "x2": 1.0, "y2": 0.85}

# Weakness/resistance/retreat: bottom strip
_BOTTOM_REGION = {"x1": 0.0, "y1": 0.83, "x2": 1.0, "y2": 0.95}


# ---------------------------------------------------------------------------
# Lazy-loaded EasyOCR reader (heavy init, ~2s first call)
# ---------------------------------------------------------------------------
_easyocr_reader = None


def _get_easyocr_reader():
    """Lazy-load the EasyOCR reader singleton."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            logger.info("Initializing EasyOCR reader (first call, may take a few seconds)...")
            _easyocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        except ImportError:
            logger.warning("easyocr not installed; will fall back to pytesseract")
            return None
    return _easyocr_reader


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _crop_region(img: np.ndarray, region: dict) -> np.ndarray:
    """Crop a fractional region from an image."""
    h, w = img.shape[:2]
    x1 = int(region["x1"] * w)
    y1 = int(region["y1"] * h)
    x2 = int(region["x2"] * w)
    y2 = int(region["y2"] * h)
    return img[y1:y2, x1:x2]


def _preprocess_otsu(crop: np.ndarray, upscale: int = 4) -> np.ndarray:
    """Grayscale, upscale, Otsu threshold.  Works well on clean and binder images."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if upscale > 1:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _preprocess_otsu_inv(crop: np.ndarray, upscale: int = 4) -> np.ndarray:
    """Inverted Otsu -- for light text on dark/colorful backgrounds (full-art cards)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if upscale > 1:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)
    inv = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _preprocess_clahe(crop: np.ndarray, upscale: int = 4) -> np.ndarray:
    """CLAHE contrast enhancement -- helps with washed-out or glare-affected cards."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    if upscale > 1:
        enhanced = cv2.resize(enhanced, None, fx=upscale, fy=upscale,
                              interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# OCR runners
# ---------------------------------------------------------------------------

def _ocr_easyocr(crop: np.ndarray) -> list[tuple[str, float]]:
    """Run EasyOCR on a BGR crop.  Returns list of (text, confidence)."""
    reader = _get_easyocr_reader()
    if reader is None:
        return []
    try:
        results = reader.readtext(crop)
        return [(r[1], float(r[2])) for r in results]
    except Exception as e:
        logger.warning("EasyOCR failed: %s", e)
        return []


def _ocr_tesseract(binary: np.ndarray, config: str = "--psm 7") -> str:
    """Run pytesseract on a preprocessed binary image.  Returns raw text."""
    try:
        import pytesseract
    except ImportError:
        return ""
    try:
        return pytesseract.image_to_string(binary, config=config).strip()
    except Exception as e:
        logger.warning("Tesseract failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# HP parsing
# ---------------------------------------------------------------------------

def _extract_valid_hps(text: str) -> list[int]:
    """Extract all valid HP values from a text string.

    Handles noisy OCR like "570" (should find 70), "7130" (should find 130),
    "Gi70" (should find 170 -- "G"/"C"/"E" is a misread "1").

    Strategy: find all digit runs, try exact matches and suffixes.
    Also try interpreting leading letters as misread digits (common on
    full-art cards where "1" is OCR'd as "I", "l", "G", "C", etc.).
    """
    results = []
    digit_runs = re.findall(r'\d+', text)

    for digits in digit_runs:
        if len(digits) < 2:
            continue
        # Try the full number first (exact match)
        if len(digits) <= 3:
            val = int(digits)
            if _is_valid_hp(val) and val not in results:
                results.append(val)
        # Try suffixes (right-to-left), since OCR noise is prefix garbage.
        for length in (3, 2):
            if len(digits) > length:
                suffix = digits[-length:]
                val = int(suffix)
                if _is_valid_hp(val) and val >= 30 and val not in results:
                    results.append(val)

    # Heuristic: if the text has a specific letter immediately before a
    # 2-digit number (e.g. "Gi70", "C170", "Ei70"), the letter might be
    # a misread "1".  Only certain letters are plausible misreads of "1":
    # I, l, |, C, G, E, F (stylized fonts on full-art cards).
    # The optional "i"/"I" handles "Gi70" where "G" = noise, "i" = "1".
    m = re.search(r'[ICGEFlL|][iIlL1]?(\d{2})\b', text)
    if m:
        suffix_digits = m.group(1)
        val = 100 + int(suffix_digits)  # assume the letter was "1"
        if _is_valid_hp(val) and val not in results:
            results.append(val)

    return results


def _normalize_ocr_digits(text: str) -> str:
    """Normalize common OCR letter-digit confusions.

    EasyOCR frequently misreads digits as visually similar letters:
      0 -> O, o, D, Q    1 -> I, l, |, i    8 -> B
      5 -> S, s           6 -> G, b          3 -> E
    This is especially common on binder-sleeve photos where text is
    slightly blurred or at an angle.
    """
    table = str.maketrans("OoDQIil|BsSbGE", "00001111855630")
    return text.translate(table)


def _parse_hp_from_texts(texts: list[tuple[str, float]]) -> Optional[int]:
    """Given a list of (text, confidence) from OCR, extract the HP value.

    EasyOCR often returns HP as a separate detection like "200" or "Kp60"
    or "#70" or "570" (noise prefix) or "3i70" or "I0o" (=100).
    We try explicit HP patterns first, then extract all valid HP numbers
    and pick the best one.
    """
    # Pass 1: look for explicit "HP" + number patterns (highest confidence)
    for text, conf in texts:
        clean = text.upper().strip()

        m = re.search(r'HP\s*(\d{2,3})', clean)
        if m and _is_valid_hp(int(m.group(1))):
            return int(m.group(1))

        m = re.search(r'(\d{2,3})\s*HP', clean)
        if m and _is_valid_hp(int(m.group(1))):
            return int(m.group(1))

        # Also try after digit normalization (e.g. "I00 HP" -> "100 HP")
        norm = _normalize_ocr_digits(clean)
        m = re.search(r'HP\s*(\d{2,3})', norm)
        if m and _is_valid_hp(int(m.group(1))):
            return int(m.group(1))
        m = re.search(r'(\d{2,3})\s*HP', norm)
        if m and _is_valid_hp(int(m.group(1))):
            return int(m.group(1))

    # Pass 2: extract all valid HP candidates from all text fragments.
    # Try both raw text and digit-normalized text.
    # Score by: prefer exact 2-3 digit matches, prefer higher values
    # (HP is usually the largest number in the top-right region),
    # prefer higher OCR confidence.
    candidates = []
    for text, conf in texts:
        for hp_val in _extract_valid_hps(text):
            candidates.append((hp_val, conf))
        # Also try with digit normalization
        norm = _normalize_ocr_digits(text)
        if norm != text:
            for hp_val in _extract_valid_hps(norm):
                if hp_val not in [c[0] for c in candidates]:
                    candidates.append((hp_val, conf * 0.9))  # slight penalty

    if candidates:
        # Prefer higher HP values (the actual HP is usually the biggest
        # number in the top-right region).  Among ties, prefer higher
        # confidence.
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][0]

    return None


def _is_valid_hp(val: int) -> bool:
    """Check if a value is a plausible Pokemon card HP.

    Pokemon HP values are always multiples of 10, ranging from 30 to 340+.
    """
    return 10 <= val <= 400 and val % 10 == 0


# ---------------------------------------------------------------------------
# Damage parsing
# ---------------------------------------------------------------------------

def _parse_damage_from_texts(texts: list[tuple[str, float]]) -> list[int]:
    """Extract damage values from OCR text fragments."""
    damages = []
    for text, conf in texts:
        for m in re.finditer(r'(\d{1,3})\s*[+x]?', text):
            val = int(m.group(1))
            if _is_valid_damage(val):
                damages.append(val)
    # Deduplicate preserving order
    seen = set()
    result = []
    for d in damages:
        if d not in seen:
            seen.add(d)
            result.append(d)
    return result


def _is_valid_damage(val: int) -> bool:
    """Check if a value is a plausible Pokemon attack damage."""
    return 10 <= val <= 400 and val % 5 == 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_hp(image_path: str) -> Optional[int]:
    """Extract the HP value from a Pokemon card image.

    Crops the top-right area of the card and runs OCR to find the HP number.
    Uses EasyOCR (preferred) or falls back to tesseract with Otsu preprocessing.

    Parameters
    ----------
    image_path : str or Path
        Path to the card image.

    Returns
    -------
    int or None
        The HP value (e.g. 60, 100, 200), or None if not detected.
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Could not read image: %s", image_path)
        return None

    crop = _crop_region(img, _HP_REGION)
    if crop.size == 0:
        return None

    # Try multiple crop regions: narrow first (less noise), then wider.
    # HP is in the top-right; narrower crops give cleaner OCR.
    h, w = img.shape[:2]
    crops = [
        ("narrow", img[0:int(h * 0.10), int(w * 0.55):int(w * 0.95)]),
        ("default", crop),
        # Delta species / older cards: HP text can sit slightly lower or more
        # to the right than modern cards.
        ("wide_right", img[0:int(h * 0.13), int(w * 0.45):w]),
    ]

    for crop_name, hp_crop in crops:
        if hp_crop.size == 0:
            continue

        # Strategy 1: EasyOCR on the raw crop
        texts = _ocr_easyocr(hp_crop)
        if texts:
            logger.debug("HP EasyOCR (%s): %s", crop_name, texts)
            hp = _parse_hp_from_texts(texts)
            if hp is not None:
                return hp

        # Strategy 2: Tesseract with multiple preprocessing methods
        for preprocess_fn in (_preprocess_otsu, _preprocess_otsu_inv, _preprocess_clahe):
            binary = preprocess_fn(hp_crop)
            text = _ocr_tesseract(binary, config="--psm 7")
            logger.debug("HP tesseract %s (%s): %r", crop_name, preprocess_fn.__name__, text)
            if text:
                hp = _parse_hp_from_texts([(text, 1.0)])
                if hp is not None:
                    return hp

    # Strategy 3: Wider fallback regions (whole top strip)
    # Delta species and older cards sometimes have HP text lower or more
    # embedded in the artwork.  Try progressively taller strips.
    for pct in (0.15, 0.20):
        top_strip = img[0:int(h * pct), :]
        texts = _ocr_easyocr(top_strip)
        if texts:
            logger.debug("HP fallback EasyOCR (top %.0f%%): %s", pct * 100, texts)
            hp = _parse_hp_from_texts(texts)
            if hp is not None:
                return hp

    return None


def detect_damage(image_path: str) -> list[int]:
    """Extract attack damage values from a Pokemon card image.

    Damage values appear on the right side of the attack description area.

    Parameters
    ----------
    image_path : str or Path
        Path to the card image.

    Returns
    -------
    list[int]
        Damage values found (e.g. [20, 60]), empty list if none detected.
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        return []

    crop = _crop_region(img, _DAMAGE_REGION)
    if crop.size == 0:
        return []

    # EasyOCR on raw crop
    texts = _ocr_easyocr(crop)
    if texts:
        logger.debug("Damage EasyOCR: %s", texts)
        damages = _parse_damage_from_texts(texts)
        if damages:
            return damages

    # Fallback: tesseract with Otsu
    binary = _preprocess_otsu(crop)
    text = _ocr_tesseract(binary, config="--psm 6 -c tessedit_char_whitelist=0123456789+x ")
    if text:
        logger.debug("Damage tesseract: %r", text)
        damages = _parse_damage_from_texts([(text, 1.0)])
        if damages:
            return damages

    return []


def detect_weakness_resistance(image_path: str) -> dict:
    """Extract weakness/resistance text from the bottom of a Pokemon card.

    Parameters
    ----------
    image_path : str or Path
        Path to the card image.

    Returns
    -------
    dict
        Keys: 'weakness_text', 'resistance_text', 'retreat_count'.
        Values are str or int or None.
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        return {"weakness_text": None, "resistance_text": None, "retreat_count": None}

    crop = _crop_region(img, _BOTTOM_REGION)
    if crop.size == 0:
        return {"weakness_text": None, "resistance_text": None, "retreat_count": None}

    result = {"weakness_text": None, "resistance_text": None, "retreat_count": None}

    # Try EasyOCR first
    texts = _ocr_easyocr(crop)
    all_text = " ".join(t for t, c in texts) if texts else ""
    logger.debug("Bottom EasyOCR: %s", texts)

    if not all_text:
        # Fallback to tesseract
        binary = _preprocess_otsu(crop)
        all_text = _ocr_tesseract(binary, config="--psm 6")
        logger.debug("Bottom tesseract: %r", all_text)

    if all_text:
        # Weakness multiplier: "x2", "*2"
        wk_match = re.search(r'[xX*]\s*(\d)', all_text)
        if wk_match:
            result["weakness_text"] = "x{}".format(wk_match.group(1))

        # Resistance: "-20", "-30"
        res_match = re.search(r'-\s*(\d{2})', all_text)
        if res_match:
            result["resistance_text"] = "-{}".format(res_match.group(1))

        # Retreat cost
        text_upper = all_text.upper()
        ret_match = re.search(r'(?:RETREAT)\D*(\d)', text_upper)
        if ret_match:
            result["retreat_count"] = int(ret_match.group(1))

    return result


def detect_card_name(image_path: str) -> Optional[str]:
    """Extract the card name from the top-left area of a Pokemon card.

    Parameters
    ----------
    image_path : str or Path
        Path to the card image.

    Returns
    -------
    str or None
        The card name as detected by OCR, or None.
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        return None

    crop = _crop_region(img, _NAME_REGION)
    if crop.size == 0:
        return None

    # EasyOCR on raw crop
    texts = _ocr_easyocr(crop)
    if texts:
        logger.debug("Name EasyOCR: %s", texts)
        # Pick the longest text fragment with reasonable confidence
        # (name is usually the most prominent text in this region)
        best = None
        for text, conf in texts:
            cleaned = _clean_card_name(text)
            if cleaned and (best is None or len(cleaned) > len(best)):
                best = cleaned
        if best:
            return best

    # Fallback: tesseract
    for preprocess_fn in (_preprocess_otsu, _preprocess_otsu_inv):
        binary = preprocess_fn(crop)
        text = _ocr_tesseract(binary, config="--psm 7")
        if text:
            cleaned = _clean_card_name(text)
            if cleaned:
                return cleaned

    return None


def _clean_card_name(text: str) -> Optional[str]:
    """Clean up an OCR-detected card name."""
    name = text.strip()
    # Remove stage indicators
    name = re.sub(r'^(BASIC|Stage\s*[12]|STAGE\s*[12])\s*', '', name, flags=re.IGNORECASE)
    # Remove leading/trailing non-alpha junk
    name = re.sub(r'^[^a-zA-Z]+', '', name)
    name = re.sub(r'[^a-zA-Z\s\'-]+$', '', name).strip()
    if name and len(name) >= 2:
        return name
    return None


def extract_all_features(image_path: str) -> dict:
    """Extract all OCR-based features from a Pokemon card image.

    Convenience function that runs all detectors and returns a combined dict.

    Parameters
    ----------
    image_path : str or Path
        Path to the card image.

    Returns
    -------
    dict
        Keys: hp, damage_values, name, weakness_text, resistance_text,
        retreat_count.
    """
    features = {
        "hp": detect_hp(image_path),
        "damage_values": detect_damage(image_path),
        "name": detect_card_name(image_path),
    }
    wr = detect_weakness_resistance(image_path)
    features.update(wr)

    logger.info(
        "OCR features for %s: HP=%s, damage=%s, name=%r, weakness=%s, resistance=%s",
        Path(image_path).name,
        features["hp"],
        features["damage_values"],
        features["name"],
        features["weakness_text"],
        features["resistance_text"],
    )
    return features


# ---------------------------------------------------------------------------
# CLI test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m cardprice.ml.hp_detector <image_path> [image_path ...]")
        print("       python -m cardprice.ml.hp_detector --page2   (test on binder page 2 segments)")
        sys.exit(1)

    if sys.argv[1] == "--page2":
        # Test on page 2 segments
        segments_dir = Path(__file__).resolve().parent.parent.parent / "data" / "test_binder_pages" / "binder_page_02_cards"
        if not segments_dir.exists():
            print("Segments dir not found: {}".format(segments_dir))
            sys.exit(1)

        # Ground truth based on visual inspection of actual segment images
        # (segment order may differ from ground_truth.json due to detection order)
        ground_truth = [
            ("bw10-7", "Shelmet", 60, [20]),
            ("sv3-144", "Bronzor", 70, [30]),
            ("sv2-239", "Dedenne", 170, [170]),
            ("swsh11-30", "Poliwag", 60, [10]),
            ("sv8-219", "Pikachu ex", 200, [300]),
            ("dp6-97", "Gloom", 80, [30]),
            ("sv5-174", "Baxcalibur", 130, [20, 130]),
            ("xy6-53", "Altaria", 80, [30]),
            ("xy1-146", "Xerneas EX", 170, [60, 140]),
        ]

        correct_hp = 0
        total = 0
        for i, card_file in enumerate(sorted(segments_dir.glob("card_*.png"))):
            gt_id, gt_name, gt_hp, gt_dmg = ground_truth[i] if i < len(ground_truth) else ("?", "?", 0, [])
            total += 1
            print("\n{}".format("=" * 60))
            print("Card {}: {}  (expected: {} {} HP={} dmg={})".format(
                i, card_file.name, gt_id, gt_name, gt_hp, gt_dmg))
            print("=" * 60)
            features = extract_all_features(str(card_file))
            for k, v in features.items():
                match = ""
                if k == "hp" and v == gt_hp:
                    match = "  << CORRECT"
                    correct_hp += 1
                elif k == "hp" and v is not None:
                    match = "  << WRONG (expected {})".format(gt_hp)
                elif k == "hp":
                    match = "  << MISS (expected {})".format(gt_hp)
                print("  {:20s}: {}{}".format(k, v, match))

        print("\n" + "=" * 60)
        print("HP accuracy: {}/{} ({:.0%})".format(correct_hp, total, correct_hp / total if total else 0))

    else:
        for path in sys.argv[1:]:
            print("\n{}".format("=" * 60))
            print("File: {}".format(path))
            print("=" * 60)
            features = extract_all_features(path)
            for k, v in features.items():
                print("  {:20s}: {}".format(k, v))
