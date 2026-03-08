"""Attack/move name OCR for card disambiguation.

When card name + type leaves multiple candidate cards, reading the attack
names from the card image narrows it down further -- attack combinations
are nearly unique per card printing.

Pipeline:
  1. Crop to attack region (roughly 40-75% of card height, center 80% width)
  2. Preprocess: upscale, CLAHE, optional sharpen
  3. Run RapidOCR on the region (preferred) or EasyOCR (fallback)
  4. Filter OCR fragments: keep likely attack names, discard damage numbers,
     energy costs, description text, and other noise
  5. Fuzzy-match surviving fragments against the attack_index.pkl
  6. Return ranked list of (card_id, match_score) tuples

Attack OCR gets ~56% recall on modern cards, ~0% on older EX-delta cards.
The bottom half of the card has attacks in bold/larger text; attack
descriptions are in smaller text and should be filtered out.
"""

from __future__ import annotations

import logging
import pickle
import re
from difflib import SequenceMatcher

from rapidfuzz import fuzz as _rfuzz
from rapidfuzz import process as _rprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ATTACK_INDEX_PATH = _PROJECT_ROOT / "data" / "attack_index.pkl"
_ATTACK_DB_PATH = _PROJECT_ROOT / "data" / "attack_db.json"
_STRUCTURED_ATTACKS_PATH = _PROJECT_ROOT / "data" / "structured_attacks.json"

# Lazy-loaded globals
_attack_index: dict | None = None
_attack_db: dict | None = None
_structured_atk_to_cards: dict | None = None
_structured_card_to_atks: dict | None = None

# ---------------------------------------------------------------------------
# Attack index
# ---------------------------------------------------------------------------

def _load_attack_index() -> dict:
    """Lazy-load the attack index from pickle."""
    global _attack_index
    if _attack_index is not None:
        return _attack_index

    if not _ATTACK_INDEX_PATH.exists():
        logger.warning("Attack index not found at %s", _ATTACK_INDEX_PATH)
        _attack_index = {"attack_to_cards": {}, "card_to_attacks": {}}
        return _attack_index

    with open(_ATTACK_INDEX_PATH, "rb") as f:
        _attack_index = pickle.load(f)
    logger.info(
        "Loaded attack index: %d attacks, %d cards",
        len(_attack_index.get("attack_to_cards", {})),
        len(_attack_index.get("card_to_attacks", {})),
    )
    return _attack_index


def _load_attack_db() -> dict:
    """Lazy-load the precomputed attack DB (OCR'd reference card attacks)."""
    global _attack_db
    if _attack_db is not None:
        return _attack_db

    if not _ATTACK_DB_PATH.exists():
        logger.warning("Attack DB not found at %s", _ATTACK_DB_PATH)
        _attack_db = {}
        return _attack_db

    import json
    with open(_ATTACK_DB_PATH) as f:
        _attack_db = json.load(f)
    logger.info("Loaded attack DB: %d cards", len(_attack_db))
    return _attack_db


def _load_structured_attacks() -> tuple[dict, dict]:
    """Lazy-load structured_attacks.json and build reverse index.

    The file has 19,895 cards with clean attack + ability names scraped
    from the official API.  We build two dicts (same shape as attack_index.pkl):
      - atk_to_cards: {attack_name_lower: [card_id, ...]}
      - card_to_atks: {card_id: [attack_name_lower, ...]}

    Card IDs in the structured file are base IDs (e.g. "ecard3-116").
    We store them as-is (no "/normal" suffix) so callers that strip the
    variant suffix can still match.  We *also* store with "/normal" so
    the existing scoring loop (which uses suffixed IDs from the pkl
    index) finds them.

    Returns
    -------
    tuple of (atk_to_cards, card_to_atks)
    """
    global _structured_atk_to_cards, _structured_card_to_atks
    if _structured_atk_to_cards is not None:
        return _structured_atk_to_cards, _structured_card_to_atks

    if not _STRUCTURED_ATTACKS_PATH.exists():
        logger.warning(
            "Structured attacks not found at %s", _STRUCTURED_ATTACKS_PATH
        )
        _structured_atk_to_cards = {}
        _structured_card_to_atks = {}
        return _structured_atk_to_cards, _structured_card_to_atks

    import json

    with open(_STRUCTURED_ATTACKS_PATH) as f:
        data = json.load(f)

    atk_to_cards: dict[str, list[str]] = {}
    card_to_atks: dict[str, list[str]] = {}

    for base_id, entry in data.items():
        names: list[str] = []
        for atk in entry.get("attacks", []):
            name = atk.get("name", "").strip()
            if name:
                names.append(name.lower())
        for abi in entry.get("abilities", []):
            name = abi.get("name", "").strip()
            if name:
                names.append(name.lower())

        if not names:
            continue

        # Store under both base ID and /normal-suffixed ID
        for cid in (base_id, f"{base_id}/normal"):
            card_to_atks[cid] = names
            for atk_name in names:
                atk_to_cards.setdefault(atk_name, []).append(cid)

    _structured_atk_to_cards = atk_to_cards
    _structured_card_to_atks = card_to_atks
    logger.info(
        "Loaded structured attacks: %d attacks, %d cards",
        len(_structured_atk_to_cards),
        len(_structured_card_to_atks),
    )
    return _structured_atk_to_cards, _structured_card_to_atks


def _get_all_attack_names() -> list[str]:
    """Return sorted list of all known attack names (lowercase).

    Merges names from both attack_index.pkl and structured_attacks.json.
    """
    idx = _load_attack_index()
    names = set(idx.get("attack_to_cards", {}).keys())
    struct_atk, _ = _load_structured_attacks()
    names.update(struct_atk.keys())
    return sorted(names)


# ---------------------------------------------------------------------------
# EasyOCR reader
# ---------------------------------------------------------------------------

def _get_reader():
    """Get the shared EasyOCR reader from ocr_matcher (saves ~500MB RAM)."""
    from cardprice.ml.ocr_matcher import get_easyocr_reader
    return get_easyocr_reader()


# ---------------------------------------------------------------------------
# Image cropping and preprocessing
# ---------------------------------------------------------------------------

def crop_attack_region(img: np.ndarray) -> np.ndarray:
    """Crop to the attack text region of a Pokemon card.

    Attack names are in the middle-to-lower portion:
    - Vertically: roughly 38%-78% of card height
      (above the flavor text / dex entry, below the illustration)
    - Horizontally: center 80% to avoid border and energy cost symbols

    Parameters
    ----------
    img : np.ndarray
        Full card image (BGR or grayscale).

    Returns
    -------
    np.ndarray
        Cropped attack region.
    """
    h, w = img.shape[:2]
    y_start = int(h * 0.38)
    y_end = int(h * 0.78)
    x_start = int(w * 0.10)
    x_end = int(w * 0.90)
    return img[y_start:y_end, x_start:x_end]


def preprocess_attack_region(img: np.ndarray) -> np.ndarray:
    """Enhance the attack region crop for better OCR.

    Steps:
      1. Convert to grayscale
      2. Upscale if small (binder segments are ~630x880)
      3. Apply CLAHE for local contrast
      4. Light sharpen

    Parameters
    ----------
    img : np.ndarray
        Attack region crop (BGR or grayscale).

    Returns
    -------
    np.ndarray
        Preprocessed grayscale image.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape[:2]
    if h < 400:
        scale = 2
        gray = cv2.resize(
            gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
        )

    # CLAHE for contrast normalization (helps with holo glare)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


# ---------------------------------------------------------------------------
# OCR fragment filtering
# ---------------------------------------------------------------------------

# Patterns that indicate a fragment is NOT an attack name
_NOISE_PATTERNS = [
    # Damage numbers: "30", "50+", "120x", "80-"
    re.compile(r"^\d{1,3}[\+x\-]?$"),
    # Energy costs: single letters/symbols common in cost text
    re.compile(r"^[RWFGDMLPCSY]{1,2}$", re.IGNORECASE),
    # Weakness/resistance markers
    re.compile(r"^[\+\-x]\d{1,3}$"),
    # Very short fragments (1-2 chars)
    re.compile(r"^.{0,2}$"),
    # Common non-attack text fragments
    re.compile(
        r"^(weakness|resistance|retreat|cost|hp|lv\b|stage|basic|"
        r"pokemon|power|poke-body|poke-power|pok\u00e9-body|pok\u00e9-power|ability|pokedex|"
        r"illustrator|illus|rarity|\d+/\d+|no\.|put|this|the|and|"
        r"your|each|from|does|coin|flip|damage|energy|attach|"
        r"discard|shuffle|deck|hand|active|bench|turn|"
        r"opponent|defender|asleep|confused|paralyzed|poisoned|burned)$",
        re.IGNORECASE,
    ),
    # Fragments that are mostly digits
    re.compile(r"^\d[\d\s/\-\.]{2,}$"),
    # Stage indicators: "STAGE 1", "STAGE 2" (matches "strange bell" at 0.632)
    re.compile(r"^stage\s*\d?$", re.IGNORECASE),
]

# Additional heuristics for description text (smaller font, longer phrases)
_MAX_ATTACK_NAME_WORDS = 5
_MIN_ATTACK_NAME_LEN = 3


def _is_likely_attack_name(text: str, confidence: float) -> bool:
    """Filter an OCR fragment to determine if it's likely an attack name.

    Attack names are:
    - Typically 1-4 words
    - Bold/larger text (higher OCR confidence)
    - Not damage numbers, energy costs, or description text

    Parameters
    ----------
    text : str
        OCR text fragment.
    confidence : float
        OCR confidence (0-1).

    Returns
    -------
    bool
        True if the fragment looks like an attack name.
    """
    text = text.strip()

    # Too short
    if len(text) < _MIN_ATTACK_NAME_LEN:
        return False

    # Very low confidence usually means noise
    if confidence < 0.15:
        return False

    # Check noise patterns
    for pattern in _NOISE_PATTERNS:
        if pattern.match(text):
            return False

    # Too many words -> likely a description sentence
    words = text.split()
    if len(words) > _MAX_ATTACK_NAME_WORDS:
        return False

    # Description text heuristic: if it contains common filler words
    # in the middle, it's probably a description rather than an attack name
    text_lower = text.lower()
    description_markers = [
        " the ", " this ", " your ", " each ", " from ",
        " does ", " coin ", " flip ", " if ", " may ",
        " all ", " any ", " into ", " then ", " also ",
        " during ", " between ", " opponent",
    ]
    marker_count = sum(1 for m in description_markers if m in text_lower)
    if marker_count >= 2:
        return False

    # Single-word fragments that start with lowercase are usually
    # description fragments (attack names are capitalized on cards)
    if len(words) == 1 and text[0].islower() and confidence < 0.5:
        return False

    return True


def _recombine_fragments(
    results: list,
) -> list[tuple[str, float, list]]:
    """Recombine OCR fragments that are on the same text line.

    EasyOCR sometimes splits multi-word attack names into separate
    fragments (e.g. "Body" and "Slam" on the same line). We detect
    fragments at similar Y positions and merge them left-to-right.

    Parameters
    ----------
    results : list
        Raw EasyOCR results: [(bbox, text, confidence), ...].

    Returns
    -------
    list of (text, confidence, bbox_list) tuples
        Merged fragments grouped by text line.
    """
    if not results:
        return []

    # Sort by Y center, then X center
    def center(bbox):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return (sum(ys) / len(ys), sum(xs) / len(xs))

    items = []
    for bbox, text, conf in results:
        cy, cx = center(bbox)
        h = max(p[1] for p in bbox) - min(p[1] for p in bbox)
        items.append({
            "bbox": bbox, "text": text.strip(), "conf": float(conf),
            "cy": cy, "cx": cx, "h": max(h, 1),
        })

    items.sort(key=lambda it: (it["cy"], it["cx"]))

    # Group by Y proximity (same line if centers within half the text height)
    lines: list[list[dict]] = []
    current_line = [items[0]]
    for it in items[1:]:
        prev = current_line[-1]
        threshold = max(prev["h"], it["h"]) * 0.6
        if abs(it["cy"] - prev["cy"]) <= threshold:
            current_line.append(it)
        else:
            lines.append(current_line)
            current_line = [it]
    lines.append(current_line)

    # Merge each line into a single text
    merged = []
    for line in lines:
        line.sort(key=lambda it: it["cx"])
        combined_text = " ".join(it["text"] for it in line if it["text"])
        avg_conf = sum(it["conf"] for it in line) / len(line)
        bboxes = [it["bbox"] for it in line]
        merged.append((combined_text, avg_conf, bboxes))

    return merged


def extract_attack_names(
    image_path: str | Path,
) -> list[tuple[str, float]]:
    """Extract likely attack names from a card image using OCR.

    Steps:
    1. Crop to attack region
    2. Preprocess (grayscale, upscale, CLAHE)
    3. Run EasyOCR
    4. Recombine fragments on the same text line
    5. Filter to likely attack names

    Parameters
    ----------
    image_path : str or Path
        Path to the card segment image.

    Returns
    -------
    list of (text, confidence) tuples
        Extracted attack name candidates with OCR confidence.
    """
    image_path = str(Path(image_path).resolve())

    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Failed to read image: %s", image_path)
        return []

    # Crop to attack region
    attack_crop = crop_attack_region(img)
    processed = preprocess_attack_region(attack_crop)

    # Run EasyOCR
    reader = _get_reader()
    results = reader.readtext(processed, detail=1, paragraph=False, batch_size=8)

    if not results:
        return []

    # Recombine fragments on the same text line
    merged_lines = _recombine_fragments(results)

    # Also keep individual fragments as separate candidates
    # (sometimes a line merges attack name with description)
    individual = [(text.strip(), float(conf))
                  for _, text, conf in results if text.strip()]

    # Filter merged lines to likely attack names
    candidates = []
    seen = set()
    for text, conf, _ in merged_lines:
        text = text.strip()
        if text and _is_likely_attack_name(text, conf):
            key = text.lower()
            if key not in seen:
                candidates.append((text, conf))
                seen.add(key)
                logger.debug("  KEEP (merged): '%s' (conf=%.2f)", text, conf)

    # Also add individual fragments that pass the filter
    for text, conf in individual:
        if _is_likely_attack_name(text, conf):
            key = text.lower()
            if key not in seen:
                candidates.append((text, conf))
                seen.add(key)
                logger.debug("  KEEP (single): '%s' (conf=%.2f)", text, conf)

    return candidates


# ---------------------------------------------------------------------------
# RapidOCR-based attack extraction (replaces PaddleOCR, 10x faster than EasyOCR)
# ---------------------------------------------------------------------------

_rapid_ocr_engine = None


def _get_rapid_ocr():
    """Lazy singleton for the RapidOCR engine."""
    global _rapid_ocr_engine
    if _rapid_ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _rapid_ocr_engine = RapidOCR()
    return _rapid_ocr_engine


def _recombine_rapid_fragments(
    fragments: list[dict],
) -> list[tuple[str, float, list]]:
    """Recombine RapidOCR fragments that are on the same text line.

    RapidOCR returns per-line detections, but occasionally splits
    multi-word text. We group fragments at similar Y positions and
    merge them left-to-right, same logic as the EasyOCR recombiner.

    Parameters
    ----------
    fragments : list of dict
        Each dict has keys: text, conf, cx, cy, h.

    Returns
    -------
    list of (text, confidence, bbox_list) tuples
        Merged fragments grouped by text line.
    """
    if not fragments:
        return []

    items = sorted(fragments, key=lambda it: (it["cy"], it["cx"]))

    lines: list[list[dict]] = []
    current_line = [items[0]]
    for it in items[1:]:
        prev = current_line[-1]
        threshold = max(prev["h"], it["h"]) * 0.6
        if abs(it["cy"] - prev["cy"]) <= threshold:
            current_line.append(it)
        else:
            lines.append(current_line)
            current_line = [it]
    lines.append(current_line)

    merged = []
    for line in lines:
        line.sort(key=lambda it: it["cx"])
        combined_text = " ".join(it["text"] for it in line if it["text"])
        avg_conf = sum(it["conf"] for it in line) / len(line)
        merged.append((combined_text, avg_conf, []))

    return merged


def extract_attack_names_paddle(
    image_path: str | Path,
    det_model=None,
    rec_model=None,
) -> list[tuple[str, float]]:
    """Extract likely attack names using RapidOCR (faster than EasyOCR).

    Uses the same crop region, preprocessing, and filtering as
    extract_attack_names(), but runs RapidOCR detection + recognition
    instead of EasyOCR.

    The det_model and rec_model parameters are accepted for backward
    compatibility but ignored (RapidOCR uses a single engine).

    Parameters
    ----------
    image_path : str or Path
        Path to the card segment image.
    det_model : ignored
        Kept for backward compatibility with callers.
    rec_model : ignored
        Kept for backward compatibility with callers.

    Returns
    -------
    list of (text, confidence) tuples
        Extracted attack name candidates with OCR confidence.
    """
    image_path = str(Path(image_path).resolve())

    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Failed to read image: %s", image_path)
        return []

    # Crop to attack region
    attack_crop = crop_attack_region(img)
    processed = preprocess_attack_region(attack_crop)

    # RapidOCR expects 3-channel BGR input
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    # Upscale for better detection (same rationale as name OCR)
    h, w = processed.shape[:2]
    if h < 600:
        scale = max(2, 600 // max(h, 1))
        processed = cv2.resize(
            processed, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
        )

    # Pad edges so text isn't at the very border
    processed = cv2.copyMakeBorder(
        processed, 20, 20, 20, 20, cv2.BORDER_REPLICATE
    )

    # Run RapidOCR (combined detection + recognition)
    try:
        engine = _get_rapid_ocr()
        result, _ = engine(processed)
    except Exception as e:
        logger.warning("RapidOCR attack detection failed: %s", e)
        return []

    if not result:
        return []

    # Parse RapidOCR results: each item is [box, text, confidence]
    # box is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    fragments = []
    for box, text, conf in result:
        text = text.strip()
        conf = float(conf)
        if not text or len(text) < 2 or conf < 0.2:
            continue
        # Compute center and height from polygon box
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        cy = sum(ys) / len(ys)
        cx = sum(xs) / len(xs)
        bh = max(ys) - min(ys)
        fragments.append({
            "text": text, "conf": conf,
            "cx": cx, "cy": cy, "h": max(bh, 1),
        })

    if not fragments:
        return []

    # Recombine fragments on the same text line
    merged_lines = _recombine_rapid_fragments(fragments)

    # Also keep individual fragments as separate candidates
    individual = [(f["text"], f["conf"]) for f in fragments if f["text"]]

    # Filter merged lines to likely attack names
    candidates = []
    seen = set()
    for text, conf, _ in merged_lines:
        text = text.strip()
        if text and _is_likely_attack_name(text, conf):
            key = text.lower()
            if key not in seen:
                candidates.append((text, conf))
                seen.add(key)
                logger.debug("  KEEP (merged/rapid): '%s' (conf=%.2f)", text, conf)

    # Also add individual fragments that pass the filter
    for text, conf in individual:
        if _is_likely_attack_name(text, conf):
            key = text.lower()
            if key not in seen:
                candidates.append((text, conf))
                seen.add(key)
                logger.debug("  KEEP (single/rapid): '%s' (conf=%.2f)", text, conf)

    return candidates


# ---------------------------------------------------------------------------
# Fuzzy matching against attack index
# ---------------------------------------------------------------------------

def _fuzzy_ratio(a: str, b: str) -> float:
    """SequenceMatcher ratio between two lowercased strings."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def fuzzy_match_attacks(
    ocr_candidates: list[tuple[str, float]],
    known_attacks: list[str] | None = None,
    threshold: float = 0.60,
) -> list[tuple[str, str, float]]:
    """Fuzzy-match OCR fragments against known attack names.

    Parameters
    ----------
    ocr_candidates : list of (text, confidence) tuples
        OCR-extracted attack name candidates.
    known_attacks : list of str, optional
        Known attack names to match against. If None, uses all attacks
        from the attack index.
    threshold : float
        Minimum fuzzy ratio to accept a match.

    Returns
    -------
    list of (ocr_text, matched_attack, score) tuples
        Matched attacks with their fuzzy match scores.
    """
    if not ocr_candidates:
        return []

    if known_attacks is None:
        known_attacks = _get_all_attack_names()

    # Build set of multi-word attack names for n-gram matching
    multiword_attacks = [a for a in known_attacks if " " in a]

    # Generate n-gram candidates from consecutive OCR fragments.
    # EasyOCR often splits "Body Slam" into separate "Body" and "Slam"
    # fragments. Combining adjacent fragments recovers multi-word names.
    ngram_candidates = list(ocr_candidates)  # start with originals
    for n in (2, 3):
        for i in range(len(ocr_candidates) - n + 1):
            words = [ocr_candidates[i + j][0] for j in range(n)]
            confs = [ocr_candidates[i + j][1] for j in range(n)]
            combined = " ".join(words)
            avg_conf = sum(confs) / len(confs)
            ngram_candidates.append((combined, avg_conf))

    matches = []
    used_attacks = set()
    # Track best match per attack (prefer higher-scoring n-gram matches)
    best_per_attack: dict[str, tuple[str, float]] = {}

    for raw_ocr_text, ocr_conf in ngram_candidates:
        # Skip low-confidence OCR fragments — garbled text (e.g. Japanese
        # misread as Latin) produces false fuzzy matches against real attacks.
        if ocr_conf < 0.40:
            continue
        # Strip trailing damage numbers that OCR merged with attack name
        # e.g. "Bite 20" -> "Bite", "Slash 30+" -> "Slash"
        ocr_text = re.sub(r'\s*\d{1,3}[+x]?\s*$', '', raw_ocr_text).strip()
        if not ocr_text:
            continue
        best_attack = None
        best_score = 0.0

        word_count = len(ocr_text.split())

        # For short single-word fragments, require higher threshold
        # and only match single-word attacks (prevents "Call" -> "recall")
        effective_threshold = threshold
        search_attacks = known_attacks
        if word_count == 1 and len(ocr_text) <= 3:
            # Very short fragments need exact match — "Cut" is only 3 chars,
            # fuzzy ratio("cunt","cut")=85% is a false match.
            search_attacks = [a for a in known_attacks if " " not in a]
            effective_threshold = 1.0
        elif word_count == 1 and len(ocr_text) <= 6:
            search_attacks = [a for a in known_attacks if " " not in a]
            effective_threshold = max(threshold, 0.80)
        elif word_count >= 2:
            # Multi-word OCR fragments: search ALL attacks, not just multi-word.
            # OCR noise can append garbage to single-word attack names
            # (e.g. "Foresight L"), so we must not exclude single-word attacks.
            search_attacks = known_attacks

        result = _rprocess.extractOne(
            ocr_text.lower().strip(), search_attacks,
            scorer=_rfuzz.ratio,
            score_cutoff=effective_threshold * 100,
            processor=lambda s: s.lower().strip(),
        )
        if result:
            best_attack, best_score = result[0], result[1] / 100.0

        # Fallback to all attacks if no good match in filtered set
        if best_score < effective_threshold and search_attacks is not known_attacks:
            result2 = _rprocess.extractOne(
                ocr_text.lower().strip(), known_attacks,
                scorer=_rfuzz.ratio,
                score_cutoff=effective_threshold * 100,
                processor=lambda s: s.lower().strip(),
            )
            if result2 and result2[1] / 100.0 > best_score:
                best_attack, best_score = result2[0], result2[1] / 100.0

        if best_attack and best_score >= effective_threshold:
            # Keep the best-scoring OCR fragment for each attack
            prev = best_per_attack.get(best_attack)
            if prev is None or best_score > prev[1]:
                best_per_attack[best_attack] = (ocr_text, best_score)

    # Convert to output format
    for attack, (ocr_text, score) in best_per_attack.items():
        matches.append((ocr_text, attack, score))

    # Sort by score descending for stable output
    matches.sort(key=lambda x: -x[2])
    return matches


# ---------------------------------------------------------------------------
# Main entry point: OCR attacks -> candidate card IDs
# ---------------------------------------------------------------------------

def identify_by_attacks(
    image_path: str | Path,
    pokemon_name: str | None = None,
    candidate_card_ids: list[str] | None = None,
    fuzzy_threshold: float = 0.60,
    precomputed_ocr_candidates: list | None = None,
) -> list[tuple[str, float]]:
    """Identify a card by OCR-reading its attack names and matching the index.

    End-to-end pipeline:
      1. Crop attack region from the card image
      2. Run EasyOCR to extract text fragments
      3. Filter to likely attack names
      4. Fuzzy-match against attack_index.pkl
      5. Score candidate cards by attack overlap

    Parameters
    ----------
    image_path : str or Path
        Path to the card segment image.
    pokemon_name : str, optional
        If provided, boosts candidates whose DB name matches.
    candidate_card_ids : list of str, optional
        If provided, only score these card IDs (for narrowing an
        existing candidate set).
    fuzzy_threshold : float
        Minimum fuzzy ratio for accepting an OCR-to-attack match.

    Returns
    -------
    list of (card_id, score) tuples
        Candidates sorted by descending score. Score is 0.0-1.0.
    """
    idx = _load_attack_index()
    atk_to_cards_pkl = idx.get("attack_to_cards", {})
    card_to_atks_pkl = idx.get("card_to_attacks", {})

    # Load structured attacks (clean API data: attacks + abilities)
    struct_atk_to_cards, struct_card_to_atks = _load_structured_attacks()

    # Merged lookup helpers — check both sources
    def _merged_card_attacks(cid: str) -> set[str]:
        """Get all known attacks for a card from both indices."""
        attacks = set(card_to_atks_pkl.get(cid, []))
        attacks.update(struct_card_to_atks.get(cid, []))
        # Also check base ID (without variant suffix)
        base = cid.split("/")[0]
        attacks.update(struct_card_to_atks.get(base, []))
        return attacks

    def _merged_atk_cards(atk_name: str) -> set[str]:
        """Get all card IDs that have a given attack from both indices."""
        cards = set(atk_to_cards_pkl.get(atk_name, []))
        cards.update(struct_atk_to_cards.get(atk_name, []))
        return cards

    # Step 1-3: Extract attack name candidates from image
    ocr_candidates = precomputed_ocr_candidates if precomputed_ocr_candidates is not None else extract_attack_names(image_path)
    if not ocr_candidates:
        logger.info("No attack candidates extracted from %s", Path(image_path).name)
        return []

    logger.info(
        "Attack OCR candidates from %s: %s",
        Path(image_path).name,
        [t for t, _ in ocr_candidates],
    )

    # Step 4: Determine which known attacks to match against
    if candidate_card_ids:
        # Only match against attacks belonging to the candidate cards
        relevant_attacks = set()
        for cid in candidate_card_ids:
            relevant_attacks.update(_merged_card_attacks(cid))
        known_attacks = sorted(relevant_attacks) if relevant_attacks else None
    else:
        known_attacks = None  # match against all

    matched = fuzzy_match_attacks(
        ocr_candidates, known_attacks=known_attacks, threshold=fuzzy_threshold
    )

    if not matched:
        logger.info("No attacks matched above threshold %.2f", fuzzy_threshold)
        return []

    matched_attack_names = [atk for _, atk, _ in matched]
    logger.info(
        "Matched attacks: %s",
        [(ocr, atk, f"{s:.2f}") for ocr, atk, s in matched],
    )

    # Step 5: Score candidate cards by attack overlap
    # Find all cards that have any of the matched attacks
    if candidate_card_ids:
        cards_to_score = set(candidate_card_ids)
    else:
        cards_to_score = set()
        for atk_name in matched_attack_names:
            cards_to_score.update(_merged_atk_cards(atk_name))

    if not cards_to_score:
        return []

    # Score each candidate
    scored: list[tuple[str, float]] = []
    for cid in cards_to_score:
        card_attacks = _merged_card_attacks(cid)
        if not card_attacks:
            continue

        matched_set = set(matched_attack_names)

        # Overlap: how many of the card's attacks did we find?
        intersection = card_attacks & matched_set
        if not intersection:
            continue

        # Attack overlap score: matched / total card attacks
        overlap_ratio = len(intersection) / len(card_attacks)

        # Precision: how many of our matched attacks belong to this card?
        precision = len(intersection) / len(matched_set) if matched_set else 0

        # Combined score
        score = 0.4 * overlap_ratio + 0.4 * precision

        # Bonus for matching ALL card attacks (strong signal), but only
        # for cards with 2+ attacks — a single-attack match is too weak
        # to warrant a bonus since many unrelated cards share one attack.
        if intersection == card_attacks and len(card_attacks) >= 2:
            score += 0.15

        # Bonus for absolute match count: matching 2 attacks is stronger
        # evidence than matching 1, even if both are 100% overlap.
        score += 0.05 * min(len(intersection), 3)

        # Bonus for fuzzy match quality
        avg_fuzzy = 0.0
        for _, atk, fuzz_score in matched:
            if atk in intersection:
                avg_fuzzy += fuzz_score
        if intersection:
            avg_fuzzy /= len(intersection)
        score += 0.05 * avg_fuzzy

        # Name boost if pokemon_name provided
        if pokemon_name:
            # Quick check: does the card_id base match the pokemon name?
            # card_id format: "base1-4/normal"
            name_lower = pokemon_name.lower().strip()
            name_lower = re.sub(
                r"\s*(ex|δ|delta|v|vstar|vmax|gx|lv\.\w+|star)\s*$",
                "", name_lower,
            ).strip()
            # We'd need DB access for proper name matching, so just use
            # a lightweight heuristic
            cid_base = cid.split("/")[0]  # "ex15-92"
            # No easy name check without DB, skip for now

        scored.append((cid, round(min(score, 1.0), 4)))

    # Sort by score descending
    scored.sort(key=lambda x: -x[1])

    logger.info(
        "Attack-based candidates (top 5): %s",
        scored[:5],
    )

    return scored


def narrow_candidates(
    image_path: str | Path,
    candidate_card_ids: list[str],
    fuzzy_threshold: float = 0.55,
) -> list[tuple[str, float]]:
    """Narrow an existing candidate set using attack OCR.

    This is the intended use case: when name + type matching produces
    multiple candidates (e.g. 5 different Pikachu printings), run attack
    OCR to disambiguate.

    Parameters
    ----------
    image_path : str or Path
        Path to the card segment image.
    candidate_card_ids : list of str
        Card IDs to narrow down.
    fuzzy_threshold : float
        Minimum fuzzy ratio for attack matching.

    Returns
    -------
    list of (card_id, score) tuples
        Re-ranked candidates. If attack OCR produces no signal, returns
        empty list (caller should fall back to other methods).
    """
    if not candidate_card_ids:
        return []

    return identify_by_attacks(
        image_path,
        candidate_card_ids=candidate_card_ids,
        fuzzy_threshold=fuzzy_threshold,
    )
