#!/usr/bin/env python3
"""
Research script: test detect_pokemon_name on page 1 and page 2 cards,
then try a series of improvements on any failures.

Usage:
    python scripts/test_name_ocr_research.py [--page1] [--page2] [--debug]
"""

import sys
import os
import argparse
import time

# Add project root to path
sys.path.insert(0, "/home/godli/cardprice")

# Suppress TF/CUDA noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Ground truth (from project eval logs / known binder content)
# ---------------------------------------------------------------------------
PAGE1_GROUND_TRUTH = {
    # e-card era cards, page 20260228_195512
    "card_00.png": None,   # fill in after first run
    "card_01.png": None,
    "card_02.png": None,
    "card_03.png": None,
    "card_04.png": None,
    "card_05.png": None,
    "card_06.png": None,
    "card_07.png": None,
    "card_08.png": None,
}

PAGE2_GROUND_TRUTH = {
    "card_00.png": None,
    "card_01.png": None,
    "card_02.png": None,
    "card_03.png": None,
    "card_04.png": None,
    "card_05.png": None,
    "card_06.png": None,
    "card_07.png": None,
    "card_08.png": None,
}

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
from cardprice.ml.ocr_matcher import detect_pokemon_name, _easyocr_reader


def get_reader():
    """Get or init the global EasyOCR reader."""
    global _easyocr_reader
    import cardprice.ml.ocr_matcher as ocr_mod
    if ocr_mod._easyocr_reader is None:
        import easyocr
        print("  [init] Loading EasyOCR reader...")
        ocr_mod._easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return ocr_mod._easyocr_reader


# ---------------------------------------------------------------------------
# Preprocessing variants to test
# ---------------------------------------------------------------------------
def preprocess_sharpen(crop_up: np.ndarray) -> np.ndarray:
    """Apply unsharp masking / sharpening to the upscaled crop."""
    # Unsharp mask: sharp = original + alpha*(original - blur)
    blur = cv2.GaussianBlur(crop_up, (0, 0), 2.0)
    sharp = cv2.addWeighted(crop_up, 1.5, blur, -0.5, 0)
    return sharp


def preprocess_bilateral(crop_up: np.ndarray) -> np.ndarray:
    """Bilateral filter: reduces noise while preserving edges."""
    # d=9 is a strong but still fast bilateral filter
    return cv2.bilateralFilter(crop_up, d=9, sigmaColor=75, sigmaSpace=75)


def preprocess_clahe_gray(crop_up: np.ndarray) -> np.ndarray:
    """CLAHE on grayscale — classic low-contrast enhancer."""
    gray = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_sharpen_then_bilateral(crop_up: np.ndarray) -> np.ndarray:
    """Sharpen first, then bilateral to clean up ringing."""
    sharpened = preprocess_sharpen(crop_up)
    return cv2.bilateralFilter(sharpened, d=7, sigmaColor=50, sigmaSpace=50)


def preprocess_gamma(crop_up: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    """Gamma correction to brighten dark name regions."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
                     dtype=np.uint8)
    return cv2.LUT(crop_up, table)


def preprocess_contrast_stretch(crop_up: np.ndarray) -> np.ndarray:
    """Per-channel linear stretch to [0,255]."""
    out = crop_up.copy().astype(np.float32)
    for c in range(3):
        ch = out[:, :, c]
        lo, hi = ch.min(), ch.max()
        if hi > lo:
            out[:, :, c] = (ch - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------
def run_variant(reader, crop_up: np.ndarray, preprocessor,
                name: str, text_threshold: float = 0.3,
                low_text: float = 0.3, paragraph: bool = False) -> list[tuple]:
    """Run EasyOCR with a given preprocessed crop, return raw results."""
    img = preprocessor(crop_up)
    results = reader.readtext(
        img, detail=1, paragraph=paragraph,
        text_threshold=text_threshold, low_text=low_text,
    )
    out = []
    for item in results:
        # paragraph=True returns (text, conf) 2-tuples; detail=1 returns (bbox, text, conf)
        if len(item) == 3:
            _bbox, text, conf = item
        elif len(item) == 2:
            text, conf = item
        else:
            continue
        text = str(text).strip()
        if len(text) >= 2:
            out.append((text, float(conf)))
    return out


def test_card_baseline(card_path: Path, debug: bool = False) -> dict:
    """Run the current detect_pokemon_name on a card and return results."""
    t0 = time.time()
    name, conf = detect_pokemon_name(card_path, debug=debug)
    elapsed = time.time() - t0
    return {"name": name, "conf": conf, "elapsed": elapsed}


def test_card_variants(card_path: Path, debug: bool = False) -> dict:
    """
    Test multiple preprocessing variants on the name region.
    Returns a dict of variant -> [(text, conf), ...]

    Focused on the most likely improvements for e-card era failures:
    - Different crop heights (e-card name is slightly lower)
    - 4x upscale
    - Sharpening and bilateral denoising
    - CLAHE grayscale
    - Lowered thresholds (0.2)
    """
    reader = get_reader()
    img = cv2.imread(str(card_path))
    if img is None:
        return {}
    h, w = img.shape[:2]

    # Focused crop specs: baseline + e-card era variants
    crop_specs = {
        "top12":    (0,  12, 3, 97),
        "top15":    (0,  15, 5, 95),
        "top20":    (0,  20, 5, 95),
        "top25":    (0,  25, 3, 97),
        "skip2_18": (2, 18, 5, 95),
        "skip5_20": (5, 20, 5, 95),
    }

    preprocessors = {
        "raw":        lambda x: x,
        "sharpen":    preprocess_sharpen,
        "bilateral":  preprocess_bilateral,
        "clahe_gray": preprocess_clahe_gray,
        "sharp_bilat": preprocess_sharpen_then_bilateral,
        "gamma":      preprocess_gamma,
        "contrast":   preprocess_contrast_stretch,
    }

    results = {}

    for crop_name, (t, b, l, r) in crop_specs.items():
        y1, y2 = int(h * t / 100), int(h * b / 100)
        x1, x2 = int(w * l / 100), int(w * r / 100)
        crop = img[y1:y2, x1:x2]

        for fx in [3, 4]:
            crop_up = cv2.resize(crop, None, fx=fx, fy=fx,
                                 interpolation=cv2.INTER_LANCZOS4)
            for pp_name, pp_fn in preprocessors.items():
                key = f"{crop_name}_{fx}x_{pp_name}"
                # Test both threshold levels
                for thresh in [0.3, 0.2]:
                    vkey = key if thresh == 0.3 else f"{key}_t02"
                    ocr_out = run_variant(reader, crop_up, pp_fn,
                                         name=vkey, text_threshold=thresh,
                                         low_text=thresh)
                    if ocr_out:
                        results[vkey] = ocr_out

    return results


_STATIC_NAMES_CACHE: list[str] | None = None

def _load_names_from_json() -> list[str]:
    """Load unique Pokemon names from card_attacks.json (no DB needed)."""
    global _STATIC_NAMES_CACHE
    if _STATIC_NAMES_CACHE is not None:
        return _STATIC_NAMES_CACHE
    import json
    json_path = "/home/godli/cardprice/data/card_attacks.json"
    with open(json_path) as f:
        data = json.load(f)
    seen = set()
    names = []
    for v in data.values():
        name = v.get("name", "")
        if name and name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    names.sort()
    _STATIC_NAMES_CACHE = names
    return names


def fuzzy_match_name(text: str) -> tuple[str | None, float]:
    """Do a quick fuzzy match of an OCR string against the Pokemon name DB."""
    from rapidfuzz import fuzz, process
    from cardprice.ml.ocr_matcher import _clean_name_ocr

    cleaned = _clean_name_ocr(text)
    if not cleaned or len(cleaned) < 3:
        return None, 0.0
    alpha = sum(1 for c in cleaned if c.isalpha())
    if alpha / len(cleaned) < 0.6:
        return None, 0.0

    unique_names = _load_names_from_json()
    name_list_lower = [n.lower() for n in unique_names]
    lower_to_original = {n.lower(): n for n in unique_names}

    q = cleaned.lower()
    if q in lower_to_original:
        return lower_to_original[q], 1.0

    m = process.extractOne(q, name_list_lower, scorer=fuzz.ratio, score_cutoff=60.0)
    if m:
        matched_lower, score, _ = m
        return lower_to_original[matched_lower], score / 100.0

    m2 = process.extractOne(q, name_list_lower, scorer=fuzz.partial_ratio, score_cutoff=85.0)
    if m2:
        matched_lower, score, _ = m2
        return lower_to_original[matched_lower], (score * 0.85) / 100.0

    return None, 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_page(page_dir: Path, page_label: str, debug: bool = False):
    cards = sorted(page_dir.glob("card_*.png"))
    if not cards:
        print(f"  No cards found in {page_dir}")
        return

    print(f"\n{'='*70}")
    print(f"  PAGE: {page_label}  ({len(cards)} cards)")
    print(f"  Dir:  {page_dir}")
    print(f"{'='*70}")

    failures = []

    for card_path in cards:
        print(f"\n--- {card_path.name} ---")
        result = test_card_baseline(card_path, debug=debug)
        name = result["name"]
        conf = result["conf"]
        elapsed = result["elapsed"]
        print(f"  BASELINE: name={name!r:30s}  conf={conf:.3f}  t={elapsed:.2f}s")

        if name is None or conf < 0.65:
            print(f"  *** FAILURE — running variant analysis ***")
            failures.append(card_path)

            variants = test_card_variants(card_path, debug=debug)
            # For each variant, try fuzzy matching the OCR text
            best_variant = None
            best_matched = None
            best_matched_conf = 0.0

            for vkey, ocr_results in variants.items():
                for raw_text, ocr_conf in ocr_results:
                    matched_name, matched_conf = fuzzy_match_name(raw_text)
                    if matched_conf > best_matched_conf:
                        best_matched_conf = matched_conf
                        best_matched = matched_name
                        best_variant = (vkey, raw_text, ocr_conf, matched_name, matched_conf)

            if best_variant:
                vk, rt, oc, mn, mc = best_variant
                print(f"  BEST VARIANT: [{vk}]")
                print(f"    raw OCR: {rt!r} (ocr_conf={oc:.2f})")
                print(f"    fuzzy -> {mn!r} (fuzzy_conf={mc:.3f})")

                # Show all variants that found something
                print(f"  All variants with matches:")
                variant_matches = {}
                for vkey, ocr_results in variants.items():
                    for raw_text, ocr_conf in ocr_results:
                        matched_name, matched_conf = fuzzy_match_name(raw_text)
                        if matched_conf >= 0.65:
                            k = f"{matched_name}|{matched_conf:.2f}"
                            if k not in variant_matches:
                                variant_matches[k] = []
                            variant_matches[k].append(f"{vkey}:{raw_text!r}({ocr_conf:.2f})")
                for k, sources in sorted(variant_matches.items(), key=lambda x: -float(x[0].split('|')[1])):
                    print(f"    {k:40s} <- {sources[:3]}")
            else:
                print(f"  NO VARIANT could match a name (card may be truly unreadable)")
                # Print raw OCR from all variants to understand what EasyOCR sees
                print(f"  All raw OCR output across variants (top results):")
                seen_texts = set()
                for vkey, ocr_results in sorted(variants.items())[:10]:
                    for raw_text, ocr_conf in ocr_results:
                        if raw_text not in seen_texts and ocr_conf > 0.3:
                            seen_texts.add(raw_text)
                            print(f"    [{vkey}] {raw_text!r} conf={ocr_conf:.2f}")

    print(f"\n{'='*70}")
    print(f"  SUMMARY for {page_label}")
    print(f"  Total cards: {len(cards)}")
    print(f"  Failures (no name or conf<0.65): {len(failures)}")
    print(f"  Success rate: {(len(cards)-len(failures))/len(cards)*100:.0f}%")
    if failures:
        print(f"  Failed cards: {[p.name for p in failures]}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page1", action="store_true", default=False)
    parser.add_argument("--page2", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    # Default: run both
    if not args.page1 and not args.page2:
        args.page1 = True
        args.page2 = True

    # Pre-load names from JSON into the ocr_matcher cache so no DB needed
    print("Loading Pokemon names from card_attacks.json...")
    names = _load_names_from_json()
    import cardprice.ml.ocr_matcher as ocr_mod
    ocr_mod._unique_pokemon_names = names
    # Also pre-load the card_names cache with (card_id, name, set_id) tuples
    import json
    with open("/home/godli/cardprice/data/card_attacks.json") as f:
        _attack_data = json.load(f)
    ocr_mod._card_names_cache = [
        (card_id, v["name"], card_id.split("-")[0])
        for card_id, v in _attack_data.items()
        if v.get("name")
    ]
    print(f"  Loaded {len(names)} unique names, {len(ocr_mod._card_names_cache)} card entries.\n")

    # Pre-warm the EasyOCR reader once
    print("Pre-warming EasyOCR reader...")
    get_reader()
    print("  Reader ready.\n")

    if args.page1:
        run_page(
            Path("/home/godli/cardprice/data/inbox/page_20260228_195512_cards"),
            "PAGE1 (e-card era non-holo)",
            debug=args.debug,
        )

    if args.page2:
        run_page(
            Path("/home/godli/cardprice/data/inbox/page_20260228_202134_cards"),
            "PAGE2 (mixed era)",
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
