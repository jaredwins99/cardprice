#!/usr/bin/env python3
"""Test whether OCR can read card numbers from binder page segments.

Card numbers (e.g. "92/101", "H32/H32") are printed at the bottom of Pokemon
cards in very small text. At our 1008x1530 segment resolution, the number text
is only ~8-12px tall. This script tests whether PaddleOCR and/or EasyOCR can
read these numbers, with various upscaling and preprocessing strategies.

Tested on:
  - Xatu H32 (ecard3-H32) vs Xatu 35 (ecard3-35) -- identical artwork variants
  - Venusaur 13/127 (pl3-13) -- modern card with clearer printing
  - Suicune 19/132 (dp3-19) -- DP-era card
  - Several other cards from eval set

Usage:
    python scripts/test_card_number_ocr.py
    python scripts/test_card_number_ocr.py --paddle-only
    python scripts/test_card_number_ocr.py --easyocr-only
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIRES_BASE = PROJECT_ROOT / "data" / "inbox"
DEBUG_DIR = PROJECT_ROOT / "data" / "debug_card_number"

# Test cards: (segment_path, expected_number, description)
TEST_CARDS = [
    # Page 1 (e-card era) -- the critical test cases
    (HIRES_BASE / "page_20260228_195512_cards_hires/card_01.png",
     "H32/H32", "Xatu H32 (our unsolvable failure)"),
    (HIRES_BASE / "page_20260228_195512_cards_hires/card_04.png",
     "35/144", "Xatu 35/144 (identical artwork)"),
    (HIRES_BASE / "page_20260228_195512_cards_hires/card_00.png",
     "80/144", "Natu ecard3-80 (pos 0,0)"),
    (HIRES_BASE / "page_20260228_195512_cards_hires/card_05.png",
     "90/144", "Rattata ecard3-90"),
    (HIRES_BASE / "page_20260228_195512_cards_hires/card_07.png",
     "89/144", "Raticate ecard3-89"),
    (HIRES_BASE / "page_20260228_195512_cards_hires/card_08.png",
     "51/144", "Ditto ecard3-51"),
    # Page 2 (mixed DP/Platinum era)
    (HIRES_BASE / "page_20260228_202134_cards_hires/card_03.png",
     "13/127", "Venusaur pl3-13 (modern, clear)"),
    (HIRES_BASE / "page_20260228_202134_cards_hires/card_04.png",
     "5/127", "Flygon pl2-5 (holo)"),
    (HIRES_BASE / "page_20260228_202134_cards_hires/card_05.png",
     "16/132", "Raikou dp3-16 (reverse holo)"),
    (HIRES_BASE / "page_20260228_202134_cards_hires/card_07.png",
     "19/132", "Suicune dp3-19"),
    (HIRES_BASE / "page_20260228_202134_cards_hires/card_08.png",
     "16/130", "Staraptor dp1-16 (reverse holo)"),
]


# ---------------------------------------------------------------------------
# Crop / preprocess helpers
# ---------------------------------------------------------------------------

def crop_bottom_region(img: np.ndarray, top_pct: float = 0.87,
                       bot_pct: float = 0.97) -> np.ndarray:
    """Crop a wide bottom strip to capture the card number region."""
    h, w = img.shape[:2]
    y0 = int(h * top_pct)
    y1 = int(h * bot_pct)
    # Trim 5% margins to skip binder sleeve edges
    x0 = int(w * 0.03)
    x1 = int(w * 0.97)
    return img[y0:y1, x0:x1]


def make_variants(crop: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """Generate preprocessed variants at multiple upscale levels."""
    variants = []
    h, w = crop.shape[:2]

    for scale in [4, 6, 8]:
        big = cv2.resize(crop, (w * scale, h * scale),
                         interpolation=cv2.INTER_CUBIC)

        # A: plain upscale
        variants.append((big, f"color_{scale}x"))

        # B: CLAHE on grayscale
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enh = clahe.apply(gray)
        variants.append((cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR),
                         f"clahe_{scale}x"))

        # C: sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharp = cv2.filter2D(big, -1, kernel)
        variants.append((sharp, f"sharp_{scale}x"))

        # D: Otsu binarize
        _, otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
                         f"otsu_{scale}x"))

        # E: unsharp mask
        blur = cv2.GaussianBlur(big, (0, 0), 3)
        unsharp = cv2.addWeighted(big, 1.5, blur, -0.5, 0)
        variants.append((unsharp, f"unsharp_{scale}x"))

    return variants


# ---------------------------------------------------------------------------
# OCR engines
# ---------------------------------------------------------------------------

def run_paddleocr(variants: list[tuple[np.ndarray, str]],
                  det_model, rec_model) -> list[tuple[str, float, str]]:
    """Run PaddleOCR det+rec on each variant. Returns [(text, conf, label)]."""
    results = []
    for img, label in variants:
        try:
            det_out = list(det_model.predict(img))
            if not det_out or not det_out[0]:
                continue
            res = det_out[0]
            if not hasattr(res, 'boxes') or res.boxes is None:
                continue
            for box in res.boxes:
                pts = np.array(box, dtype=np.float32)
                x0 = max(0, int(pts[:, 0].min()))
                x1 = min(img.shape[1], int(pts[:, 0].max()))
                y0 = max(0, int(pts[:, 1].min()))
                y1 = min(img.shape[0], int(pts[:, 1].max()))
                if x1 <= x0 or y1 <= y0:
                    continue
                box_crop = img[y0:y1, x0:x1]
                if box_crop.shape[0] < 3 or box_crop.shape[1] < 3:
                    continue
                rec_out = list(rec_model.predict(box_crop))
                if rec_out and rec_out[0]:
                    for item in rec_out:
                        if hasattr(item, '__iter__'):
                            for sub in item:
                                if hasattr(sub, 'text'):
                                    results.append((sub.text, sub.score, label))
        except Exception as e:
            results.append((f"ERROR:{e}", 0.0, label))
    return results


def run_easyocr(variants: list[tuple[np.ndarray, str]],
                reader) -> list[tuple[str, float, str]]:
    """Run EasyOCR on each variant. Returns [(text, conf, label)]."""
    results = []
    for img, label in variants:
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 and img.shape[2] == 3 else img
            out = reader.readtext(
                rgb, detail=1, paragraph=False,
                text_threshold=0.3, low_text=0.3,
            )
            for bbox, text, conf in out:
                results.append((text.strip(), float(conf), label))
        except Exception as e:
            results.append((f"ERROR:{e}", 0.0, label))
    return results


# ---------------------------------------------------------------------------
# Also test the existing extract_card_number function from ocr_matcher.py
# ---------------------------------------------------------------------------

def run_existing_extractor(image_path: Path) -> tuple:
    """Run the existing extract_card_number from ocr_matcher.py."""
    try:
        from cardprice.ml.ocr_matcher import extract_card_number
        return extract_card_number(str(image_path))
    except Exception as e:
        return None, None, 0.0


# ---------------------------------------------------------------------------
# Matching / scoring
# ---------------------------------------------------------------------------

def find_card_number_in_texts(
    texts: list[tuple[str, float, str]], expected: str
) -> tuple[bool, str | None, float, str | None]:
    """Search OCR results for a card number pattern matching expected.

    Returns (matched, found_text, confidence, variant_label).
    """
    # Try to find X/Y patterns in all texts
    for text, conf, label in texts:
        # Strict: digits/digits
        for m in re.finditer(r'(\d{1,3})\s*/\s*(\d{2,3})', text):
            found = f"{m.group(1)}/{m.group(2)}"
            if matches_expected(found, expected):
                return True, found, conf, label

        # H-prefix for e-card holos: H32/H32
        for m in re.finditer(r'([Hh]\d{1,3})\s*/\s*([Hh]\d{1,3})', text):
            found = f"{m.group(1)}/{m.group(2)}"
            if matches_expected(found, expected):
                return True, found, conf, label

    # Pass 2: OCR char substitutions
    for text, conf, label in texts:
        fixed = text
        for old, new in [('O', '0'), ('o', '0'), ('l', '1'), ('I', '1'),
                         ('S', '5'), ('B', '8'), ('D', '0')]:
            fixed = fixed.replace(old, new)
        for m in re.finditer(r'(\d{1,3})\s*/\s*(\d{2,3})', fixed):
            found = f"{m.group(1)}/{m.group(2)}"
            if matches_expected(found, expected):
                return True, found, conf * 0.8, label

    # Pass 3: slash-like separators
    for text, conf, label in texts:
        for m in re.finditer(r'(\d{1,3})\s*[/|\\]\s*(\d{2,3})', text):
            found = f"{m.group(1)}/{m.group(2)}"
            if matches_expected(found, expected):
                return True, found, conf * 0.7, label

    return False, None, 0.0, None


def matches_expected(found: str, expected: str) -> bool:
    """Check if found number matches expected, ignoring case."""
    return found.lower().replace(" ", "") == expected.lower().replace(" ", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    paddle_only = "--paddle-only" in sys.argv
    easy_only = "--easyocr-only" in sys.argv
    use_paddle = not easy_only
    use_easy = not paddle_only

    print("=" * 72)
    print("CARD NUMBER OCR FEASIBILITY TEST")
    print("=" * 72)
    print(f"Segment resolution: 1008x1530")
    print(f"Bottom crop: 87-97% of height = ~153px tall")
    print(f"Text height at 1x: ~8-12px")
    print(f"Upscale factors tested: 4x, 6x, 8x")
    print(f"Preprocessing variants per scale: 5 (color, clahe, sharp, otsu, unsharp)")
    print(f"Total variants per card: 15")
    print(f"Test cards: {len(TEST_CARDS)}")
    print(f"OCR engines: {'PaddleOCR' if use_paddle else ''} {'EasyOCR' if use_easy else ''}")
    print()

    # Validate images
    missing = [t for t in TEST_CARDS if not t[0].exists()]
    if missing:
        for path, _, desc in missing:
            print(f"MISSING: {path} ({desc})")
        sys.exit(1)
    print("All test images found.")

    # Init OCR engines
    det_model = rec_model = reader = None
    if use_paddle:
        print("\nLoading PaddleOCR...")
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        from cardprice.ml.ocr_matcher import get_paddle_engines
        det_model, rec_model = get_paddle_engines()
        print("  PaddleOCR ready.")

    if use_easy:
        print("Loading EasyOCR...")
        import easyocr
        reader = easyocr.Reader(['en'], gpu=True)
        print("  EasyOCR ready.")

    DEBUG_DIR.mkdir(exist_ok=True)

    # Track results
    paddle_scores = {}
    easy_scores = {}
    existing_scores = {}

    for seg_path, expected, desc in TEST_CARDS:
        print(f"\n{'#' * 72}")
        print(f"# {desc}")
        print(f"# Expected: {expected}")
        print(f"# File: {seg_path.name}")
        print(f"{'#' * 72}")

        img = cv2.imread(str(seg_path))
        h, w = img.shape[:2]

        # Crop bottom region
        crop = crop_bottom_region(img)
        ch, cw = crop.shape[:2]
        print(f"  Image: {w}x{h}, crop: {cw}x{ch}")

        # Save debug crops
        safe = desc.replace(" ", "_").replace("/", "-")[:35]
        cv2.imwrite(str(DEBUG_DIR / f"{safe}_crop.png"), crop)

        # Generate variants
        variants = make_variants(crop)
        # Save a couple debug variants
        for vi, (vimg, vlabel) in enumerate(variants[:3]):
            cv2.imwrite(str(DEBUG_DIR / f"{safe}_{vlabel}.png"), vimg)

        # --- PaddleOCR ---
        if use_paddle:
            t0 = time.time()
            paddle_texts = run_paddleocr(variants, det_model, rec_model)
            elapsed = time.time() - t0
            matched, found, conf, vlabel = find_card_number_in_texts(
                paddle_texts, expected)

            print(f"\n  PaddleOCR ({elapsed:.1f}s):")
            # Show all unique texts found
            seen = set()
            for text, c, lab in paddle_texts:
                key = text.strip()
                if key and key not in seen:
                    seen.add(key)
                    print(f"    [{lab:18s}] '{key}' (conf={c:.3f})")

            if matched:
                print(f"  >>> MATCH: '{found}' via {vlabel} (conf={conf:.3f})")
            else:
                print(f"  >>> NO MATCH for '{expected}'")
            paddle_scores[desc] = (matched, found, conf)

        # --- EasyOCR ---
        if use_easy:
            t0 = time.time()
            easy_texts = run_easyocr(variants, reader)
            elapsed = time.time() - t0
            matched, found, conf, vlabel = find_card_number_in_texts(
                easy_texts, expected)

            print(f"\n  EasyOCR ({elapsed:.1f}s):")
            seen = set()
            for text, c, lab in easy_texts:
                key = text.strip()
                if key and key not in seen:
                    seen.add(key)
                    print(f"    [{lab:18s}] '{key}' (conf={c:.3f})")

            if matched:
                print(f"  >>> MATCH: '{found}' via {vlabel} (conf={conf:.3f})")
            else:
                print(f"  >>> NO MATCH for '{expected}'")
            easy_scores[desc] = (matched, found, conf)

        # --- Existing extract_card_number ---
        t0 = time.time()
        num, total, conf = run_existing_extractor(seg_path)
        elapsed = time.time() - t0
        found_str = f"{num}/{total}" if num else "(none)"
        matched = False
        if num and total:
            matched = matches_expected(f"{num}/{total}", expected)
        print(f"\n  Existing extractor ({elapsed:.1f}s): {found_str} (conf={conf:.2f}) "
              f"{'MATCH' if matched else 'NO MATCH'}")
        existing_scores[desc] = (matched, found_str, conf)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n\n{'=' * 72}")
    print("SUMMARY TABLE")
    print(f"{'=' * 72}\n")

    header = f"  {'Card':<38} {'Expected':<10}"
    if use_paddle:
        header += f" {'Paddle':<8} {'Found':<12}"
    if use_easy:
        header += f" {'Easy':<8} {'Found':<12}"
    header += f" {'Existing':<8}"
    print(header)
    print(f"  {'-' * 38} {'-' * 10}", end="")
    if use_paddle:
        print(f" {'-' * 8} {'-' * 12}", end="")
    if use_easy:
        print(f" {'-' * 8} {'-' * 12}", end="")
    print(f" {'-' * 8}")

    p_total = p_ok = e_total = e_ok = x_total = x_ok = 0
    for _, expected, desc in TEST_CARDS:
        line = f"  {desc:<38} {expected:<10}"

        if use_paddle and desc in paddle_scores:
            m, f, c = paddle_scores[desc]
            line += f" {'YES' if m else 'NO':<8} {(f or ''):12}"
            p_total += 1
            p_ok += int(m)

        if use_easy and desc in easy_scores:
            m, f, c = easy_scores[desc]
            line += f" {'YES' if m else 'NO':<8} {(f or ''):12}"
            e_total += 1
            e_ok += int(m)

        if desc in existing_scores:
            m, f, c = existing_scores[desc]
            line += f" {'YES' if m else 'NO':<8}"
            x_total += 1
            x_ok += int(m)

        print(line)

    print()
    if use_paddle:
        print(f"  PaddleOCR:  {p_ok}/{p_total} matched")
    if use_easy:
        print(f"  EasyOCR:    {e_ok}/{e_total} matched")
    print(f"  Existing:   {x_ok}/{x_total} matched")

    # -----------------------------------------------------------------------
    # Resolution analysis
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("RESOLUTION ANALYSIS")
    print(f"{'=' * 72}")
    print("""
At 1008x1530 segments (current):
  - Bottom crop (87-97%) = ~153px tall, ~950px wide
  - Card number text is ~8-12px tall at native resolution
  - At 4x upscale: ~32-48px tall (marginal for OCR)
  - At 6x upscale: ~48-72px tall (borderline readable)
  - At 8x upscale: ~64-96px tall (should be readable if sharp enough)

Key factors affecting readability:
  1. Binder page scan blur (phone camera at distance)
  2. Perspective distortion at card edges
  3. Glare from binder sleeve plastic
  4. Low contrast (faint gray text on card background)
  5. Font size varies by card era (e-card smallest)

For reliable card number OCR, we would need:
  - Original segments at ~2500x3800 (2.5x current) for 20-30px text, OR
  - Targeted high-res phone photo of just the card bottom, OR
  - Super-resolution ML model (Real-ESRGAN) before OCR
""")

    # -----------------------------------------------------------------------
    # Xatu-specific analysis
    # -----------------------------------------------------------------------
    print(f"{'=' * 72}")
    print("XATU H32 vs 35/144 ANALYSIS")
    print(f"{'=' * 72}")
    xatu_h32 = paddle_scores.get("Xatu H32 (our unsolvable failure)") if use_paddle else None
    xatu_35 = paddle_scores.get("Xatu 35/144 (identical artwork)") if use_paddle else None
    if xatu_h32 and xatu_35:
        if xatu_h32[0] and xatu_35[0]:
            print("  Both Xatu variants readable -- card number OCR CAN solve this!")
        elif xatu_h32[0] or xatu_35[0]:
            print("  Partial: one variant readable, one not.")
        else:
            print("  Neither Xatu variant readable at current resolution.")
    print()

    # -----------------------------------------------------------------------
    # Ultra-tight crop experiment
    # -----------------------------------------------------------------------
    print(f"{'=' * 72}")
    print("ULTRA-TIGHT CROP EXPERIMENT (12x upscale, number region only)")
    print(f"{'=' * 72}")
    print("  Testing whether isolating just the number area helps...\n")

    tight_tests = [
        # (path, expected, desc, y0%, y1%, x0%, x1%)
        (HIRES_BASE / "page_20260228_195512_cards_hires/card_04.png",
         "35/144", "Xatu 35/144", 0.875, 0.895, 0.78, 0.96),
        (HIRES_BASE / "page_20260228_195512_cards_hires/card_01.png",
         "H32/H32", "Xatu H32", 0.875, 0.895, 0.78, 0.96),
        (HIRES_BASE / "page_20260228_202134_cards_hires/card_03.png",
         "13/127", "Venusaur", 0.945, 0.965, 0.72, 0.93),
    ]

    for tpath, texpected, tdesc, ty0, ty1, tx0, tx1 in tight_tests:
        timg = cv2.imread(str(tpath))
        th, tw = timg.shape[:2]
        tiny = timg[int(th*ty0):int(th*ty1), int(tw*tx0):int(tw*tx1)]
        print(f"  {tdesc}: tiny crop {tiny.shape[1]}x{tiny.shape[0]} -> 12x = "
              f"{tiny.shape[1]*12}x{tiny.shape[0]*12}")

        big = cv2.resize(tiny, None, fx=12, fy=12, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        clahe_obj = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        enh = clahe_obj.apply(gray)
        enh_bgr = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)

        if use_paddle:
            tight_paddle = run_paddleocr([(enh_bgr, "tight_12x")], det_model, rec_model)
            tm, tf, _, _ = find_card_number_in_texts(tight_paddle, texpected)
            texts_str = "; ".join(f"'{t}'" for t, c, l in tight_paddle) or "(none)"
            print(f"    PaddleOCR: {texts_str} -> {'MATCH' if tm else 'NO MATCH'}")

        if use_easy:
            tight_easy = run_easyocr([(enh_bgr, "tight_12x")], reader)
            tm2, tf2, _, _ = find_card_number_in_texts(tight_easy, texpected)
            texts_str2 = "; ".join(f"'{t}'" for t, c, l in tight_easy) or "(none)"
            print(f"    EasyOCR:   {texts_str2} -> {'MATCH' if tm2 else 'NO MATCH'}")

    print()
    print(f"Debug crops saved to: {DEBUG_DIR}/")


if __name__ == "__main__":
    main()
