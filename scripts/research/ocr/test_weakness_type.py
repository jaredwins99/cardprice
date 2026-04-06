#!/usr/bin/env python3
"""Prototype: Weakness type symbol detection from Pokemon card segments.

Research question: Can we extract the weakness type icon from the bottom
of Pokemon card segments as a second type signal?

Pokemon cards show a weakness type in the bottom section:
  - y: ~82-92% of card height
  - x: ~5-22% of card width
  - Contains a small type symbol + "x2" or "+N" multiplier text

Weakness -> Type mapping:
  Fire weakness     -> Grass or Metal
  Water weakness    -> Fire
  Grass weakness    -> Water
  Lightning weakness -> Water or Colorless
  Psychic weakness  -> Psychic or Fighting
  Fighting weakness -> Grass or Psychic
  Darkness weakness -> Fighting
  Metal weakness    -> Fire

FINDINGS SUMMARY
================
After extensive testing across 15 eval segment cards (3 binder pages),
weakness symbol detection via computer vision is NOT VIABLE at current
segment resolutions.

Key findings:
1. The weakness symbol is ~25px diameter on 1008x1530 hires segments,
   and ~15px on 880x630 v6 segments. This is too small for reliable
   color classification.

2. Approach 1 - HoughCircles: Finds circles but mostly on wrong features
   (text, binder edges, e-Card stamps). 23% accuracy on hires, 0% on v6.

3. Approach 2 - Dark blob detection: The symbol is NOT reliably darker
   than the card background. On dark cards (Water, dark lighting), the
   entire region is dark. 27% accuracy.

4. Approach 3 - K-means clustering: The symbol's ~400 pixels are
   overwhelmed by the 30,000+ background pixels. 13% accuracy.

5. Approach 4 - Background subtraction: Finding pixels that differ from
   the dominant background color picks up text and noise, not the symbol.

6. The reference card images from pokemontcg.io are cropped and do NOT
   include the weakness row, so we can't use them for template matching.

7. Psychic-type cards with Psychic weakness are particularly hard because
   both the card background and the weakness symbol are purple.

RECOMMENDATION: This approach should be abandoned for binder scan segments.
The weakness symbol is simply too small at practical scan resolutions.
Alternative approaches that might work:
  - Higher resolution segments (e.g., 2000x3000) -- but doubles processing time
  - OCR of the "weakness" text label + "x2" multiplier text
  - Claude Vision could read the weakness type directly
  - Use the known weakness-type mapping table as a VALIDATION signal
    after identifying the card, not as a detection signal
"""

import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Weakness -> Pokemon Type mapping
# If we detect a weakness type, these are the possible Pokemon types
# ---------------------------------------------------------------------------
WEAKNESS_TO_TYPE = {
    "Fire": ["Grass", "Metal"],
    "Water": ["Fire"],
    "Grass": ["Water"],
    "Lightning": ["Water", "Colorless"],
    "Psychic": ["Psychic", "Fighting"],
    "Fighting": ["Colorless", "Grass", "Psychic"],
    "Darkness": ["Fighting"],
    "Metal": ["Fire"],
}


# ---------------------------------------------------------------------------
# Ground truth for eval segments
# (card_name, card_type, expected_weakness_type)
# ---------------------------------------------------------------------------
GROUND_TRUTH_HIRES = {
    "page_20260228_202134_cards_hires/card_05.png": ("Raikou", "Lightning", "Fighting"),
    "page_20260228_202134_cards_hires/card_03.png": ("Venusaur", "Grass", "Fire"),
    "page_20260228_202134_cards_hires/card_06.png": ("Kingdra", "Water", "Lightning"),
    "page_20260228_202134_cards_hires/card_07.png": ("Suicune", "Water", "Lightning"),
    "page_20260228_202134_cards_hires/card_08.png": ("Staraptor", "Colorless", "Lightning"),
    "page_20260228_202134_cards_hires/card_04.png": ("Flygon", "Colorless", "Colorless"),
    "page_20260228_195512_cards_hires/card_00.png": ("Natu", "Psychic", "Psychic"),
    "page_20260228_195512_cards_hires/card_01.png": ("Xatu", "Psychic", "Psychic"),
    "page_20260228_195512_cards_hires/card_02.png": ("Mr. Mime", "Psychic", "Psychic"),
    "page_20260228_195512_cards_hires/card_03.png": ("Natu", "Psychic", "Psychic"),
    "page_20260228_195512_cards_hires/card_04.png": ("Xatu H32", "Psychic", "Psychic"),
    "page_20260228_195512_cards_hires/card_05.png": ("Rattata", "Colorless", "Fighting"),
    "page_20260228_195512_cards_hires/card_06.png": ("Rattata", "Colorless", "Fighting"),
    "page_20260228_195512_cards_hires/card_07.png": ("Raticate", "Colorless", "Fighting"),
    "page_20260228_195512_cards_hires/card_08.png": ("Ditto", "Colorless", "Fighting"),
}

GROUND_TRUTH_V6 = {
    k.replace("_hires", "_v6"): v
    for k, v in GROUND_TRUTH_HIRES.items()
}


# ---------------------------------------------------------------------------
# Approach 1: HoughCircles - find circular symbols
# ---------------------------------------------------------------------------
def detect_weakness_hough(img, debug=False):
    """Attempt to find the weakness symbol using Hough circle detection.

    Accuracy: 23% on hires (1008x1530), 0% on v6 (880x630).
    """
    h, w = img.shape[:2]
    expected_d = int(w * 0.025)
    min_r = max(expected_d // 3, 5)
    max_r = max(expected_d, 15)

    y1, y2 = int(h * 0.82), int(h * 0.93)
    x1, x2 = int(w * 0.04), int(w * 0.30)
    strip = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

    best_circles = None
    for blur_size in [3, 5, 7]:
        gray_blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 1.0)
        for p1 in [40, 50, 60]:
            for p2 in [20, 25, 30]:
                circles = cv2.HoughCircles(
                    gray_blur, cv2.HOUGH_GRADIENT,
                    dp=1.2, minDist=15,
                    param1=p1, param2=p2,
                    minRadius=min_r, maxRadius=max_r,
                )
                if circles is not None:
                    if best_circles is None or len(circles[0]) > len(best_circles):
                        best_circles = circles[0]

    if best_circles is None:
        return None

    circles = np.round(best_circles).astype(int)
    sh, sw = strip.shape[:2]

    scored = []
    for cx, cy, r in circles:
        if cx > sw * 0.65:
            continue
        mask = np.zeros(strip.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), max(r - 2, 3), 255, -1)
        pixels = hsv[mask > 0]
        if len(pixels) < 5:
            continue
        mean_h = float(np.mean(pixels[:, 0]))
        mean_s = float(np.mean(pixels[:, 1]))
        mean_v = float(np.mean(pixels[:, 2]))
        y_score = 1.0 if cy > sh * 0.35 else 0.5
        score = y_score * (mean_s / 255) * min(mean_v / 120, 1.0) * (r / max_r)
        scored.append((score, mean_h, mean_s, mean_v))

    if not scored:
        return None

    scored.sort(reverse=True)
    _, mh, ms, mv = scored[0]
    return _classify_weakness_hsv(mh, ms, mv)


# ---------------------------------------------------------------------------
# Approach 2: Dark blob detection
# ---------------------------------------------------------------------------
def detect_weakness_dark_blob(img, debug=False):
    """Find the weakness symbol as a dark, colored blob.

    Accuracy: 27% on hires. Fails when background is already dark (Water cards).
    """
    h, w = img.shape[:2]
    y1, y2 = int(h * 0.82), int(h * 0.93)
    x1, x2 = int(w * 0.04), int(w * 0.22)
    strip = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

    bg_brightness = float(np.median(gray))
    dark_threshold = max(bg_brightness - 40, 50)
    dark_mask = gray < dark_threshold
    colored_dark = dark_mask & (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 30)

    if colored_dark.sum() < 10:
        dark_threshold = max(bg_brightness - 20, 40)
        dark_mask = gray < dark_threshold
        colored_dark = dark_mask & (hsv[:, :, 1] > 25) & (hsv[:, :, 2] > 25)
        if colored_dark.sum() < 10:
            return None

    symbol_pixels = hsv[colored_dark]
    mean_h = float(np.mean(symbol_pixels[:, 0]))
    mean_s = float(np.mean(symbol_pixels[:, 1]))
    mean_v = float(np.mean(symbol_pixels[:, 2]))
    return _classify_weakness_hsv(mean_h, mean_s, mean_v)


# ---------------------------------------------------------------------------
# Approach 3: K-means clustering
# ---------------------------------------------------------------------------
def detect_weakness_kmeans(img, debug=False):
    """Use K-means to find the symbol as a distinct color cluster.

    Accuracy: 13% on hires. Symbol pixels too few to form their own cluster.
    """
    h, w = img.shape[:2]
    y1, y2 = int(h * 0.82), int(h * 0.93)
    x1, x2 = int(w * 0.04), int(w * 0.22)
    strip = img[y1:y2, x1:x2]

    pixels = strip.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    hsv_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

    clusters = []
    for i in range(5):
        mask = labels == i
        frac = mask.sum() / len(labels)
        bgr = centers[i]
        pix = np.array([[bgr.astype(np.uint8)]], dtype=np.uint8)
        hsv_val = cv2.cvtColor(pix, cv2.COLOR_BGR2HSV)[0, 0]
        clusters.append({
            "hsv": (float(hsv_val[0]), float(hsv_val[1]), float(hsv_val[2])),
            "frac": frac,
        })

    bg = max(clusters, key=lambda c: c["frac"])
    bg_h = bg["hsv"][0]

    best_score = 0
    best_type = None
    for c in clusters:
        if c == bg or c["frac"] < 0.01:
            continue
        ch, cs, cv_val = c["hsv"]
        if cv_val < 30:
            continue
        hue_diff = min(abs(ch - bg_h), 180 - abs(ch - bg_h))
        sat_diff = abs(cs - bg["hsv"][1])
        score = hue_diff + sat_diff * 0.5
        if score > best_score:
            best_score = score
            best_type = _classify_weakness_hsv(ch, cs, cv_val)

    return best_type


# ---------------------------------------------------------------------------
# HSV -> Weakness type classifier
# ---------------------------------------------------------------------------
def _classify_weakness_hsv(h: float, s: float, v: float) -> str:
    """Classify HSV color to a Pokemon weakness type name."""
    if s < 25:
        return "Colorless" if v > 80 else "Darkness"
    if h >= 165 or h <= 12:
        return "Fire"
    if 13 <= h <= 22:
        return "Fighting"
    if 23 <= h <= 35:
        return "Lightning"
    if 36 <= h <= 85:
        return "Grass"
    if 86 <= h <= 115:
        return "Water"
    if 116 <= h <= 164:
        return "Psychic"
    return "Colorless"


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------
def run_eval(data_root: str = "data/inbox", ground_truth: dict = None):
    """Run all three approaches against the eval ground truth."""
    root = Path(data_root)

    approaches = [
        ("HoughCircles", detect_weakness_hough),
        ("DarkBlob", detect_weakness_dark_blob),
        ("K-Means", detect_weakness_kmeans),
    ]

    for approach_name, detect_fn in approaches:
        correct = 0
        total = 0
        results = []

        for rel_path, (card_name, card_type, expected_weakness) in ground_truth.items():
            full_path = root / rel_path
            if not full_path.exists():
                results.append(f"  SKIP {card_name}: file missing")
                continue

            img = cv2.imread(str(full_path))
            if img is None:
                results.append(f"  SKIP {card_name}: unreadable")
                continue

            total += 1
            detected = detect_fn(img)
            is_correct = detected == expected_weakness

            if is_correct:
                correct += 1

            marker = "OK  " if is_correct else "MISS"
            det_str = detected if detected else "NONE"
            results.append(
                f"  [{marker}] {card_name:15s} type={card_type:12s} "
                f"expected_weak={expected_weakness:10s} detected={det_str}"
            )

        pct = correct / total * 100 if total else 0
        print(f"\n{'='*70}")
        print(f"Approach: {approach_name} -- Accuracy: {correct}/{total} = {pct:.0f}%")
        print(f"{'='*70}")
        for r in results:
            print(r)

    return


# ---------------------------------------------------------------------------
# Visual analysis: save cropped weakness regions for manual inspection
# ---------------------------------------------------------------------------
def save_weakness_crops(data_root: str = "data/inbox", output_dir: str = "/tmp/weakness_crops"):
    """Save cropped weakness regions from eval segments for manual inspection."""
    root = Path(data_root)
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    for rel_path, (card_name, _, _) in GROUND_TRUTH_HIRES.items():
        full_path = root / rel_path
        if not full_path.exists():
            continue

        img = cv2.imread(str(full_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        y1, y2 = int(h * 0.80), int(h * 0.93)
        x1, x2 = int(w * 0.03), int(w * 0.35)
        crop = img[y1:y2, x1:x2]

        # Save at 2x scale for readability
        big = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                         interpolation=cv2.INTER_LINEAR)

        safe_name = card_name.replace(" ", "_").replace(".", "")
        cv2.imwrite(str(out / f"{safe_name}.png"), big)

    print(f"Saved weakness crops to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)

    print("\n" + "#" * 70)
    print("# EVALUATION: HIRES segments (1008x1530)")
    print("#" * 70)
    run_eval("data/inbox", GROUND_TRUTH_HIRES)

    print("\n\n" + "#" * 70)
    print("# EVALUATION: V6 segments (880x630)")
    print("#" * 70)
    run_eval("data/inbox", GROUND_TRUTH_V6)

    # Also save crops for visual inspection
    print("\n")
    save_weakness_crops()
