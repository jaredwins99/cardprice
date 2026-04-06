#!/usr/bin/env python3
"""
Template matching for EX-era set stamps.

Renders text templates ("DRAGON FRONTIERS", "CRYSTAL GUARDIANS", etc.) at various
angles and scales, then uses normalized cross-correlation (cv2.matchTemplate with
TM_CCOEFF_NORMED) to detect stamps in the artwork region of binder-scan cards.

The stamps appear as diagonal gold/white text in the bottom-right quadrant of the
card artwork, typically at ~20-35 degree angles.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys

# EX-era set stamp texts (the text that appears on stamped cards)
EX_SET_STAMPS = {
    "ex7": "TEAM ROCKET RETURNS",
    "ex8": "DEOXYS",
    "ex9": "EMERALD",
    "ex10": "UNSEEN FORCES",
    "ex11": "DELTA SPECIES",
    "ex12": "LEGEND MAKER",
    "ex13": "HOLON PHANTOMS",
    "ex14": "CRYSTAL GUARDIANS",
    "ex15": "DRAGON FRONTIERS",
    "ex16": "POWER KEEPERS",
}

# Font for rendering templates
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Matching parameters
ANGLES = [15, 20, 25, 30, 35, 40]  # degrees, stamps are typically 20-35
SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
FONT_SIZES = [14, 18, 22, 26]  # try multiple sizes to match binder scan resolution


def render_text_template(text: str, font_size: int, angle: float, color: int = 255) -> np.ndarray:
    """Render text at an angle as a grayscale template image.

    Creates white text on black background, rotated to the given angle.
    Returns a grayscale numpy array.
    """
    font = ImageFont.truetype(FONT_PATH, font_size)

    # Measure text size
    dummy = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4

    # Render text on black background
    img = Image.new("L", (tw, th), 0)
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), text, fill=color, font=font)

    # Rotate (expand=True to avoid clipping)
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True)

    return np.array(rotated)


def extract_stamp_region(card_img: np.ndarray) -> np.ndarray:
    """Extract the artwork region where stamps appear.

    Stamps are in the bottom-right of the artwork, which is roughly
    the right half of the top 55% of the card (below the name bar,
    above the attack text).

    We take a generous region: right 70%, vertical 20%-60% of card height.
    """
    h, w = card_img.shape[:2]
    # Stamp region: bottom-right of artwork
    y1, y2 = int(h * 0.15), int(h * 0.55)
    x1, x2 = int(w * 0.25), w
    return card_img[y1:y2, x1:x2]


def match_stamp_template(card_path: str, stamp_texts: list[str] = None,
                         verbose: bool = False) -> dict:
    """Run template matching for stamp detection on a card image.

    Args:
        card_path: Path to card image
        stamp_texts: List of stamp texts to try. If None, tries all EX stamps.
        verbose: Print detailed match info

    Returns:
        Dict with best match info: {text, angle, scale, font_size, score, location}
    """
    if stamp_texts is None:
        stamp_texts = list(EX_SET_STAMPS.values())

    card_bgr = cv2.imread(card_path)
    if card_bgr is None:
        raise FileNotFoundError(f"Cannot read: {card_path}")

    # Convert to grayscale
    card_gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)

    # Extract stamp region
    stamp_region = extract_stamp_region(card_gray)
    rh, rw = stamp_region.shape[:2]

    best = {"text": None, "angle": 0, "scale": 1.0, "font_size": 0,
            "score": -1.0, "location": (0, 0)}

    for text in stamp_texts:
        for font_size in FONT_SIZES:
            for angle in ANGLES:
                # Render template
                template = render_text_template(text, font_size, angle)

                for scale in SCALES:
                    # Scale template
                    th, tw = template.shape[:2]
                    new_w = int(tw * scale)
                    new_h = int(th * scale)

                    # Skip if template is bigger than search region
                    if new_w >= rw or new_h >= rh:
                        continue
                    # Skip tiny templates
                    if new_w < 10 or new_h < 5:
                        continue

                    scaled = cv2.resize(template, (new_w, new_h),
                                       interpolation=cv2.INTER_AREA)

                    # Normalized cross-correlation
                    result = cv2.matchTemplate(stamp_region, scaled,
                                              cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    if max_val > best["score"]:
                        best = {
                            "text": text,
                            "angle": angle,
                            "scale": scale,
                            "font_size": font_size,
                            "score": max_val,
                            "location": max_loc,
                        }
                        if verbose and max_val > 0.25:
                            print(f"  New best: {max_val:.4f} "
                                  f"text={text} angle={angle} "
                                  f"scale={scale} font={font_size}")

    # Also try inverted (dark text on light background)
    # Some stamps appear as darker text on lighter artwork
    stamp_region_inv = 255 - stamp_region

    for text in stamp_texts:
        for font_size in FONT_SIZES:
            for angle in ANGLES:
                template = render_text_template(text, font_size, angle)

                for scale in SCALES:
                    th, tw = template.shape[:2]
                    new_w = int(tw * scale)
                    new_h = int(th * scale)

                    if new_w >= rw or new_h >= rh or new_w < 10 or new_h < 5:
                        continue

                    scaled = cv2.resize(template, (new_w, new_h),
                                       interpolation=cv2.INTER_AREA)

                    result = cv2.matchTemplate(stamp_region_inv, scaled,
                                              cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    if max_val > best["score"]:
                        best = {
                            "text": text + " (inv)",
                            "angle": angle,
                            "scale": scale,
                            "font_size": font_size,
                            "score": max_val,
                            "location": max_loc,
                        }
                        if verbose and max_val > 0.25:
                            print(f"  New best (inv): {max_val:.4f} "
                                  f"text={text} angle={angle} "
                                  f"scale={scale} font={font_size}")

    # Additionally try edge-based matching: Canny edges on both template and region
    # This is more robust to color/brightness variations
    stamp_edges = cv2.Canny(stamp_region, 50, 150)

    edge_best = {"text": None, "angle": 0, "scale": 1.0, "font_size": 0,
                 "score": -1.0, "location": (0, 0)}

    for text in stamp_texts:
        for font_size in FONT_SIZES:
            for angle in ANGLES:
                template = render_text_template(text, font_size, angle)
                # Edge-detect the template too
                template_edges = cv2.Canny(template, 50, 150)

                for scale in SCALES:
                    th, tw = template_edges.shape[:2]
                    new_w = int(tw * scale)
                    new_h = int(th * scale)

                    if new_w >= rw or new_h >= rh or new_w < 10 or new_h < 5:
                        continue

                    scaled = cv2.resize(template_edges, (new_w, new_h),
                                       interpolation=cv2.INTER_AREA)

                    result = cv2.matchTemplate(stamp_edges, scaled,
                                              cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    if max_val > edge_best["score"]:
                        edge_best = {
                            "text": text + " (edge)",
                            "angle": angle,
                            "scale": scale,
                            "font_size": font_size,
                            "score": max_val,
                            "location": max_loc,
                        }

    return {"pixel": best, "edge": edge_best}


def main():
    base = Path("/home/godli/cardprice/data/inbox")

    # Test cards with ground truth
    test_cards = [
        # Dragon Frontiers page: card_00 (Chikorita) and card_02 (Meganium) are stamped
        ("page_20260305_094228_cards/card_00.png", True,  "Chikorita - Dragon Frontiers STAMPED"),
        ("page_20260305_094228_cards/card_01.png", False, "Bayleef - Dragon Frontiers unstamped"),
        ("page_20260305_094228_cards/card_02.png", True,  "Meganium - Dragon Frontiers STAMPED"),
        ("page_20260305_094228_cards/card_03.png", False, "Totodile - Dragon Frontiers unstamped"),
        ("page_20260305_094228_cards/card_04.png", False, "Croconaw - Dragon Frontiers unstamped"),
        ("page_20260305_094228_cards/card_05.png", False, "Feraligatr - Dragon Frontiers unstamped"),
        ("page_20260305_094228_cards/card_06.png", False, "Cyndaquil - Dragon Frontiers unstamped"),
        ("page_20260305_094228_cards/card_07.png", False, "Quilava - Dragon Frontiers unstamped"),
        ("page_20260305_094228_cards/card_08.png", False, "Typhlosion - Dragon Frontiers unstamped"),
        # Crystal Guardians / Dragon Frontiers stamps from another page
        ("page_20260228_174819_cards/card_01.png", True,  "Skitty - Crystal Guardians STAMPED"),
        ("page_20260228_174819_cards/card_05.png", True,  "Vibrava - Dragon Frontiers STAMPED"),
    ]

    # Focus on the most likely stamp texts for these cards
    focus_texts = ["DRAGON FRONTIERS", "CRYSTAL GUARDIANS"]

    print("=" * 90)
    print(f"{'Card':<50} {'Stamped':>7} {'Pixel':>8} {'Edge':>8} {'Best Text'}")
    print("=" * 90)

    results = []
    for rel_path, is_stamped, label in test_cards:
        card_path = str(base / rel_path)
        print(f"\nProcessing: {label}...")

        res = match_stamp_template(card_path, stamp_texts=focus_texts, verbose=True)

        pixel_score = res["pixel"]["score"]
        edge_score = res["edge"]["score"]
        best_text = res["pixel"]["text"] if pixel_score >= edge_score else res["edge"]["text"]
        best_score = max(pixel_score, edge_score)

        results.append((label, is_stamped, pixel_score, edge_score, best_text))

        marker = "STAMPED" if is_stamped else "      "
        print(f"  => pixel={pixel_score:.4f}  edge={edge_score:.4f}  "
              f"best_text={best_text}")

    print("\n" + "=" * 90)
    print(f"\n{'Card':<50} {'Truth':>7} {'Pixel':>8} {'Edge':>8} {'Best Match'}")
    print("-" * 90)
    for label, is_stamped, pixel_score, edge_score, best_text in results:
        truth = "YES" if is_stamped else "no"
        print(f"{label:<50} {truth:>7} {pixel_score:>8.4f} {edge_score:>8.4f} {best_text}")

    # Analyze separation
    stamped_pixel = [r[2] for r in results if r[1]]
    unstamped_pixel = [r[2] for r in results if not r[1]]
    stamped_edge = [r[3] for r in results if r[1]]
    unstamped_edge = [r[3] for r in results if not r[1]]

    print(f"\n--- Separation Analysis ---")
    print(f"Pixel scores:  stamped={[f'{s:.4f}' for s in stamped_pixel]}")
    print(f"               unstamped={[f'{s:.4f}' for s in unstamped_pixel]}")
    print(f"  stamped min={min(stamped_pixel):.4f}  unstamped max={max(unstamped_pixel):.4f}  "
          f"gap={min(stamped_pixel) - max(unstamped_pixel):.4f}")

    print(f"Edge scores:   stamped={[f'{s:.4f}' for s in stamped_edge]}")
    print(f"               unstamped={[f'{s:.4f}' for s in unstamped_edge]}")
    print(f"  stamped min={min(stamped_edge):.4f}  unstamped max={max(unstamped_edge):.4f}  "
          f"gap={min(stamped_edge) - max(unstamped_edge):.4f}")


if __name__ == "__main__":
    main()
