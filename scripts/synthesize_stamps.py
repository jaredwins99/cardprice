#!/usr/bin/env python3
"""Synthesize stamped card images from clean reference images for EX era sets.

EX-era Pokemon cards (2004-2007) had "stamped" reverse holo variants with a
semi-transparent foil stamp of the set logo/name embossed on the artwork area.

This script creates synthetic stamped images for training a stamp detector.

Usage:
    python scripts/synthesize_stamps.py                  # all sets, all cards
    python scripts/synthesize_stamps.py --set ex7        # single set
    python scripts/synthesize_stamps.py --limit 20       # 20 cards per set
    python scripts/synthesize_stamps.py --preview        # show first image only
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Set definitions: set_id -> stamp text
STAMP_DEFS = {
    "ex7":  "TEAM ROCKET",
    "ex8":  "DEOXYS",
    "ex9":  "EMERALD",
    "ex10": "UNSEEN FORCES",
    "ex11": "\u03b4 DELTA SPECIES",
    "ex12": "LEGEND MAKER",
    "ex13": "HOLON PHANTOMS",
    "ex14": "CRYSTAL GUARDIANS",
    "ex15": "DRAGON FRONTIERS",
    "ex16": "POWER KEEPERS",
}

# Stamp color palettes (R, G, B) - gold/silver variations
STAMP_COLORS = [
    (212, 175, 55),   # gold
    (207, 181, 59),   # darker gold
    (192, 192, 192),  # silver
    (218, 195, 80),   # warm gold
    (180, 160, 50),   # antique gold
]

BASE_DIR = Path(__file__).resolve().parent.parent
CARD_IMAGES_DIR = BASE_DIR / "data" / "card_images"
OUTPUT_DIR = BASE_DIR / "data" / "condition_training" / "stamps"


def find_font(size: int) -> ImageFont.FreeTypeFont:
    """Find a bold font available on the system."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    # Fallback to default
    return ImageFont.load_default()


def create_stamp_overlay(
    card_w: int,
    card_h: int,
    stamp_text: str,
    angle_deg: float = 35.0,
    opacity: float = 0.35,
    color: tuple = (212, 175, 55),
) -> Image.Image:
    """Create a semi-transparent stamp overlay for the artwork region.

    Returns an RGBA image the same size as the card with the stamp rendered
    in the lower-right quadrant of the artwork area.
    """
    # Artwork region: ~15-75% of height, ~8-92% of width
    art_top = int(card_h * 0.15)
    art_bot = int(card_h * 0.75)
    art_left = int(card_w * 0.08)
    art_right = int(card_w * 0.92)
    art_w = art_right - art_left
    art_h = art_bot - art_top

    # Stamp goes in the lower-right quadrant of artwork
    # Center of stamp placement area
    stamp_cx = art_left + int(art_w * 0.65)
    stamp_cy = art_top + int(art_h * 0.65)

    # Size the font to fit roughly 60-70% of artwork width
    target_text_w = int(art_w * 0.65)
    font_size = 8
    font = find_font(font_size)

    # Binary search for right font size
    lo, hi = 6, 60
    while lo < hi:
        mid = (lo + hi + 1) // 2
        test_font = find_font(mid)
        bbox = test_font.getbbox(stamp_text)
        tw = bbox[2] - bbox[0]
        if tw <= target_text_w:
            lo = mid
        else:
            hi = mid - 1
    font_size = lo
    font = find_font(font_size)
    bbox = font.getbbox(stamp_text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Render text on a larger canvas so rotation doesn't clip
    pad = max(text_w, text_h)
    canvas_size = text_w + text_h + pad  # plenty of room
    text_img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_img)

    # Draw text centered on canvas
    tx = (canvas_size - text_w) // 2
    ty = (canvas_size - text_h) // 2

    alpha_byte = int(255 * opacity)

    # Emboss effect: highlight above-left, shadow below-right, main on top
    emboss_offset = max(1, font_size // 10)

    # Highlight (lighter, above-left)
    light = tuple(min(255, c + 90) for c in color)
    highlight_alpha = int(alpha_byte * 0.5)
    draw.text(
        (tx - emboss_offset, ty - emboss_offset),
        stamp_text,
        font=font,
        fill=(*light, highlight_alpha),
    )

    # Main stamp text
    draw.text((tx, ty), stamp_text, font=font, fill=(*color, alpha_byte))

    # Shadow (darker, below-right)
    dark = tuple(max(0, c - 70) for c in color)
    shadow_alpha = int(alpha_byte * 0.6)
    draw.text(
        (tx + emboss_offset, ty + emboss_offset),
        stamp_text,
        font=font,
        fill=(*dark, shadow_alpha),
    )

    # Rotate
    text_img = text_img.rotate(angle_deg, resample=Image.BICUBIC, expand=False)

    # Slight blur for foil/emboss feel
    text_img = text_img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Crop to bounding box of non-transparent pixels
    arr = np.array(text_img)
    alpha_ch = arr[:, :, 3]
    rows = np.any(alpha_ch > 0, axis=1)
    cols = np.any(alpha_ch > 0, axis=0)
    if not rows.any() or not cols.any():
        # Degenerate: return empty overlay
        return Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    text_img = text_img.crop((c_min, r_min, c_max + 1, r_max + 1))
    stamp_w, stamp_h = text_img.size

    # Position stamp on full card overlay, centered at (stamp_cx, stamp_cy)
    overlay = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    paste_x = stamp_cx - stamp_w // 2
    paste_y = stamp_cy - stamp_h // 2

    # Clamp to card bounds
    paste_x = max(0, min(paste_x, card_w - stamp_w))
    paste_y = max(0, min(paste_y, card_h - stamp_h))

    overlay.paste(text_img, (paste_x, paste_y), text_img)
    return overlay


def apply_stamp(card_img: Image.Image, stamp_text: str, set_id: str) -> Image.Image:
    """Apply a synthetic stamp to a card image.

    Randomly varies angle, color, and opacity for diversity.
    """
    card_w, card_h = card_img.size

    # Random variations for realism
    angle = random.uniform(28, 45)
    opacity = random.uniform(0.35, 0.50)
    color = random.choice(STAMP_COLORS)

    overlay = create_stamp_overlay(
        card_w, card_h, stamp_text,
        angle_deg=angle, opacity=opacity, color=color,
    )

    # Composite: paste stamp overlay onto card
    if card_img.mode != "RGBA":
        card_img = card_img.convert("RGBA")
    result = Image.alpha_composite(card_img, overlay)
    return result


def get_card_images(set_id: str, limit: int | None = None) -> list[tuple[str, Path]]:
    """Get list of (card_id, path) for a set's reference images."""
    set_dir = CARD_IMAGES_DIR / set_id
    if not set_dir.exists():
        print(f"WARNING: {set_dir} does not exist, skipping")
        return []

    results = []
    for fname in sorted(set_dir.iterdir()):
        if not fname.name.endswith("_normal.png"):
            continue
        # card_id like "ex7-100/normal" from filename "ex7-100_normal.png"
        base = fname.stem  # "ex7-100_normal"
        card_id_part = base.replace("_normal", "")  # "ex7-100"
        card_id = f"{card_id_part}/normal"
        results.append((card_id, fname))

    if limit is not None:
        results = results[:limit]
    return results


def process_set(
    set_id: str,
    stamp_text: str,
    limit: int | None = None,
    preview: bool = False,
) -> list[dict]:
    """Process all cards in a set, generating stamped + clean copies.

    Returns list of label entries.
    """
    cards = get_card_images(set_id, limit)
    if not cards:
        return []

    out_dir = OUTPUT_DIR / set_id
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    for i, (card_id, img_path) in enumerate(cards):
        card_img = Image.open(img_path).convert("RGBA")
        stamped_img = apply_stamp(card_img, stamp_text, set_id)

        # File names
        card_id_safe = card_id.replace("/", "_")  # "ex7-100_normal"
        stamped_fname = f"{card_id_safe}_stamped.png"
        clean_fname = f"{card_id_safe}_clean.png"

        stamped_rel = str(Path(set_id) / stamped_fname)
        clean_rel = str(Path(set_id) / clean_fname)

        if preview:
            # Show first image and exit
            print(f"Preview: {card_id} from {set_id}")
            print(f"  Stamp text: {stamp_text}")
            print(f"  Card size: {card_img.size}")

            # Save a temp preview
            preview_path = OUTPUT_DIR / "preview.png"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # Side-by-side comparison
            w, h = card_img.size
            comparison = Image.new("RGBA", (w * 2 + 10, h), (40, 40, 40, 255))
            comparison.paste(card_img, (0, 0))
            comparison.paste(stamped_img, (w + 10, 0))
            comparison.save(preview_path)
            print(f"  Preview saved to: {preview_path}")
            return []

        # Save stamped
        stamped_path = out_dir / stamped_fname
        stamped_img.save(stamped_path)

        # Save clean copy
        clean_path = out_dir / clean_fname
        card_img.save(clean_path)

        # Label entries
        labels.append({
            "image": stamped_rel,
            "stamped": True,
            "set_id": set_id,
            "card_id": card_id,
            "stamp_text": stamp_text,
        })
        labels.append({
            "image": clean_rel,
            "stamped": False,
            "set_id": set_id,
            "card_id": card_id,
        })

        if (i + 1) % 25 == 0:
            print(f"  {set_id}: {i + 1}/{len(cards)} cards processed")

    print(f"  {set_id}: {len(cards)} cards done -> {out_dir}")
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize stamped card images for EX era sets"
    )
    parser.add_argument(
        "--set",
        type=str,
        default=None,
        help="Process only this set (e.g., ex7). Default: all ex7-ex16.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max cards per set. Default: all.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show first image as preview instead of processing all.",
    )
    args = parser.parse_args()

    # Determine which sets to process
    if args.set:
        if args.set not in STAMP_DEFS:
            print(f"ERROR: unknown set '{args.set}'. Valid: {list(STAMP_DEFS.keys())}")
            sys.exit(1)
        sets_to_process = {args.set: STAMP_DEFS[args.set]}
    else:
        sets_to_process = STAMP_DEFS

    random.seed(42)  # Reproducible

    all_labels = []
    for set_id, stamp_text in sets_to_process.items():
        print(f"Processing {set_id}: \"{stamp_text}\"")
        labels = process_set(set_id, stamp_text, args.limit, args.preview)
        all_labels.extend(labels)

        if args.preview and not labels:
            # Preview mode exits after first image
            break

    if args.preview:
        return

    # Write labels file
    if all_labels:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        labels_path = OUTPUT_DIR / "labels.jsonl"
        with open(labels_path, "w") as f:
            for entry in all_labels:
                f.write(json.dumps(entry) + "\n")
        print(f"\nLabels written to {labels_path}")
        print(f"Total images: {len(all_labels)} ({len(all_labels)//2} stamped + {len(all_labels)//2} clean)")


if __name__ == "__main__":
    main()
