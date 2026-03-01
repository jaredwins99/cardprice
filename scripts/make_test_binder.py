"""Create synthetic binder page test images by compositing real card images into a 3x3 grid.

Produces ground-truth test data for the card segmenter and grid detector.
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CARD_IMG_DIR = DATA_DIR / "card_images"

# Pick 9 cards from 9 different sets for variety
CARD_PATHS = [
    CARD_IMG_DIR / "base1" / "base1-100_normal.png",
    CARD_IMG_DIR / "base2" / "base2-10_normal.png",
    CARD_IMG_DIR / "base3" / "base3-10_normal.png",
    CARD_IMG_DIR / "base4" / "base4-10_normal.png",
    CARD_IMG_DIR / "base5" / "base5-10_normal.png",
    CARD_IMG_DIR / "base6" / "base6-10_normal.png",
    CARD_IMG_DIR / "g1" / "g1-13_normal.png",
    CARD_IMG_DIR / "dp1" / "dp1-10_normal.png",
    CARD_IMG_DIR / "bw1" / "bw1-10_normal.png",
]

# Standard Pokemon card aspect ratio: 63mm x 88mm
# Target card size in the composite (pixels)
CARD_W = 245
CARD_H = 342

# Grid layout
COLS, ROWS = 3, 3
GAP = 20  # pixels between cards
MARGIN = 40  # border around the grid

BG_COLOR = (30, 30, 35)  # dark binder page background


def load_and_resize_card(path: Path) -> Image.Image:
    """Load a card image and resize to standard dimensions."""
    img = Image.open(path).convert("RGBA")
    img = img.resize((CARD_W, CARD_H), Image.LANCZOS)
    return img


def composite_grid(cards: list[Image.Image], rotate: bool = False,
                   seed: int = 42) -> Image.Image:
    """Composite card images into a 3x3 grid on a dark background.

    Args:
        cards: List of 9 card images (RGBA).
        rotate: If True, apply slight random rotation (1-3 degrees) per card.
        seed: Random seed for reproducibility.

    Returns:
        Composited binder page image (RGB).
    """
    rng = random.Random(seed)

    canvas_w = 2 * MARGIN + COLS * CARD_W + (COLS - 1) * GAP
    canvas_h = 2 * MARGIN + ROWS * CARD_H + (ROWS - 1) * GAP
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)

    for idx, card in enumerate(cards):
        row, col = divmod(idx, COLS)
        x = MARGIN + col * (CARD_W + GAP)
        y = MARGIN + row * (CARD_H + GAP)

        if rotate:
            angle = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
            # Rotate with expand to avoid clipping, then paste centered
            rotated = card.rotate(angle, resample=Image.BICUBIC, expand=True)
            # Center the rotated card on the target position
            dx = (rotated.width - CARD_W) // 2
            dy = (rotated.height - CARD_H) // 2
            canvas.paste(rotated, (x - dx, y - dy), rotated)
        else:
            canvas.paste(card, (x, y), card)

    return canvas


def main():
    # Verify all card images exist
    missing = [p for p in CARD_PATHS if not p.exists()]
    if missing:
        print(f"Missing card images: {missing}")
        # Try to find alternatives
        for i, p in enumerate(CARD_PATHS):
            if not p.exists():
                set_dir = p.parent
                if set_dir.exists():
                    alt = next(set_dir.glob("*.png"), None)
                    if alt:
                        print(f"  Substituting {alt.name} for {p.name}")
                        CARD_PATHS[i] = alt

    # Load all 9 cards
    cards = [load_and_resize_card(p) for p in CARD_PATHS]
    print(f"Loaded {len(cards)} cards from {len(set(p.parent.name for p in CARD_PATHS))} sets")

    # Clean grid (no rotation)
    clean = composite_grid(cards, rotate=False)
    clean_path = DATA_DIR / "test_binder_clean.jpg"
    clean.save(clean_path, quality=92)
    print(f"Saved clean grid: {clean_path} ({clean.size[0]}x{clean.size[1]})")

    # Rotated grid (simulating real placement)
    rotated = composite_grid(cards, rotate=True)
    rotated_path = DATA_DIR / "test_binder_3x3.jpg"
    rotated.save(rotated_path, quality=92)
    print(f"Saved rotated grid: {rotated_path} ({rotated.size[0]}x{rotated.size[1]})")


if __name__ == "__main__":
    main()
