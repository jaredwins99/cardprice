"""Generate synthetic binder page test images by compositing real card images.

Produces ground-truth test data for the card segmenter and grid detector.

Usage:
    python scripts/make_test_binder.py           # generate 5 test images
    python scripts/make_test_binder.py --test     # generate + run segmenter
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CARD_IMG_DIR = DATA_DIR / "card_images"
OUTPUT_DIR = DATA_DIR / "test_binder_pages"

# Standard Pokemon card aspect ratio: 63mm x 88mm
CARD_W = 245
CARD_H = 342

COLS, ROWS = 3, 3
GAP = 18
BORDER = 50
BORDER_COLOR_BGR = (30, 120, 220)  # orange in BGR

SEED = 2026
NUM_IMAGES = 5


def collect_card_files() -> list[Path]:
    """Gather all PNG card images from data/card_images/."""
    files = sorted(CARD_IMG_DIR.rglob("*.png"))
    if not files:
        print(f"ERROR: No card images found in {CARD_IMG_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} card images in {CARD_IMG_DIR}")
    return files


def pick_cards(all_files: list[Path], rng: random.Random) -> list[Path]:
    """Pick 9 random card images."""
    return rng.sample(all_files, 9)


def load_card(path: Path) -> np.ndarray:
    """Load a card image as BGR and resize to standard dimensions."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read: {path}")
    return cv2.resize(img, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)


def make_grid(cards: list[np.ndarray]) -> np.ndarray:
    """Arrange 9 card images in a 3x3 grid with gaps and orange border."""
    canvas_w = 2 * BORDER + COLS * CARD_W + (COLS - 1) * GAP
    canvas_h = 2 * BORDER + ROWS * CARD_H + (ROWS - 1) * GAP
    canvas = np.full((canvas_h, canvas_w, 3), BORDER_COLOR_BGR, dtype=np.uint8)

    for idx, card in enumerate(cards):
        row, col = divmod(idx, COLS)
        x = BORDER + col * (CARD_W + GAP)
        y = BORDER + row * (CARD_H + GAP)
        canvas[y:y + CARD_H, x:x + CARD_W] = card

    return canvas


def apply_perspective(image: np.ndarray, rng: random.Random,
                      strength_pct: float = None) -> np.ndarray:
    """Apply random perspective distortion (2-5% of image dimensions)."""
    if strength_pct is None:
        strength_pct = rng.uniform(2.0, 5.0)
    h, w = image.shape[:2]
    max_shift_x = int(w * strength_pct / 100)
    max_shift_y = int(h * strength_pct / 100)

    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = np.array([
        [rng.randint(0, max_shift_x), rng.randint(0, max_shift_y)],
        [w - rng.randint(0, max_shift_x), rng.randint(0, max_shift_y)],
        [w - rng.randint(0, max_shift_x), h - rng.randint(0, max_shift_y)],
        [rng.randint(0, max_shift_x), h - rng.randint(0, max_shift_y)],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h),
                               borderMode=cv2.BORDER_REPLICATE)


def apply_rotation(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply random rotation (1-3 degrees)."""
    angle = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_noise(image: np.ndarray, rng: random.Random,
                sigma: float = None) -> np.ndarray:
    """Add Gaussian noise."""
    if sigma is None:
        sigma = rng.uniform(5.0, 15.0)
    noise = np.zeros(image.shape, dtype=np.int16)
    # Use numpy RNG seeded from our rng for reproducibility
    np_rng = np.random.RandomState(rng.randint(0, 2**31))
    np_rng.randn(*noise.shape).astype(np.float64)
    noise = (np_rng.randn(*image.shape) * sigma).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def generate_images(all_files: list[Path]) -> dict:
    """Generate NUM_IMAGES test binder pages and return ground truth."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    ground_truth = {}

    for i in range(NUM_IMAGES):
        card_paths = pick_cards(all_files, rng)
        cards = [load_card(p) for p in card_paths]
        grid = make_grid(cards)

        # Apply augmentations with increasing intensity
        aug = grid.copy()
        aug = apply_perspective(aug, rng)
        aug = apply_rotation(aug, rng)
        aug = apply_noise(aug, rng)

        filename = f"binder_page_{i:02d}.jpg"
        out_path = OUTPUT_DIR / filename
        cv2.imwrite(str(out_path), aug, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Record ground truth (relative paths from data/)
        gt_cards = [str(p.relative_to(DATA_DIR)) for p in card_paths]
        ground_truth[filename] = {
            "cards": gt_cards,
            "grid": [ROWS, COLS],
            "expected_count": ROWS * COLS,
        }
        print(f"  [{i+1}/{NUM_IMAGES}] {out_path.name}  "
              f"({aug.shape[1]}x{aug.shape[0]})")

    # Write ground truth
    gt_path = OUTPUT_DIR / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"\nGround truth: {gt_path}")

    return ground_truth


def run_segmenter_test(ground_truth: dict):
    """Run the card segmenter on each generated image and report results."""
    # Import here so generation works even if ML deps are missing
    from cardprice.ml.card_segmenter import segment_cards

    print("\n--- Segmenter Test Results ---\n")
    total_expected = 0
    total_detected = 0

    for filename, gt in ground_truth.items():
        image_path = OUTPUT_DIR / filename
        expected = gt["expected_count"]

        # Run segmenter, output to a temp dir alongside the image
        out_dir = OUTPUT_DIR / f"{Path(filename).stem}_cards"
        try:
            results = segment_cards(
                image_path,
                output_dir=out_dir,
                expected_grid=(gt["grid"][0], gt["grid"][1]),
            )
            detected = len(results)
        except Exception as e:
            print(f"  {filename}: ERROR - {e}")
            detected = 0

        status = "OK" if detected == expected else "MISS"
        print(f"  {filename}: {detected}/{expected} cards detected  [{status}]")
        total_expected += expected
        total_detected += detected

    pct = (total_detected / total_expected * 100) if total_expected else 0
    print(f"\n  Total: {total_detected}/{total_expected} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic binder page test images")
    parser.add_argument("--test", action="store_true",
                        help="Run segmenter on generated images and report")
    args = parser.parse_args()

    # Check card images exist
    all_files = collect_card_files()

    print(f"\nGenerating {NUM_IMAGES} test binder pages -> {OUTPUT_DIR}\n")
    ground_truth = generate_images(all_files)

    if args.test:
        run_segmenter_test(ground_truth)

    print("\nDone.")


if __name__ == "__main__":
    main()
