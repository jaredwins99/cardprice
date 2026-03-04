#!/usr/bin/env python3
"""Generate synthetic phone-camera-like augmentations of reference card images.

Creates training pairs (augmented_image, original_card_id) for fine-tuning
a DINOv2 projection head. Simulates realistic phone-camera capture conditions:
perspective distortion, defocus blur, lighting variation, JPEG artifacts,
white-balance shift, sensor noise, slight rotation, and glare spots.

Usage:
    # Test mode: 100 diverse cards, saves visual samples
    python scripts/build_training_pairs.py --test

    # Full run: all 20k cards
    python scripts/build_training_pairs.py

    # Custom settings
    python scripts/build_training_pairs.py --num-augments 4 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CARD_IMAGES_DIR = PROJECT_ROOT / "data" / "card_images"
OUTPUT_DIR = PROJECT_ROOT / "data" / "training_pairs"
SAMPLES_DIR = OUTPUT_DIR / "samples"


# ---------------------------------------------------------------------------
# Card-ID derivation (mirrors dino_matcher.py / clip_matcher.py logic)
# ---------------------------------------------------------------------------
def path_to_card_id(img_path: Path) -> str:
    """Convert image path to canonical card_id (DB format).

    data/card_images/sv8/sv8-162_normal.png  ->  sv8-162/normal

    Uses filename only (not parent dir) to produce the DB-canonical
    card_id format: ``{pokemontcg_id}/{variant}``.
    """
    stem = img_path.stem  # e.g. "sv8-162_normal"
    last_under = stem.rfind("_")
    if last_under != -1:
        card_id = stem[:last_under] + "/" + stem[last_under + 1:]
    else:
        card_id = stem
    return card_id


# ---------------------------------------------------------------------------
# Glare simulation (custom transform)
# ---------------------------------------------------------------------------
class GlareSimulation(A.ImageOnlyTransform):
    """Simulate specular glare spots on a card surface.

    Adds 1-3 bright, low-saturation elliptical patches to mimic overhead
    light reflections on glossy card stock or binder sleeve plastic.
    """

    def __init__(
        self,
        num_spots: tuple[int, int] = (1, 3),
        intensity_range: tuple[float, float] = (0.3, 0.7),
        radius_fraction: tuple[float, float] = (0.05, 0.20),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.num_spots = num_spots
        self.intensity_range = intensity_range
        self.radius_fraction = radius_fraction

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        result = img.copy()
        h, w = result.shape[:2]
        num = random.randint(*self.num_spots)

        for _ in range(num):
            # Random center, biased toward the middle
            cx = int(random.gauss(w / 2, w / 4))
            cy = int(random.gauss(h / 2, h / 4))
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))

            # Elliptical radius
            max_dim = max(h, w)
            rx = int(random.uniform(*self.radius_fraction) * max_dim)
            ry = int(random.uniform(*self.radius_fraction) * max_dim * random.uniform(0.5, 1.0))

            # Rotation angle for the ellipse
            angle = random.uniform(0, 360)
            intensity = random.uniform(*self.intensity_range)

            # Create a glare mask
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)
            # Gaussian blur the mask for soft edges
            ksize = max(rx, ry) * 2 + 1
            ksize = ksize if ksize % 2 == 1 else ksize + 1
            ksize = min(ksize, min(h, w) | 1)  # ensure odd and <= image dim
            if ksize >= 3:
                mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

            # Normalize mask to [0, intensity]
            mask_max = mask.max()
            if mask_max > 0:
                mask = mask / mask_max * intensity

            # Blend: push toward white (desaturated bright spot)
            mask_3ch = mask[:, :, np.newaxis]
            white = np.full_like(result, 255, dtype=np.uint8)
            result = np.clip(
                result.astype(np.float32) * (1 - mask_3ch) + white.astype(np.float32) * mask_3ch,
                0, 255,
            ).astype(np.uint8)

        return result

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("num_spots", "intensity_range", "radius_fraction")


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------
def build_augmentation_pipeline() -> A.Compose:
    """Build the phone-camera simulation augmentation pipeline."""
    return A.Compose([
        # 1. Perspective distortion (phone angle to binder page)
        A.Perspective(
            scale=(0.02, 0.08),
            border_mode=cv2.BORDER_REPLICATE,
            p=0.7,
        ),
        # 7. Slight rotation (+-5 degrees, card not perfectly aligned)
        A.Rotate(
            limit=(-5, 5),
            border_mode=cv2.BORDER_REPLICATE,
            p=0.6,
        ),
        # 3. Random brightness/contrast (indoor lighting variation)
        A.RandomBrightnessContrast(
            brightness_limit=(-0.25, 0.25),
            contrast_limit=(-0.20, 0.20),
            p=0.8,
        ),
        # 5. Color shift (phone white-balance differences)
        A.ColorJitter(
            brightness=(0.9, 1.1),
            contrast=(0.9, 1.1),
            saturation=(0.8, 1.2),
            hue=(-0.04, 0.04),
            p=0.7,
        ),
        # 2. Gaussian blur (phone defocus)
        A.GaussianBlur(
            blur_limit=(3, 7),
            p=0.5,
        ),
        # 6. Gaussian noise (sensor noise in low light)
        A.GaussNoise(
            std_range=(5 / 255, 25 / 255),
            p=0.5,
        ),
        # 4. JPEG compression artifacts
        A.ImageCompression(
            quality_range=(40, 85),
            p=0.6,
        ),
        # 8. Glare simulation (bright spots, low saturation)
        GlareSimulation(
            num_spots=(1, 3),
            intensity_range=(0.2, 0.5),
            radius_fraction=(0.05, 0.15),
            p=0.4,
        ),
    ])


# ---------------------------------------------------------------------------
# Card selection
# ---------------------------------------------------------------------------
def select_test_cards(n: int = 100) -> list[Path]:
    """Select n cards from diverse sets for testing.

    Picks cards spread across different sets to ensure variety.
    """
    set_dirs = sorted([d for d in CARD_IMAGES_DIR.iterdir() if d.is_dir()])
    if not set_dirs:
        raise FileNotFoundError(f"No set directories found in {CARD_IMAGES_DIR}")

    # Spread picks across sets
    cards_per_set = max(1, n // len(set_dirs))
    selected: list[Path] = []

    for set_dir in set_dirs:
        pngs = sorted(set_dir.glob("*.png"))
        if not pngs:
            continue
        pick_count = min(cards_per_set, len(pngs))
        # Deterministic sample for reproducibility
        rng = random.Random(42)
        picked = rng.sample(pngs, pick_count)
        selected.extend(picked)
        if len(selected) >= n:
            break

    # If we haven't reached n yet (few cards per set), do another pass
    if len(selected) < n:
        all_cards = sorted(CARD_IMAGES_DIR.rglob("*.png"))
        rng = random.Random(42)
        rng.shuffle(all_cards)
        existing = set(selected)
        for card in all_cards:
            if card not in existing:
                selected.append(card)
                if len(selected) >= n:
                    break

    return selected[:n]


def select_all_cards() -> list[Path]:
    """Return all card images."""
    return sorted(CARD_IMAGES_DIR.rglob("*.png"))


# ---------------------------------------------------------------------------
# Augmentation generation
# ---------------------------------------------------------------------------
def generate_augmented(
    img_path: Path,
    pipeline: A.Compose,
    num_augments: int,
    output_dir: Path,
) -> list[tuple[str, str]]:
    """Generate augmented versions of a single card image.

    Returns list of (augmented_relative_path, card_id) pairs.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Could not read image: %s", img_path)
        return []

    card_id = path_to_card_id(img_path)
    # Derive output subdirectory mirroring input structure
    rel = img_path.relative_to(CARD_IMAGES_DIR)
    stem = rel.stem  # e.g. "sv8-162_normal"

    pairs: list[tuple[str, str]] = []
    for aug_idx in range(num_augments):
        augmented = pipeline(image=img)["image"]

        # Output path: data/training_pairs/augmented/{set_id}/{stem}_aug{idx}.jpg
        out_subdir = output_dir / "augmented" / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_name = f"{stem}_aug{aug_idx}.jpg"
        out_path = out_subdir / out_name

        cv2.imwrite(str(out_path), augmented, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Relative path from project root for portability
        rel_out = out_path.relative_to(PROJECT_ROOT)
        pairs.append((str(rel_out), card_id))

    return pairs


def save_side_by_side_sample(
    img_path: Path,
    pipeline: A.Compose,
    sample_dir: Path,
    sample_idx: int,
) -> None:
    """Save a side-by-side comparison (original | augmented) for visual inspection."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return

    augmented = pipeline(image=img)["image"]

    # Resize both to same height for side-by-side
    target_h = 400
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)

    orig_resized = cv2.resize(img, (new_w, target_h))
    aug_resized = cv2.resize(augmented, (new_w, target_h))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(orig_resized, "Original", (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(aug_resized, "Augmented", (10, 30), font, 0.8, (0, 0, 255), 2)

    # Separator line
    sep = np.full((target_h, 4, 3), 128, dtype=np.uint8)

    combined = np.hstack([orig_resized, sep, aug_resized])

    card_id = path_to_card_id(img_path).replace("/", "_")
    out_path = sample_dir / f"sample_{sample_idx:03d}_{card_id}.jpg"
    cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate phone-camera augmented training pairs for DINOv2."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: 100 diverse cards, save visual samples",
    )
    parser.add_argument(
        "--num-augments", type=int, default=3,
        help="Number of augmented views per card (default: 3)",
    )
    parser.add_argument(
        "--num-test-cards", type=int, default=100,
        help="Number of cards in test mode (default: 100)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=20,
        help="Number of side-by-side visual samples to save (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/training_pairs)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select cards
    if args.test:
        cards = select_test_cards(args.num_test_cards)
        logger.info("Test mode: selected %d cards from diverse sets", len(cards))
    else:
        cards = select_all_cards()
        logger.info("Full mode: processing all %d cards", len(cards))

    pipeline = build_augmentation_pipeline()

    # Generate visual samples first (always, but more in test mode)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    num_samples = min(args.num_samples, len(cards))
    sample_cards = random.sample(cards, num_samples)

    logger.info("Saving %d side-by-side samples to %s", num_samples, samples_dir)
    for i, card_path in enumerate(sample_cards):
        save_side_by_side_sample(card_path, pipeline, samples_dir, i)
    logger.info("Samples saved.")

    # Generate all augmented pairs
    all_pairs: list[tuple[str, str]] = []
    total = len(cards)

    for idx, card_path in enumerate(cards):
        pairs = generate_augmented(card_path, pipeline, args.num_augments, output_dir)
        all_pairs.extend(pairs)

        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            logger.info(
                "Progress: %d/%d cards (%d pairs generated)",
                idx + 1, total, len(all_pairs),
            )

    # Save manifest as CSV
    manifest_path = output_dir / "pairs.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["augmented_path", "card_id"])
        writer.writerows(all_pairs)

    # Also save as JSON for convenience
    manifest_json = output_dir / "pairs.json"
    records = [{"augmented_path": p, "card_id": cid} for p, cid in all_pairs]
    with open(manifest_json, "w") as f:
        json.dump(records, f, indent=2)

    # Summary
    unique_cards = len(set(cid for _, cid in all_pairs))
    sets_covered = len(set(cid.split("-")[0].split("/")[0] for _, cid in all_pairs))

    logger.info("=" * 60)
    logger.info("Done! Generated %d training pairs", len(all_pairs))
    logger.info("  Unique cards: %d", unique_cards)
    logger.info("  Sets covered: %d", sets_covered)
    logger.info("  Augments per card: %d", args.num_augments)
    logger.info("  Manifest CSV: %s", manifest_path)
    logger.info("  Manifest JSON: %s", manifest_json)
    logger.info("  Visual samples: %s", samples_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
