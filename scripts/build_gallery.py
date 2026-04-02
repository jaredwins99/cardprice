#!/usr/bin/env python3
"""Build a ground truth image gallery for visual verification of variant detection.

Creates data/ground_truth_gallery/ with resized images organized by variant type.
Max 12 representative images per variant, resized to 400px max width.
"""

import shutil
from pathlib import Path
from PIL import Image

SRC = Path("/home/godli/cardprice/data/condition_training/ground_truth_variants")
DST = Path("/home/godli/cardprice/data/ground_truth_gallery")
MAX_WIDTH = 400
MAX_PER_VARIANT = 20

# Map source dirs to gallery dirs (curated subset, readable names)
VARIANT_MAP = {
    "1st_edition": "1st_edition",
    "shadowless": "shadowless",
    "unlimited": "unlimited",
    "holo_nonholo": "holo_vs_nonholo",
    "reverse_holo": "reverse_holo",
    "holo_patterns": "holo_patterns",
    "prerelease": "prerelease",
    "promo": "promo",
    "shining": "shining",
    "crystal": "crystal",
    "break": "break",
    "build_battle": "build_battle",
    "crosshatch": "crosshatch",
    "grey_stamp": "grey_stamp",
    "no_symbol": "no_symbol_error",
    "errors": "errors_misprints",
    "mcdonalds": "mcdonalds",
    "pokemon_center": "pokemon_center",
    "retailer_stamps": "retailer_stamps",
}

# Files to skip (not actual card images, or metadata)
SKIP_FILES = {"labels.json", "labels.jsonl", ".DS_Store"}
SKIP_PREFIXES = ("comparison_",)  # skip only side-by-side comparison images


def resize_image(src_path: Path, dst_path: Path):
    """Resize image to MAX_WIDTH, preserving aspect ratio. Convert to PNG."""
    try:
        img = Image.open(src_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        w, h = img.size
        if w > MAX_WIDTH:
            new_h = int(h * MAX_WIDTH / w)
            img = img.resize((MAX_WIDTH, new_h), Image.LANCZOS)
        # Save as JPEG for smaller size
        out = dst_path.with_suffix(".jpg")
        img.save(out, "JPEG", quality=85)
        return out
    except Exception as e:
        print(f"  SKIP {src_path.name}: {e}")
        return None


def pick_representative(files: list[Path], max_n: int) -> list[Path]:
    """Pick up to max_n representative files, preferring variety."""
    # Filter out non-image, metadata, and comparison/editorial images
    images = []
    for f in sorted(files):
        if f.name in SKIP_FILES:
            continue
        if f.name.startswith(SKIP_PREFIXES):
            continue
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        if f.is_dir():
            continue
        images.append(f)

    if len(images) <= max_n:
        return images

    # Take evenly spaced samples
    step = len(images) / max_n
    return [images[int(i * step)] for i in range(max_n)]


def build_readme(gallery_dir: Path, stats: dict):
    """Create README.md for the gallery."""
    lines = [
        "# Variant Detection Ground Truth Gallery",
        "",
        "Visual reference images for Pokemon card variant types.",
        "Images resized to 400px max width for git-friendliness.",
        "",
        f"**Total images:** {sum(stats.values())}",
        "",
        "## Variant Types",
        "",
    ]
    for variant, count in sorted(stats.items()):
        desc = VARIANT_DESCRIPTIONS.get(variant, "")
        lines.append(f"### [{variant}/]({variant}/)")
        if desc:
            lines.append(f"{desc}")
        lines.append(f"*{count} images*")
        lines.append("")
        # Show first 3 images inline
        img_dir = gallery_dir / variant
        imgs = sorted(img_dir.glob("*.jpg"))[:3]
        for img in imgs:
            lines.append(f"![{img.stem}]({variant}/{img.name})")
        lines.append("")

    (gallery_dir / "README.md").write_text("\n".join(lines))


VARIANT_DESCRIPTIONS = {
    "1st_edition": "Cards with the 1st Edition stamp (black circle with '1' and 'EDITION'). Base Set through Neo Destiny.",
    "shadowless": "Base Set cards printed without the shadow border on the right side of the card art frame. Rarer than Unlimited.",
    "unlimited": "Standard Base Set 2 / Legendary Collection prints with shadow border. The most common variant.",
    "holo_vs_nonholo": "Same card exists in both holofoil and non-holo versions (e.g., Team Rocket set).",
    "reverse_holo": "Cards with holographic foil on the non-art portions. Pattern varies by era (e.g., Legendary Collection stamped, EX series rainbow).",
    "holo_patterns": "Different holographic foil patterns: cosmos, cracked ice, confetti, starlight, linear, galaxy.",
    "prerelease": "Cards with PRERELEASE or STAFF stamp from pre-release tournaments.",
    "promo": "Promotional cards distributed through events, products, or special offers. Black star promo symbol.",
    "shining": "Neo-era cards with 'Shining' prefix and holographic Pokemon art on dark background.",
    "crystal": "e-Card era Crystal-type cards with translucent/crystal artwork effect. Very rare.",
    "break": "XY-era BREAK evolution cards with horizontal/landscape orientation and golden border.",
    "build_battle": "Stamped promo cards from Build & Battle boxes at pre-release events.",
    "crosshatch": "Cards with crosshatch holographic pattern, typically from League promos.",
    "grey_stamp": "Cards with grey/silver authentication stamps (CGC, SNCB). Manufacturing mark, not a variant per se.",
    "no_symbol_error": "Jungle/Fossil cards mistakenly printed without the set symbol. Error cards.",
    "errors_misprints": "Various printing errors: wrong energy symbols, missing damage, color errors.",
    "mcdonalds": "McDonald's Happy Meal promotional Pokemon cards with confetti holo pattern.",
    "pokemon_center": "Cards exclusive to Pokemon Center stores/online.",
    "retailer_stamps": "Cards with retailer-specific stamps (Toys R Us, Build-A-Bear).",
}


def main():
    # Clean and create
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    stats = {}

    for src_name, dst_name in VARIANT_MAP.items():
        src_dir = SRC / src_name
        if not src_dir.exists():
            print(f"SKIP {src_name}: not found")
            continue

        dst_dir = DST / dst_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        files = list(src_dir.iterdir())
        selected = pick_representative(files, MAX_PER_VARIANT)

        count = 0
        for f in selected:
            result = resize_image(f, dst_dir / f.stem)
            if result:
                count += 1

        stats[dst_name] = count
        print(f"{dst_name}: {count} images")

    # Also add a few 1st edition stamp closeups
    stamp_dir = SRC.parent.parent / "1st_edition_stamps" / "samples"
    if stamp_dir.exists():
        dst_stamps = DST / "1st_edition"
        for f in sorted(stamp_dir.iterdir())[:4]:
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                result = resize_image(f, dst_stamps / f"stamp_{f.stem}")
                if result:
                    stats["1st_edition"] = stats.get("1st_edition", 0) + 1

    build_readme(DST, stats)

    total = sum(stats.values())
    print(f"\nGallery built: {total} images in {len(stats)} categories")
    print(f"Location: {DST}")


if __name__ == "__main__":
    main()
