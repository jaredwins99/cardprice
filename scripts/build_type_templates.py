#!/usr/bin/env python3
"""Build energy-type symbol templates by cropping the top-right energy icon
from Pokemon card reference images.

Reads card type info from data/card_names.json (works without PostgreSQL).
Saves 64x64 crops to data/type_templates/{TypeName}/.

Usage:
    python scripts/build_type_templates.py
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CARD_IMAGES_DIR = ROOT / "data" / "card_images"
CARD_NAMES_JSON = ROOT / "data" / "card_names.json"
EVAL_JSON = ROOT / "data" / "eval" / "binder_eval.json"
OUTPUT_DIR = ROOT / "data" / "type_templates"
TEMPLATE_SIZE = 64

# Supertypes that do NOT have energy symbols (Trainer, Supporter, Stadium, Energy)
# We only want Pokemon cards.
SKIP_SUPERTYPES = {"Trainer", "Energy", "Supporter", "Stadium", "Item", "Tool"}

# --------------------------------------------------------------------------
# Era detection: symbol position varies by card era
# --------------------------------------------------------------------------
# WotC era (Base Set through Neo/e-Card): wider border, symbol slightly left
# Modern era (EX series onward): tighter layout
#
# Set prefixes by era:
#   WotC:  base1-6, gym1-2, neo1-4, ecard1-3, basep, si1, bp
#   ex:    ex1-16, tk1a, tk2a, pop1-9
#   DP+:   dp1-7, pl1-4, hgss1-4, bw1-11, xy0-13, sm1-12, sv1-8, swsh1-13
WOTC_PREFIXES = (
    "base", "gym", "neo", "si1", "bp",
    # Base set promotional
    "basep",
)

EX_PREFIXES = (
    # e-Card series has symbol BEFORE HP number, similar to EX era
    "ecard",
    "ex", "tk1", "tk2", "pop",
)


def get_era(set_id: str) -> str:
    """Determine card era from set_id prefix."""
    set_lower = set_id.lower()
    for prefix in WOTC_PREFIXES:
        if set_lower.startswith(prefix):
            return "wotc"
    for prefix in EX_PREFIXES:
        if set_lower.startswith(prefix):
            return "ex"
    return "modern"


def get_symbol_crop_region(img_w: int, img_h: int, era: str):
    """Return (x1, y1, x2, y2) for the energy symbol region.

    The energy type symbol sits in the HP/name bar at the top of the card.
    Its exact position varies by era:

      - WotC (Base-Neo): "120 HP [fire]" -- symbol AFTER HP text, at far right
      - e-Card: "[psychic] 70 HP" -- symbol BEFORE HP number, slightly left
      - EX era: similar to e-Card but variable
      - Modern (DP-SV): "HP 90 [lightning]" or "[grass] HP 30" -- near HP text

    We crop a generous region (right ~25% of width, top ~9% of height) which
    reliably contains the symbol across all eras. The 64x64 resize captures
    the symbol plus some HP text context, which aids template matching.
    """
    if era == "wotc":
        # WotC: symbol at far right after "HP" text
        # Right 20% of width, top 9% of height
        x1 = int(img_w * 0.80)
        y1 = int(img_h * 0.01)
        x2 = int(img_w * 0.99)
        y2 = int(img_h * 0.09)
    elif era == "ex":
        # EX/e-Card era: symbol before HP number, slightly more left
        # Right 25% of width to catch the symbol which can be further left
        x1 = int(img_w * 0.75)
        y1 = int(img_h * 0.01)
        x2 = int(img_w * 0.99)
        y2 = int(img_h * 0.09)
    else:
        # Modern (DP through SV): symbol near HP text in top-right
        # Right 25% of width, top 8.5% of height
        x1 = int(img_w * 0.75)
        y1 = int(img_h * 0.005)
        x2 = int(img_w * 0.99)
        y2 = int(img_h * 0.085)

    return x1, y1, x2, y2


def card_id_to_image_path(card_id: str) -> Path | None:
    """Convert card_id like 'base1-4/normal' to its image file path."""
    # card_id format: "set_id-num/variant"
    if "/" not in card_id:
        return None
    base, variant = card_id.split("/", 1)
    # set_id is everything before the last dash+number
    # e.g. base1-4 -> set_id=base1, but stored in dir base1/
    parts = base.split("-")
    if len(parts) < 2:
        return None
    set_id = parts[0]
    filename = f"{base}_{variant}.png"
    path = CARD_IMAGES_DIR / set_id / filename
    if path.exists():
        return path
    # Try .jpg
    path_jpg = path.with_suffix(".jpg")
    if path_jpg.exists():
        return path_jpg
    return None


def is_pokemon_card(card_entry: list) -> bool:
    """Check if card_names.json entry is a Pokemon card (has types)."""
    # Format: [card_id, name, set_id, hp, types_list]
    if len(card_entry) < 5:
        return False
    types = card_entry[4]
    if not types or not isinstance(types, list) or len(types) == 0:
        return False
    # Skip if HP is empty (likely trainer/energy)
    hp = card_entry[3]
    if not hp:
        return False
    return True


def crop_energy_symbol(img_path: Path, era: str) -> np.ndarray | None:
    """Crop the energy symbol from the top-right of a card image.

    Returns a 64x64 BGR numpy array, or None if the crop looks bad.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    x1, y1, x2, y2 = get_symbol_crop_region(w, h, era)

    # Clamp
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Resize to TEMPLATE_SIZE x TEMPLATE_SIZE
    crop_resized = cv2.resize(crop, (TEMPLATE_SIZE, TEMPLATE_SIZE),
                               interpolation=cv2.INTER_AREA)

    # Basic quality check: reject if too uniform (likely blank/white border)
    gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)
    if std < 10:
        # Very uniform -- likely no symbol here
        return None

    return crop_resized


def build_type_index(cards_data: list) -> dict[str, list]:
    """Build a mapping from type_name -> list of (card_id, set_id) entries.

    Only includes Pokemon cards that have images on disk.
    """
    type_index = defaultdict(list)
    for entry in cards_data:
        if not is_pokemon_card(entry):
            continue
        card_id = entry[0]
        set_id = entry[2]
        types = entry[4]
        # Use primary type (first in list)
        primary_type = types[0]
        img_path = card_id_to_image_path(card_id)
        if img_path is not None:
            type_index[primary_type].append((card_id, set_id, img_path))
    return dict(type_index)


def main():
    # Load card data
    logger.info("Loading card data from %s", CARD_NAMES_JSON)
    with open(CARD_NAMES_JSON) as f:
        cards_data = json.load(f)
    logger.info("Loaded %d card entries", len(cards_data))

    # Build type index
    type_index = build_type_index(cards_data)
    logger.info("Type index built: %s",
                {t: len(v) for t, v in sorted(type_index.items())})

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # Step 1: Process eval set cards
    # --------------------------------------------------------------------------
    eval_card_ids = set()
    if EVAL_JSON.exists():
        logger.info("Loading eval set from %s", EVAL_JSON)
        with open(EVAL_JSON) as f:
            eval_data = json.load(f)
        for page in eval_data.get("pages", []):
            for card in page.get("cards", []):
                cid = card.get("card_id")
                if cid:
                    eval_card_ids.add(cid)
        logger.info("Found %d eval cards (with card_id)", len(eval_card_ids))

    # Map eval card_ids to their types
    card_type_map = {}
    for entry in cards_data:
        card_id = entry[0]
        if card_id in eval_card_ids and is_pokemon_card(entry):
            card_type_map[card_id] = entry[4][0]  # primary type

    # Process eval cards
    eval_saved = 0
    for card_id, type_name in card_type_map.items():
        img_path = card_id_to_image_path(card_id)
        if img_path is None:
            logger.warning("No image for eval card %s", card_id)
            continue

        set_id = card_id.split("-")[0]
        era = get_era(set_id)
        crop = crop_energy_symbol(img_path, era)
        if crop is None:
            logger.warning("Bad crop for eval card %s (era=%s)", card_id, era)
            continue

        # Save
        type_dir = OUTPUT_DIR / type_name
        type_dir.mkdir(parents=True, exist_ok=True)
        safe_id = card_id.replace("/", "_")
        out_path = type_dir / f"{safe_id}.png"
        cv2.imwrite(str(out_path), crop)
        eval_saved += 1

    logger.info("Saved %d eval card symbol crops", eval_saved)

    # --------------------------------------------------------------------------
    # Step 2: Sample 20 cards per type from broader reference set
    # --------------------------------------------------------------------------
    SAMPLES_PER_TYPE = 20
    total_saved = eval_saved

    for type_name, card_entries in sorted(type_index.items()):
        type_dir = OUTPUT_DIR / type_name
        type_dir.mkdir(parents=True, exist_ok=True)

        # Count how many we already have from eval
        existing = set(p.stem for p in type_dir.glob("*.png"))

        # Filter out cards already saved
        available = [
            (cid, sid, path) for cid, sid, path in card_entries
            if cid.replace("/", "_") not in existing
        ]

        # Sample from different eras for variety
        era_buckets = defaultdict(list)
        for cid, sid, path in available:
            era = get_era(sid)
            era_buckets[era].append((cid, sid, path))

        # Try to get roughly equal samples from each era
        needed = SAMPLES_PER_TYPE - len(existing)
        if needed <= 0:
            logger.info("  %s: already have %d templates, skipping",
                        type_name, len(existing))
            continue

        samples = []
        eras = list(era_buckets.keys())
        random.seed(42)  # Reproducible sampling

        if eras:
            per_era = max(1, needed // len(eras))
            for era in eras:
                bucket = era_buckets[era]
                random.shuffle(bucket)
                samples.extend(bucket[:per_era])

            # Fill remainder if not enough
            if len(samples) < needed:
                all_remaining = [
                    item for era in eras for item in era_buckets[era]
                    if item not in samples
                ]
                random.shuffle(all_remaining)
                samples.extend(all_remaining[:needed - len(samples)])

        samples = samples[:needed]

        saved_count = 0
        skipped = 0
        for cid, sid, img_path in samples:
            era = get_era(sid)
            crop = crop_energy_symbol(img_path, era)
            if crop is None:
                skipped += 1
                continue

            safe_id = cid.replace("/", "_")
            out_path = type_dir / f"{safe_id}.png"
            cv2.imwrite(str(out_path), crop)
            saved_count += 1
            total_saved += 1

        logger.info("  %s: saved %d new templates (skipped %d bad crops, "
                     "%d existing)", type_name, saved_count, skipped,
                     len(existing))

    logger.info("Total templates saved: %d across %d types",
                total_saved, len(type_index))

    # Print summary
    print("\n=== Type Template Summary ===")
    for type_dir in sorted(OUTPUT_DIR.iterdir()):
        if type_dir.is_dir():
            count = len(list(type_dir.glob("*.png")))
            print(f"  {type_dir.name:15s}: {count:3d} templates")


if __name__ == "__main__":
    main()
