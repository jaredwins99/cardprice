#!/usr/bin/env python3
"""Pre-compute DINOv2 embeddings for Japanese card images.

Maps each JP image to a `jp_<tcg_product_id>` card_id from `dim_cards_jp`,
then extracts a 768-dim L2-normalized DINOv2 embedding (CLS token, matching
build_ref_embeddings.py) and saves a dict to data/ref_embeddings_jp.pkl.

Mapping strategy
----------------
1. TCGdex-style dirs (e.g. `SV10_ロケット団の栄光/SV10-001.png`):
   - Directory prefix before the first '_' is the TCGdex set code ("SV10").
   - Match against `dim_sets_jp.abbreviation` -> set_id.
   - Filename suffix "-NNN" gives the card number; match against
     `dim_cards_jp.card_number` rows for that set whose number prefix
     (split on '/') equals the zero-padded number.
2. tcgplayer-style dirs (e.g. `glory_of_team_rocket_jp/Pineco_001_098_628642.jpg`,
   `team_rocket_jp/Abra_N_A_575712.jpg`):
   - The trailing integer before the extension is the tcg_product_id.
   - card_id = "jp_<tcg_product_id>".

Usage:
    python scripts/build_jp_embeddings.py [--limit N] [--output PATH]
"""

import argparse
import json
import logging
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_IMAGE_ROOT = Path("data/card_images_jp")

# Trailing _<digits> before extension (tcgplayer filenames)
_TRAILING_PRODUCT_RE = re.compile(r"_(\d{5,})$")
# TCGdex filename: "<SET>-<NUM>" e.g. SV10-001
_TCGDEX_NAME_RE = re.compile(r"^([A-Za-z0-9]+)-(\d+[a-zA-Z]?)$")


def _build_db_indexes():
    """Build in-memory indexes from dim_cards_jp / dim_sets_jp.

    Returns
    -------
    set_code_to_id : dict[str, str]   abbreviation (lowercased) -> set_id
    set_card_index : dict[str, dict[str, str]]
        set_id -> { numeric_part_of_card_number: card_id }
    valid_card_ids : set[str]   all jp_<id> card_ids in dim_cards_jp
    """
    from cardprice.db.session import SessionLocal
    from sqlalchemy import text

    session = SessionLocal()
    try:
        rows = session.execute(text(
            "SELECT set_id, abbreviation FROM dim_sets_jp WHERE abbreviation IS NOT NULL"
        )).fetchall()
        set_code_to_id = {r[1].lower(): r[0] for r in rows if r[1]}

        rows = session.execute(text(
            "SELECT card_id, set_id, card_number FROM dim_cards_jp"
        )).fetchall()
        set_card_index: dict[str, dict[str, str]] = {}
        valid_card_ids: set[str] = set()
        for card_id, set_id, card_number in rows:
            valid_card_ids.add(card_id)
            if not set_id or not card_number:
                continue
            # Extract numeric prefix: "001/098" -> "1", "TG12/TG30" -> "TG12"
            num_part = card_number.split("/")[0].strip()
            # Strip leading zeros for numeric matching
            try:
                key = str(int(num_part))
            except ValueError:
                key = num_part.lower()
            set_card_index.setdefault(set_id, {})[key] = card_id
        return set_code_to_id, set_card_index, valid_card_ids
    finally:
        session.close()


def _resolve_card_id(
    img_path: Path,
    set_code_to_id: dict[str, str],
    set_card_index: dict[str, dict[str, str]],
    valid_card_ids: set[str],
) -> str | None:
    """Map an image path to a jp_<product_id> card_id, or None if unmappable."""
    stem = img_path.stem
    parent_name = img_path.parent.name

    # Strategy 1: trailing _<product_id>
    m = _TRAILING_PRODUCT_RE.search(stem)
    if m:
        candidate = f"jp_{m.group(1)}"
        if candidate in valid_card_ids:
            return candidate

    # Strategy 2: TCGdex SET-NUM filename
    m = _TCGDEX_NAME_RE.match(stem)
    if m:
        set_code = m.group(1).lower()
        num_str = m.group(2)
        set_id = set_code_to_id.get(set_code)
        if set_id is None:
            # Try parent directory prefix as fallback
            parent_code = parent_name.split("_", 1)[0].lower()
            set_id = set_code_to_id.get(parent_code)
        if set_id and set_id in set_card_index:
            try:
                key = str(int(num_str))
            except ValueError:
                key = num_str.lower()
            cid = set_card_index[set_id].get(key)
            if cid:
                return cid
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=str(_IMAGE_ROOT))
    parser.add_argument("--output", default="data/ref_embeddings_jp.pkl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Process only first N images (0 = all)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_path = Path(args.output)

    if not image_dir.is_dir():
        logger.error("Image directory not found: %s", image_dir)
        sys.exit(1)

    logger.info("Building DB indexes from dim_cards_jp / dim_sets_jp ...")
    set_code_to_id, set_card_index, valid_card_ids = _build_db_indexes()
    logger.info(
        "Indexes: %d set codes, %d sets with cards, %d total JP cards",
        len(set_code_to_id), len(set_card_index), len(valid_card_ids),
    )

    # Collect images and resolve card_ids
    image_files = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    logger.info("Found %d JP images under %s", len(image_files), image_dir)

    resolved: list[tuple[Path, str]] = []
    unmapped = 0
    for p in image_files:
        cid = _resolve_card_id(p, set_code_to_id, set_card_index, valid_card_ids)
        if cid:
            resolved.append((p, cid))
        else:
            unmapped += 1
    logger.info("Resolved %d / %d images (%d unmapped)", len(resolved), len(image_files), unmapped)

    if args.limit > 0:
        resolved = resolved[: args.limit]
        logger.info("Limited to first %d resolved images for testing", len(resolved))

    if not resolved:
        logger.error("No resolvable images; aborting.")
        sys.exit(1)

    # Lazy imports for fast --help
    import torch
    from PIL import Image
    from torchvision import transforms

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading DINOv2 ViT-B/14 on %s ...", device)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.to(device)
    model.eval()
    logger.info("Model loaded.")

    embeddings: dict[str, np.ndarray] = {}
    failed = 0
    t0 = time.time()
    bs = args.batch_size

    for batch_start in range(0, len(resolved), bs):
        batch = resolved[batch_start : batch_start + bs]
        tensors, ids = [], []
        for img_path, cid in batch:
            try:
                img = Image.open(img_path).convert("RGB")
                tensors.append(transform(img))
                ids.append(cid)
            except Exception as e:
                logger.warning("Failed to load %s: %s", img_path, e)
                failed += 1

        if not tensors:
            continue

        try:
            t = torch.stack(tensors).to(device)
            with torch.no_grad():
                out = model(t)
            arr = out.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.warning("Inference failed at %d: %s", batch_start, e)
            failed += len(tensors)
            continue

        for i, cid in enumerate(ids):
            v = arr[i]
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            embeddings[cid] = v

        processed = batch_start + len(batch)
        if processed % (bs * 5) == 0 or processed >= len(resolved):
            elapsed = time.time() - t0
            rate = len(embeddings) / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d / %d (%.1f img/s, %d failed, %d unique ids)",
                processed, len(resolved), rate, failed, len(embeddings),
            )

    elapsed = time.time() - t0
    logger.info(
        "Done: %d embeddings in %.1fs (%d failures, %d unmapped images skipped earlier)",
        len(embeddings), elapsed, failed, unmapped,
    )

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved %s (%.1f MB)", output_path, size_mb)

    # Sample lookups
    sample_keys = list(embeddings.keys())[:5]
    for k in sample_keys:
        v = embeddings[k]
        logger.info("  sample %s: shape=%s norm=%.4f", k, v.shape, float(np.linalg.norm(v)))


if __name__ == "__main__":
    main()
