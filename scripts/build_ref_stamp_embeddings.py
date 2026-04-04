#!/usr/bin/env python3
"""Pre-compute DINOv2 stamp+control region embeddings for EX-era reference cards.

For each card in ex7-ex16, crops two regions from the reference image:
  - Stamp region (x: 0.55-0.90, y: 0.35-0.55) — bottom-right artwork
  - Control region (x: 0.10-0.55, y: 0.10-0.35) — top-center artwork

Runs DINOv2 on each crop and saves:
    data/ref_stamp_embeddings.pkl: dict[card_id -> {"stamp": np.array(768), "control": np.array(768)}]

Usage:
    python scripts/build_ref_stamp_embeddings.py
"""

import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_EX_STAMPED_SETS = [
    "ex7", "ex8", "ex9", "ex10", "ex11",
    "ex12", "ex13", "ex14", "ex15", "ex16",
]

# Same regions as _check_ex_stamp_dino in stamp_detection.py
STAMP_REGION = (0.55, 0.35, 0.90, 0.55)
CONTROL_REGION = (0.10, 0.10, 0.55, 0.35)


def _crop_region(img, x0, y0, x1, y1):
    """Crop a fractional region from a PIL Image."""
    w, h = img.size
    return img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


def _card_id_from_path(img_path: Path, image_dir: Path) -> str:
    """Derive card_id from relative image path.

    Example: data/card_images/ex15/ex15-26_normal.png -> "ex15-26/normal"
    """
    rel = img_path.relative_to(image_dir).with_suffix("")
    card_id = str(rel).replace(os.sep, "/")
    parts = card_id.split("/")
    if len(parts) == 2:
        card_id = parts[1]
    last_under = card_id.rfind("_")
    if last_under != -1:
        card_id = card_id[:last_under] + "/" + card_id[last_under + 1:]
    return card_id


def main():
    image_dir = Path("data/card_images")
    output_path = Path("data/ref_stamp_embeddings.pkl")
    batch_size = 64  # 2 crops per card, so 32 cards per batch

    if not image_dir.is_dir():
        logger.error("Image directory not found: %s", image_dir)
        sys.exit(1)

    # Collect images for ex7-ex16 only
    image_files = []
    for set_id in _EX_STAMPED_SETS:
        set_dir = image_dir / set_id
        if not set_dir.is_dir():
            logger.warning("Set directory not found: %s", set_dir)
            continue
        for p in sorted(set_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
                image_files.append(p)

    logger.info("Found %d reference images across ex7-ex16", len(image_files))
    if not image_files:
        logger.error("No images found.")
        sys.exit(1)

    # Import heavy dependencies after setup
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

    embeddings: dict[str, dict[str, np.ndarray]] = {}
    failed = 0
    t0 = time.time()

    # Process in batches. Each card produces 2 crops (stamp + control).
    # We batch all crops together for GPU efficiency.
    cards_per_batch = batch_size // 2  # 2 crops per card

    for batch_start in range(0, len(image_files), cards_per_batch):
        batch_paths = image_files[batch_start:batch_start + cards_per_batch]
        batch_tensors = []
        batch_card_ids = []

        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                stamp_crop = _crop_region(img, *STAMP_REGION)
                control_crop = _crop_region(img, *CONTROL_REGION)
                batch_tensors.append(transform(stamp_crop))
                batch_tensors.append(transform(control_crop))
                batch_card_ids.append(_card_id_from_path(img_path, image_dir))
            except Exception as e:
                logger.warning("Failed to load %s: %s", img_path, e)
                failed += 1

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)
        try:
            with torch.no_grad():
                batch_emb = model(batch)  # (2*N, 768)
            batch_np = batch_emb.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.warning("Batch inference failed at index %d: %s", batch_start, e)
            failed += len(batch_paths)
            continue

        # L2-normalize
        norms = np.linalg.norm(batch_np, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        batch_np /= norms

        # Unpack: even indices = stamp, odd indices = control
        for i, card_id in enumerate(batch_card_ids):
            embeddings[card_id] = {
                "stamp": batch_np[2 * i],
                "control": batch_np[2 * i + 1],
            }

        processed = batch_start + len(batch_paths)
        if processed % 100 < cards_per_batch or processed >= len(image_files):
            elapsed = time.time() - t0
            rate = len(embeddings) / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d / %d cards (%.1f cards/s, %d failed)",
                len(embeddings), len(image_files), rate, failed,
            )

    elapsed = time.time() - t0
    logger.info(
        "Done: %d card embeddings (2 crops each) in %.1fs (%d failures)",
        len(embeddings), elapsed, failed,
    )

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved to %s (%.1f MB, %d cards)", output_path, size_mb, len(embeddings))


if __name__ == "__main__":
    main()
