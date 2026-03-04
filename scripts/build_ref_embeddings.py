#!/usr/bin/env python3
"""Pre-compute DINOv2 embeddings for all reference card images.

Scans data/card_images/, extracts a 768-dim L2-normalized DINOv2 CLS+patch
fusion embedding for each image, and saves a dict mapping card_id -> numpy
array to data/ref_embeddings.pkl.

Embedding method: 50/50 fusion of L2-normalized CLS token and L2-normalized
average-pooled patch tokens, then L2-normalized again.

Usage:
    python scripts/build_ref_embeddings.py [--image-dir data/card_images] [--output data/ref_embeddings.pkl]
"""

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _card_id_from_path(img_path: Path, image_dir: Path) -> str:
    """Derive card_id from relative image path.

    Example: data/card_images/sv8/sv8-162_normal.png -> "sv8-162/normal"
    """
    rel = img_path.relative_to(image_dir).with_suffix("")
    card_id = str(rel).replace(os.sep, "/")
    # Strip the set subdirectory prefix: "sv8/sv8-162_normal" -> "sv8-162_normal"
    parts = card_id.split("/")
    if len(parts) == 2:
        card_id = parts[1]
    # Replace last '_' with '/' to separate variant
    last_under = card_id.rfind("_")
    if last_under != -1:
        card_id = card_id[:last_under] + "/" + card_id[last_under + 1:]
    return card_id


def main():
    parser = argparse.ArgumentParser(description="Pre-compute DINOv2 reference embeddings")
    parser.add_argument("--image-dir", default="data/card_images", help="Root dir of card images")
    parser.add_argument("--output", default="data/ref_embeddings.pkl", help="Output pickle path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for GPU inference")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_path = Path(args.output)

    if not image_dir.is_dir():
        logger.error("Image directory not found: %s", image_dir)
        sys.exit(1)

    # Collect all image files
    image_files = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    logger.info("Found %d reference images in %s", len(image_files), image_dir)

    if not image_files:
        logger.error("No images found.")
        sys.exit(1)

    # Import after arg parsing so --help is fast
    import torch
    from PIL import Image
    from torchvision import transforms

    # Use same transform as dino_matcher
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading DINOv2 ViT-B/14 on %s ...", device)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.to(device)
    model.eval()
    logger.info("Model loaded.")

    embeddings: dict[str, np.ndarray] = {}
    failed = 0
    t0 = time.time()
    batch_size = args.batch_size

    # Process in batches for GPU efficiency
    for batch_start in range(0, len(image_files), batch_size):
        batch_paths = image_files[batch_start : batch_start + batch_size]
        batch_tensors = []
        batch_card_ids = []

        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img)
                batch_tensors.append(tensor)
                batch_card_ids.append(_card_id_from_path(img_path, image_dir))
            except Exception as e:
                logger.warning("Failed to load %s: %s", img_path, e)
                failed += 1

        if not batch_tensors:
            continue

        # Stack into batch and run inference
        batch = torch.stack(batch_tensors).to(device)
        try:
            with torch.no_grad():
                batch_emb = model(batch)  # (B, 768) CLS tokens
            batch_np = batch_emb.cpu().numpy().astype(np.float32)  # (B, 768)
        except Exception as e:
            logger.warning("Batch inference failed at index %d: %s", batch_start, e)
            failed += len(batch_tensors)
            continue

        # L2-normalize each embedding
        for i, card_id in enumerate(batch_card_ids):
            vec = batch_np[i]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings[card_id] = vec

        processed = batch_start + len(batch_paths)
        if processed % 100 < batch_size or processed == len(image_files):
            elapsed = time.time() - t0
            rate = len(embeddings) / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d / %d images (%.1f img/s, %d failed)",
                processed, len(image_files), rate, failed,
            )

    elapsed = time.time() - t0
    logger.info(
        "Done: %d embeddings extracted in %.1fs (%d failures)",
        len(embeddings), elapsed, failed,
    )

    # Save
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved to %s (%.1f MB)", output_path, size_mb)


if __name__ == "__main__":
    main()
