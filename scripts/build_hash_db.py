#!/usr/bin/env python3
"""Build perceptual hash database from downloaded card images."""
from cardprice.ml.hash_matcher import build_hash_database
from pathlib import Path
import sys

image_dir = Path("data/card_images")
if not image_dir.exists() or not any(image_dir.rglob("*.png")):
    print("No images found in data/card_images/. Run download-images first.")
    sys.exit(1)

# Count available images
count = sum(1 for _ in image_dir.rglob("*.png"))
print(f"Found {count} images")

result = build_hash_database(str(image_dir), "data/hash_db.pkl")
print(f"Hash database built: {len(result)} cards")
