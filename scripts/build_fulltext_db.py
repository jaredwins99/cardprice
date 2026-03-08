#!/usr/bin/env python3
"""Build fulltext_db.json by OCR-ing ALL text from ~20k reference card images.

Unlike build_attack_db.py which filters to attack names only, this extracts
every piece of text from each card image using RapidOCR.

Usage:
    python scripts/build_fulltext_db.py [--workers 1] [--resume]

Outputs data/fulltext_db.json:
    {"base1-1": "full ocr text here...", ...}
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CARD_IMAGES_DIR = PROJECT_ROOT / "data" / "card_images"
OUTPUT_PATH = PROJECT_ROOT / "data" / "fulltext_db.json"
CHECKPOINT_INTERVAL = 500


def find_all_images() -> list[tuple[str, Path]]:
    """Return (card_id, path) for every card image."""
    results = []
    for set_dir in sorted(CARD_IMAGES_DIR.iterdir()):
        if not set_dir.is_dir():
            continue
        for img_path in sorted(set_dir.glob("*_normal.png")):
            card_id = img_path.stem.replace("_normal", "")
            results.append((card_id, img_path))
    return results


def process_one(args: tuple[str, str]) -> tuple[str, str | None]:
    """Worker function: extract ALL text from one card image via RapidOCR.

    Args are (card_id, image_path_str) to stay picklable.
    Returns (card_id, full_text) or (card_id, None) on error.
    """
    card_id, image_path_str = args
    try:
        import cv2
        from rapidocr_onnxruntime import RapidOCR

        # Each worker gets its own engine (stored on function attribute to reuse)
        if not hasattr(process_one, "_engine"):
            process_one._engine = RapidOCR()

        img = cv2.imread(image_path_str)
        if img is None:
            return card_id, None

        result, elapse = process_one._engine(img)
        if result:
            texts = [text for box, text, conf in result]
            full_text = " ".join(texts).lower()
            # Normalize whitespace
            full_text = " ".join(full_text.split())
            return card_id, full_text
        else:
            return card_id, ""
    except Exception as e:
        return card_id, None


def main():
    parser = argparse.ArgumentParser(description="Build fulltext_db.json from card images")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (1 recommended to avoid OOM)")
    parser.add_argument("--resume", action="store_true", help="Skip card_ids already in output file")
    args = parser.parse_args()

    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # Load existing results for resume
    existing: dict[str, str] = {}
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} cards already processed")

    # Gather all images, skip already-done
    all_images = find_all_images()
    print(f"Found {len(all_images)} card images")

    to_process = [(cid, str(p)) for cid, p in all_images if cid not in existing]
    print(f"To process: {len(to_process)} (skipping {len(all_images) - len(to_process)})")

    if not to_process:
        print("Nothing to do.")
        return

    results = dict(existing)
    done = 0
    errors = 0
    t0 = time.time()

    with mp.Pool(processes=args.workers) as pool:
        for card_id, full_text in pool.imap_unordered(process_one, to_process, chunksize=4):
            done += 1
            if full_text is None:
                errors += 1
            else:
                results[card_id] = full_text

            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (len(to_process) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(to_process)} ({rate:.1f}/s, ~{remaining:.0f}s left, {errors} errors)")

            # Checkpoint
            if done % CHECKPOINT_INTERVAL == 0:
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(results, f, separators=(",", ":"))
                print(f"  [checkpoint] saved {len(results)} cards")

    # Final write (sorted for stable output)
    sorted_results = dict(sorted(results.items()))
    with open(OUTPUT_PATH, "w") as f:
        json.dump(sorted_results, f, indent=1)

    elapsed = time.time() - t0
    print(f"\nDone: {done} cards in {elapsed:.1f}s ({done/elapsed:.1f}/s)")
    print(f"  Errors: {errors}")
    print(f"  Cards with text: {sum(1 for v in sorted_results.values() if v)}")
    print(f"  Cards empty: {sum(1 for v in sorted_results.values() if not v)}")
    print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
