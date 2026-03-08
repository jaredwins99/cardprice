#!/usr/bin/env python3
"""Build attack_db.json by OCR-ing all ~20k reference card images.

Usage:
    python scripts/build_attack_db.py [--workers 4] [--resume]

Outputs data/attack_db.json:
    {"base1-1": ["Confuse Ray", "Dark Mind"], ...}
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CARD_IMAGES_DIR = PROJECT_ROOT / "data" / "card_images"
OUTPUT_PATH = PROJECT_ROOT / "data" / "attack_db.json"


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


def _init_worker():
    """Ensure project root is on sys.path in each worker process."""
    import sys
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def process_one(args: tuple[str, str]) -> tuple[str, list[str] | None]:
    """Worker function: extract attack names from one card image.

    Args are (card_id, image_path_str) to stay picklable.
    Returns (card_id, list_of_attack_names) or (card_id, None) on error.
    """
    card_id, image_path_str = args
    try:
        from cardprice.ml.attack_ocr import extract_attack_names_paddle
        candidates = extract_attack_names_paddle(image_path_str)
        attacks = [name for name, _conf in candidates]
        return card_id, attacks
    except Exception as e:
        print(f"  ERROR {card_id}: {e}", file=sys.stderr)
        return card_id, None


def main():
    parser = argparse.ArgumentParser(description="Build attack_db.json from card images")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true", help="Skip card_ids already in output file")
    args = parser.parse_args()

    # Load existing results for resume
    existing: dict[str, list[str]] = {}
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

    with mp.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for card_id, attacks in pool.imap_unordered(process_one, to_process, chunksize=8):
            done += 1
            if attacks is None:
                errors += 1
            else:
                results[card_id] = attacks

            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (len(to_process) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(to_process)} ({rate:.1f}/s, ~{remaining:.0f}s left, {errors} errors)")

            # Checkpoint every 1000
            if done % 1000 == 0:
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(results, f, separators=(",", ":"))

    # Final write (sorted for stable output)
    sorted_results = dict(sorted(results.items()))
    with open(OUTPUT_PATH, "w") as f:
        json.dump(sorted_results, f, indent=1)

    elapsed = time.time() - t0
    print(f"\nDone: {done} cards in {elapsed:.1f}s ({done/elapsed:.1f}/s)")
    print(f"  Errors: {errors}")
    print(f"  Cards with attacks: {sum(1 for v in results.values() if v)}")
    print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
