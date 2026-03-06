#!/usr/bin/env python3
"""Benchmark EasyOCR readtext batch_size=1 vs batch_size=8.

Measures the actual impact of the CRAFT text detection batch size parameter
on card segment images. Note: batch_size in readtext() controls the CRAFT
text detection network's internal batching, NOT recognition batching.

Also tests readtext_batched() if available in the installed EasyOCR version
(added in newer releases for true multi-image batching).

Usage:
    python scripts/bench_easyocr_batch.py
"""

from __future__ import annotations

import glob
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Find test images
# ---------------------------------------------------------------------------

def find_test_images(n: int = 9) -> list[str]:
    """Find up to n card segment images for benchmarking."""
    # Prefer binder segments (consistent card crops)
    segments_dir = PROJECT_ROOT / "data" / "test_binder_segments"
    candidates = sorted(glob.glob(str(segments_dir / "card_*.png")))

    if len(candidates) >= n:
        return candidates[:n]

    # Fall back to ebay test images (skip binder pages)
    ebay_dir = PROJECT_ROOT / "data" / "eval" / "ebay_test_images"
    ebay_images = sorted(
        p for p in glob.glob(str(ebay_dir / "*"))
        if not Path(p).name.startswith("cg_binder_page")
        and not Path(p).name.startswith("df_master_set")
        and Path(p).suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    candidates.extend(ebay_images)
    return candidates[:n]


def load_images(paths: list[str]) -> list[np.ndarray]:
    """Load images as numpy arrays (BGR)."""
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)
        else:
            print(f"  WARNING: could not read {p}")
    return images


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

def bench_readtext(
    reader,
    images: list[np.ndarray],
    names: list[str],
    batch_size: int,
) -> tuple[list[float], float]:
    """Time readtext() on each image individually with the given batch_size.

    Returns per-image times and total wall time.
    """
    per_image: list[float] = []
    total_start = time.perf_counter()

    for img, name in zip(images, names):
        t0 = time.perf_counter()
        results = reader.readtext(img, detail=1, paragraph=False, batch_size=batch_size)
        elapsed = time.perf_counter() - t0
        per_image.append(elapsed)
        n_detections = len(results) if results else 0
        texts = [r[1] for r in results] if results else []
        print(f"    {name:30s}  {elapsed:6.3f}s  ({n_detections} detections: {texts[:5]})")

    total = time.perf_counter() - total_start
    return per_image, total


def bench_readtext_batched(
    reader,
    images: list[np.ndarray],
    names: list[str],
) -> tuple[float, int] | None:
    """Try readtext_batched() if available. Returns (total_time, n_results) or None."""
    if not hasattr(reader, "readtext_batched"):
        return None

    print("\n  readtext_batched() is available -- benchmarking...")
    t0 = time.perf_counter()
    try:
        batch_results = reader.readtext_batched(images, detail=1, paragraph=False)
        elapsed = time.perf_counter() - t0
        total_detections = sum(len(r) for r in batch_results)
        for name, results in zip(names, batch_results):
            texts = [r[1] for r in results] if results else []
            print(f"    {name:30s}  ({len(results)} detections: {texts[:5]})")
        return elapsed, total_detections
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"    readtext_batched() failed after {elapsed:.3f}s: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EasyOCR batch_size Benchmark")
    print("=" * 70)

    # Find images
    image_paths = find_test_images(9)
    if not image_paths:
        print("ERROR: No test images found.")
        return

    names = [Path(p).name for p in image_paths]
    print(f"\nFound {len(image_paths)} test images:")
    for p in image_paths:
        img = cv2.imread(p)
        h, w = img.shape[:2] if img is not None else (0, 0)
        print(f"  {Path(p).name:30s}  {w}x{h}")

    # Load images into memory so file I/O doesn't affect timing
    print("\nLoading images into memory...")
    images = load_images(image_paths)
    names = names[:len(images)]
    print(f"  Loaded {len(images)} images.\n")

    # Initialize EasyOCR reader (include init time separately)
    print("Initializing EasyOCR reader (gpu=True)...")
    import easyocr
    t_init = time.perf_counter()
    reader = easyocr.Reader(["en"], gpu=True)
    init_time = time.perf_counter() - t_init
    print(f"  Reader initialized in {init_time:.2f}s\n")

    # Warmup: run one image to trigger any lazy CUDA/model loading
    print("Warmup run (1 image, batch_size=1)...")
    _ = reader.readtext(images[0], detail=1, paragraph=False, batch_size=1)
    print("  Done.\n")

    # --- Benchmark batch_size=1 ---
    print("-" * 70)
    print("Benchmark: batch_size=1 (default)")
    print("-" * 70)
    times_bs1, total_bs1 = bench_readtext(reader, images, names, batch_size=1)

    # --- Benchmark batch_size=8 ---
    print()
    print("-" * 70)
    print("Benchmark: batch_size=8")
    print("-" * 70)
    times_bs8, total_bs8 = bench_readtext(reader, images, names, batch_size=8)

    # --- Try readtext_batched ---
    print()
    print("-" * 70)
    print("Benchmark: readtext_batched() (true multi-image batch)")
    print("-" * 70)
    batched_result = bench_readtext_batched(reader, images, names)

    # --- Summary ---
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    # Per-image comparison
    print(f"{'Image':<30s}  {'bs=1':>8s}  {'bs=8':>8s}  {'Diff':>8s}")
    print("-" * 60)
    for name, t1, t8 in zip(names, times_bs1, times_bs8):
        diff = t8 - t1
        print(f"{name:<30s}  {t1:7.3f}s  {t8:7.3f}s  {diff:+7.3f}s")

    print("-" * 60)
    avg_bs1 = sum(times_bs1) / len(times_bs1)
    avg_bs8 = sum(times_bs8) / len(times_bs8)
    print(f"{'Average':<30s}  {avg_bs1:7.3f}s  {avg_bs8:7.3f}s  {avg_bs8 - avg_bs1:+7.3f}s")
    print(f"{'Total':<30s}  {total_bs1:7.3f}s  {total_bs8:7.3f}s  {total_bs8 - total_bs1:+7.3f}s")

    print()
    if total_bs8 > 0:
        speedup = total_bs1 / total_bs8
        print(f"Speedup ratio (bs=1 / bs=8): {speedup:.3f}x")
        if speedup > 1.05:
            print("  -> batch_size=8 is FASTER")
        elif speedup < 0.95:
            print("  -> batch_size=1 is FASTER (batch overhead exceeds benefit)")
        else:
            print("  -> No meaningful difference")

    if batched_result is not None:
        batched_time, batched_detections = batched_result
        print(f"\nreadtext_batched() total time: {batched_time:.3f}s ({batched_detections} detections)")
        if total_bs1 > 0:
            print(f"  vs sequential bs=1: {total_bs1 / batched_time:.3f}x speedup")
        if total_bs8 > 0:
            print(f"  vs sequential bs=8: {total_bs8 / batched_time:.3f}x speedup")
    else:
        print("\nreadtext_batched() is NOT available in this EasyOCR version.")
        print("  (This method provides true multi-image batching in newer releases.)")

    print()
    print("NOTE: batch_size in readtext() controls the CRAFT text detection")
    print("network's internal batch processing, NOT recognition batching.")
    print("For single card images with few text regions, the effect is often")
    print("minimal. True speedup comes from readtext_batched() which batches")
    print("multiple images through both detection and recognition stages.")


if __name__ == "__main__":
    main()
