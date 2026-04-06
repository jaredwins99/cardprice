#!/usr/bin/env python3
"""Test multiple parallelism strategies for OCR on card images.

Compares sequential, threaded, process pool with initializer,
pre-fork, and persistent (warm) pool approaches.

IMPORTANT: Process-based strategies use 'spawn' context to avoid
deadlocks from forking after importing onnxruntime/torch.
"""

import glob
import multiprocessing
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Force CPU-only to avoid CUDA issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

TEST_IMAGES = sorted(glob.glob(
    os.path.join(PROJECT_ROOT, "data/inbox/page_20260228_195512_cards/card_*.png")
))
assert len(TEST_IMAGES) == 9, f"Expected 9 test images, got {len(TEST_IMAGES)}"

N_PROC_WORKERS = 3


# ---------------------------------------------------------------------------
# Shared worker functions (must be top-level for pickling with 'spawn')
# ---------------------------------------------------------------------------
_worker_engine = None


def _worker_init():
    """Called once per worker process to pre-load the RapidOCR engine."""
    global _worker_engine
    # Suppress CUDA warnings in child processes
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from cardprice.ml.ocr_matcher import get_rapid_engine
    _worker_engine = get_rapid_engine()
    from cardprice.ml.ocr_matcher import _load_unique_pokemon_names
    _load_unique_pokemon_names()


def _pool_ocr_worker(image_path: str) -> dict:
    """OCR worker that uses the pre-loaded global engine."""
    import cv2
    import re
    from cardprice.ml.ocr_matcher import (
        get_rapid_engine, _unsharp_mask_ocr,
        _load_unique_pokemon_names,
    )
    from cardprice.ml.preprocess import upscale_for_ocr
    from rapidfuzz import fuzz, process

    global _worker_engine
    engine = _worker_engine if _worker_engine is not None else get_rapid_engine()

    img = cv2.imread(str(image_path))
    if img is None:
        return {"ocr_name": None, "ocr_conf": 0.0, "ocr_raw": None, "hp_value": None}

    h, w = img.shape[:2]
    y1, y2 = 0, int(h * 0.25)
    x1, x2 = int(w * 0.03), int(w * 0.97)
    crop = img[y1:y2, x1:x2]

    pad = 30
    crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = _unsharp_mask_ocr(crop)
    crop_up = upscale_for_ocr(crop, scale=2)

    ocr_result, _ = engine(crop_up)
    if not ocr_result:
        return {"ocr_name": None, "ocr_conf": 0.0, "ocr_raw": None, "hp_value": None}

    texts = [text for _bbox, text, _conf in ocr_result]
    raw_text = " | ".join(texts)

    pokemon_names = _load_unique_pokemon_names()
    _NON_NAME_WORDS = {"stage", "basic", "hp", "trainer", "supporter",
                       "pokemon", "item", "energy", "stadium", "tool"}

    best_name = None
    best_conf = 0.0
    hp_value = None

    for _bbox, text, conf in ocr_result:
        clean = text.strip()
        if clean.lower() in _NON_NAME_WORDS:
            continue
        hp_match = re.search(r'(\d{2,3})\s*HP|HP\s*(\d{2,3})', clean, re.IGNORECASE)
        if hp_match:
            hp_val = int(hp_match.group(1) or hp_match.group(2))
            if 10 <= hp_val <= 340 and hp_val % 10 == 0:
                hp_value = hp_val
        match = process.extractOne(clean, pokemon_names, scorer=fuzz.ratio, score_cutoff=60)
        if match and match[1] > best_conf:
            best_name = match[0]
            best_conf = match[1] / 100.0

    return {
        "ocr_name": best_name,
        "ocr_conf": best_conf,
        "ocr_raw": raw_text,
        "hp_value": hp_value,
    }


# ---------------------------------------------------------------------------
# Strategy 1: Baseline sequential
# ---------------------------------------------------------------------------
def run_baseline():
    from cardprice.ml import _name_ocr_worker
    results = []
    for img in TEST_IMAGES:
        results.append(_name_ocr_worker(img))
    return results


# ---------------------------------------------------------------------------
# Strategy 2: ThreadPool(9)
# ---------------------------------------------------------------------------
def run_threadpool():
    from cardprice.ml import _name_ocr_worker
    with ThreadPoolExecutor(max_workers=9) as pool:
        results = list(pool.map(_name_ocr_worker, TEST_IMAGES))
    return results


# ---------------------------------------------------------------------------
# Strategy 3: ProcessPool with initializer (spawn context)
# ---------------------------------------------------------------------------
def run_process_pool_init():
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(N_PROC_WORKERS, initializer=_worker_init) as pool:
        results = pool.map(_pool_ocr_worker, TEST_IMAGES)
    return results


# ---------------------------------------------------------------------------
# Strategy 4: Pre-fork approach
# Fork BEFORE importing heavy libs, so children inherit nothing and load fresh.
# This tests whether fork+COW is faster than spawn for engine sharing.
# ---------------------------------------------------------------------------
def run_prefork():
    # On Linux, default is fork. The engine singleton loaded in parent
    # is inherited by children via COW. get_rapid_engine() returns it
    # immediately without re-loading.
    # NOTE: We must NOT have imported onnxruntime in the parent for this
    # to work safely. Since strategies 1&2 already imported it, we run
    # this via subprocess instead (see main).
    ctx = multiprocessing.get_context("fork")
    from cardprice.ml.ocr_matcher import get_rapid_engine
    get_rapid_engine()  # Load in parent
    with ctx.Pool(N_PROC_WORKERS) as pool:
        results = pool.map(_pool_ocr_worker, TEST_IMAGES)
    return results


# ---------------------------------------------------------------------------
# Strategy 5: Persistent pool (warm vs cold)
# ---------------------------------------------------------------------------
def run_persistent_pool():
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(N_PROC_WORKERS, initializer=_worker_init)
    timings = []

    for batch_num in range(3):
        t0 = time.perf_counter()
        results = pool.map(_pool_ocr_worker, TEST_IMAGES)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        names = [r.get("ocr_name", "?") for r in results]
        print(f"    Batch {batch_num + 1}: {elapsed:.2f}s  names={names}")
        sys.stdout.flush()

    pool.close()
    pool.join()
    return timings


# ---------------------------------------------------------------------------
# Main: run each strategy, skip 4 (fork after onnxruntime import is unsafe)
# ---------------------------------------------------------------------------
def main():
    print(f"Test images: {len(TEST_IMAGES)} cards")
    print(f"Process workers: {N_PROC_WORKERS}")
    print(f"CPU count: {os.cpu_count()}")
    print()

    all_times = {}

    strategies = [
        ("1. Baseline (sequential)", run_baseline),
        ("2. ThreadPool(9)", run_threadpool),
        ("3. ProcessPool({}, spawn) + initializer".format(N_PROC_WORKERS), run_process_pool_init),
        # Strategy 4 skipped: fork after onnxruntime import deadlocks.
        # In a real deployment, you'd fork BEFORE importing any ML libs.
    ]

    for label, fn in strategies:
        print(f"--- {label} ---")
        sys.stdout.flush()
        t0 = time.perf_counter()
        results = fn()
        elapsed = time.perf_counter() - t0
        names = [r.get("ocr_name", "?") for r in results]
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Names: {names}")
        print()
        sys.stdout.flush()
        all_times[label] = elapsed

    print("--- 4. Pre-fork (skipped: fork after onnxruntime import deadlocks) ---")
    print("  In production, you'd fork BEFORE importing ML libs.")
    print()

    print("--- 5. Persistent pool (3 batches: cold then warm) ---")
    sys.stdout.flush()
    timings = run_persistent_pool()
    print(f"  Cold: {timings[0]:.2f}s  Warm1: {timings[1]:.2f}s  Warm2: {timings[2]:.2f}s")
    print()

    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for label, t in all_times.items():
        print(f"  {label}: {t:.2f}s")
    print()
    print("Strategy 5 (Persistent pool) warm vs cold:")
    print(f"  Cold start (init + work): {timings[0]:.2f}s")
    warm_avg = sum(timings[1:]) / len(timings[1:])
    print(f"  Warm average:             {warm_avg:.2f}s")
    if warm_avg > 0:
        print(f"  Speedup (cold/warm):      {timings[0] / warm_avg:.1f}x")


if __name__ == "__main__":
    main()
