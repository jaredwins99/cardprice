#!/usr/bin/env python3
"""Build type symbol reference histograms from reference card images.

Extracts the type-symbol ROI (top-right corner) from each reference image,
computes a color histogram in HSV space, and averages per type.  The result
is saved to ``data/type_symbol_refs.pkl`` for use by ``type_detector.py``.

Type labels come from ``data/card_names.json`` (column 4 = list of types).
Reference images live in ``data/card_images/{set}/{set}-{num}_{variant}.png``
at approximately 240x330 pixels.

ROI strategy
------------
The type symbol sits in the top-right of the card, to the right of the HP
text.  Its exact pixel position shifts between eras (Base Set vs DP vs SV)
but in proportional coordinates it reliably occupies:

    x: 88-98% of card width
    y:  1-8% of card height

We extract this ROI, convert to HSV, and compute a normalised 3-D colour
histogram (H=18 bins, S=8 bins, V=4 bins).  By averaging hundreds or
thousands of histograms per type we get a robust per-type signature.

At query time, the same ROI + histogram pipeline is applied to a binder-scan
segment and compared against the reference histograms using histogram
correlation (``cv2.compareHist``).

Usage::

    python scripts/build_type_reference.py          # full build
    python scripts/build_type_reference.py --quick   # sample ≤50 per type
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CARD_IMAGES_DIR = ROOT / "data" / "card_images"
CARD_NAMES_JSON = ROOT / "data" / "card_names.json"
OUTPUT_PKL = ROOT / "data" / "type_symbol_refs.pkl"

# HSV histogram parameters
H_BINS = 18   # 180 / 18 = 10-degree bins
S_BINS = 8    # 256 / 8  = 32-value bins
V_BINS = 4    # 256 / 4  = 64-value bins
HIST_SIZE = [H_BINS, S_BINS, V_BINS]
H_RANGE = [0, 180]
S_RANGE = [0, 256]
V_RANGE = [0, 256]

# ROI proportional coordinates (relative to card image dimensions)
ROI_X_START = 0.88
ROI_X_END   = 0.98
ROI_Y_START = 0.01
ROI_Y_END   = 0.08

logger = logging.getLogger(__name__)


def _card_id_to_image_path(card_id: str) -> Path | None:
    """Convert a card_id like 'base1-44/normal' to its image file path."""
    parts = card_id.split("/")
    if len(parts) != 2:
        return None
    base_id, variant = parts
    set_id = base_id.rsplit("-", 1)[0]
    fname = f"{base_id}_{variant}.png"
    path = CARD_IMAGES_DIR / set_id / fname
    return path if path.exists() else None


def _extract_symbol_roi(img: np.ndarray) -> np.ndarray:
    """Extract the type-symbol ROI from a card image (BGR).

    Returns the ROI as a BGR ndarray, or an empty array if extraction fails.
    """
    h, w = img.shape[:2]
    x1 = int(w * ROI_X_START)
    x2 = int(w * ROI_X_END)
    y1 = int(h * ROI_Y_START)
    y2 = int(h * ROI_Y_END)

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)

    return img[y1:y2, x1:x2]


def _compute_hsv_histogram(roi_bgr: np.ndarray) -> np.ndarray | None:
    """Compute a normalised 3-D HSV histogram from a BGR ROI.

    Returns a flattened float32 histogram, or None if the ROI is empty.
    """
    if roi_bgr.size == 0:
        return None

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [roi_hsv], [0, 1, 2], None,
        HIST_SIZE, H_RANGE + S_RANGE + V_RANGE,
    )
    # Normalise so total sums to 1
    total = hist.sum()
    if total == 0:
        return None
    hist = (hist / total).astype(np.float32)
    return hist.flatten()


def _load_type_labels() -> dict[str, str]:
    """Load card_id -> primary type from card_names.json.

    Cards with multiple types use only the first type.
    Cards without types are skipped.
    """
    with open(CARD_NAMES_JSON) as f:
        rows = json.load(f)

    labels: dict[str, str] = {}
    for row in rows:
        card_id = row[0]
        types = row[4]
        if types:
            labels[card_id] = types[0]
    return labels


def build_type_references(
    max_per_type: int = 0,
    verbose: bool = False,
) -> dict:
    """Build per-type average histograms from reference card images.

    Parameters
    ----------
    max_per_type : int
        If > 0, sample at most this many images per type (for speed).
    verbose : bool
        Print per-card progress.

    Returns
    -------
    dict with keys:
        "histograms"  : dict[str, np.ndarray]  -- type -> avg histogram (flat)
        "hist_size"   : list[int]               -- [H_BINS, S_BINS, V_BINS]
        "ranges"      : list[int]               -- [H_RANGE + S_RANGE + V_RANGE]
        "roi"         : dict                    -- proportional ROI coords
        "counts"      : dict[str, int]          -- type -> number of images used
        "mean_bgr"    : dict[str, list]         -- type -> [B, G, R] mean of ROI
        "mean_hsv"    : dict[str, list]         -- type -> [H, S, V] mean of ROI
    """
    labels = _load_type_labels()
    logger.info("Loaded type labels for %d cards", len(labels))

    # Group card IDs by type
    type_groups: dict[str, list[str]] = defaultdict(list)
    for card_id, ptype in labels.items():
        type_groups[ptype].append(card_id)

    logger.info("Types: %s", {t: len(ids) for t, ids in sorted(type_groups.items())})

    # Collect histograms per type
    type_hists: dict[str, list[np.ndarray]] = defaultdict(list)
    type_bgr_sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    type_hsv_sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    type_counts: dict[str, int] = defaultdict(int)

    skipped = 0
    processed = 0
    t0 = time.time()

    for ptype, card_ids in sorted(type_groups.items()):
        sample = card_ids
        if max_per_type > 0 and len(sample) > max_per_type:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(sample), max_per_type, replace=False)
            sample = [sample[i] for i in indices]

        for card_id in sample:
            img_path = _card_id_to_image_path(card_id)
            if img_path is None:
                skipped += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            roi = _extract_symbol_roi(img)
            hist = _compute_hsv_histogram(roi)
            if hist is None:
                skipped += 1
                continue

            type_hists[ptype].append(hist)
            type_counts[ptype] += 1

            # Accumulate mean BGR and HSV for diagnostics
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mean_bgr = roi.mean(axis=(0, 1))
            mean_hsv = roi_hsv.mean(axis=(0, 1))
            for i in range(3):
                type_bgr_sums[ptype][i] += float(mean_bgr[i])
                type_hsv_sums[ptype][i] += float(mean_hsv[i])

            processed += 1
            if verbose and processed % 500 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "  %d processed, %d skipped (%.1f img/s)",
                    processed, skipped, processed / elapsed,
                )

    elapsed = time.time() - t0
    logger.info(
        "Processed %d images in %.1fs (%.0f img/s), skipped %d",
        processed, elapsed, processed / max(elapsed, 0.001), skipped,
    )

    # Average histograms per type
    avg_histograms: dict[str, np.ndarray] = {}
    mean_bgr: dict[str, list[float]] = {}
    mean_hsv: dict[str, list[float]] = {}

    for ptype, hists in sorted(type_hists.items()):
        if not hists:
            logger.warning("No histograms for type %s, skipping", ptype)
            continue
        stacked = np.stack(hists, axis=0)
        avg = stacked.mean(axis=0).astype(np.float32)
        # Re-normalise after averaging
        total = avg.sum()
        if total > 0:
            avg /= total
        avg_histograms[ptype] = avg

        n = type_counts[ptype]
        mean_bgr[ptype] = [round(type_bgr_sums[ptype][i] / n, 1) for i in range(3)]
        mean_hsv[ptype] = [round(type_hsv_sums[ptype][i] / n, 1) for i in range(3)]

        logger.info(
            "  %-12s  %5d images  mean_HSV=(%5.1f, %5.1f, %5.1f)  mean_BGR=(%5.1f, %5.1f, %5.1f)",
            ptype, n,
            mean_hsv[ptype][0], mean_hsv[ptype][1], mean_hsv[ptype][2],
            mean_bgr[ptype][0], mean_bgr[ptype][1], mean_bgr[ptype][2],
        )

    result = {
        "histograms": avg_histograms,
        "hist_size": HIST_SIZE,
        "ranges": H_RANGE + S_RANGE + V_RANGE,
        "roi": {
            "x_start": ROI_X_START,
            "x_end": ROI_X_END,
            "y_start": ROI_Y_START,
            "y_end": ROI_Y_END,
        },
        "counts": dict(type_counts),
        "mean_bgr": mean_bgr,
        "mean_hsv": mean_hsv,
    }
    return result


def match_type_from_histogram(
    roi_bgr: np.ndarray,
    refs: dict,
    *,
    top_n: int = 3,
    method: int = cv2.HISTCMP_CORREL,
) -> list[tuple[str, float]]:
    """Match a type-symbol ROI against reference histograms.

    Parameters
    ----------
    roi_bgr : np.ndarray
        The type-symbol ROI in BGR.
    refs : dict
        The reference data from ``build_type_references()`` / pickle.
    top_n : int
        Number of top matches to return.
    method : int
        OpenCV histogram comparison method (default: correlation).

    Returns
    -------
    list of (type_name, score) sorted by score descending.
    """
    hist = _compute_hsv_histogram(roi_bgr)
    if hist is None:
        return [("Colorless", 0.0)]

    hist_size = refs["hist_size"]
    hist_3d = hist.reshape(hist_size)

    scores: list[tuple[str, float]] = []
    for ptype, ref_hist_flat in refs["histograms"].items():
        ref_3d = ref_hist_flat.reshape(hist_size)
        score = cv2.compareHist(
            hist_3d.astype(np.float32),
            ref_3d.astype(np.float32),
            method,
        )
        scores.append((ptype, float(score)))

    # For CORREL and INTERSECT, higher is better
    # For CHI_SQR and BHATTACHARYYA, lower is better
    if method in (cv2.HISTCMP_CORREL, cv2.HISTCMP_INTERSECT):
        scores.sort(key=lambda x: x[1], reverse=True)
    else:
        scores.sort(key=lambda x: x[1])

    return scores[:top_n]


def main():
    parser = argparse.ArgumentParser(
        description="Build type-symbol reference histograms from card images",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Sample at most 50 images per type (fast test mode)",
    )
    parser.add_argument(
        "--max-per-type", type=int, default=0,
        help="Max images per type (0 = all)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After building, run self-test on a sample of reference images",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    max_per_type = args.max_per_type
    if args.quick:
        max_per_type = 50

    refs = build_type_references(max_per_type=max_per_type, verbose=True)

    # Save
    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(refs, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_images = sum(refs["counts"].values())
    logger.info(
        "Saved %d types (%d total images) to %s",
        len(refs["histograms"]), total_images, OUTPUT_PKL,
    )

    # --- Optional verification ---
    if args.verify:
        _run_verification(refs)


def _run_verification(refs: dict) -> None:
    """Run a self-test: classify a sample of reference images and report accuracy."""
    labels = _load_type_labels()
    type_groups: dict[str, list[str]] = defaultdict(list)
    for card_id, ptype in labels.items():
        type_groups[ptype].append(card_id)

    rng = np.random.RandomState(99)
    correct = 0
    total = 0
    per_type_correct: dict[str, int] = defaultdict(int)
    per_type_total: dict[str, int] = defaultdict(int)

    # Sample up to 20 per type
    sample_n = 20
    for ptype, card_ids in sorted(type_groups.items()):
        if ptype not in refs["histograms"]:
            continue
        sample = card_ids
        if len(sample) > sample_n:
            indices = rng.choice(len(sample), sample_n, replace=False)
            sample = [sample[i] for i in indices]

        for card_id in sample:
            img_path = _card_id_to_image_path(card_id)
            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            roi = _extract_symbol_roi(img)
            results = match_type_from_histogram(roi, refs, top_n=3)
            predicted = results[0][0] if results else "?"
            is_correct = predicted == ptype

            per_type_total[ptype] += 1
            total += 1
            if is_correct:
                correct += 1
                per_type_correct[ptype] += 1
            else:
                logger.warning(
                    "  WRONG: %s expected=%s predicted=%s (scores: %s)",
                    card_id, ptype, predicted,
                    ", ".join(f"{t}={s:.3f}" for t, s in results),
                )

    logger.info("\n--- Verification Results ---")
    logger.info("Overall: %d / %d = %.1f%%", correct, total, 100.0 * correct / max(total, 1))
    for ptype in sorted(per_type_total.keys()):
        n = per_type_total[ptype]
        c = per_type_correct[ptype]
        logger.info("  %-12s  %d / %d = %.0f%%", ptype, c, n, 100.0 * c / max(n, 1))


if __name__ == "__main__":
    main()
