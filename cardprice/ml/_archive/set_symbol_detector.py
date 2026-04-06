"""Set symbol detection and matching for Pokemon card identification.

Every Pokemon card has a small set symbol icon in the bottom-right area,
near the collector number (e.g. "16/132").  This module provides:

1. **extract_set_symbol()** -- Crop the set symbol region from a card image.
2. **build_set_symbol_index()** -- Build a per-set binary symbol index from
   reference card images.
3. **match_set_symbol()** -- Match a scanned card's symbol region against the
   index to narrow down candidate sets.
4. **rerank_candidates_by_set_symbol()** -- Rerank cascade candidates using
   set symbol similarity as a disambiguation signal.
5. **download_set_symbols()** -- Download official set symbol PNGs from
   pokemontcg.io.

Usage as disambiguation signal
------------------------------
The set symbol is most effective when used to **disambiguate** between a small
number of candidate sets (5-15), not as a standalone identifier across all 167
sets.  In testing:

- Against 15 candidate sets: correct set ranks #1-3 consistently.
- Against all 167 sets: correct set may rank #20-50 due to noise.

This matches the intended use: the DINOv2/CLIP cascade gives ~10 candidates,
then set symbol matching narrows to 2-3 candidate sets.  Combined with OCR
card name, this gives near-exact identification.

Set symbol location on Pokemon cards
-------------------------------------
The set symbol appears in the bottom info bar:
- **Reference cards** (245x342): x: 76-88%, y: 90-97%
- **Phone scans** (630x880): varies by card position in sleeve/binder.
  Multiple crop presets are tried to handle this variation.
"""

from __future__ import annotations

import logging
import os
import pickle
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CARD_IMAGES_DIR = _PROJECT_ROOT / "data" / "card_images"
_SET_SYMBOLS_DIR = _PROJECT_ROOT / "data" / "set_symbols"
_INDEX_PATH = _PROJECT_ROOT / "data" / "set_symbol_index.pkl"

# ---------------------------------------------------------------------------
# Crop region presets (fractional coordinates: y1, y2, x1, x2)
#
# Reference cards have consistent layout, but scanned binder page cards vary
# slightly due to sleeve position, angle, and card cropping.  We try multiple
# presets and use the best match.
# ---------------------------------------------------------------------------

# Reference card images (clean digital renders, 245x342)
_REF_REGIONS = [
    (0.90, 0.97, 0.76, 0.88),  # Modern cards: bottom-right
]

# Scanned/phone photo cards (from binder pages, ~630x880)
_SCAN_REGIONS = [
    (0.885, 0.915, 0.77, 0.84),  # Calibrated from DP-era binder scans
    (0.86, 0.90, 0.75, 0.85),    # Slightly higher / different sleeve position
    (0.87, 0.92, 0.76, 0.86),    # Alternative position
    (0.84, 0.89, 0.74, 0.84),    # Cards positioned higher in sleeve
]

# Size to resize symbol crops for comparison (width, height)
_SYMBOL_SIZE = 32


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _extract_binary(img: np.ndarray, y1f: float, y2f: float,
                    x1f: float, x2f: float,
                    size: int = _SYMBOL_SIZE) -> np.ndarray | None:
    """Extract and binarize the set symbol region from a card image.

    Returns a binary (0/255) image of size (size, size), or None if the
    crop region is too small / empty.
    """
    h, w = img.shape[:2]
    y1, y2 = int(y1f * h), int(y2f * h)
    x1, x2 = int(x1f * w), int(x2f * w)

    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

    # Adaptive binarization to handle varying backgrounds
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 4,
    )
    return binary


def _extract_multi_preset(img: np.ndarray,
                          regions: list[tuple[float, float, float, float]],
                          ) -> list[np.ndarray]:
    """Extract binary symbols using multiple crop presets.

    Returns a list of binary images (one per successful preset).
    """
    results = []
    for y1f, y2f, x1f, x2f in regions:
        binary = _extract_binary(img, y1f, y2f, x1f, x2f)
        if binary is not None:
            results.append(binary)
    return results


# ---------------------------------------------------------------------------
# Public API: extract_set_symbol
# ---------------------------------------------------------------------------

def extract_set_symbol(
    image_path: str | Path,
    *,
    is_scan: bool = True,
    output_size: tuple[int, int] | None = (64, 64),
) -> np.ndarray:
    """Crop the set-symbol region from a Pokemon card image.

    Parameters
    ----------
    image_path : str or Path
        Path to the card image (JPEG/PNG).
    is_scan : bool
        True for phone photos, False for reference card images.
    output_size : tuple or None
        Resize the crop to this (width, height).  None returns raw crop.

    Returns
    -------
    numpy.ndarray
        BGR crop of the set-symbol region.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    return extract_set_symbol_from_array(img, is_scan=is_scan,
                                          output_size=output_size)


def extract_set_symbol_from_array(
    img: np.ndarray,
    *,
    is_scan: bool = True,
    output_size: tuple[int, int] | None = (64, 64),
) -> np.ndarray:
    """Crop the set-symbol region from a card image array.

    Tries multiple crop presets and returns the one with the most edge content
    (proxy for "contains a symbol rather than blank background").
    """
    regions = _SCAN_REGIONS if is_scan else _REF_REGIONS
    h, w = img.shape[:2]

    best_crop = None
    best_score = -1

    for y1f, y2f, x1f, x2f in regions:
        y1, y2 = int(y1f * h), int(y2f * h)
        x1, x2 = int(x1f * w), int(x2f * w)
        crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0 or crop.shape[0] < 4:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if score > best_score:
            best_score = score
            best_crop = crop

    if best_crop is None:
        # Fallback to first region
        y1f, y2f, x1f, x2f = regions[0]
        best_crop = img[int(y1f * h):int(y2f * h), int(x1f * w):int(x2f * w)]

    if output_size is not None:
        best_crop = cv2.resize(best_crop, output_size, interpolation=cv2.INTER_AREA)

    return best_crop


# ---------------------------------------------------------------------------
# Public API: build_set_symbol_index
# ---------------------------------------------------------------------------

def build_set_symbol_index(
    card_images_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    samples_per_set: int = 5,
) -> dict:
    """Build a set symbol binary image index from reference card images.

    Samples up to *samples_per_set* cards from each set directory, extracts
    binarized symbol regions, and stores them for matchShapes comparison.

    The index is compact (~2 MB for 167 sets at 5 samples each) because
    each sample is only a 32x32 binary image.

    Parameters
    ----------
    card_images_dir : path, optional
        Directory containing per-set subdirectories of card images.
        Defaults to data/card_images/.
    output_path : path, optional
        Where to save the pickled index.  Defaults to data/set_symbol_index.pkl.
    samples_per_set : int
        Maximum number of reference cards to sample per set.

    Returns
    -------
    dict with keys:
        - "set_binaries": dict[str, list[list]] -- per-set binarized symbol images
        - "version": int -- index format version
    """
    if card_images_dir is None:
        card_images_dir = _CARD_IMAGES_DIR
    card_images_dir = Path(card_images_dir)

    if output_path is None:
        output_path = _INDEX_PATH
    output_path = Path(output_path)

    set_bins: dict[str, list] = {}
    total = 0

    for set_id in sorted(os.listdir(card_images_dir)):
        set_dir = card_images_dir / set_id
        if not set_dir.is_dir():
            continue

        images = sorted(f for f in os.listdir(set_dir) if f.endswith(".png"))
        # Sample evenly across the set to get representative symbols
        step = max(1, len(images) // samples_per_set)
        sample = images[::step][:samples_per_set]

        bins = []
        for fname in sample:
            img = cv2.imread(str(set_dir / fname))
            if img is None:
                continue

            for y1f, y2f, x1f, x2f in _REF_REGIONS:
                binary = _extract_binary(img, y1f, y2f, x1f, x2f)
                if binary is not None:
                    bins.append(binary.tolist())
                    total += 1
                    break  # One region per card

        if bins:
            set_bins[set_id] = bins

    index = {
        "set_binaries": set_bins,
        "version": 1,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(index, f)

    logger.info(
        "Built set symbol index: %d sets, %d samples -> %s (%.0f KB)",
        len(set_bins), total, output_path,
        output_path.stat().st_size / 1024,
    )
    return index


# ---------------------------------------------------------------------------
# Public API: match_set_symbol
# ---------------------------------------------------------------------------

_cached_index: dict | None = None


def _load_index(index_path: Path | None = None) -> dict | None:
    """Load the set symbol index, with caching."""
    global _cached_index
    if _cached_index is not None:
        return _cached_index

    path = index_path or _INDEX_PATH
    if not path.exists():
        return None

    with open(path, "rb") as f:
        _cached_index = pickle.load(f)

    # Convert stored lists back to numpy arrays
    for set_id in _cached_index.get("set_binaries", {}):
        _cached_index["set_binaries"][set_id] = [
            np.array(b, dtype=np.uint8) for b in _cached_index["set_binaries"][set_id]
        ]

    n_sets = len(_cached_index.get("set_binaries", {}))
    logger.info("Loaded set symbol index (%d sets)", n_sets)
    return _cached_index


def match_set_symbol(
    image_path: str | Path | None = None,
    *,
    img: np.ndarray | None = None,
    preloaded_index: dict | None = None,
    top_k: int = 10,
    candidate_sets: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Match a card's set symbol region against the index.

    Uses cv2.matchShapes() (Hu moments) to compare the binarized symbol
    region of the query card against each set's reference samples.

    When *candidate_sets* is provided, only those sets are scored (much faster
    and more accurate for disambiguation).

    Parameters
    ----------
    image_path : path, optional
        Path to the card image.
    img : ndarray, optional
        Card image as a numpy array (alternative to image_path).
    preloaded_index : dict, optional
        Pre-loaded index to avoid re-reading from disk.
    top_k : int
        Number of top matches to return.
    candidate_sets : list of str, optional
        If provided, only score these sets (for disambiguation).

    Returns
    -------
    list of (set_id, distance) tuples, sorted by ascending distance.
    Lower distance = better match.  Distance is from cv2.matchShapes().
    """
    if img is None:
        if image_path is None:
            raise ValueError("Provide either image_path or img")
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not decode image: {image_path}")

    index = preloaded_index or _load_index()
    if index is None:
        logger.warning("Set symbol index not found. Build with: "
                       "python -m cardprice.ml.set_symbol_detector build_index")
        return []

    set_bins = index.get("set_binaries", {})
    sets_to_check = candidate_sets or list(set_bins.keys())

    # Extract binary symbols from the query using multiple crop presets
    query_binaries = _extract_multi_preset(img, _SCAN_REGIONS)
    if not query_binaries:
        # Fallback to reference regions
        query_binaries = _extract_multi_preset(img, _REF_REGIONS)
    if not query_binaries:
        return []

    # Score each candidate set
    results = []
    for set_id in sets_to_check:
        if set_id not in set_bins:
            continue

        ref_bins = set_bins[set_id]
        if not ref_bins:
            continue

        # Find the minimum distance across all query x reference combinations
        best_dist = float("inf")
        for q_bin in query_binaries:
            for r_bin in ref_bins:
                dist = cv2.matchShapes(q_bin, r_bin, cv2.CONTOURS_MATCH_I2, 0)
                best_dist = min(best_dist, dist)

        results.append((set_id, best_dist))

    results.sort(key=lambda x: x[1])
    return results[:top_k]


def rerank_candidates_by_set_symbol(
    candidates: list[tuple[str, float]],
    img: np.ndarray,
    preloaded_index: dict | None = None,
    boost_factor: float = 0.05,
) -> list[tuple[str, float]]:
    """Rerank identification candidates using set symbol matching.

    Takes a list of (card_id, score) candidates from the main identification
    pipeline and adjusts scores based on set symbol similarity.

    Cards from sets that match the symbol region get a score boost;
    cards from non-matching sets are unchanged.

    This is the primary integration point with the cascade pipeline.

    Parameters
    ----------
    candidates : list of (card_id, score)
        Candidate identifications from DINOv2/CLIP/etc.
    img : ndarray
        The card image (BGR).
    preloaded_index : dict, optional
        Pre-loaded set symbol index.
    boost_factor : float
        Maximum score boost for the best-matching set (default 0.05 = 5%).

    Returns
    -------
    Re-sorted list of (card_id, adjusted_score).
    """
    if not candidates or len(candidates) < 2:
        return candidates

    # Extract unique sets from candidates
    candidate_sets = list({
        cid.split("-")[0].split("/")[0]
        for cid, _ in candidates
    })

    if len(candidate_sets) < 2:
        return candidates  # All from same set, nothing to disambiguate

    # Match set symbol against candidate sets only
    set_matches = match_set_symbol(
        img=img,
        preloaded_index=preloaded_index,
        candidate_sets=candidate_sets,
        top_k=len(candidate_sets),
    )

    if not set_matches:
        return candidates

    # Convert distances to scores: lower distance = higher score
    # Normalize to [0, 1] range within the candidate set
    distances = {sid: dist for sid, dist in set_matches}
    max_dist = max(distances.values())
    min_dist = min(distances.values())
    dist_range = max_dist - min_dist

    if dist_range < 1e-6:
        return candidates  # All sets look the same

    # Build set -> normalized symbol score (1.0 = best match, 0.0 = worst)
    sym_scores = {}
    for sid, dist in distances.items():
        sym_scores[sid] = 1.0 - (dist - min_dist) / dist_range

    # Apply boost
    adjusted = []
    for card_id, orig_score in candidates:
        set_id = card_id.split("-")[0].split("/")[0]
        sym_score = sym_scores.get(set_id, 0.0)
        boost = boost_factor * sym_score
        adjusted.append((card_id, orig_score + boost))

    adjusted.sort(key=lambda x: -x[1])

    logger.debug(
        "Set symbol reranking: %d candidates, %d sets. "
        "Best set symbol match: %s (dist=%.6f)",
        len(candidates), len(candidate_sets),
        set_matches[0][0], set_matches[0][1],
    )

    return adjusted


# ---------------------------------------------------------------------------
# Public API: download_set_symbols
# ---------------------------------------------------------------------------

def download_set_symbols(
    set_ids: list[str] | None = None,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Download official set symbol PNGs from pokemontcg.io.

    These are high-resolution (~500x465) black-on-transparent PNGs of each
    set's symbol.  Useful for visual reference but NOT directly used for
    matching (the symbols on actual cards are too small for template matching).

    Parameters
    ----------
    set_ids : list of str, optional
        Set IDs to download.  Defaults to all sets in data/card_images/.
    output_dir : path, optional
        Where to save symbol PNGs.  Defaults to data/set_symbols/.
    force : bool
        Re-download even if file exists.

    Returns
    -------
    dict mapping set_id -> Path of downloaded symbol PNG.
    """
    if output_dir is None:
        output_dir = _SET_SYMBOLS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if set_ids is None:
        if _CARD_IMAGES_DIR.exists():
            set_ids = sorted(
                d for d in os.listdir(_CARD_IMAGES_DIR)
                if (_CARD_IMAGES_DIR / d).is_dir()
            )
        else:
            set_ids = []

    downloaded = {}
    failed = 0

    for set_id in set_ids:
        path = output_dir / f"{set_id}.png"
        if path.exists() and not force:
            downloaded[set_id] = path
            continue

        url = f"https://images.pokemontcg.io/{set_id}/symbol.png"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                path.write_bytes(data)
                downloaded[set_id] = path
        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning("Failed to download %s: %s", set_id, e)

    logger.info("Downloaded %d set symbols (%d failed)", len(downloaded), failed)
    return downloaded


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli():
    """Command-line interface for building and testing the set symbol index."""
    import sys
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m cardprice.ml.set_symbol_detector <command>")
        print()
        print("Commands:")
        print("  build_index     Build set symbol feature index from reference cards")
        print("  download        Download set symbol PNGs from pokemontcg.io")
        print("  match <image>   Match a card image against the index (all sets)")
        print("  test            Run disambiguation test on known binder page cards")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "build_index":
        t0 = time.time()
        index = build_set_symbol_index()
        print(f"Built index in {time.time() - t0:.1f}s: "
              f"{len(index['set_binaries'])} sets")

    elif cmd == "download":
        t0 = time.time()
        result = download_set_symbols()
        print(f"Downloaded {len(result)} symbols in {time.time() - t0:.1f}s")

    elif cmd == "match":
        if len(sys.argv) < 3:
            print("Usage: match <image_path> [candidate_set1,set2,...]")
            sys.exit(1)
        image_path = sys.argv[2]
        candidate_sets = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        matches = match_set_symbol(
            image_path, candidate_sets=candidate_sets, top_k=20,
        )
        if matches:
            label = f"Top matches (from {len(candidate_sets)} candidates)" if candidate_sets else "Top matches (all sets)"
            print(f"{label}:")
            for set_id, dist in matches:
                print(f"  {set_id}: {dist:.6f}")
        else:
            print("No matches (index may not exist)")

    elif cmd == "test":
        _run_test()

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


def _run_test():
    """Test set symbol disambiguation on known binder page cards.

    Simulates the real use case: given ~15 candidate sets from the cascade,
    can the set symbol narrow it down to the correct set?
    """
    import time

    cards_dir = _PROJECT_ROOT / "data" / "inbox" / "page_20260228_202134_cards"
    if not cards_dir.exists():
        print(f"Test cards not found: {cards_dir}")
        return

    # Known card -> expected set + realistic candidate sets
    test_cases = [
        {
            "file": "card_05.png",
            "name": "Raikou",
            "expected_set": "dp3",
            # Realistic candidates from cascade (visually similar Raikou cards)
            "candidate_sets": [
                "dp3", "dp1", "dp2", "pl3", "pl1", "pl2",
                "hgss1", "hgss2", "bw1", "sm1", "sv1",
                "base1", "neo1", "xy1", "ex1",
            ],
        },
        {
            "file": "card_03.png",
            "name": "Venusaur",
            "expected_set": "pl3",
            "candidate_sets": [
                "pl3", "pl1", "pl2", "dp3", "dp1", "dp2",
                "base1", "base2", "hgss1", "bw1", "sm1",
                "sv1", "xy1", "ex1", "neo1",
            ],
        },
    ]

    # Build index if needed
    if not _INDEX_PATH.exists():
        print("Building index first...")
        build_set_symbol_index()

    index = _load_index()
    if index is None:
        print("Failed to load index")
        return

    for case in test_cases:
        card_path = cards_dir / case["file"]
        if not card_path.exists():
            continue

        expected = case["expected_set"]
        candidates = case["candidate_sets"]

        t0 = time.time()
        # Disambiguation mode: only score candidate sets
        matches = match_set_symbol(
            str(card_path),
            preloaded_index=index,
            candidate_sets=candidates,
            top_k=len(candidates),
        )
        elapsed = time.time() - t0

        print(f"\n=== {case['name']} - {case['file']} "
              f"(expected: {expected}) [{elapsed:.3f}s] ===")
        print(f"Scoring {len(candidates)} candidate sets:")
        for i, (set_id, dist) in enumerate(matches):
            marker = " <-- CORRECT" if set_id == expected else ""
            print(f"  #{i + 1:2d} {set_id:8s}: {dist:.6f}{marker}")

        # Also test all-sets mode for comparison
        all_matches = match_set_symbol(
            str(card_path), preloaded_index=index, top_k=10,
        )
        print(f"\nAll-sets mode (top 10 of 167):")
        for i, (set_id, dist) in enumerate(all_matches):
            marker = " <-- CORRECT" if set_id == expected else ""
            print(f"  #{i + 1:2d} {set_id:8s}: {dist:.6f}{marker}")


if __name__ == "__main__":
    _cli()
