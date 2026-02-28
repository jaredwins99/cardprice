"""Fast card identification using perceptual image hashing.

Computes multiple perceptual hashes (pHash, dHash, aHash, wHash) for card
images and matches unknown cards against a prebuilt hash database using
Hamming distance.

Distance thresholds (pHash):
    <5  = confident match
    <10 = likely match
    <15 = possible match
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_HASH_DB_PATH = "data/hash_db.pkl"

# Confidence tiers based on Hamming distance
CONFIDENT_THRESHOLD = 5
LIKELY_THRESHOLD = 10
POSSIBLE_THRESHOLD = 15


def compute_hashes(image_path: str | Path) -> dict[str, imagehash.ImageHash]:
    """Compute pHash, dHash, aHash, and wHash for a card image.

    Parameters
    ----------
    image_path : str or Path
        Path to the card image file.

    Returns
    -------
    dict
        Keys: "phash", "dhash", "ahash", "whash".
        Values: corresponding ``imagehash.ImageHash`` objects.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    PIL.UnidentifiedImageError
        If the file cannot be opened as an image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)

    return {
        "phash": imagehash.phash(img),
        "dhash": imagehash.dhash(img),
        "ahash": imagehash.average_hash(img),
        "whash": imagehash.whash(img),
    }


def build_hash_database(
    image_dir: str | Path,
    output_path: str | Path = DEFAULT_HASH_DB_PATH,
) -> dict[str, dict[str, imagehash.ImageHash]]:
    """Process all card images in a directory and build a hash database.

    Card IDs are derived from filenames by stripping the extension, so a file
    named ``base1-4_holofoil.png`` produces card_id ``base1-4_holofoil``.

    Parameters
    ----------
    image_dir : str or Path
        Directory containing card images (png, jpg, jpeg, webp).
    output_path : str or Path
        Where to serialize the hash database (pickle).

    Returns
    -------
    dict
        Mapping ``{card_id: {"phash": …, "dhash": …, "ahash": …, "whash": …}}``.
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path)

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in valid_extensions
    )

    if not image_files:
        logger.warning("No images found in %s", image_dir)
        return {}

    logger.info("Building hash database from %d images in %s", len(image_files), image_dir)

    hash_db: dict[str, dict[str, imagehash.ImageHash]] = {}
    errors = 0

    for i, img_path in enumerate(image_files, 1):
        card_id = img_path.stem
        try:
            hash_db[card_id] = compute_hashes(img_path)
        except Exception:
            logger.warning("Failed to hash %s", img_path, exc_info=True)
            errors += 1
            continue

        if i % 500 == 0:
            logger.info("Hashed %d / %d images", i, len(image_files))

    logger.info(
        "Hash database complete: %d cards, %d errors", len(hash_db), errors
    )

    # Serialize
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(hash_db, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Hash database saved to %s", output_path)

    return hash_db


def _load_hash_db(
    hash_db_path: str | Path,
) -> dict[str, dict[str, imagehash.ImageHash]]:
    """Load a pickled hash database from disk."""
    hash_db_path = Path(hash_db_path)
    if not hash_db_path.exists():
        raise FileNotFoundError(
            f"Hash database not found: {hash_db_path}. "
            "Run build_hash_database() first."
        )
    with open(hash_db_path, "rb") as f:
        return pickle.load(f)


def match_card(
    image_path: str | Path,
    hash_db_path: str | Path = DEFAULT_HASH_DB_PATH,
    threshold: int = POSSIBLE_THRESHOLD,
    *,
    hash_db: dict[str, dict[str, imagehash.ImageHash]] | None = None,
) -> list[tuple[str, int]]:
    """Find the closest card(s) to an input image by perceptual hash distance.

    Matching is performed on pHash (perceptual hash), which is the most
    robust to scaling and minor colour shifts.  Additional hashes are stored
    for potential multi-hash voting in the future.

    Parameters
    ----------
    image_path : str or Path
        Path to the query image.
    hash_db_path : str or Path
        Path to the pickled hash database.
    threshold : int
        Maximum Hamming distance to include in results (default 15).
    hash_db : dict, optional
        Pre-loaded hash database.  When supplied, *hash_db_path* is ignored.
        Useful for batch operations to avoid repeated disk reads.

    Returns
    -------
    list of (card_id, distance)
        Matches sorted by ascending Hamming distance.  Distance < 5 is a
        confident match, < 10 is likely, < 15 is possible.
    """
    if hash_db is None:
        hash_db = _load_hash_db(hash_db_path)

    query_hashes = compute_hashes(image_path)
    query_phash = query_hashes["phash"]

    matches: list[tuple[str, int]] = []
    for card_id, stored_hashes in hash_db.items():
        distance = query_phash - stored_hashes["phash"]
        if distance <= threshold:
            matches.append((card_id, distance))

    matches.sort(key=lambda m: m[1])
    return matches


def classify_match(distance: int) -> str:
    """Return a human-readable confidence label for a Hamming distance.

    Returns
    -------
    str
        One of "confident", "likely", "possible", or "no_match".
    """
    if distance < CONFIDENT_THRESHOLD:
        return "confident"
    if distance < LIKELY_THRESHOLD:
        return "likely"
    if distance < POSSIBLE_THRESHOLD:
        return "possible"
    return "no_match"


def batch_match(
    image_dir: str | Path,
    hash_db_path: str | Path = DEFAULT_HASH_DB_PATH,
    threshold: int = POSSIBLE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Match all images in a directory against the hash database.

    Parameters
    ----------
    image_dir : str or Path
        Directory containing query images.
    hash_db_path : str or Path
        Path to the pickled hash database.
    threshold : int
        Maximum Hamming distance for matches.

    Returns
    -------
    list of dict
        Each dict contains:
        - ``"query_image"``: filename of the query image
        - ``"matches"``: list of ``(card_id, distance)`` tuples
        - ``"best_match"``: card_id of the top match, or ``None``
        - ``"best_distance"``: distance of the top match, or ``None``
        - ``"confidence"``: one of "confident", "likely", "possible", "no_match"
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in valid_extensions
    )

    if not image_files:
        logger.warning("No images found in %s", image_dir)
        return []

    # Load hash DB once for all queries
    hash_db = _load_hash_db(hash_db_path)
    logger.info(
        "Batch matching %d images against %d-card hash database",
        len(image_files),
        len(hash_db),
    )

    results: list[dict[str, Any]] = []

    for i, img_path in enumerate(image_files, 1):
        try:
            matches = match_card(
                img_path, threshold=threshold, hash_db=hash_db
            )
        except Exception:
            logger.warning("Failed to match %s", img_path, exc_info=True)
            results.append({
                "query_image": img_path.name,
                "matches": [],
                "best_match": None,
                "best_distance": None,
                "confidence": "no_match",
            })
            continue

        best_id = matches[0][0] if matches else None
        best_dist = matches[0][1] if matches else None
        confidence = classify_match(best_dist) if best_dist is not None else "no_match"

        results.append({
            "query_image": img_path.name,
            "matches": matches,
            "best_match": best_id,
            "best_distance": best_dist,
            "confidence": confidence,
        })

        if i % 100 == 0:
            logger.info("Matched %d / %d images", i, len(image_files))

    # Summary
    confident = sum(1 for r in results if r["confidence"] == "confident")
    likely = sum(1 for r in results if r["confidence"] == "likely")
    possible = sum(1 for r in results if r["confidence"] == "possible")
    no_match = sum(1 for r in results if r["confidence"] == "no_match")
    logger.info(
        "Batch complete: %d confident, %d likely, %d possible, %d no_match",
        confident, likely, possible, no_match,
    )

    return results
