"""Reference-image matching: narrow candidates via attributes, then compare embeddings.

Pipeline:
1. Cheap classifiers (Claude vision / OCR) identify pokemon name, HP, type.
2. DB query finds all cards matching those attributes -> typically 2-20 candidates.
3. Load reference images for each candidate from data/card_images/{set_id}/{card_id}_normal.png.
4. Compute DINOv2 embedding similarity between query image and each candidate reference.
5. Best match wins.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Default reference image root
_REF_IMAGE_DIR = Path("data/card_images")

# Pre-computed reference embeddings
_REF_EMBEDDINGS_PATH = Path("data/ref_embeddings.pkl")
_ref_embeddings: Optional[dict[str, np.ndarray]] = None
_ref_embeddings_lock = __import__("threading").Lock()

# JSON fallback for card names when DB is unavailable
_CARD_NAMES_JSON = Path(__file__).resolve().parent.parent.parent / "data" / "card_names.json"
_card_names_fallback: Optional[list[tuple[str, str, str]]] = None


def _get_session():
    """Get a DB session, or None if DB is unavailable."""
    try:
        from cardprice.db.session import SessionLocal
        return SessionLocal()
    except Exception:
        return None


def _load_card_names_fallback() -> list[tuple]:
    """Load card names from JSON fallback.

    Returns list of (card_id, name, set_id, hp_str, types_list).
    """
    global _card_names_fallback
    if _card_names_fallback is not None:
        return _card_names_fallback
    import json
    if _CARD_NAMES_JSON.exists():
        with open(_CARD_NAMES_JSON) as f:
            entries = json.load(f)
        # Format: [card_id, name, set_id, hp, types]
        _card_names_fallback = [tuple(e) for e in entries]
        logger.info("Loaded %d card names from JSON fallback", len(_card_names_fallback))
    else:
        _card_names_fallback = []
        logger.error("No card_names.json fallback found at %s", _CARD_NAMES_JSON)
    return _card_names_fallback


def _load_ref_embeddings() -> dict[str, np.ndarray]:
    """Lazy-load pre-computed DINOv2 reference embeddings from pickle.

    Returns an empty dict if the pickle file does not exist, allowing
    graceful fallback to on-the-fly computation.
    """
    global _ref_embeddings
    if _ref_embeddings is not None:
        return _ref_embeddings

    with _ref_embeddings_lock:
        # Double-check after acquiring lock
        if _ref_embeddings is not None:
            return _ref_embeddings

        if not _REF_EMBEDDINGS_PATH.is_file():
            logger.warning(
                "Pre-computed embeddings not found at %s — will compute on the fly. "
                "Run 'python scripts/build_ref_embeddings.py' to pre-compute.",
                _REF_EMBEDDINGS_PATH,
            )
            _ref_embeddings = {}
            return _ref_embeddings

        logger.info("Loading pre-computed reference embeddings from %s ...", _REF_EMBEDDINGS_PATH)
        with open(_REF_EMBEDDINGS_PATH, "rb") as f:
            _ref_embeddings = pickle.load(f)
        logger.info("Loaded %d pre-computed embeddings.", len(_ref_embeddings))
        return _ref_embeddings


# ---------------------------------------------------------------------------
# Candidate lookup
# ---------------------------------------------------------------------------

def get_candidate_card_ids(
    pokemon_name: str,
    hp: Optional[int] = None,
    card_type: Optional[str] = None,
    session=None,
) -> list[str]:
    """Query dim_cards for card_ids matching the given attributes.

    Parameters
    ----------
    pokemon_name : str
        Pokemon name to search for (case-insensitive ILIKE match).
    hp : int, optional
        HP value to filter on.  Exact match when provided.
    card_type : str, optional
        Pokemon type (e.g. "Fire", "Water").  Matched against dim_pokemon.types
        via the pokemon_id foreign key.
    session : sqlalchemy Session, optional
        Existing DB session.  One is created (and closed) if not provided.

    Returns
    -------
    list[str]
        Card IDs matching the criteria, e.g. ["base1-4/normal", "base1-4/holofoil"].
    """
    # Try DB first
    db_ok = False
    own_session = session is None
    if own_session:
        session = _get_session()

    if session is not None:
        try:
            conditions = [
                "(LOWER(c.name) = LOWER(:name)"
                " OR LOWER(c.name) LIKE LOWER(:name_space)"
                " OR LOWER(c.name) LIKE LOWER(:name_dash))"
            ]
            params: dict = {
                "name": pokemon_name,
                "name_space": pokemon_name + " %",
                "name_dash": pokemon_name + "-%",
            }

            if hp is not None:
                conditions.append("c.hp = :hp")
                params["hp"] = hp

            where_clause = " AND ".join(conditions)

            if card_type:
                query_str = f"""
                    SELECT c.card_id
                    FROM dim_cards c
                    JOIN dim_pokemon p ON c.pokemon_id = p.pokemon_id
                    WHERE {where_clause}
                      AND :card_type = ANY(p.types)
                    ORDER BY c.card_id
                """
                params["card_type"] = card_type
            else:
                query_str = f"""
                    SELECT c.card_id
                    FROM dim_cards c
                    WHERE {where_clause}
                    ORDER BY c.card_id
                """

            rows = session.execute(text(query_str), params).fetchall()
            card_ids = [row[0] for row in rows]
            db_ok = True
            logger.info(
                "get_candidate_card_ids(name=%r, hp=%s, type=%s) -> %d candidates",
                pokemon_name, hp, card_type, len(card_ids),
            )
            return card_ids
        except Exception as e:
            logger.warning("DB query failed: %s — using JSON fallback", e)
        finally:
            if own_session and session is not None:
                try:
                    session.close()
                except Exception:
                    pass

    # JSON fallback with HP/type filtering
    if not db_ok:
        entries = _load_card_names_fallback()
        name_lower = pokemon_name.lower()
        card_ids = []
        for entry in entries:
            card_id, name = entry[0], entry[1]
            entry_hp = entry[3] if len(entry) > 3 else None
            entry_types = entry[4] if len(entry) > 4 else []
            nl = name.lower()
            if not (nl == name_lower or nl.startswith(name_lower + " ") or nl.startswith(name_lower + "-")):
                continue
            # HP filter
            if hp is not None and entry_hp:
                try:
                    if int(entry_hp) != hp:
                        continue
                except (ValueError, TypeError):
                    pass
            # Type filter
            if card_type and entry_types:
                if card_type not in entry_types:
                    continue
            card_ids.append(card_id)
        logger.info(
            "get_candidate_card_ids(name=%r, hp=%s, type=%s) -> %d candidates (JSON fallback)",
            pokemon_name, hp, card_type, len(card_ids),
        )
        return card_ids


# ---------------------------------------------------------------------------
# Reference image path resolution
# ---------------------------------------------------------------------------

def get_reference_image_path(
    card_id: str,
    image_dir: Path | str = _REF_IMAGE_DIR,
) -> Optional[Path]:
    """Map a card_id to its reference image file on disk.

    Card IDs look like "base1-4/normal".  The reference image lives at:
        data/card_images/base1/base1-4_normal.png

    The set_id is everything before the first hyphen-digit boundary in the
    card portion (before the '/').  In practice, set_id == card_id.split('/')[0]
    split at the last '-' that precedes the card number, but simpler: the
    set_id is stored in dim_cards and is the directory name.  We derive it
    from the card_id: everything before the first '-' followed by a digit,
    but to keep it simple we just use the portion before the '/'.

    Actual convention: card_id "base1-4/normal"
        -> set_id from card_id portion before '/': "base1-4"
        -> but the set_id directory is "base1" (the set), not "base1-4"
        -> We need to extract set_id: split the card portion on '-' and
           reassemble all parts except the last (which is the card number).

    Examples:
        "base1-4/normal"      -> set_id="base1",  file="base1-4_normal.png"
        "sv8-162/normal"      -> set_id="sv8",     file="sv8-162_normal.png"
        "swsh12pt5-160/normal" -> set_id="swsh12pt5", file="swsh12pt5-160_normal.png"

    Parameters
    ----------
    card_id : str
        Card ID in the format "{set_and_number}/{variant}".
    image_dir : Path or str
        Root directory for reference images (default: data/card_images).

    Returns
    -------
    Path or None
        Path to the reference image, or None if the file does not exist.
    """
    image_dir = Path(image_dir)

    # Split card_id into card portion and variant
    if "/" not in card_id:
        logger.warning("card_id %r has no variant separator '/'", card_id)
        return None

    card_portion, variant = card_id.split("/", 1)

    # Extract set_id: everything up to the last '-' in the card portion
    # e.g. "base1-4" -> "base1", "swsh12pt5-160" -> "swsh12pt5"
    last_dash = card_portion.rfind("-")
    if last_dash == -1:
        set_id = card_portion
    else:
        set_id = card_portion[:last_dash]

    # Build filename: "{card_portion}_{variant}.png"
    filename = f"{card_portion}_{variant}.png"
    ref_path = image_dir / set_id / filename

    if ref_path.is_file():
        return ref_path

    # Try .jpg as fallback
    for ext in (".jpg", ".jpeg", ".webp"):
        alt = ref_path.with_suffix(ext)
        if alt.is_file():
            return alt

    logger.debug("Reference image not found: %s", ref_path)
    return None


# ---------------------------------------------------------------------------
# Embedding similarity computation
# ---------------------------------------------------------------------------

def compute_embedding_similarity(
    query_path: str | Path,
    ref_paths: list[Path],
    ref_card_ids: Optional[list[str]] = None,
    query_embedding: Optional[np.ndarray] = None,
) -> list[float]:
    """Compute DINOv2 cosine similarity between a query image and reference images.

    Uses the same DINOv2 model and embedding extraction as dino_matcher.
    When pre-computed embeddings are available (via data/ref_embeddings.pkl),
    reference embeddings are looked up instead of recomputed, which is
    dramatically faster.

    Parameters
    ----------
    query_path : str or Path
        Path to the query (camera photo) image.
    ref_paths : list[Path]
        Paths to reference card images.
    ref_card_ids : list[str], optional
        Card IDs corresponding to each ref_path, used to look up pre-computed
        embeddings.  If not provided, embeddings are always computed on the fly.
    query_embedding : np.ndarray, optional
        Pre-computed 768-dim L2-normalized DINOv2 embedding for the query image.
        When provided, skips GPU extraction entirely (used for batch processing).

    Returns
    -------
    list[float]
        Cosine similarity scores, one per reference image.  Values in [-1, 1].
    """
    from cardprice.ml.dino_matcher import extract_embedding

    if not ref_paths:
        return []

    # Use pre-computed embedding or extract on demand
    if query_embedding is not None:
        query_emb = query_embedding
    else:
        query_emb = extract_embedding(query_path)  # (768,), L2-normalized

    # Try to use pre-computed reference embeddings
    precomputed = _load_ref_embeddings()

    similarities = []
    for i, ref_path in enumerate(ref_paths):
        try:
            # Look up pre-computed embedding by card_id if available
            ref_emb = None
            if ref_card_ids and i < len(ref_card_ids):
                card_id = ref_card_ids[i]
                ref_emb = precomputed.get(card_id)
                if ref_emb is not None:
                    logger.debug("Using pre-computed embedding for %s", card_id)

            # Fall back to on-the-fly computation
            if ref_emb is None:
                ref_emb = extract_embedding(ref_path)  # (768,), L2-normalized

            # Cosine similarity = dot product of L2-normalized vectors
            sim = float(np.dot(query_emb, ref_emb))
            similarities.append(sim)
        except Exception:
            logger.warning("Failed to extract embedding for %s", ref_path, exc_info=True)
            similarities.append(-1.0)

    return similarities


# ---------------------------------------------------------------------------
# Main matching function
# ---------------------------------------------------------------------------

def match_by_reference(
    query_image_path: str | Path,
    pokemon_name: Optional[str] = None,
    hp: Optional[int] = None,
    card_type: Optional[str] = None,
    session=None,
) -> tuple[Optional[str], float]:
    """Match a query image against reference images for candidate cards.

    This is the main entry point.  Given a camera photo and attribute hints
    (from OCR or Claude vision), narrow the search space via DB query, then
    do direct DINOv2 embedding comparison against each candidate's reference
    image.

    Parameters
    ----------
    query_image_path : str or Path
        Path to the camera photo of the card.
    pokemon_name : str, optional
        Pokemon name identified by OCR/vision.  Required for candidate lookup.
    hp : int, optional
        HP value identified from the card.
    card_type : str, optional
        Pokemon type (e.g. "Fire").
    session : sqlalchemy Session, optional
        Existing DB session.

    Returns
    -------
    tuple[str | None, float]
        (card_id, confidence) where card_id is the best match or None if no
        candidates were found, and confidence is the cosine similarity score.
    """
    if pokemon_name is None:
        logger.warning("match_by_reference called without pokemon_name; cannot query candidates")
        return None, 0.0

    # Step 1: Get candidate card_ids from DB
    candidates = get_candidate_card_ids(
        pokemon_name=pokemon_name,
        hp=hp,
        card_type=card_type,
        session=session,
    )

    if not candidates:
        logger.info("No candidates found for name=%r hp=%s type=%s", pokemon_name, hp, card_type)
        return None, 0.0

    # Step 2: Resolve reference image paths
    ref_paths: list[Path] = []
    ref_card_ids: list[str] = []
    for cid in candidates:
        path = get_reference_image_path(cid)
        if path is not None:
            ref_paths.append(path)
            ref_card_ids.append(cid)

    if not ref_paths:
        logger.info(
            "No reference images found for %d candidates (name=%r)",
            len(candidates), pokemon_name,
        )
        return None, 0.0

    logger.info(
        "Comparing query against %d reference images (of %d candidates) for %r",
        len(ref_paths), len(candidates), pokemon_name,
    )

    # Step 3: Compute similarities (pass card_ids for pre-computed embedding lookup)
    similarities = compute_embedding_similarity(query_image_path, ref_paths, ref_card_ids)

    # Step 4: Find best match
    best_idx = int(np.argmax(similarities))
    best_card_id = ref_card_ids[best_idx]
    best_score = similarities[best_idx]

    logger.info(
        "Best reference match: %s (similarity=%.4f) out of %d candidates",
        best_card_id, best_score, len(ref_paths),
    )

    # Log top-3 for debugging
    ranked = sorted(
        zip(ref_card_ids, similarities), key=lambda x: x[1], reverse=True
    )
    for i, (cid, sim) in enumerate(ranked[:3]):
        logger.debug("  #%d: %s (%.4f)", i + 1, cid, sim)

    return best_card_id, best_score
