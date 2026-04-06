"""Text-based card matching using TF-IDF on structured attack/ability text."""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_STRUCTURED_PATH = _PROJECT_ROOT / "data" / "structured_attacks.json"

_tfidf_index = None  # Lazy singleton


def _build_index():
    """Build TF-IDF index from structured attacks data."""
    global _tfidf_index
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not _STRUCTURED_PATH.exists():
        logger.warning("text_matcher: %s not found", _STRUCTURED_PATH)
        _tfidf_index = {"card_ids": [], "matrix": None, "vectorizer": None, "ref_texts": {}}
        return _tfidf_index

    with open(_STRUCTURED_PATH) as f:
        sa = json.load(f)

    card_ids = []
    texts = []
    ref_texts = {}

    for cid, data in sa.items():
        parts = []
        for atk in data.get("attacks", []):
            parts.extend([atk.get("name", ""), atk.get("text", ""), str(atk.get("damage", ""))])
        for ab in data.get("abilities", []):
            parts.extend([ab.get("name", ""), ab.get("text", "")])
        text = " ".join(parts).lower().strip()
        if text:
            card_ids.append(cid)
            texts.append(text)
            ref_texts[cid] = text

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000)
    matrix = vectorizer.fit_transform(texts)

    _tfidf_index = {
        "card_ids": card_ids,
        "matrix": matrix,
        "vectorizer": vectorizer,
        "ref_texts": ref_texts,
    }
    logger.info(
        "text_matcher: built TF-IDF index with %d cards, %d features",
        len(card_ids),
        matrix.shape[1],
    )
    return _tfidf_index


def get_index():
    """Return the TF-IDF index, building it on first call."""
    global _tfidf_index
    if _tfidf_index is None:
        _build_index()
    return _tfidf_index


def search_by_text(query_text: str, top_n: int = 50) -> list[tuple[str, float]]:
    """Search for cards by free-text query using TF-IDF cosine similarity.

    Args:
        query_text: OCR-extracted text (attack names, descriptions, damage values).
        top_n: Number of top results to return.

    Returns:
        List of (card_id, score) tuples sorted by descending similarity.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    idx = get_index()
    if idx["matrix"] is None or len(idx["card_ids"]) == 0:
        return []

    query_vec = idx["vectorizer"].transform([query_text.lower().strip()])
    scores = cosine_similarity(query_vec, idx["matrix"]).flatten()

    top_indices = np.argsort(scores)[::-1][:top_n]
    results = []
    for i in top_indices:
        if scores[i] > 0:
            results.append((idx["card_ids"][i], float(scores[i])))

    return results


def text_similarity_score(query_text: str, card_id: str) -> float:
    """Compute text similarity between query text and a specific card.

    Uses rapidfuzz token_set_ratio for fuzzy OCR-tolerant matching against the
    card's reference text, normalized to 0.0-1.0.

    Args:
        query_text: OCR-extracted text from the card.
        card_id: The card_id to compare against.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    from rapidfuzz import fuzz

    idx = get_index()
    ref = idx["ref_texts"].get(card_id)
    if ref is None:
        return 0.0

    score = fuzz.token_set_ratio(query_text.lower().strip(), ref) / 100.0
    return score
