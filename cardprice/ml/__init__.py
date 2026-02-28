"""ML modules for card identification and price prediction."""

"""Cascade card identification pipeline.

Tries identification methods in order of cost/speed:
1. Perceptual hash (free, instant) — accept if distance < 5
2. DINOv2 + FAISS (free, ~1s) — accept if similarity > 0.95
3. Claude Haiku vision API ($0.0015/card) — accept if confidence > 0.8
"""

import logging
import os
from pathlib import Path

from cardprice.utils.image_convert import ensure_compatible

logger = logging.getLogger(__name__)


def identify_card(image_path, session=None):
    """Identify a card using the cascade pipeline.

    Returns dict with: card_id, confidence, method, raw_response
    """
    image_path = ensure_compatible(str(image_path))
    result = {"card_id": None, "confidence": 0.0, "method": None, "raw_response": {}}

    # Tier 1: Perceptual hash (fastest, cheapest)
    try:
        from cardprice.ml.hash_matcher import match_card, classify_match, CONFIDENT_THRESHOLD
        hash_db_path = Path("data/hash_db.pkl")
        if hash_db_path.exists():
            matches = match_card(image_path, str(hash_db_path))
            if matches and matches[0][1] < CONFIDENT_THRESHOLD:
                result["card_id"] = matches[0][0]
                result["confidence"] = max(0, 1.0 - matches[0][1] / 15.0)
                result["method"] = "hash"
                result["raw_response"] = {"matches": matches[:5]}
                logger.info("Hash match: %s (distance=%d)", matches[0][0], matches[0][1])
                return result
    except Exception as e:
        logger.debug("Hash matching skipped: %s", e)

    # Tier 2: DINOv2 + FAISS (good accuracy, no API cost)
    try:
        from cardprice.ml.dino_matcher import identify_card as dino_identify
        index_path = Path("data/dino_index.faiss")
        if index_path.exists():
            matches = dino_identify(image_path)
            if matches and matches[0][1] > 0.95:
                result["card_id"] = matches[0][0]
                result["confidence"] = matches[0][1]
                result["method"] = "dino"
                result["raw_response"] = {"top_matches": matches[:5]}
                logger.info("DINOv2 match: %s (similarity=%.4f)", matches[0][0], matches[0][1])
                return result
    except Exception as e:
        logger.debug("DINOv2 matching skipped: %s", e)

    # Tier 3: Claude Haiku vision API (highest accuracy, costs money)
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            from cardprice.ml.claude_scanner import scan_card, match_to_database
            scan_result = scan_card(image_path, model="claude-haiku-4-5")
            matched_id, match_conf = match_to_database(scan_result, session)
            if matched_id and match_conf > 0.5:
                result["card_id"] = matched_id
                result["confidence"] = match_conf
                result["method"] = "claude"
                result["raw_response"] = scan_result
                logger.info("Claude match: %s (confidence=%.2f)", matched_id, match_conf)
                return result
    except Exception as e:
        logger.debug("Claude vision skipped: %s", e)

    logger.info("No confident match found for %s", image_path)
    return result
