"""ML modules for card identification and price prediction.

Cascade card identification pipeline.

Tries identification methods in order of cost/speed:
1. Perceptual hash (free, instant) -- accept if distance < 5
2. DINOv2 + FAISS (free, ~1s) -- accept if similarity > 0.65
2.5. CLIP image-to-image (free, ~2s) -- accept if similarity > 0.75
3. Claude Haiku vision API ($0.0015/card) -- accept if db-match confidence > 0.5
"""

import hashlib
import logging
import os
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

# In-memory LRU cache for identify_card results, keyed by md5 of file contents.
_scan_cache: OrderedDict = OrderedDict()
_SCAN_CACHE_MAX = 100

# Resolve data paths relative to the project root (two levels up from this file).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_HASH_DB_PATH = _PROJECT_ROOT / "data" / "hash_db.pkl"
_DINO_INDEX_PATH = _PROJECT_ROOT / "data" / "dino_index.faiss"
_CLIP_IMAGE_INDEX_PATH = _PROJECT_ROOT / "data" / "clip_image_index.pkl"


def _cache_store(file_hash, result):
    """Store a result in the LRU cache, evicting oldest if over capacity."""
    if file_hash is None:
        return
    _scan_cache[file_hash] = result
    _scan_cache.move_to_end(file_hash)
    while len(_scan_cache) > _SCAN_CACHE_MAX:
        _scan_cache.popitem(last=False)


def identify_card(image_path, session=None):
    """Identify a card using the cascade pipeline.

    Returns dict with keys: card_id, confidence, method, explanation, raw_response.
    Results are cached by md5 of the image file contents (up to 100 entries).
    """
    # Check in-memory cache keyed by file content hash.
    try:
        file_hash = hashlib.md5(Path(image_path).read_bytes()).hexdigest()
        if file_hash in _scan_cache:
            logger.info("Cache HIT for %s (md5=%s)", image_path, file_hash)
            _scan_cache.move_to_end(file_hash)
            return _scan_cache[file_hash]
    except Exception as e:
        logger.warning("Could not hash image file for cache lookup: %s", e)
        file_hash = None

    # Convert HEIC/HEIF if needed; tolerate conversion failures.
    try:
        from cardprice.utils.image_convert import ensure_compatible
        image_path = ensure_compatible(str(image_path))
    except Exception as e:
        logger.warning("Image conversion failed, using original path: %s", e)
        image_path = str(image_path)

    result = {"card_id": None, "confidence": 0.0, "method": None, "explanation": None, "raw_response": {}}

    # Tier 1: Perceptual hash (fastest, cheapest)
    try:
        from cardprice.ml.hash_matcher import match_card, CONFIDENT_THRESHOLD
        if _HASH_DB_PATH.exists():
            logger.info("Tier 1 (hash): searching hash database ...")
            matches = match_card(image_path, str(_HASH_DB_PATH))
            if matches and matches[0][1] < CONFIDENT_THRESHOLD:
                # Hash DB stores card_ids with underscore (filename stem).
                # Convert last '_' to '/' to match dim_cards card_id format.
                raw_cid = matches[0][0]
                last_under = raw_cid.rfind("_")
                card_id = (raw_cid[:last_under] + "/" + raw_cid[last_under + 1:]) if last_under != -1 else raw_cid
                distance = matches[0][1]
                result["card_id"] = card_id
                result["confidence"] = float(max(0.0, 1.0 - distance / 15.0))
                result["method"] = "hash"
                result["explanation"] = f"Exact visual match (perceptual hash distance: {distance})"
                result["raw_response"] = {
                    "matches": [(cid, int(d)) for cid, d in matches[:5]],
                    "top_alternatives": [(cid, int(d)) for cid, d in matches[1:4]] if len(matches) > 1 else []
                }
                logger.info("Tier 1 (hash): MATCH %s (distance=%d)", card_id, distance)
                _cache_store(file_hash, result)
                return result
            elif matches:
                logger.info("Tier 1 (hash): best distance=%d >= threshold %d, falling through",
                            matches[0][1], CONFIDENT_THRESHOLD)
            else:
                logger.info("Tier 1 (hash): no matches within threshold")
        else:
            logger.info("Tier 1 (hash): SKIPPED -- hash DB not found at %s "
                        "(build with: python -m cardprice.cli build-hash-index)", _HASH_DB_PATH)
    except ImportError as e:
        logger.info("Tier 1 (hash): SKIPPED -- missing dependency: %s", e)
    except Exception as e:
        logger.warning("Tier 1 (hash): ERROR -- %s", e)

    # Tier 2: DINOv2 + FAISS (good accuracy, no API cost)
    try:
        from cardprice.ml.dino_matcher import identify_card as dino_identify
        if _DINO_INDEX_PATH.exists():
            logger.info("Tier 2 (dino): searching FAISS index ...")
            matches = dino_identify(image_path)
            if matches:
                # DINOv2 index stores card_ids with set dir prefix: "bw5/bw5-107/normal"
                # Strip the first path segment to get "bw5-107/normal"
                raw_cid = matches[0][0]
                parts = raw_cid.split("/", 1)
                card_id = parts[1] if len(parts) > 1 else raw_cid
                similarity = float(matches[0][1])
                # Phone photos score ~0.4-0.6 against digital refs;
                # digital-to-digital scores ~0.8+. Use 0.65 as threshold.
                if similarity > 0.65:
                    # Build explanation with top alternatives
                    alt_list = []
                    for alt_raw, alt_score in matches[1:4]:
                        alt_parts = alt_raw.split("/", 1)
                        alt_id = alt_parts[1] if len(alt_parts) > 1 else alt_raw
                        alt_list.append((alt_id, float(alt_score)))

                    alt_str = ", ".join(f"{a[0]} ({a[1]:.0%})" for a in alt_list)
                    result["card_id"] = card_id
                    result["confidence"] = similarity
                    result["method"] = "dino"
                    result["explanation"] = f"Visual similarity match ({similarity:.0%}). Top alternatives: {alt_str}" if alt_str else f"Visual similarity match ({similarity:.0%})"
                    result["raw_response"] = {
                        "top_matches": matches[:5],
                        "top_alternatives": alt_list
                    }
                    logger.info("Tier 2 (dino): MATCH %s (similarity=%.4f)", card_id, similarity)
                    if file_hash:
                        _scan_cache[file_hash] = result
                    return result
                else:
                    logger.info("Tier 2 (dino): best similarity=%.4f < threshold 0.65, falling through",
                                similarity)
            else:
                logger.info("Tier 2 (dino): no matches found")
        else:
            logger.info("Tier 2 (dino): SKIPPED -- FAISS index not found at %s "
                        "(build with: python -m cardprice.cli build-dino-index)", _DINO_INDEX_PATH)
    except ImportError as e:
        logger.info("Tier 2 (dino): SKIPPED -- missing dependency: %s", e)
    except Exception as e:
        logger.warning("Tier 2 (dino): ERROR -- %s", e)

    # Tier 2.5: CLIP image-to-image (good on phone photos, no API cost)
    try:
        from cardprice.ml.clip_matcher import identify_card_by_image
        if _CLIP_IMAGE_INDEX_PATH.exists():
            logger.info("Tier 2.5 (clip): searching CLIP image index ...")
            matches = identify_card_by_image(image_path, index_path=str(_CLIP_IMAGE_INDEX_PATH))
            if matches:
                # CLIP image index may store card_ids with set dir prefix like DINOv2:
                # "bw5/bw5-107/normal" -> strip first segment to get "bw5-107/normal"
                raw_cid = matches[0][0]
                parts = raw_cid.split("/", 1)
                card_id = parts[1] if len(parts) > 1 and "/" in parts[1] else raw_cid
                similarity = float(matches[0][1])
                if similarity > 0.75:
                    # Build explanation with top alternatives
                    alt_list = []
                    for alt_raw, alt_score in matches[1:4]:
                        alt_parts = alt_raw.split("/", 1)
                        alt_id = alt_parts[1] if len(alt_parts) > 1 else alt_raw
                        alt_list.append((alt_id, float(alt_score)))

                    alt_str = ", ".join(f"{a[0]} ({a[1]:.0%})" for a in alt_list)
                    result["card_id"] = card_id
                    result["confidence"] = similarity
                    result["method"] = "clip"
                    result["explanation"] = f"Visual similarity match ({similarity:.0%}). Top alternatives: {alt_str}" if alt_str else f"Visual similarity match ({similarity:.0%})"
                    result["raw_response"] = {
                        "top_matches": matches[:5],
                        "top_alternatives": alt_list
                    }
                    logger.info("Tier 2.5 (clip): MATCH %s (similarity=%.4f)", card_id, similarity)
                    if file_hash:
                        _scan_cache[file_hash] = result
                    return result
                else:
                    logger.info("Tier 2.5 (clip): best similarity=%.4f < threshold 0.75, falling through",
                                similarity)
            else:
                logger.info("Tier 2.5 (clip): no matches found")
        else:
            logger.info("Tier 2.5 (clip): SKIPPED -- image index not found at %s "
                        "(build with: python -m cardprice.ml.clip_matcher build_image_index)",
                        _CLIP_IMAGE_INDEX_PATH)
    except ImportError as e:
        logger.info("Tier 2.5 (clip): SKIPPED -- missing dependency: %s", e)
    except Exception as e:
        logger.warning("Tier 2.5 (clip): ERROR -- %s", e)

    # Tier 3: Claude Haiku vision API (highest accuracy, costs money)
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.info("Tier 3 (claude): SKIPPED -- ANTHROPIC_API_KEY not set")
        else:
            from cardprice.ml.claude_scanner import scan_card, match_to_database
            logger.info("Tier 3 (claude): calling Claude Haiku vision API ...")
            scan_result = scan_card(image_path, model="claude-haiku-4-5")
            matched_id, match_conf = match_to_database(scan_result, session)
            if matched_id and match_conf > 0.5:
                result["card_id"] = matched_id
                result["confidence"] = float(match_conf)
                result["method"] = "claude"
                result["explanation"] = "Identified by AI vision"
                result["raw_response"] = {
                    "scan_result": scan_result,
                    "top_alternatives": []
                }
                logger.info("Tier 3 (claude): MATCH %s (confidence=%.2f)", matched_id, match_conf)
                _cache_store(file_hash, result)
                return result
            elif matched_id:
                logger.info("Tier 3 (claude): identified %s but low confidence=%.2f (threshold=0.5)",
                            matched_id, match_conf)
                result["raw_response"] = {
                    "scan_result": scan_result,
                    "top_alternatives": []
                }
            else:
                logger.info("Tier 3 (claude): API responded but no DB match found")
                result["raw_response"] = {
                    "scan_result": scan_result,
                    "top_alternatives": []
                }
    except ImportError as e:
        logger.info("Tier 3 (claude): SKIPPED -- missing dependency: %s", e)
    except Exception as e:
        logger.warning("Tier 3 (claude): ERROR -- %s", e)

    logger.info("No confident match found for %s", image_path)
    if result.get("raw_response") and "top_alternatives" not in result["raw_response"]:
        result["raw_response"]["top_alternatives"] = []
    if result["card_id"] and not result.get("explanation"):
        result["explanation"] = f"No confident match. Best guess: {result['card_id']} at {result['confidence']:.0%}"
    _cache_store(file_hash, result)
    return result
