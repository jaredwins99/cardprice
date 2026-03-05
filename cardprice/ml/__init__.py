"""ML modules for card identification and price prediction.

Cascade card identification pipeline.

Tries identification methods in order of cost/speed:
1. Perceptual hash (free, instant) -- accept if distance < 5
2. DINOv2 + FAISS (free, ~1s) -- accept if similarity > 0.65
2.5. CLIP image-to-image (free, ~2s) -- accept if similarity > 0.75
2.7. OCR name reading (free, ~1s) -- accept if fuzzy score >= 90 and confidence > 0.70
2.8. DP-era level detection (free, ~1s) -- OCR "LV.XX" + level map matching
3. Claude Haiku vision API ($0.0015/card) -- accept if db-match confidence > 0.5
"""

import hashlib
import logging
import os
import pickle
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

# Thread lock for OCR engines (PaddleOCR/EasyOCR are not thread-safe)
_ocr_lock = threading.Lock()
_jp_easyocr_reader = None  # Cached Japanese EasyOCR reader (slow to init)

# In-memory LRU cache for identify_card results, keyed by md5 of file contents.
_scan_cache: OrderedDict = OrderedDict()
_SCAN_CACHE_MAX = 100
_ROBUST_CONFIDENCE_THRESHOLD = 0.65

# Resolve data paths relative to the project root (two levels up from this file).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_HASH_DB_PATH = _PROJECT_ROOT / "data" / "hash_db.pkl"
_DINO_INDEX_PATH = _PROJECT_ROOT / "data" / "dino_index.faiss"
_DINO_CARD_IDS_PATH = _PROJECT_ROOT / "data" / "dino_card_ids.pkl"
_CLIP_IMAGE_INDEX_PATH = _PROJECT_ROOT / "data" / "clip_image_index.pkl"
_CLIP_AUGMENTED_INDEX_PATH = _PROJECT_ROOT / "data" / "clip_augmented_index.pkl"

# ---------------------------------------------------------------------------
# Lazy-loaded singletons for heavy ML resources
# ---------------------------------------------------------------------------
_hash_db = None
_dino_faiss_index = None
_dino_card_ids = None
_clip_image_index = None


def _get_hash_db():
    """Lazy-load and cache the perceptual hash database."""
    global _hash_db
    if _hash_db is None:
        if not _HASH_DB_PATH.exists():
            return None
        logger.info("Loading hash DB from %s ...", _HASH_DB_PATH)
        with open(_HASH_DB_PATH, "rb") as f:
            _hash_db = pickle.load(f)
        logger.info("Hash DB loaded (%d entries).", len(_hash_db))
    return _hash_db


def _get_dino_index():
    """Lazy-load and cache the FAISS index and card-ID mapping for DINOv2."""
    global _dino_faiss_index, _dino_card_ids
    if _dino_faiss_index is None:
        if not _DINO_INDEX_PATH.exists() or not _DINO_CARD_IDS_PATH.exists():
            return None, None
        import faiss
        logger.info("Loading FAISS index from %s ...", _DINO_INDEX_PATH)
        _dino_faiss_index = faiss.read_index(str(_DINO_INDEX_PATH))
        with open(_DINO_CARD_IDS_PATH, "rb") as f:
            _dino_card_ids = pickle.load(f)
        logger.info("FAISS index loaded (%d vectors).", _dino_faiss_index.ntotal)
    return _dino_faiss_index, _dino_card_ids


def _get_clip_image_index():
    """Lazy-load and cache the CLIP image embedding index.

    Prefers the augmented index (clip_augmented_index.pkl) if it exists,
    as it bridges the domain gap between clean digital reference images
    and phone photos. Falls back to the standard clip_image_index.pkl.
    """
    global _clip_image_index
    if _clip_image_index is None:
        # Prefer augmented index (better for phone photos of binder pages)
        if _CLIP_AUGMENTED_INDEX_PATH.exists():
            idx_path = _CLIP_AUGMENTED_INDEX_PATH
            label = "augmented CLIP image index"
        elif _CLIP_IMAGE_INDEX_PATH.exists():
            idx_path = _CLIP_IMAGE_INDEX_PATH
            label = "CLIP image index"
        else:
            return None
        logger.info("Loading %s from %s ...", label, idx_path)
        with open(idx_path, "rb") as f:
            _clip_image_index = pickle.load(f)
        logger.info("%s loaded (%d entries).", label.capitalize(),
                    len(_clip_image_index["card_ids"]))
    return _clip_image_index


def _cache_store(file_hash, result):
    """Store a result in the LRU cache, evicting oldest if over capacity."""
    if file_hash is None:
        return
    _scan_cache[file_hash] = result
    _scan_cache.move_to_end(file_hash)
    while len(_scan_cache) > _SCAN_CACHE_MAX:
        _scan_cache.popitem(last=False)


def identify_card(image_path, session=None, page_context=None):
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
        hash_db = _get_hash_db()
        if hash_db is not None:
            logger.info("Tier 1 (hash): searching hash database ...")
            matches = match_card(image_path, str(_HASH_DB_PATH), hash_db=hash_db)
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
                        "(build with: python -m cardprice.cli build-hash-index)",
                        _HASH_DB_PATH)
    except ImportError as e:
        logger.info("Tier 1 (hash): SKIPPED -- missing dependency: %s", e)
    except Exception as e:
        logger.warning("Tier 1 (hash): ERROR -- %s", e)

    # Tier 2: DINOv2 + FAISS (good accuracy, no API cost)
    _preproc_tmp = None
    try:
        from cardprice.ml.dino_matcher import identify_card as dino_identify
        dino_idx, dino_cids = _get_dino_index()
        if dino_idx is not None:
            # Preprocess for DINOv2: CLAHE + glare removal + border crop
            # improves scores by ~+0.05 on phone photos with sleeves.
            dino_query_path = image_path
            try:
                from cardprice.ml.preprocess import preprocess_for_matching
                _preproc_tmp = preprocess_for_matching(image_path)
                dino_query_path = _preproc_tmp
                logger.info("Tier 2 (dino): using preprocessed image")
            except Exception as e:
                logger.debug("Tier 2 (dino): preprocessing skipped: %s", e)

            logger.info("Tier 2 (dino): searching FAISS index ...")
            matches = dino_identify(dino_query_path, faiss_index=dino_idx, card_ids_list=dino_cids)
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
                        "(build with: python -m cardprice.cli build-dino-index)",
                        _DINO_INDEX_PATH)
    except ImportError as e:
        logger.info("Tier 2 (dino): SKIPPED -- missing dependency: %s", e)
    except Exception as e:
        logger.warning("Tier 2 (dino): ERROR -- %s", e)
    finally:
        if _preproc_tmp:
            try:
                os.unlink(_preproc_tmp)
            except OSError:
                pass

    # Tier 2.5: CLIP image-to-image (good on phone photos, no API cost)
    try:
        from cardprice.ml.clip_matcher import identify_card_by_image
        clip_idx = _get_clip_image_index()
        if clip_idx is not None:
            logger.info("Tier 2.5 (clip): searching CLIP image index ...")
            matches = identify_card_by_image(image_path, preloaded_index=clip_idx)
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

    # Tier 2.7: OCR card name reading (free, fast, complements visual matchers)
    try:
        from cardprice.ml.ocr_matcher import identify_card_by_ocr
        logger.info("Tier 2.7 (ocr): reading card name via OCR ...")
        ocr_matches = identify_card_by_ocr(image_path, top_k=5, page_context=page_context)
        if ocr_matches:
            best_id, best_conf, best_details = ocr_matches[0]
            fuzzy_score = best_details["fuzzy_score"]
            # Accept if fuzzy score >= 90 (strong name match) and overall conf > 0.70
            if fuzzy_score >= 90 and best_conf > 0.70:
                # Build alternatives from remaining matches
                alt_list = [
                    (m[0], m[2]["matched_name"], m[1])
                    for m in ocr_matches[1:4]
                ]
                alt_str = ", ".join(
                    f"{a[1]} ({a[2]:.0%})" for a in alt_list
                )
                result["card_id"] = best_id
                result["confidence"] = best_conf
                result["method"] = "ocr"
                result["explanation"] = (
                    f"Card name read via OCR: {best_details['ocr_cleaned']!r} "
                    f"-> {best_details['matched_name']} "
                    f"(fuzzy={fuzzy_score:.0f})"
                )
                if alt_str:
                    result["explanation"] += f". Alternatives: {alt_str}"
                result["raw_response"] = {
                    "ocr_raw": best_details["ocr_raw"],
                    "ocr_cleaned": best_details["ocr_cleaned"],
                    "matched_name": best_details["matched_name"],
                    "fuzzy_score": fuzzy_score,
                    "ocr_confidence": best_details["ocr_confidence"],
                    "top_alternatives": alt_list,
                }
                logger.info(
                    "Tier 2.7 (ocr): MATCH %s (name=%s, fuzzy=%d, conf=%.4f)",
                    best_id, best_details["matched_name"], fuzzy_score, best_conf,
                )
                _cache_store(file_hash, result)
                return result
            else:
                logger.info(
                    "Tier 2.7 (ocr): best=%s fuzzy=%d conf=%.2f "
                    "(need fuzzy>=90 and conf>0.70), falling through",
                    best_details["matched_name"], fuzzy_score, best_conf,
                )
        else:
            logger.info("Tier 2.7 (ocr): no matches found")
    except ImportError as e:
        logger.info("Tier 2.7 (ocr): SKIPPED -- missing dependency: %s", e)
    except Exception as e:
        logger.warning("Tier 2.7 (ocr): ERROR -- %s", e)

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


def identify_page(card_image_paths, session=None):
    """Identify all cards on a binder page with context-aware two-pass strategy.

    Pass 1: Run identify_card on all cards to get initial results.
    Pass 2: Build page context from high-confidence results, then re-run
            low-confidence cards with page_context for set disambiguation.

    Args:
        card_image_paths: List of paths to individual card segment images.
        session: Optional DB session.

    Returns:
        List of result dicts (same format as identify_card), one per card.
    """
    from cardprice.ml.page_context import identify_page_context

    RERUN_THRESHOLD = 0.70  # re-run cards below this with context

    # Pass 1: run hybrid (multisignal when strong, ensemble fallback) on all cards
    results = []
    for path in card_image_paths:
        r = identify_card_hybrid(str(path), session=session)
        results.append(r)

    # Build page context from confident results
    ctx = identify_page_context(results)
    logger.info("Page context: sets=%s era=%s confidence=%.2f",
                ctx.get("likely_sets", [])[:3], ctx.get("era"), ctx.get("confidence", 0))

    if not ctx.get("likely_sets") or ctx["confidence"] < 0.3:
        return results  # not enough context to help

    # Guard: only apply page context when the page is coherent (single set/era).
    # On mixed-era pages (e.g. EX + DP + Platinum), context hurts by steering
    # cards toward whatever wrong era dominates the initial guesses.
    skip_pass2 = ctx["confidence"] < 0.65
    if skip_pass2:
        logger.info("Page context too weak (%.2f < 0.65), skipping pass 2", ctx["confidence"])

    # Pass 2: re-run cards with ensemble + leave-one-out page context
    # Re-run when: low confidence, OR methods disagreed (non-"agree" result)
    if not skip_pass2:
        for i, (path, result) in enumerate(zip(card_image_paths, results)):
            needs_rerun = (
                result["confidence"] < RERUN_THRESHOLD
                or "(agree)" not in (result.get("method") or "")
            )
            if needs_rerun:
                # Build leave-one-out context (exclude current card to avoid self-reinforcing errors)
                loo_results = results[:i] + results[i+1:]
                loo_ctx = identify_page_context(loo_results)
                if not loo_ctx.get("likely_sets") or loo_ctx["confidence"] < 0.3:
                    continue

                logger.info("Pass 2: re-running card %d (%s, conf=%.2f, method=%s) with LOO context (sets=%s)",
                            i, result.get("card_id"), result["confidence"], result.get("method"),
                            loo_ctx["likely_sets"][:3])
                new_result = identify_card_hybrid(str(path), session=session, page_context=loo_ctx)
                # Accept context result if:
                # - It changed the card to one matching the page context, OR
                # - It has higher confidence
                old_set = (result.get("card_id") or "").split("-")[0].split("/")[0]
                new_set = (new_result.get("card_id") or "").split("-")[0].split("/")[0]
                ctx_sets = set(loo_ctx.get("likely_sets", []))
                context_match = new_set in ctx_sets and old_set not in ctx_sets
                if context_match or new_result["confidence"] > result["confidence"]:
                    new_result["explanation"] = (new_result.get("explanation") or "") + " (with page context)"
                    results[i] = new_result

    # Pass 3: Claude vision verification/override
    # Claude identifies the Pokemon name, HP, era with near-perfect accuracy.
    # We combine that with the ML pipeline's visual candidates:
    #   - If ML already has the right Pokemon, keep it (agreement)
    #   - If Claude disagrees on the Pokemon name, search DB for Claude's
    #     name+HP and cross-reference with ML's visual candidate pool
    #   - Fall back to pure DB lookup if no ML candidates match
    use_vision = os.environ.get("CARDPRICE_VISION", "1") != "0"
    if use_vision:
        try:
            from cardprice.ml.claude_vision import (
                identify_cards_vision_parallel,
                match_vision_to_db,
            )
            from sqlalchemy import text as sa_text

            logger.info("Pass 3: Running Claude vision on %d cards ...",
                        len(card_image_paths))
            vision_results = identify_cards_vision_parallel(
                card_image_paths, model="sonnet", max_workers=4,
            )

            # Need a DB session for candidate lookups
            from cardprice.db.session import SessionLocal
            own_sess = session is None
            sess = session or SessionLocal()

            try:
                for i, (vr, ml_result) in enumerate(zip(vision_results, results)):
                    if vr is None:
                        continue

                    vision_name = (vr.get("pokemon_name") or "").strip()
                    vision_conf = vr.get("confidence", 0)
                    vision_hp = vr.get("hp")
                    vision_num = vr.get("card_number")
                    # Skip low-confidence or unrecognized vision results
                    if not vision_name or len(vision_name) < 2:
                        continue
                    if vision_name.lower() in ("unknown", "none", "card back"):
                        continue
                    if vision_conf < 0.50:
                        logger.debug("Card %d: vision=%s conf=%.2f too low, skipping",
                                     i, vision_name, vision_conf)
                        continue

                    ml_card_id = ml_result.get("card_id") or ""

                    # Check if ML's pick already matches Claude's name
                    ml_name = ""
                    if ml_card_id:
                        row = sess.execute(
                            sa_text("SELECT name FROM dim_cards WHERE card_id = :cid"),
                            {"cid": ml_card_id.split("/")[0] + "/normal"},
                        ).fetchone()
                        if row:
                            ml_name = row[0]

                    # Check agreement: compare base Pokemon names
                    # Claude often drops suffixes (ex, δ, LV.X) so compare
                    # the base name portion
                    import re as _re
                    def _base_name(n):
                        return _re.sub(
                            r'\s*(ex|EX|δ|delta|V|VSTAR|VMAX|GX|LV\.\w+|Star)\s*$',
                            '', n,
                        ).strip().lower()

                    names_agree = (
                        _base_name(ml_name) == _base_name(vision_name)
                        or vision_name.lower() in ml_name.lower()
                        or ml_name.lower() in vision_name.lower()
                    )

                    if names_agree:
                        # ML already has the right Pokemon — confirm it
                        results[i]["confidence"] = min(
                            ml_result.get("confidence", 0) + 0.10, 1.0,
                        )
                        results[i]["explanation"] = (
                            (results[i].get("explanation") or "") +
                            " (confirmed by Claude vision)"
                        )
                        logger.info("Card %d: ML+vision agree on %s (ml=%s)",
                                    i, vision_name, ml_name)
                        continue

                    # Claude disagrees on the Pokemon name.
                    # Override when: Claude is confident AND the names are
                    # truly different (not just suffix differences).
                    # But be cautious: only override high-confidence ML when
                    # Claude also has high confidence AND a card number.
                    ml_conf = ml_result.get("confidence", 0)
                    # Only override ML when we're very confident in vision:
                    # - Vision needs conf >= 0.80 to override any ML result
                    # - Vision needs a card number to override high-confidence ML
                    if vision_conf < 0.80:
                        logger.debug("Card %d: vision=%s(%.2f) vs ML=%s(%.2f), "
                                     "vision conf too low to override, keeping ML",
                                     i, vision_name, vision_conf, ml_name, ml_conf)
                        continue

                    # Search DB for cards matching Claude's identification
                    num_only = None
                    if vision_num:
                        num_only = vision_num.split("/")[0].strip().lstrip("0") or None

                    # Build query: name + optional HP + optional number
                    q = """
                        SELECT c.card_id, c.name, c.hp, c.card_number,
                               s.name as set_name
                        FROM dim_cards c
                        JOIN dim_sets s ON c.set_id = s.set_id
                        WHERE LOWER(c.name) = LOWER(:name)
                    """
                    params = {"name": vision_name}
                    if vision_hp and isinstance(vision_hp, (int, float)) and vision_hp >= 30:
                        q += " AND c.hp = :hp"
                        params["hp"] = str(int(vision_hp))
                    if num_only:
                        q += " AND LTRIM(c.card_number, '0') = :num"
                        params["num"] = num_only

                    db_candidates = sess.execute(sa_text(q), params).fetchall()

                    if not db_candidates:
                        # Relax: try without HP/number filters
                        db_candidates = sess.execute(
                            sa_text("""
                                SELECT c.card_id, c.name, c.hp, c.card_number,
                                       s.name as set_name
                                FROM dim_cards c
                                JOIN dim_sets s ON c.set_id = s.set_id
                                WHERE LOWER(c.name) = LOWER(:name)
                            """),
                            {"name": vision_name},
                        ).fetchall()

                    if not db_candidates:
                        logger.debug("Card %d: vision=%s, no DB candidates", i, vision_name)
                        continue

                    # Cross-reference with ML's visual candidate pool
                    ml_candidates = set()
                    raw = ml_result.get("raw_response", {})
                    for key in ("scored_candidates", "top_matches", "top_alternatives"):
                        for item in raw.get(key, []):
                            if isinstance(item, dict):
                                cid = item.get("card_id", "")
                            elif isinstance(item, (list, tuple)):
                                cid = str(item[0])
                            elif isinstance(item, str):
                                cid = item
                            else:
                                continue
                            # Normalize: strip variant suffix for matching
                            base = cid.split("/")[0] if "/" in cid else cid
                            ml_candidates.add(base)

                    # Find best DB candidate that also appears in ML visual pool
                    best_id = None
                    for row in db_candidates:
                        cid_base = row[0].split("/")[0]
                        if cid_base in ml_candidates:
                            best_id = row[0]
                            break

                    # If no overlap, just take the first DB candidate
                    if not best_id and len(db_candidates) == 1:
                        best_id = db_candidates[0][0]
                    elif not best_id and num_only:
                        # If we had a card number match, trust it
                        best_id = db_candidates[0][0]
                    elif not best_id:
                        # Fall back to match_vision_to_db scoring
                        best_id, _ = match_vision_to_db(vr, session=sess)

                    if best_id:
                        results[i] = {
                            "card_id": best_id,
                            "confidence": 0.85,
                            "method": "claude_vision",
                            "explanation": (
                                f"Claude vision: {vision_name} "
                                f"(ML had {ml_name or ml_card_id})"
                            ),
                            "raw_response": {
                                "vision_result": vr,
                                "ml_result": ml_result,
                            },
                        }
                        logger.info("Card %d: vision OVERRIDE %s -> %s (%s)",
                                    i, ml_card_id, best_id, vision_name)
                    else:
                        logger.debug("Card %d: vision=%s, couldn't resolve DB match",
                                     i, vision_name)
            finally:
                if own_sess:
                    sess.close()
        except Exception as e:
            logger.warning("Pass 3 (Claude vision) failed: %s", e)

    return results


def identify_page_vision_first(card_image_paths, session=None):
    """Identify cards using ML pipeline + multi-step Claude vision.

    Runs ML (DINOv2/CLIP) and multi-step Claude vision (name, attacks,
    number, era, HP) in parallel.  Combines signals:

    1. If ML and vision agree on name → confirm ML's pick (boost confidence)
    2. If they disagree → search ML candidates for vision's name
    3. If no ML candidate matches → try attack-based DB matching
    4. If attack match works → use that (vision name + attacks = strong)
    5. Last resort → use vision's name + number for DB lookup

    Args:
        card_image_paths: List of paths to card segment images.
        session: Optional DB session.

    Returns:
        List of result dicts, one per card.
    """
    import re as _re
    from concurrent.futures import ThreadPoolExecutor
    from cardprice.ml.claude_vision import (
        identify_cards_multi_step_parallel,
        match_attacks_to_db,
        match_multi_step_to_db,
    )
    from cardprice.db.session import SessionLocal
    from sqlalchemy import text as sa_text

    own_sess = session is None
    sess = session or SessionLocal()

    try:
        # Run ML pipeline and multi-step Claude vision in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            ml_future = pool.submit(identify_page, card_image_paths, sess)
            vision_future = pool.submit(
                identify_cards_multi_step_parallel,
                card_image_paths, "sonnet", 45, 4,
            )
            ml_results = ml_future.result()
            vision_results = vision_future.result()

        def _base_name(name):
            return _re.sub(
                r'\s*(ex|EX|δ|delta|V|VSTAR|VMAX|GX|LV\.\w+|Star)\s*$',
                '', name,
            ).strip().lower()

        results = list(ml_results)  # start with ML results

        for i, vr in enumerate(vision_results):
            if vr is None:
                continue
            vision_name = (vr.get("pokemon_name") or "").strip()
            vision_conf = vr.get("confidence", 0)
            vision_attacks = vr.get("attacks", [])
            vision_number = vr.get("card_number")
            vision_hp = vr.get("hp")

            if not vision_name or len(vision_name) < 2:
                continue
            if vision_name.lower() in ("unknown", "none", "card back",
                                        "pokemon", "pokémon"):
                # Detected card back or unreadable
                if vision_name.lower() in ("card back",):
                    results[i] = {
                        "card_id": None,
                        "confidence": 0.90,
                        "method": "vision_cardback",
                        "explanation": "Claude vision detected card back",
                    }
                continue

            ml_card_id = results[i].get("card_id") or ""
            ml_conf = results[i].get("confidence", 0)

            # Look up ML's pick's name from DB
            ml_name = ""
            if ml_card_id:
                base_cid = ml_card_id.split("/")[0] + "/normal"
                row = sess.execute(
                    sa_text("SELECT name FROM dim_cards WHERE card_id = :cid"),
                    {"cid": base_cid},
                ).fetchone()
                if row:
                    ml_name = row[0]

            # --- Case 1: Names agree → confirm ML's pick ---
            if (ml_name and (
                _base_name(ml_name) == _base_name(vision_name)
                or vision_name.lower() in ml_name.lower()
                or ml_name.lower() in vision_name.lower()
            )):
                results[i]["confidence"] = min(
                    results[i].get("confidence", 0) + 0.10, 1.0,
                )
                results[i]["explanation"] = (
                    (results[i].get("explanation") or "") +
                    " (confirmed by Claude vision)"
                )
                logger.info("Card %d: ML+vision agree on %s", i, vision_name)
                continue

            # --- Case 2: Names disagree → search ML candidate pool ---
            raw = results[i].get("raw_response", {})
            ml_candidates = []
            for key in ("scored_candidates", "top_matches", "top_alternatives"):
                for item in raw.get(key, []):
                    if isinstance(item, dict):
                        ml_candidates.append(item.get("card_id", ""))
                    elif isinstance(item, (list, tuple)):
                        ml_candidates.append(str(item[0]))
                    elif isinstance(item, str):
                        ml_candidates.append(item)

            best_match = None
            for cid in ml_candidates:
                if "/" not in cid:
                    cid_norm = cid + "/normal"
                else:
                    cid_norm = cid
                row = sess.execute(
                    sa_text("SELECT name FROM dim_cards WHERE card_id = :cid"),
                    {"cid": cid_norm},
                ).fetchone()
                if row and (
                    _base_name(row[0]) == _base_name(vision_name)
                    or vision_name.lower() in row[0].lower()
                ):
                    best_match = cid_norm
                    break

            if best_match:
                results[i] = {
                    "card_id": best_match,
                    "confidence": 0.85,
                    "method": "vision+ml_rerank",
                    "explanation": (
                        f"Claude vision ({vision_name}) reranked ML candidates"
                    ),
                    "raw_response": {
                        "vision_result": vr,
                        "ml_result": ml_results[i],
                    },
                }
                logger.info("Card %d: vision+ML rerank: %s -> %s",
                            i, ml_card_id, best_match)
                continue

            # --- Case 3: Try attack-based matching ---
            if vision_attacks and len(vision_attacks) >= 1:
                atk_id, atk_conf = match_attacks_to_db(
                    vision_attacks, pokemon_name=vision_name,
                    hp=vision_hp, card_number=vision_number,
                    session=sess,
                )
                if atk_id and atk_conf >= 0.55:
                    results[i] = {
                        "card_id": atk_id,
                        "confidence": atk_conf,
                        "method": "vision_attacks",
                        "explanation": (
                            f"Claude vision ({vision_name}) + "
                            f"attack match [{', '.join(vision_attacks)}]"
                        ),
                        "raw_response": {
                            "vision_result": vr,
                            "ml_result": ml_results[i],
                        },
                    }
                    logger.info("Card %d: attack match: %s (%s) -> %s",
                                i, vision_name, vision_attacks, atk_id)
                    continue

            # --- Case 4: Fall back to vision name+number DB lookup ---
            db_id, db_conf = match_multi_step_to_db(vr, session=sess)
            if db_id and db_conf >= 0.60:
                results[i] = {
                    "card_id": db_id,
                    "confidence": db_conf,
                    "method": "vision_db",
                    "explanation": (
                        f"Claude vision ({vision_name}) DB match"
                    ),
                    "raw_response": {
                        "vision_result": vr,
                        "ml_result": ml_results[i],
                    },
                }
                logger.info("Card %d: vision DB match: %s -> %s (conf=%.2f)",
                            i, vision_name, db_id, db_conf)
            else:
                logger.debug("Card %d: vision=%s, no override found, "
                             "keeping ML=%s", i, vision_name, ml_card_id)

    finally:
        if own_sess:
            sess.close()

    return results


def identify_card_robust(image_path, session=None):
    """Identify a card, trying rotations if the initial attempt is low-confidence.

    Tries identify_card at 0 degrees first.  If confidence is below
    _ROBUST_CONFIDENCE_THRESHOLD, also tries 90, 180, and 270 degree
    rotations and returns whichever attempt produced the highest confidence.

    Temporary rotated images are cleaned up after use.
    """
    from PIL import Image

    best = identify_card(image_path, session=session)
    if best["confidence"] >= _ROBUST_CONFIDENCE_THRESHOLD:
        return best

    logger.info("Robust: 0deg confidence=%.4f < %.2f, trying rotations ...",
                best["confidence"], _ROBUST_CONFIDENCE_THRESHOLD)

    tmp_files = []
    try:
        for angle in (90, 180, 270):
            try:
                img = Image.open(image_path)
                rotated = img.rotate(-angle, expand=True)  # negative = clockwise
                suffix = Path(image_path).suffix or ".png"
                fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix=f"card_rot{angle}_")
                os.close(fd)
                tmp_files.append(tmp_path)
                rotated.save(tmp_path)
                rotated.close()
                img.close()

                candidate = identify_card(tmp_path, session=session)
                logger.info("Robust: %ddeg -> confidence=%.4f method=%s card=%s",
                            angle, candidate["confidence"], candidate.get("method"), candidate.get("card_id"))

                if candidate["confidence"] > best["confidence"]:
                    best = candidate
                    best["explanation"] = (best.get("explanation") or "") + f" (matched at {angle}deg rotation)"

                if best["confidence"] >= _ROBUST_CONFIDENCE_THRESHOLD:
                    break
            except Exception as e:
                logger.warning("Robust: rotation %ddeg failed: %s", angle, e)
    finally:
        for tmp_path in tmp_files:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    logger.info("Robust: best result confidence=%.4f method=%s card=%s",
                best["confidence"], best.get("method"), best.get("card_id"))
    return best


# ---------------------------------------------------------------------------
# Helper: normalize card_id from index format to DB format
# ---------------------------------------------------------------------------

def _normalize_card_id(raw_cid: str) -> str:
    """Normalize a raw card_id from DINOv2/CLIP index to DB format.

    Index format: "set/set-num/variant" (e.g. "bw5/bw5-107/normal")
    DB format:    "set-num/variant"     (e.g. "bw5-107/normal")

    If the raw_cid has three path segments (set/id/variant), strip the first.
    Otherwise return as-is.
    """
    parts = raw_cid.split("/")
    if len(parts) >= 3:
        # "bw5/bw5-107/normal" -> "bw5-107/normal"
        return "/".join(parts[1:])
    if len(parts) == 2:
        return raw_cid
    # Single segment -- unlikely but handle gracefully
    return raw_cid


# ---------------------------------------------------------------------------
# Ensemble: DINOv2 + CLIP parallel voting
# ---------------------------------------------------------------------------

# Thresholds for the ensemble voter
_ENSEMBLE_AGREEMENT_CONFIDENCE = 0.85   # both top-1 agree -> this confidence floor
_ENSEMBLE_BOOST_FACTOR = 0.10           # bonus for appearing in both top-10 lists
_ENSEMBLE_MIN_ACCEPT = 0.55             # minimum ensemble score to accept


def _run_dino(image_path: str) -> list[tuple[str, float]]:
    """Run DINOv2 identification, returning normalized (card_id, score) top-10.

    Applies CLAHE + glare removal preprocessing when available, matching
    the cascade's Tier 2 behaviour.
    """
    from cardprice.ml.dino_matcher import identify_card as dino_identify
    dino_idx, dino_cids = _get_dino_index()
    if dino_idx is None:
        return []
    query_path = image_path
    preproc_tmp = None
    try:
        from cardprice.ml.preprocess import preprocess_for_matching
        preproc_tmp = preprocess_for_matching(image_path)
        query_path = preproc_tmp
    except Exception:
        pass
    try:
        matches = dino_identify(query_path, faiss_index=dino_idx, card_ids_list=dino_cids, top_k=10)
        return [(_normalize_card_id(cid), score) for cid, score in matches]
    finally:
        if preproc_tmp:
            try:
                os.unlink(preproc_tmp)
            except OSError:
                pass


def _run_clip(image_path: str) -> list[tuple[str, float]]:
    """Run CLIP image-to-image identification, returning normalized (card_id, score) top-10."""
    from cardprice.ml.clip_matcher import identify_card_by_image
    clip_idx = _get_clip_image_index()
    if clip_idx is None:
        return []
    matches = identify_card_by_image(image_path, preloaded_index=clip_idx, top_k=10)
    return [(_normalize_card_id(cid), score) for cid, score in matches]


def identify_card_ensemble(image_path, session=None, page_context=None):
    """Identify a card using DINOv2 + CLIP ensemble voting.

    Runs both methods in parallel, then combines their results:
    1. Get top-10 from each method
    2. Apply page context reranking if available
    2.5. Run border color analysis to filter candidates by era
         (e.g. DP-era yellow border filters out EX/BW candidates)
    3. Cards appearing in both lists get a score boost
    4. If both top-1 agree, assign high confidence regardless of individual scores
    5. If they disagree, use the method with higher relative margin (top1 - top2)

    Returns dict with keys: card_id, confidence, method, explanation, raw_response.
    """
    image_path = str(image_path)

    # Convert HEIC/HEIF if needed
    try:
        from cardprice.utils.image_convert import ensure_compatible
        image_path = ensure_compatible(image_path)
    except Exception as e:
        logger.warning("Image conversion failed, using original path: %s", e)

    # --- Phase 1: Run DINOv2 and CLIP in parallel ---
    dino_results = []
    clip_results = []
    dino_error = None
    clip_error = None

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_run_dino, image_path): "dino",
            pool.submit(_run_clip, image_path): "clip",
        }
        for future in as_completed(futures):
            method = futures[future]
            try:
                results = future.result()
                if method == "dino":
                    dino_results = results
                else:
                    clip_results = results
            except Exception as e:
                logger.warning("Ensemble: %s failed: %s", method, e)
                if method == "dino":
                    dino_error = e
                else:
                    clip_error = e

    # Apply page context reranking if available
    if page_context and page_context.get("likely_sets"):
        try:
            from cardprice.ml.page_context import rerank_with_context
            if dino_results:
                dino_results = rerank_with_context(dino_results, page_context)
            if clip_results:
                clip_results = rerank_with_context(clip_results, page_context)
            logger.info("Ensemble: applied page context (sets=%s)", page_context["likely_sets"][:3])
        except Exception as e:
            logger.debug("Ensemble: page context reranking failed: %s", e)

    # --- Phase 1.5: Border analysis to filter by era ---
    # When DINOv2 and CLIP disagree, border color can eliminate wrong-era candidates.
    border_info = None
    _BORDER_FILTER_MIN_CONFIDENCE = 0.35
    try:
        from cardprice.ml.border_analyzer import analyze_border, SET_TO_ERA
        border_info = analyze_border(image_path=image_path)
        logger.info("Ensemble: border analysis: color=%s era=%s confidence=%.2f sets=%d",
                     border_info["border_color"], border_info["era"],
                     border_info["confidence"], len(border_info["era_sets"]))

        if border_info["confidence"] >= _BORDER_FILTER_MIN_CONFIDENCE:
            era_set_ids = set(border_info["era_sets"])

            def _card_id_to_set(card_id: str) -> str:
                """Extract set ID from card_id like 'dp1-4/normal' -> 'dp1'."""
                # card_id format: "set-num/variant" e.g. "dp1-4/normal"
                base = card_id.split("/")[0]  # "dp1-4"
                # Set ID is everything before the last hyphen-number segment
                # e.g. "dp1-4" -> "dp1", "ex6-112" -> "ex6", "bw5-105" -> "bw5"
                parts = base.rsplit("-", 1)
                return parts[0] if len(parts) == 2 else base

            def _filter_by_era(results, era_sets):
                """Filter results to only cards from matching era sets."""
                return [(cid, score) for cid, score in results
                        if _card_id_to_set(cid) in era_sets]

            dino_filtered = _filter_by_era(dino_results, era_set_ids) if dino_results else []
            clip_filtered = _filter_by_era(clip_results, era_set_ids) if clip_results else []

            # Safety: only apply filtering if it doesn't eliminate ALL candidates
            # from BOTH lists. If one list is fully eliminated, that's fine (the
            # other method was probably right).
            if dino_filtered or clip_filtered:
                d_removed = len(dino_results) - len(dino_filtered)
                c_removed = len(clip_results) - len(clip_filtered)
                if d_removed > 0 or c_removed > 0:
                    logger.info("Ensemble: border filter removed %d/%d dino, %d/%d clip candidates "
                                "(era=%s, confidence=%.2f)",
                                d_removed, len(dino_results),
                                c_removed, len(clip_results),
                                border_info["era"], border_info["confidence"])
                    dino_results = dino_filtered if dino_filtered else dino_results
                    clip_results = clip_filtered if clip_filtered else clip_results
            else:
                logger.info("Ensemble: border filter would remove ALL candidates, skipping "
                            "(era=%s, confidence=%.2f)",
                            border_info["era"], border_info["confidence"])
    except ImportError as e:
        logger.info("Ensemble: border analysis SKIPPED -- missing dependency: %s", e)
    except Exception as e:
        logger.warning("Ensemble: border analysis ERROR -- %s", e)

    logger.info("Ensemble: DINOv2 returned %d results, CLIP returned %d results",
                len(dino_results), len(clip_results))

    # If both failed, return empty result
    if not dino_results and not clip_results:
        logger.info("Ensemble: both methods returned no results")
        return {
            "card_id": None, "confidence": 0.0, "method": "ensemble",
            "explanation": "Both DINOv2 and CLIP failed to produce results",
            "raw_response": {"dino_error": str(dino_error), "clip_error": str(clip_error)},
        }

    # If only one succeeded, fall back to it directly
    if not dino_results:
        logger.info("Ensemble: only CLIP available, using CLIP result directly")
        return _single_method_result(clip_results, "clip (ensemble fallback)")
    if not clip_results:
        logger.info("Ensemble: only DINOv2 available, using DINOv2 result directly")
        return _single_method_result(dino_results, "dino (ensemble fallback)")

    # --- Phase 2: Ensemble voting ---
    dino_top1_id, dino_top1_score = dino_results[0]
    clip_top1_id, clip_top1_score = clip_results[0]

    # Build lookup dicts for both sets (card_id -> score)
    dino_dict = {cid: score for cid, score in dino_results}
    clip_dict = {cid: score for cid, score in clip_results}

    # Find overlap: cards in both top-10 lists
    overlap_ids = set(dino_dict.keys()) & set(clip_dict.keys())
    logger.info("Ensemble: %d cards overlap in both top-10 lists", len(overlap_ids))

    # Compute relative margins (top1 - top2 gap)
    dino_margin = (dino_results[0][1] - dino_results[1][1]) if len(dino_results) >= 2 else dino_results[0][1]
    clip_margin = (clip_results[0][1] - clip_results[1][1]) if len(clip_results) >= 2 else clip_results[0][1]

    # --- Decision logic ---
    result = {
        "card_id": None, "confidence": 0.0, "method": "ensemble",
        "explanation": None,
        "raw_response": {
            "dino_top10": dino_results,
            "clip_top10": clip_results,
            "overlap_ids": list(overlap_ids),
            "dino_margin": dino_margin,
            "clip_margin": clip_margin,
            "border_analysis": {
                "color": border_info["border_color"],
                "era": border_info["era"],
                "confidence": border_info["confidence"],
            } if border_info else None,
        },
    }

    # Case 1: Both top-1 agree -- strong signal
    if dino_top1_id == clip_top1_id:
        # Average the scores and apply agreement bonus
        avg_score = (dino_top1_score + clip_top1_score) / 2.0
        ensemble_confidence = max(avg_score + _ENSEMBLE_BOOST_FACTOR,
                                  _ENSEMBLE_AGREEMENT_CONFIDENCE)
        result["card_id"] = dino_top1_id
        result["confidence"] = min(ensemble_confidence, 1.0)
        result["method"] = "ensemble (agree)"
        result["explanation"] = (
            f"DINOv2 and CLIP both agree on top match. "
            f"DINOv2={dino_top1_score:.3f}, CLIP={clip_top1_score:.3f}, "
            f"ensemble={result['confidence']:.3f}"
        )
        logger.info("Ensemble: AGREEMENT on %s (confidence=%.4f)",
                     result["card_id"], result["confidence"])
        return result

    # Case 2: Top-1 disagree -- check overlap and margins
    # Score each candidate by combining signals
    candidate_scores = {}

    # Score all overlap cards (appear in both top-10)
    for cid in overlap_ids:
        d_score = dino_dict[cid]
        c_score = clip_dict[cid]
        # Weighted average with overlap boost
        combined = (d_score + c_score) / 2.0 + _ENSEMBLE_BOOST_FACTOR
        candidate_scores[cid] = {
            "combined": combined,
            "dino": d_score,
            "clip": c_score,
            "in_both": True,
        }

    # Also score top-1 from each method if not already in candidates
    for cid, d_score, c_score, source in [
        (dino_top1_id, dino_top1_score, clip_dict.get(dino_top1_id), "dino"),
        (clip_top1_id, dino_dict.get(clip_top1_id), clip_top1_score, "clip"),
    ]:
        if cid not in candidate_scores:
            # Only in one list -- use that score alone (no boost)
            single_score = d_score if d_score is not None else c_score
            candidate_scores[cid] = {
                "combined": float(single_score) if single_score is not None else 0.0,
                "dino": d_score,
                "clip": c_score,
                "in_both": False,
            }

    # If any overlap exists, ALWAYS prefer the best overlap card.
    # A card appearing in both DINOv2 and CLIP top-10 is a much stronger
    # signal than a high single-method score (which may be noise).
    if overlap_ids:
        best_cid = max(
            overlap_ids,
            key=lambda k: candidate_scores[k]["combined"],
        )
        best_info = candidate_scores[best_cid]
        result["card_id"] = best_cid
        result["confidence"] = min(best_info["combined"], 1.0)
        result["method"] = "ensemble (overlap)"
        result["explanation"] = (
            f"Card found in both top-10 lists with boosted score. "
            f"DINOv2={best_info['dino']:.3f}, CLIP={best_info['clip']:.3f}, "
            f"combined={best_info['combined']:.3f}"
        )
        logger.info("Ensemble: OVERLAP winner %s (confidence=%.4f)",
                     result["card_id"], result["confidence"])
        return result

    # Neither top-1 in overlap -- try OCR tiebreaker before falling back to margin.
    # OCR reads the card name and checks if it matches either method's top-1 card.
    # If only one matches, that method wins regardless of margin.
    ocr_override = None
    try:
        from cardprice.ml.ocr_matcher import extract_card_name, _clean_ocr_text, fuzzy_match_card_name
        ocr_text, ocr_conf = extract_card_name(image_path)
        ocr_cleaned = _clean_ocr_text(ocr_text)
        logger.info("Ensemble OCR tiebreaker: raw=%r cleaned=%r conf=%.2f",
                     ocr_text, ocr_cleaned, ocr_conf)

        if ocr_cleaned and len(ocr_cleaned) >= 2:
            # Look up the card names for both candidates from the DB
            from cardprice.ml.ocr_matcher import _load_card_names
            card_names = _load_card_names()
            card_name_lookup = {cid: name for cid, name, _sid in card_names}

            dino_name = card_name_lookup.get(dino_top1_id, "")
            clip_name = card_name_lookup.get(clip_top1_id, "")

            # Fuzzy match OCR text against each candidate's name
            from rapidfuzz import fuzz
            dino_fuzzy = fuzz.token_set_ratio(ocr_cleaned.lower(), dino_name.lower()) if dino_name else 0
            clip_fuzzy = fuzz.token_set_ratio(ocr_cleaned.lower(), clip_name.lower()) if clip_name else 0

            logger.info("Ensemble OCR tiebreaker: dino=%s (%r, fuzzy=%d), clip=%s (%r, fuzzy=%d)",
                         dino_top1_id, dino_name, dino_fuzzy,
                         clip_top1_id, clip_name, clip_fuzzy)

            # OCR can help when one name clearly matches and the other doesn't.
            # Use a threshold: match >= 80, non-match < 70 (gap of 10+ needed).
            OCR_MATCH_THRESH = 80
            OCR_REJECT_THRESH = 70
            OCR_MIN_GAP = 10

            dino_matches_ocr = dino_fuzzy >= OCR_MATCH_THRESH
            clip_matches_ocr = clip_fuzzy >= OCR_MATCH_THRESH
            gap = abs(dino_fuzzy - clip_fuzzy)

            if dino_matches_ocr and not clip_matches_ocr and gap >= OCR_MIN_GAP:
                ocr_override = "dino"
                logger.info("Ensemble OCR tiebreaker: OVERRIDE -> dino (OCR matches dino name %r but not clip name %r)",
                             dino_name, clip_name)
            elif clip_matches_ocr and not dino_matches_ocr and gap >= OCR_MIN_GAP:
                ocr_override = "clip"
                logger.info("Ensemble OCR tiebreaker: OVERRIDE -> clip (OCR matches clip name %r but not dino name %r)",
                             clip_name, dino_name)
            else:
                logger.info("Ensemble OCR tiebreaker: no override (both_match=%s/%s, gap=%d)",
                             dino_matches_ocr, clip_matches_ocr, gap)

            result["raw_response"]["ocr_text"] = ocr_cleaned
            result["raw_response"]["ocr_dino_fuzzy"] = dino_fuzzy
            result["raw_response"]["ocr_clip_fuzzy"] = clip_fuzzy
            result["raw_response"]["ocr_override"] = ocr_override
    except ImportError as e:
        logger.info("Ensemble OCR tiebreaker: SKIPPED -- missing dependency: %s", e)
    except Exception as e:
        logger.warning("Ensemble OCR tiebreaker: ERROR -- %s", e)

    # Use OCR override if available, otherwise fall back to margin decision.
    # However, when margins are very close (within 0.02 of each other),
    # prefer the method with the higher absolute score.  This prevents
    # a tiny margin advantage from overriding a far more confident score.
    #
    # CRITICAL FIX: When one method's score is below its acceptance threshold,
    # do not let it win via margin. DINOv2 at 0.62 with margin 0.039 should
    # NOT override CLIP at 0.83 with margin 0.006 — the DINOv2 result is
    # essentially noise. (See card_07 Suicune bug: CLIP had correct answer
    # at rank 1 but DINOv2's margin overrode it with Kabutops.)
    DINO_ACCEPT_THRESHOLD = 0.65
    CLIP_ACCEPT_THRESHOLD = 0.75
    if ocr_override is not None:
        use_dino = ocr_override == "dino"
        decision_reason = "ocr"
    else:
        dino_below = dino_top1_score < DINO_ACCEPT_THRESHOLD
        clip_below = clip_top1_score < CLIP_ACCEPT_THRESHOLD
        if dino_below and not clip_below:
            # CLIP scores 0.78-0.82 on wrong cards (semantic matching),
            # so high CLIP score does NOT mean correct card.  DINOv2 is
            # better at exact visual matching even with lower raw scores.
            # Prefer DINOv2 when they disagree — was 0/3 trusting CLIP here.
            use_dino = True
            decision_reason = "confidence_gate"
        elif clip_below and not dino_below:
            # CLIP score is noise — trust DINOv2 regardless of margin
            use_dino = True
            decision_reason = "confidence_gate"
        elif dino_below and clip_below:
            # Both below threshold — pick the higher absolute score
            use_dino = dino_top1_score >= clip_top1_score
            decision_reason = "both_low"
        else:
            # Both above threshold — use margin tiebreaker
            margin_diff = abs(dino_margin - clip_margin)
            if margin_diff < 0.02:
                use_dino = dino_top1_score >= clip_top1_score
            else:
                use_dino = dino_margin > clip_margin
            decision_reason = "margin"

    if use_dino:
        winner_id, winner_score = dino_top1_id, dino_top1_score
        winner_method = "dino"
        winner_margin = dino_margin
        loser_method = "clip"
        loser_margin = clip_margin
    else:
        winner_id, winner_score = clip_top1_id, clip_top1_score
        winner_method = "clip"
        winner_margin = clip_margin
        loser_method = "dino"
        loser_margin = dino_margin

    result["card_id"] = winner_id
    result["confidence"] = float(winner_score)
    if decision_reason == "ocr":
        result["method"] = f"ensemble (ocr: {winner_method})"
        result["explanation"] = (
            f"DINOv2 and CLIP disagree. OCR read {ocr_cleaned!r} which matches "
            f"{winner_method}'s candidate. Using {winner_method} over {loser_method}. "
            f"DINOv2 top1: {dino_top1_id} ({dino_top1_score:.3f}), "
            f"CLIP top1: {clip_top1_id} ({clip_top1_score:.3f})"
        )
    elif decision_reason == "confidence_gate":
        result["method"] = f"ensemble (confidence_gate: {winner_method})"
        result["explanation"] = (
            f"DINOv2 and CLIP disagree. Preferring {winner_method} ({winner_score:.3f}) — "
            f"DINOv2 is more reliable for exact visual matching when they disagree. "
            f"DINOv2 top1: {dino_top1_id} ({dino_top1_score:.3f}), "
            f"CLIP top1: {clip_top1_id} ({clip_top1_score:.3f})"
        )
    elif decision_reason == "both_low":
        result["method"] = f"ensemble (both_low: {winner_method})"
        result["explanation"] = (
            f"DINOv2 and CLIP disagree, both below thresholds. Using higher score: "
            f"{winner_method} ({winner_score:.3f}). "
            f"DINOv2 top1: {dino_top1_id} ({dino_top1_score:.3f}), "
            f"CLIP top1: {clip_top1_id} ({clip_top1_score:.3f})"
        )
    else:
        result["method"] = f"ensemble (margin: {winner_method})"
        result["explanation"] = (
            f"DINOv2 and CLIP disagree. Using {winner_method} (margin={winner_margin:.3f}) "
            f"over {loser_method} (margin={loser_margin:.3f}). "
            f"DINOv2 top1: {dino_top1_id} ({dino_top1_score:.3f}), "
            f"CLIP top1: {clip_top1_id} ({clip_top1_score:.3f})"
        )
    logger.info("Ensemble: %s winner %s via %s (confidence=%.4f, margin=%.4f)",
                 decision_reason.upper(), result["card_id"], winner_method,
                 result["confidence"], winner_margin)
    return result


# ---------------------------------------------------------------------------
# Multi-signal identification: combines ALL available signals
# ---------------------------------------------------------------------------

# DB metadata cache for multi-signal scoring: {card_id: {name, hp, set_id, supertype, subtypes}}
_card_metadata_cache: dict[str, dict] | None = None


def _get_card_metadata() -> dict[str, dict]:
    """Lazy-load card metadata from dim_cards for multi-signal filtering.

    Returns dict keyed by card_id with values containing name, hp, set_id,
    supertype, and subtypes.
    """
    global _card_metadata_cache
    if _card_metadata_cache is not None:
        return _card_metadata_cache

    from cardprice.db.session import engine
    from sqlalchemy import text as sql_text

    with engine.connect() as conn:
        rows = conn.execute(sql_text(
            "SELECT card_id, name, hp, set_id, supertype, subtypes FROM dim_cards"
        )).fetchall()

    _card_metadata_cache = {}
    for r in rows:
        _card_metadata_cache[r[0]] = {
            "name": r[1],
            "hp": r[2],
            "set_id": r[3],
            "supertype": r[4],
            "subtypes": r[5] if r[5] else [],
        }
    logger.info("Loaded %d card metadata entries for multi-signal scoring.", len(_card_metadata_cache))
    return _card_metadata_cache


def identify_card_multisignal(image_path, session=None, page_context=None):
    """Identify a card by combining ALL available signals.

    This is the ultimate identification method, sitting above the ensemble.
    It runs all signal extractors in parallel, then scores candidates from
    the DINOv2+CLIP top-10 lists against every extracted signal.

    Signals used:
        1. DINOv2 top-10 (visual similarity)
        2. CLIP top-10 (visual + semantic similarity)
        3. OCR card name (fuzzy text match)
        4. HP value (from hp_detector)
        5. Card type (from type_detector, color-based)
        6. Border/era analysis (from border_analyzer)
        7. Page context (set/era prior from neighboring cards)
        8. DP-era level (OCR "LV.XX" + dp_level_map.json matching)

    Scoring approach:
        - Start with visual similarity score (avg of DINO+CLIP if both present)
        - Apply bonuses/penalties for each signal match:
            +0.15 for exact name match (fuzzy >= 90)
            +0.05 for partial name match (fuzzy >= 70)
            +0.12 for DP-era level match (name + LV.XX from OCR)
            +0.10 for HP match (only when candidate has visual overlap)
            +0.00 for HP match on single-visual candidates (skipped)
            +0.05 for type match (top-1 detected type)
            +0.08 for era/set match from border analysis
            +0.10 for page context set match
            +0.05 for page context era match
        - Candidates with signal contradictions get penalties:
            -0.15 for name mismatch when OCR is confident
            -0.05 for HP mismatch when candidate has visual overlap (mild)
            +0.00 for HP mismatch on single-visual candidates (skipped)
            -0.05 for era mismatch when border analysis is confident
        - HP discount rationale: HP OCR is noisy on binder-sleeve photos
          (glare, partial occlusion, angle). When a candidate only appears
          in one visual model's top-10, HP match/mismatch is skipped to
          prevent a misread HP from overriding visual similarity scores.

    Args:
        image_path: Path to the card image.
        session: Optional DB session.
        page_context: Optional page context dict from identify_page_context.

    Returns:
        Dict with keys: card_id, confidence, method, explanation, raw_response.
        raw_response includes all signal details for debugging.
    """
    image_path = str(image_path)

    # Convert HEIC/HEIF if needed
    try:
        from cardprice.utils.image_convert import ensure_compatible
        image_path = ensure_compatible(image_path)
    except Exception as e:
        logger.warning("Image conversion failed, using original path: %s", e)

    # --- Phase 1: Run ALL signal extractors in parallel ---
    dino_results = []
    clip_results = []
    ocr_name = None
    ocr_confidence = 0.0
    ocr_raw = None
    hp_value = None
    type_predictions = []
    border_info = None
    dp_level_name = None
    dp_level_value = None
    dp_level_candidates = []

    signal_errors = {}

    def _run_ocr_name_signal():
        from cardprice.ml.ocr_matcher import extract_card_name, _clean_ocr_text
        raw_text, conf = extract_card_name(image_path)
        cleaned = _clean_ocr_text(raw_text)
        return cleaned, conf, raw_text

    def _run_hp_signal():
        from cardprice.ml.hp_detector import detect_hp
        return detect_hp(image_path)

    def _run_type_signal():
        from cardprice.ml.type_detector import detect_type
        return detect_type(image_path, top_n=3)

    def _run_border_signal():
        from cardprice.ml.border_analyzer import analyze_border
        return analyze_border(image_path)

    def _run_dp_level_signal():
        from cardprice.ml.ocr_matcher import extract_card_name_all_fragments, _extract_level_from_ocr
        fragments = extract_card_name_all_fragments(image_path)
        if not fragments:
            return None, None, []
        ocr_texts = [t for t, c in fragments]
        name, level = _extract_level_from_ocr(ocr_texts)
        if name and level:
            from cardprice.ml.ocr_matcher import match_by_dp_level
            candidates = match_by_dp_level(name, level)
            return name, level, candidates
        return name, level, []

    with ThreadPoolExecutor(max_workers=7) as pool:
        futures = {
            pool.submit(_run_dino, image_path): "dino",
            pool.submit(_run_clip, image_path): "clip",
            pool.submit(_run_ocr_name_signal): "ocr",
            pool.submit(_run_hp_signal): "hp",
            pool.submit(_run_type_signal): "type",
            pool.submit(_run_border_signal): "border",
            pool.submit(_run_dp_level_signal): "dp_level",
        }
        for future in as_completed(futures):
            signal = futures[future]
            try:
                res = future.result()
                if signal == "dino":
                    dino_results = res
                elif signal == "clip":
                    clip_results = res
                elif signal == "ocr":
                    ocr_name, ocr_confidence, ocr_raw = res
                elif signal == "hp":
                    hp_value = res
                elif signal == "type":
                    type_predictions = res
                elif signal == "border":
                    border_info = res
                elif signal == "dp_level":
                    dp_level_name, dp_level_value, dp_level_candidates = res
            except Exception as e:
                logger.warning("Multisignal: %s extractor failed: %s", signal, e)
                signal_errors[signal] = str(e)

    logger.info(
        "Multisignal signals: dino=%d clip=%d ocr=%r hp=%s type=%s border=%s level=%s(%s)",
        len(dino_results), len(clip_results),
        ocr_name, hp_value,
        type_predictions[0] if type_predictions else None,
        border_info.get("era") if border_info else None,
        dp_level_value, dp_level_name,
    )

    # --- Phase 2: Pool all candidate card_ids ---
    candidate_visual_scores: dict[str, dict] = {}

    for cid, score in dino_results:
        if cid not in candidate_visual_scores:
            candidate_visual_scores[cid] = {"dino": 0.0, "clip": 0.0}
        candidate_visual_scores[cid]["dino"] = score

    for cid, score in clip_results:
        if cid not in candidate_visual_scores:
            candidate_visual_scores[cid] = {"dino": 0.0, "clip": 0.0}
        candidate_visual_scores[cid]["clip"] = score

    if not candidate_visual_scores:
        logger.info("Multisignal: no visual candidates, returning empty result")
        return {
            "card_id": None, "confidence": 0.0, "method": "multisignal",
            "explanation": "No visual candidates from DINOv2 or CLIP",
            "raw_response": {"signal_errors": signal_errors},
        }

    # --- Phase 2.5: Add OCR-matched candidates to the pool ---
    card_meta = _get_card_metadata()
    ocr_matches = []
    if ocr_name and len(ocr_name) >= 2:
        try:
            from cardprice.ml.ocr_matcher import fuzzy_match_card_name
            ocr_matches = fuzzy_match_card_name(ocr_name, top_k=10, score_cutoff=70.0)
            for cid, _name, _sid, _score in ocr_matches:
                if cid not in candidate_visual_scores:
                    candidate_visual_scores[cid] = {"dino": 0.0, "clip": 0.0}
        except Exception as e:
            logger.warning("Multisignal: OCR fuzzy match failed: %s", e)

    # --- Phase 2.6: Add DP-level-matched candidates to the pool ---
    dp_level_match_ids = set()
    if dp_level_candidates:
        for cid, _name, _sid, _score, _details in dp_level_candidates:
            dp_level_match_ids.add(cid)
            if cid not in candidate_visual_scores:
                candidate_visual_scores[cid] = {"dino": 0.0, "clip": 0.0}

    # Build OCR lookup for fast scoring
    ocr_match_by_id = {}
    if ocr_matches:
        for cid, name, sid, fuzzy_score in ocr_matches:
            ocr_match_by_id[cid] = {"name": name, "set_id": sid, "fuzzy_score": fuzzy_score}

    # Extract signal parameters
    detected_type = type_predictions[0][0] if type_predictions else None
    detected_type_conf = type_predictions[0][1] if type_predictions else 0.0

    era_sets = set(border_info.get("era_sets", [])) if border_info else set()
    border_conf = border_info.get("confidence", 0.0) if border_info else 0.0

    page_sets = set(page_context.get("likely_sets", [])) if page_context else set()
    page_era = page_context.get("era") if page_context else None
    page_conf = page_context.get("confidence", 0.0) if page_context else 0.0

    # --- Phase 3: Score each candidate against all signals ---
    scored_candidates = []

    for cid, vis_scores in candidate_visual_scores.items():
        meta = card_meta.get(cid)
        if meta is None:
            continue

        # Base visual score: average of available visual scores
        vis_parts = []
        if vis_scores["dino"] > 0:
            vis_parts.append(vis_scores["dino"])
        if vis_scores["clip"] > 0:
            vis_parts.append(vis_scores["clip"])
        base_score = sum(vis_parts) / len(vis_parts) if vis_parts else 0.0

        bonuses = []
        penalties = []
        total_adjustment = 0.0

        # --- OCR name matching ---
        if ocr_name and len(ocr_name) >= 2:
            cand_name = meta.get("name", "")
            if cid in ocr_match_by_id:
                fuzzy = ocr_match_by_id[cid]["fuzzy_score"]
            else:
                try:
                    from rapidfuzz import fuzz
                    fuzzy = fuzz.token_set_ratio(ocr_name.lower(), cand_name.lower())
                except ImportError:
                    fuzzy = 0

            if fuzzy >= 90:
                total_adjustment += 0.15
                bonuses.append(f"name={cand_name}(fuzzy={fuzzy:.0f})")
            elif fuzzy >= 70:
                total_adjustment += 0.05
                bonuses.append(f"name~={cand_name}(fuzzy={fuzzy:.0f})")
            elif ocr_confidence > 0.5 and fuzzy < 50:
                total_adjustment -= 0.15
                penalties.append(f"name_mismatch(ocr={ocr_name!r},card={cand_name!r},fuzzy={fuzzy:.0f})")

        # --- HP matching ---
        # Only trust HP if the detected value is plausible (>= 30).
        # OCR often misreads HP as "10" or single digits from card numbers/damage.
        #
        # Discount HP when the candidate lacks visual overlap (only appears in
        # one of DINOv2/CLIP, not both).  A single-model candidate boosted by
        # noisy HP can override a visually stronger candidate that both models
        # agree on.  Without corroboration from a second visual model, the HP
        # signal should carry less weight.  Similarly, don't heavily penalize
        # HP mismatch on candidates that DO have visual overlap -- the HP OCR
        # is often wrong on binder-sleeve photos.
        if hp_value is not None and hp_value >= 30:
            cand_hp = meta.get("hp")
            if cand_hp is not None:
                has_visual_overlap = (vis_scores["dino"] > 0 and vis_scores["clip"] > 0)

                if has_visual_overlap:
                    # Both visual models found this candidate: HP is a useful
                    # disambiguator between visually similar cards.
                    if cand_hp == hp_value:
                        total_adjustment += 0.10
                        bonuses.append(f"hp={hp_value}")
                    else:
                        # Mild penalty -- HP OCR is unreliable on binder photos
                        # so don't punish too hard when visual evidence is strong.
                        total_adjustment -= 0.05
                        penalties.append(f"hp_mismatch(detected={hp_value},card={cand_hp},mild)")
                else:
                    # Only one visual model found this candidate: HP signal is
                    # not trustworthy enough to override visual similarity scores.
                    # A misread HP can boost a wrong candidate from one model while
                    # penalizing the correct candidate from the other model,
                    # flipping the result.  Skip HP adjustment entirely for
                    # single-visual candidates to let visual scores decide.
                    if cand_hp == hp_value:
                        bonuses.append(f"hp={hp_value}(skipped,single_visual)")
                    else:
                        penalties.append(f"hp_mismatch(detected={hp_value},card={cand_hp},skipped)")

        # --- DP-era level matching ---
        if dp_level_value is not None and cid in dp_level_match_ids:
            total_adjustment += 0.12
            bonuses.append(f"dp_level={dp_level_value}(name={dp_level_name!r})")

        # --- Border/era matching ---
        if era_sets and border_conf > 0.3:
            cand_set = meta.get("set_id", "")
            if cand_set in era_sets:
                total_adjustment += 0.08
                bonuses.append(f"era_match(set={cand_set})")
            elif border_conf > 0.6:
                total_adjustment -= 0.05
                penalties.append(f"era_mismatch(set={cand_set})")

        # --- Page context matching ---
        if page_sets and page_conf > 0.3:
            cand_set = meta.get("set_id", "")
            if cand_set in page_sets:
                total_adjustment += 0.10
                bonuses.append(f"page_set={cand_set}")
            else:
                try:
                    from cardprice.ml.page_context import _era_for_set
                    cand_era = _era_for_set(cand_set)
                    if cand_era and cand_era == page_era:
                        total_adjustment += 0.05
                        bonuses.append(f"page_era={cand_era}")
                except Exception:
                    pass

        # --- Visual overlap bonus: in both DINO and CLIP top-10 ---
        if vis_scores["dino"] > 0 and vis_scores["clip"] > 0:
            total_adjustment += 0.05
            bonuses.append("visual_overlap")

        final_score = max(base_score + total_adjustment, 0.0)  # no upper cap — let bonuses differentiate

        scored_candidates.append({
            "card_id": cid,
            "final_score": final_score,
            "base_visual": base_score,
            "adjustment": total_adjustment,
            "dino_score": vis_scores["dino"],
            "clip_score": vis_scores["clip"],
            "bonuses": bonuses,
            "penalties": penalties,
            "name": meta.get("name", ""),
            "hp": meta.get("hp"),
            "set_id": meta.get("set_id", ""),
        })

    if not scored_candidates:
        return {
            "card_id": None, "confidence": 0.0, "method": "multisignal",
            "explanation": "No candidates with valid DB metadata",
            "raw_response": {"signal_errors": signal_errors},
        }

    # Sort by final score descending
    scored_candidates.sort(key=lambda c: c["final_score"], reverse=True)

    best = scored_candidates[0]
    runner_up = scored_candidates[1] if len(scored_candidates) > 1 else None

    # Build detailed explanation
    signals_used = []
    if dino_results:
        signals_used.append(f"DINOv2({best['dino_score']:.3f})")
    if clip_results:
        signals_used.append(f"CLIP({best['clip_score']:.3f})")
    if ocr_name:
        signals_used.append(f"OCR({ocr_name!r})")
    if hp_value is not None:
        signals_used.append(f"HP({hp_value})")
    if dp_level_value is not None:
        signals_used.append(f"Level(LV.{dp_level_value},{dp_level_name!r})")
    if detected_type:
        signals_used.append(f"Type({detected_type}:{detected_type_conf:.0%})")
    if border_info:
        signals_used.append(f"Era({border_info.get('era')})")
    if page_context and page_sets:
        signals_used.append(f"PageCtx({list(page_sets)[:2]})")

    explanation_parts = [
        f"Multi-signal: {best['name']} ({best['card_id']}) score={best['final_score']:.3f}",
        f"Visual={best['base_visual']:.3f} + adjustments={best['adjustment']:+.3f}",
        f"Signals: {', '.join(signals_used)}",
    ]
    if best["bonuses"]:
        explanation_parts.append(f"Bonuses: {', '.join(best['bonuses'])}")
    if best["penalties"]:
        explanation_parts.append(f"Penalties: {', '.join(best['penalties'])}")
    if runner_up:
        explanation_parts.append(
            f"Runner-up: {runner_up['name']} ({runner_up['card_id']}) "
            f"score={runner_up['final_score']:.3f}"
        )

    result = {
        "card_id": best["card_id"],
        "confidence": min(best["final_score"], 1.0),  # cap for display, raw score may exceed 1.0
        "method": "multisignal",
        "explanation": ". ".join(explanation_parts),
        "raw_response": {
            "dino_top10": dino_results,
            "clip_top10": clip_results,
            "ocr_name": ocr_name,
            "ocr_raw": ocr_raw,
            "ocr_confidence": ocr_confidence,
            "hp_detected": hp_value,
            "dp_level_name": dp_level_name,
            "dp_level_value": dp_level_value,
            "dp_level_candidates": [(c[0], c[1], c[3]) for c in dp_level_candidates[:5]] if dp_level_candidates else [],
            "type_detected": type_predictions[:3] if type_predictions else [],
            "border_info": border_info,
            "page_context": page_context,
            "scored_candidates": scored_candidates[:10],
            "signal_errors": signal_errors,
        },
    }

    logger.info(
        "Multisignal: BEST %s (%s) score=%.4f (visual=%.3f adj=%+.3f) "
        "bonuses=%s penalties=%s",
        best["card_id"], best["name"], best["final_score"],
        best["base_visual"], best["adjustment"],
        best["bonuses"], best["penalties"],
    )

    return result


def identify_card_hybrid(image_path, session=None, page_context=None):
    """Best-of-both identification: multi-signal when strong, ensemble fallback.

    Multi-signal excels when OCR/HP/level signals are available (e.g. DP-era
    cards with readable names). Ensemble is more robust when those signals are
    absent or noisy (e.g. holo glare, poor scan quality).

    Strategy:
        1. Run multi-signal (all 6 extractors).
        2. Check if the winner has meaningful non-visual bonuses
           (name match, HP match, level match — NOT just era_match or visual_overlap).
        3. If yes → use multi-signal result.
        4. If no → fall back to ensemble (confidence-gated).
    """
    ms_result = identify_card_multisignal(image_path, session=session, page_context=page_context)

    # Check if multisignal has strong non-generic bonuses
    scored = ms_result.get("raw_response", {}).get("scored_candidates", [])
    has_strong_signal = False
    if scored:
        top = scored[0]
        bonuses = top.get("bonuses", [])
        # "Strong" means at least one bonus that isn't just era_match, visual_overlap,
        # or a discounted/skipped HP bonus (which signals low-confidence HP match).
        strong_bonuses = [
            b for b in bonuses
            if not b.startswith("era_match") and not b.startswith("visual_overlap")
            and not b.startswith("page_")
            and "discounted" not in b and "skipped" not in b
        ]
        has_strong_signal = len(strong_bonuses) > 0

    if has_strong_signal:
        logger.info("Hybrid: using multi-signal (strong bonuses: %s)",
                     [b for b in scored[0].get("bonuses", []) if not b.startswith("era_match")])
        return ms_result

    # Fall back to ensemble
    logger.info("Hybrid: multi-signal has no strong bonuses, falling back to ensemble")
    ens_result = identify_card_ensemble(image_path, session=session, page_context=page_context)
    return ens_result


def _single_method_result(results: list[tuple[str, float]], method_label: str) -> dict:
    """Build a result dict from a single method's top-10 list."""
    if not results:
        return {
            "card_id": None, "confidence": 0.0, "method": method_label,
            "explanation": "No results from single method fallback",
            "raw_response": {},
        }
    card_id, score = results[0]
    alt_list = [(cid, s) for cid, s in results[1:4]]
    alt_str = ", ".join(f"{a[0]} ({a[1]:.0%})" for a in alt_list)
    return {
        "card_id": card_id,
        "confidence": float(score),
        "method": method_label,
        "explanation": f"Single method fallback ({score:.0%}). Alternatives: {alt_str}" if alt_str else f"Single method fallback ({score:.0%})",
        "raw_response": {"top_matches": results[:5], "top_alternatives": alt_list},
    }


# ---------------------------------------------------------------------------
# Reference-matching page identification pipeline
# ---------------------------------------------------------------------------

_REF_MATCH_CONFIDENCE_THRESHOLD = 0.45  # below this, fall back to cascade


def _extract_signals_for_ref(image_path: str) -> dict:
    """Run cheap classifiers in parallel for a single card image.

    Returns a dict with keys: ocr_name, ocr_confidence, hp, type_predictions,
    dino_top10, dino_name_vote.
    """
    ocr_name = None
    ocr_confidence = 0.0
    hp_value = None
    type_predictions = []
    dino_results = []

    def _do_ocr():
        from cardprice.ml.ocr_matcher import extract_card_name, _clean_ocr_text
        raw_text, conf = extract_card_name(image_path)
        cleaned = _clean_ocr_text(raw_text)
        return cleaned, conf

    def _do_hp():
        from cardprice.ml.hp_detector import detect_hp
        return detect_hp(image_path)

    def _do_type():
        from cardprice.ml.type_detector import detect_type
        return detect_type(image_path, top_n=3)

    def _do_dino():
        return _run_dino(image_path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_do_ocr): "ocr",
            pool.submit(_do_hp): "hp",
            pool.submit(_do_type): "type",
            pool.submit(_do_dino): "dino",
        }
        for future in as_completed(futures):
            signal = futures[future]
            try:
                res = future.result()
                if signal == "ocr":
                    ocr_name, ocr_confidence = res
                elif signal == "hp":
                    hp_value = res
                elif signal == "type":
                    type_predictions = res
                elif signal == "dino":
                    dino_results = res
            except Exception as e:
                logger.warning("Signal %s failed for %s: %s", signal, image_path, e)

    # DINOv2 name voting: extract Pokemon name from each top-5 result's card_id,
    # then take the plurality name.
    dino_name_vote = None
    if dino_results:
        from collections import Counter
        name_counts = Counter()
        for card_id, _score in dino_results[:5]:
            # card_id format: "set-num/variant" e.g. "base1-4/normal"
            # We need the Pokemon name from the DB.  As a fast heuristic,
            # extract the card portion and look up the name from dim_cards.
            # But DB lookups per card are slow.  Instead, group by the
            # card portion minus the variant (cards with same set-num are
            # the same Pokemon).
            card_portion = card_id.split("/")[0] if "/" in card_id else card_id
            name_counts[card_portion] += 1

        if name_counts:
            # The most common card portion in top-5 -> look up its name
            top_card_portion = name_counts.most_common(1)[0][0]
            # Query the DB for this card's name (fast single lookup)
            try:
                from sqlalchemy import text as sa_text
                from cardprice.db.session import SessionLocal
                sess = SessionLocal()
                try:
                    row = sess.execute(
                        sa_text("SELECT name FROM dim_cards WHERE card_id LIKE :pattern LIMIT 1"),
                        {"pattern": f"{top_card_portion}/%"},
                    ).fetchone()
                    if row:
                        dino_name_vote = row[0]
                finally:
                    sess.close()
            except Exception as e:
                logger.warning("DINOv2 name vote DB lookup failed: %s", e)

    return {
        "ocr_name": ocr_name,
        "ocr_confidence": ocr_confidence,
        "hp": hp_value,
        "type_predictions": type_predictions,
        "dino_top10": dino_results,
        "dino_name_vote": dino_name_vote,
    }


def _choose_best_name(signals: dict) -> tuple:
    """Pick the best Pokemon name from OCR and DINOv2 name voting.

    Returns (name, source) where source is "ocr" or "dino_vote" or None.
    Prefers OCR when confidence is high (>= 0.5 and name length >= 3).
    Falls back to DINOv2 plurality name vote.
    """
    ocr_name = signals.get("ocr_name")
    ocr_conf = signals.get("ocr_confidence", 0.0)
    dino_name = signals.get("dino_name_vote")

    # OCR is preferred when reasonably confident
    if ocr_name and len(ocr_name) >= 3 and ocr_conf >= 0.5:
        return ocr_name, "ocr"

    # Fall back to DINOv2 name voting
    if dino_name:
        return dino_name, "dino_vote"

    # Last resort: use OCR even if low confidence
    if ocr_name and len(ocr_name) >= 3:
        return ocr_name, "ocr_low"

    return None, None


def identify_card_ref_matching(image_path, session=None, page_context=None):
    """Identify a card using reference-image matching with attribute narrowing.

    Pipeline:
        1. Run cheap classifiers in parallel: OCR name, HP, type, DINOv2 top-10.
        2. Determine best name signal (OCR preferred, DINOv2 name vote fallback).
        3. Call ref_matcher.match_by_reference() with narrowed attributes.
        4. If ref_matcher confidence < 0.45, fall back to identify_card() cascade.

    Args:
        image_path: Path to the card image.
        session: Optional DB session.
        page_context: Optional page context dict (unused here, kept for API compat).

    Returns:
        Dict with keys: card_id, confidence, method, explanation, raw_response.
    """
    image_path = str(image_path)

    # Convert HEIC/HEIF if needed
    try:
        from cardprice.utils.image_convert import ensure_compatible
        image_path = ensure_compatible(image_path)
    except Exception as e:
        logger.warning("Image conversion failed, using original path: %s", e)

    # Step 1: Extract all signals in parallel
    signals = _extract_signals_for_ref(image_path)

    logger.info(
        "Ref-match signals: ocr=%r (%.2f), hp=%s, type=%s, dino_vote=%r, dino_top10=%d",
        signals["ocr_name"], signals["ocr_confidence"],
        signals["hp"],
        signals["type_predictions"][0][0] if signals["type_predictions"] else None,
        signals["dino_name_vote"],
        len(signals["dino_top10"]),
    )

    # Step 2: Choose best name
    best_name, name_source = _choose_best_name(signals)

    if best_name is None:
        logger.info("Ref-match: no name signal available, falling back to cascade")
        return identify_card(image_path, session=session, page_context=page_context)

    # Step 3: Extract HP and type for narrowing
    hp_value = signals["hp"]
    card_type = None
    if signals["type_predictions"]:
        top_type, top_type_conf = signals["type_predictions"][0]
        # Only use type if reasonably confident (>40% pixel vote share)
        if top_type_conf >= 0.40 and top_type != "Colorless":
            card_type = top_type

    logger.info(
        "Ref-match: querying candidates with name=%r (source=%s), hp=%s, type=%s",
        best_name, name_source, hp_value, card_type,
    )

    # Step 4: Run reference matching
    from cardprice.ml.ref_matcher import match_by_reference
    best_card_id, best_score = match_by_reference(
        query_image_path=image_path,
        pokemon_name=best_name,
        hp=hp_value,
        card_type=card_type,
        session=session,
    )

    # If no match found with HP+type narrowing, try relaxing constraints
    if best_card_id is None or best_score < _REF_MATCH_CONFIDENCE_THRESHOLD:
        # Try without type constraint
        if card_type is not None:
            logger.info("Ref-match: relaxing type constraint (was %s)", card_type)
            cid2, score2 = match_by_reference(
                query_image_path=image_path,
                pokemon_name=best_name,
                hp=hp_value,
                card_type=None,
                session=session,
            )
            if score2 > best_score:
                best_card_id, best_score = cid2, score2

    if best_card_id is None or best_score < _REF_MATCH_CONFIDENCE_THRESHOLD:
        # Try without HP constraint
        if hp_value is not None:
            logger.info("Ref-match: relaxing HP constraint (was %s)", hp_value)
            cid3, score3 = match_by_reference(
                query_image_path=image_path,
                pokemon_name=best_name,
                hp=None,
                card_type=None,
                session=session,
            )
            if score3 > best_score:
                best_card_id, best_score = cid3, score3

    # Step 5: Check confidence threshold
    if best_card_id is None or best_score < _REF_MATCH_CONFIDENCE_THRESHOLD:
        logger.info(
            "Ref-match: low confidence (%.3f < %.2f), falling back to cascade",
            best_score, _REF_MATCH_CONFIDENCE_THRESHOLD,
        )
        return identify_card(image_path, session=session, page_context=page_context)

    # Build result
    alt_info = ""
    if signals["dino_top10"]:
        alts = signals["dino_top10"][:3]
        alt_info = " | DINOv2 top-3: " + ", ".join(
            f"{cid} ({s:.0%})" for cid, s in alts
        )

    explanation = (
        f"Reference match via {name_source} name={best_name!r}, "
        f"hp={hp_value}, type={card_type}, "
        f"similarity={best_score:.3f}{alt_info}"
    )

    result = {
        "card_id": best_card_id,
        "confidence": float(best_score),
        "method": f"ref_match({name_source})",
        "explanation": explanation,
        "raw_response": {
            "signals": {
                "ocr_name": signals["ocr_name"],
                "ocr_confidence": signals["ocr_confidence"],
                "hp": signals["hp"],
                "type_top1": signals["type_predictions"][0] if signals["type_predictions"] else None,
                "dino_name_vote": signals["dino_name_vote"],
                "name_used": best_name,
                "name_source": name_source,
                "type_used": card_type,
            },
            "dino_top10": signals["dino_top10"],
            "ref_match_score": best_score,
            "ref_match_card_id": best_card_id,
        },
    }

    return result


def identify_page_ref_matching(card_image_paths, session=None):
    """Identify all cards on a binder page using reference-image matching.

    This pipeline runs cheap classifiers (OCR, HP, type, DINOv2) in parallel
    for each card, determines the best name signal, then does targeted
    reference-image comparison against narrowed DB candidates.

    Pipeline per card:
        1. Parallel signal extraction: OCR name, HP, type, DINOv2 FAISS top-10.
        2. Name determination: prefer OCR if confident, else DINOv2 name voting
           (plurality name from top-5 FAISS hits).
        3. ref_matcher.match_by_reference(image, name, hp, type) narrows to
           2-20 DB candidates and does DINOv2 embedding comparison vs reference
           images.
        4. If ref_matcher confidence < 0.45, fall back to identify_card() cascade.

    All cards are processed in parallel (one thread per card).

    Args:
        card_image_paths: List of paths to individual card segment images.
        session: Optional DB session.

    Returns:
        List of result dicts (same format as identify_card), one per card.
    """
    if not card_image_paths:
        return []

    n_cards = len(card_image_paths)
    logger.info("identify_page_ref_matching: processing %d cards", n_cards)

    # Process all cards in parallel
    results = [None] * n_cards

    def _process_card(idx, path):
        """Process a single card through the ref-matching pipeline."""
        try:
            return idx, identify_card_ref_matching(str(path), session=session)
        except Exception as e:
            logger.warning("Card %d ref-match failed: %s", idx, e, exc_info=True)
            # Fall back to cascade on any error
            try:
                return idx, identify_card(str(path), session=session)
            except Exception as e2:
                logger.error("Card %d cascade fallback also failed: %s", idx, e2)
                return idx, {
                    "card_id": None,
                    "confidence": 0.0,
                    "method": "ref_match_error",
                    "explanation": f"Both ref-match and cascade failed: {e}",
                    "raw_response": {},
                }

    with ThreadPoolExecutor(max_workers=min(n_cards, 6)) as pool:
        futures = [
            pool.submit(_process_card, i, path)
            for i, path in enumerate(card_image_paths)
        ]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    # Summary logging
    methods = [r.get("method", "?") for r in results if r]
    confidences = [r.get("confidence", 0) for r in results if r]
    ref_count = sum(1 for m in methods if m and m.startswith("ref_match"))
    fallback_count = len(methods) - ref_count
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    logger.info(
        "identify_page_ref_matching: %d/%d ref-matched, %d fallback, avg confidence=%.3f",
        ref_count, n_cards, fallback_count, avg_conf,
    )

    return results


# ---------------------------------------------------------------------------
# V2 Pipeline: color + name OCR + HP -> DB filter -> DINOv2 dot product
# ---------------------------------------------------------------------------

# Confidence thresholds for the v2 pipeline
_V2_DINO_ACCEPT_THRESHOLD = 0.40      # DINOv2 dot product vs filtered refs
_V2_CANDIDATE_DISAMBIGUATION_LIMIT = 3  # above this, also run attack OCR
_V2_FALLBACK_CONFIDENCE = 0.40         # below this, fall back to ensemble


def _run_color_detect(image_path: str) -> tuple:
    """Run color/type detection on a card image.

    Returns (type_name, confidence) or (None, 0.0) on failure.
    """
    try:
        from cardprice.ml.color_detector import detect_color_type
        predictions = detect_color_type(image_path, top_n=3)
        if predictions:
            return predictions[0][0], predictions[0][1]
    except Exception as e:
        logger.warning("v2 color_detect failed: %s", e)
    return None, 0.0


def _run_name_ocr(image_path: str) -> tuple:
    """Run OCR to extract the Pokemon name from a card image.

    Returns (cleaned_name, confidence, raw_text) or (None, 0.0, None).
    Uses thread lock since PaddleOCR/EasyOCR models are not thread-safe.
    """
    with _ocr_lock:
        try:
            from cardprice.ml.ocr_matcher import detect_pokemon_name
            name, conf = detect_pokemon_name(image_path)
            if name and len(name) >= 2:
                return name, conf, name
        except Exception as e:
            logger.warning("v2 name_ocr failed: %s", e)

        # Japanese OCR fallback: if English OCR found nothing,
        # try reading Japanese text and mapping to English name.
        try:
            jp_name = _try_japanese_ocr(image_path)
            if jp_name:
                return jp_name, 0.70, f"[JP]{jp_name}"
        except Exception as e:
            logger.debug("v2 japanese_ocr failed: %s", e)

    return None, 0.0, None


def _try_japanese_ocr(image_path: str) -> str | None:
    """Try Japanese OCR on the name region, return English name if found."""
    import json
    import re
    import cv2

    JP_CHAR_RE = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+')

    # Load JP→EN mapping
    jp_map_path = Path(__file__).resolve().parent.parent.parent / "data" / "jp_en_pokemon_names.json"
    if not jp_map_path.exists():
        return None
    with open(jp_map_path) as f:
        jp_en_map = json.load(f)

    # Crop name region (top 15% of card)
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    name_crop = img[0:int(h * 0.15), :]

    # Save temp crop for EasyOCR
    tmp_path = tempfile.mktemp(suffix='.png')
    try:
        cv2.imwrite(tmp_path, name_crop)

        # Try EasyOCR with Japanese (cached reader to avoid 5-10s init)
        import easyocr
        global _jp_easyocr_reader
        if _jp_easyocr_reader is None:
            _jp_easyocr_reader = easyocr.Reader(['ja', 'en'], gpu=False, verbose=False)
        reader = _jp_easyocr_reader
        results = reader.readtext(tmp_path, detail=1)

        for _bbox, text, conf in results:
            # Check for Japanese characters
            jp_matches = JP_CHAR_RE.findall(text)
            for jp_text in jp_matches:
                # Clean common OCR artifacts
                jp_clean = jp_text.rstrip('・。、')
                if jp_clean in jp_en_map:
                    en_name = jp_en_map[jp_clean]
                    logger.info("Japanese OCR: '%s' -> '%s' (conf=%.3f)", jp_clean, en_name, conf)
                    return en_name
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return None


def _run_hp_detect(image_path: str):
    """Run HP detection on a card image.

    Returns int HP value or None.
    """
    with _ocr_lock:
        try:
            from cardprice.ml.hp_detector import detect_hp
            return detect_hp(image_path)
        except Exception as e:
            logger.warning("v2 hp_detect failed: %s", e)
    return None


def _run_attack_ocr(image_path: str) -> list:
    """Run OCR to extract attack/move names from a card image.

    Returns list of attack name strings, or empty list on failure.
    """
    try:
        from cardprice.ml.attack_ocr import extract_attack_names
        candidates = extract_attack_names(image_path)
        # extract_attack_names returns [(text, confidence), ...]
        # Return just the text strings
        return [text for text, _conf in candidates if text]
    except Exception as e:
        logger.warning("v2 attack_ocr failed: %s", e)
    return []


_CARD_NAME_SUFFIXES = [
    " LV.X", " LV. X", " Lv.X",
    " VMAX", " VSTAR", " V-UNION",
    " V", " GX", " EX", " ex",
    "-GX", "-EX", "-ex",
]


def _strip_card_suffix(name: str) -> str | None:
    """Strip common Pokemon card suffixes (V, EX, LV.X etc.) from OCR name.

    Returns the base name without suffix, or None if no suffix found.
    """
    for suffix in _CARD_NAME_SUFFIXES:
        if name.endswith(suffix):
            base = name[: -len(suffix)].strip()
            if len(base) >= 2:
                return base
    return None


def _get_candidates_from_db(
    name: str,
    hp=None,
    card_type=None,
    session=None,
) -> list:
    """Query DB for candidate card_ids matching name/hp/type.

    Thin wrapper around ref_matcher.get_candidate_card_ids that also handles
    fuzzy name matching when exact match returns nothing.

    Returns list of card_id strings.
    """
    from cardprice.ml.ref_matcher import get_candidate_card_ids

    # First try exact name match
    candidates = get_candidate_card_ids(
        pokemon_name=name, hp=hp, card_type=card_type, session=session,
    )

    if candidates:
        return candidates

    # Try stripping card suffix (V, EX, LV.X, etc.) BEFORE relaxing HP.
    # "Flygon" + hp=120 is better than "Flygon LV.X" + hp=None.
    base_name = _strip_card_suffix(name)
    if base_name:
        logger.info("v2 DB lookup: stripping suffix %r -> %r", name, base_name)
        candidates = get_candidate_card_ids(
            pokemon_name=base_name, hp=hp, card_type=card_type, session=session,
        )
        if candidates:
            logger.info("v2 DB lookup: suffix-stripped -> %d candidates for %r",
                        len(candidates), base_name)
            return candidates

    # Exact match failed -- try without HP/type constraints
    if hp is not None or card_type is not None:
        candidates = get_candidate_card_ids(
            pokemon_name=name, hp=None, card_type=None, session=session,
        )
        if candidates:
            logger.info("v2 DB lookup: relaxed HP/type -> %d candidates for %r",
                        len(candidates), name)
            return candidates

    # Also try suffix-stripped base name without HP/type
    if base_name and (hp is not None or card_type is not None):
        candidates = get_candidate_card_ids(
            pokemon_name=base_name, hp=None, card_type=None, session=session,
        )
        if candidates:
            logger.info("v2 DB lookup: suffix-stripped + relaxed -> %d for %r",
                        len(candidates), base_name)
            return candidates

    # Still nothing -- try fuzzy name matching via ocr_matcher
    try:
        from cardprice.ml.ocr_matcher import fuzzy_match_card_name
        fuzzy_hits = fuzzy_match_card_name(name, top_k=20, score_cutoff=75.0)
        if fuzzy_hits:
            # fuzzy_match_card_name returns (card_id, name, set_id, score)
            # Get unique card_ids
            seen = set()
            fuzzy_cids = []
            for cid, _name, _sid, _score in fuzzy_hits:
                if cid not in seen:
                    seen.add(cid)
                    fuzzy_cids.append(cid)
            logger.info("v2 DB lookup: fuzzy match -> %d candidates for %r",
                        len(fuzzy_cids), name)
            return fuzzy_cids
    except Exception as e:
        logger.warning("v2 DB fuzzy lookup failed: %s", e)

    return []


def _filter_candidates_by_attacks(
    candidates: list,
    attack_names: list,
    session=None,
) -> list:
    """Filter candidate card_ids by attack name matching.

    Uses the attack index (data/attack_index.pkl) to check which candidates
    have attacks matching the OCR-detected attack names.

    Returns filtered list of card_ids (subset of input). If filtering would
    eliminate all candidates, returns the original list unchanged.
    """
    if not attack_names or not candidates:
        return candidates

    try:
        from cardprice.ml.attack_ocr import _load_attack_index
        idx = _load_attack_index()
        card_to_attacks = idx.get("card_to_attacks", {})
        atk_to_cards = idx.get("attack_to_cards", {})

        if not card_to_attacks and not atk_to_cards:
            logger.info("v2 attack filter: no attack index available")
            return candidates

        # Strategy: find candidates whose attacks overlap with detected attacks
        detected_lower = {a.lower().strip() for a in attack_names if a}

        scored = []
        for cid in candidates:
            # Try full card_id first (attack index keys include variant),
            # then fall back to base card_id (without variant) for compat
            card_attacks = card_to_attacks.get(cid, [])
            if not card_attacks:
                base_cid = cid.split("/")[0] if "/" in cid else cid
                card_attacks = card_to_attacks.get(base_cid, [])
            card_attacks_lower = {a.lower() for a in card_attacks}

            # Count how many detected attacks match this card's attacks
            # Use fuzzy matching for OCR noise tolerance
            matches = 0
            try:
                from rapidfuzz import fuzz
                for det_atk in detected_lower:
                    for card_atk in card_attacks_lower:
                        if fuzz.ratio(det_atk, card_atk) >= 75:
                            matches += 1
                            break
            except ImportError:
                # Fall back to exact matching
                matches = len(detected_lower & card_attacks_lower)

            if matches > 0:
                scored.append((cid, matches))

        if scored:
            # Sort by number of matching attacks (descending)
            scored.sort(key=lambda x: x[1], reverse=True)
            filtered = [cid for cid, _score in scored]
            logger.info(
                "v2 attack filter: %d/%d candidates have matching attacks "
                "(detected: %s)",
                len(filtered), len(candidates), list(detected_lower),
            )
            return filtered

        logger.info("v2 attack filter: no candidates matched attacks, keeping all %d",
                     len(candidates))
    except Exception as e:
        logger.warning("v2 attack filter failed: %s", e)

    return candidates


def _dino_dot_product_against_refs(
    image_path: str,
    candidate_card_ids: list,
) -> list:
    """Compute DINOv2 dot product between query image and reference images.

    For each candidate card_id, looks up the reference image, computes the
    DINOv2 embedding similarity (cosine = dot product of L2-normalized vectors),
    and returns results sorted by similarity.

    Returns list of (card_id, similarity_score) sorted descending.
    """
    from cardprice.ml.ref_matcher import (
        get_reference_image_path,
        compute_embedding_similarity,
    )

    # Resolve reference images for each candidate
    ref_paths = []
    ref_card_ids = []
    for cid in candidate_card_ids:
        ref_path = get_reference_image_path(cid)
        if ref_path is not None:
            ref_paths.append(ref_path)
            ref_card_ids.append(cid)

    if not ref_paths:
        logger.info("v2 DINOv2: no reference images found for %d candidates",
                     len(candidate_card_ids))
        return []

    # Preprocess query image for DINOv2 (CLAHE + glare removal)
    query_path = image_path
    preproc_tmp = None
    try:
        from cardprice.ml.preprocess import preprocess_for_matching
        preproc_tmp = preprocess_for_matching(image_path)
        query_path = preproc_tmp
    except Exception:
        pass

    try:
        similarities = compute_embedding_similarity(
            query_path, ref_paths, ref_card_ids,
        )

        # Pair up and sort
        results = list(zip(ref_card_ids, similarities))
        results.sort(key=lambda x: x[1], reverse=True)

        if results:
            logger.info(
                "v2 DINOv2: top match %s (%.4f) out of %d refs",
                results[0][0], results[0][1], len(results),
            )
            for i, (cid, sim) in enumerate(results[:3]):
                logger.debug("  #%d: %s (%.4f)", i + 1, cid, sim)

        return results
    finally:
        if preproc_tmp:
            try:
                os.unlink(preproc_tmp)
            except OSError:
                pass


def _score_candidates_combined(
    image_path: str,
    candidate_card_ids: list,
) -> list[tuple[str, float, dict]]:
    """Score candidates using both DINOv2 visual similarity and attack OCR overlap.

    Combined score = w_dino * dino_score + w_attack * attack_score
    When attack OCR finds nothing, falls back to pure DINOv2.
    """
    from cardprice.ml.attack_ocr import extract_attack_names, _load_attack_index

    dino_results = _dino_dot_product_against_refs(image_path, candidate_card_ids)
    if not dino_results:
        return []
    dino_scores = {cid: score for cid, score in dino_results}

    # Attack OCR
    ocr_candidates = []
    try:
        ocr_candidates = extract_attack_names(image_path)
    except Exception as e:
        logger.warning("v2 combined: attack OCR failed: %s", e)

    detected_attacks = [text.lower().strip() for text, _conf in ocr_candidates if text]

    idx = _load_attack_index()
    card_to_attacks = idx.get("card_to_attacks", {})

    try:
        from rapidfuzz import fuzz
        use_rapidfuzz = True
    except ImportError:
        use_rapidfuzz = False

    attack_scores = {}
    attack_details = {}
    for cid in candidate_card_ids:
        card_attacks = card_to_attacks.get(cid, [])
        if not card_attacks:
            base_cid = cid.split("/")[0] if "/" in cid else cid
            card_attacks = card_to_attacks.get(base_cid, [])

        if not card_attacks or not detected_attacks:
            attack_scores[cid] = 0.0
            attack_details[cid] = []
            continue

        card_attacks_lower = [a.lower() for a in card_attacks]
        matched = []
        for card_atk in card_attacks_lower:
            for det_atk in detected_attacks:
                # Require higher threshold for short strings to avoid
                # spurious matches like "Whap" → "Wrap"
                min_len = min(len(det_atk), len(card_atk))
                threshold = 80 if min_len <= 5 else 70
                if use_rapidfuzz:
                    if fuzz.ratio(det_atk, card_atk) >= threshold:
                        matched.append(card_atk)
                        break
                else:
                    from difflib import SequenceMatcher
                    t = 0.75 if min_len <= 5 else 0.65
                    if SequenceMatcher(None, det_atk, card_atk).ratio() >= t:
                        matched.append(card_atk)
                        break

        # Score rewards both proportion AND absolute count
        proportion = len(matched) / len(card_attacks_lower)
        count_bonus = 0.1 * min(len(matched), 3)
        attack_scores[cid] = proportion + count_bonus
        attack_details[cid] = matched

    # Dynamic weights
    any_attacks = any(s > 0 for s in attack_scores.values())
    w_dino, w_attack = (0.5, 0.5) if any_attacks else (1.0, 0.0)

    results = []
    for cid in candidate_card_ids:
        d = dino_scores.get(cid, 0.0)
        a = attack_scores.get(cid, 0.0)
        combined = w_dino * d + w_attack * a
        results.append((cid, combined, {
            "dino_score": round(d, 4),
            "attack_score": round(a, 4),
            "matched_attacks": attack_details.get(cid, []),
        }))

    results.sort(key=lambda x: x[1], reverse=True)
    if results:
        t = results[0]
        logger.info("v2 combined: top=%s score=%.4f (dino=%.4f, atk=%.4f) %d candidates",
                     t[0], t[1], t[2]["dino_score"], t[2]["attack_score"], len(results))
    return results


def identify_card_v2(image_path, session=None, page_era=None):
    """V2 card identification: color + name OCR + HP -> DB filter -> DINOv2.

    This pipeline is fundamentally different from v1 (cascade/ensemble):
    instead of searching the entire 20k-card FAISS index, it uses cheap
    classifiers to narrow candidates to 2-20, then does precise DINOv2
    dot product against only those reference images.

    Pipeline:
        1. Run in parallel: color_detect + name_ocr + hp_detect
        2. Query DB: get candidates matching (name, hp, type)
        3. If candidates <= 3: DINOv2 dot product, return best
        4. If candidates > 3: also run attack_ocr, filter by attack match,
           then DINOv2
        5. If no candidates (OCR failed): fall back to ensemble method

    Args:
        image_path: Path to the card image.
        session: Optional SQLAlchemy DB session.
        page_era: Optional era string (e.g. "ex", "e-card") from page context.
            Used to filter attack fallback candidates.

    Returns:
        Dict with keys: card_id, confidence, method, explanation, raw_response.
    """
    image_path = str(image_path)

    # Convert HEIC/HEIF if needed
    try:
        from cardprice.utils.image_convert import ensure_compatible
        image_path = ensure_compatible(image_path)
    except Exception as e:
        logger.warning("v2: image conversion failed, using original: %s", e)

    # Check cache
    try:
        file_hash = hashlib.md5(Path(image_path).read_bytes()).hexdigest()
        cache_key = f"v2_{file_hash}"
        if cache_key in _scan_cache:
            logger.info("v2: cache HIT for %s", image_path)
            _scan_cache.move_to_end(cache_key)
            return _scan_cache[cache_key]
    except Exception:
        file_hash = None
        cache_key = None

    # -----------------------------------------------------------------------
    # Step 1: Run cheap classifiers in parallel
    # -----------------------------------------------------------------------
    color_type = None
    color_conf = 0.0
    ocr_name = None
    ocr_conf = 0.0
    ocr_raw = None
    hp_value = None

    with ThreadPoolExecutor(max_workers=3) as pool:
        color_future = pool.submit(_run_color_detect, image_path)
        name_future = pool.submit(_run_name_ocr, image_path)
        hp_future = pool.submit(_run_hp_detect, image_path)

        try:
            color_type, color_conf = color_future.result(timeout=10)
        except Exception as e:
            logger.warning("v2 step1: color_detect error: %s", e)

        try:
            ocr_name, ocr_conf, ocr_raw = name_future.result(timeout=30)
        except Exception as e:
            logger.warning("v2 step1: name_ocr error: %s", e)

        try:
            hp_value = hp_future.result(timeout=10)
        except Exception as e:
            logger.warning("v2 step1: hp_detect error: %s", e)

    # Reject partial OCR names (< 3 chars) — they create bad candidate sets.
    # e.g. "tty" for Skitty, "ch" for Trapinch match wrong cards.
    if ocr_name and len(ocr_name) < 3:
        logger.info("v2: rejecting short OCR name %r (len=%d)", ocr_name, len(ocr_name))
        ocr_name = None
        ocr_conf = 0.0

    logger.info(
        "v2 step1: name=%r (conf=%.2f), hp=%s, color=%s (conf=%.2f)",
        ocr_name, ocr_conf, hp_value, color_type, color_conf,
    )

    # -----------------------------------------------------------------------
    # Step 2: Query DB for candidates
    # -----------------------------------------------------------------------
    # Only use color_type if confidence is reasonable and it's not Colorless
    # (Colorless is the fallback/default and too broad to be useful)
    use_type = None
    if color_type and color_conf >= 0.40 and color_type != "Colorless":
        use_type = color_type

    candidates = []
    if ocr_name:
        candidates = _get_candidates_from_db(
            name=ocr_name,
            hp=hp_value,
            card_type=use_type,
            session=session,
        )
        logger.info(
            "v2 step2: %d candidates for name=%r, hp=%s, type=%s",
            len(candidates), ocr_name, hp_value, use_type,
        )

    # -----------------------------------------------------------------------
    # Step 3/4: Combined DINOv2 + attack scoring for candidate disambiguation
    # -----------------------------------------------------------------------
    ref_match_result = None  # Low-confidence ref match saved for comparison
    if candidates:
        # Single candidate: quick DINOv2 sanity check
        if len(candidates) == 1:
            only_cid = candidates[0]
            dino_check = _dino_dot_product_against_refs(image_path, [only_cid])
            dino_score = dino_check[0][1] if dino_check else 0.0

            if dino_score >= 0.30:
                explanation = (
                    f"v2: single candidate match: name OCR={ocr_name!r}, hp={hp_value}, "
                    f"type={color_type} -> {only_cid} (dino={dino_score:.3f})"
                )
                result = {
                    "card_id": only_cid,
                    "confidence": max(ocr_conf, 0.70),
                    "method": "v2_single_candidate",
                    "explanation": explanation,
                    "raw_response": {
                        "ocr_name": ocr_name, "ocr_confidence": ocr_conf,
                        "hp": hp_value, "color_type": color_type,
                        "color_confidence": color_conf, "n_candidates": 1,
                        "dino_sanity": dino_score,
                    },
                }
                _cache_store(cache_key, result)
                return result
            else:
                logger.warning("v2: REJECTED single candidate %s — DINOv2 %.3f too low",
                               only_cid, dino_score)

        # Multiple candidates: combined DINOv2 + attack scoring
        elif len(candidates) >= 2:
            combined_results = _score_candidates_combined(image_path, candidates)
            if combined_results:
                best_cid, best_score, best_detail = combined_results[0]
                alt_list = [(cid, score) for cid, score, _ in combined_results[1:4]]
                alt_str = ", ".join(f"{cid} ({s:.0%})" for cid, s in alt_list)

                n_cand = len(candidates)
                if n_cand <= 3:
                    effective_threshold = 0.35
                elif n_cand <= 10:
                    effective_threshold = 0.45
                else:
                    effective_threshold = 0.50

                if best_score >= effective_threshold:
                    attack_names = best_detail.get("matched_attacks", [])
                    explanation = (
                        f"v2: name OCR={ocr_name!r}, hp={hp_value}, "
                        f"type={color_type}, {n_cand} candidates, "
                        f"combined={best_score:.3f} (dino={best_detail['dino_score']:.3f}, "
                        f"atk={best_detail['attack_score']:.3f})"
                    )
                    if attack_names:
                        explanation += f", attacks={attack_names}"
                    if alt_str:
                        explanation += f". Alts: {alt_str}"

                    ref_match_result = {
                        "card_id": best_cid,
                        "confidence": float(best_score),
                        "method": "v2_ref_match",
                        "explanation": explanation,
                        "raw_response": {
                            "ocr_name": ocr_name, "ocr_confidence": ocr_conf,
                            "ocr_raw": ocr_raw, "hp": hp_value,
                            "color_type": color_type, "color_confidence": color_conf,
                            "attack_names": attack_names, "n_candidates": n_cand,
                            "combined_results": [(c, round(s, 4), d) for c, s, d in combined_results[:5]],
                        },
                    }
                    # If DINOv2 score is low (< 0.60), OCR name might be wrong.
                    # Save result but don't return yet — also try attack path.
                    if best_detail['dino_score'] < 0.60:
                        logger.info("v2: ref_match dino=%.3f < 0.60, will also try attack path",
                                    best_detail['dino_score'])
                    else:
                        _cache_store(cache_key, ref_match_result)
                        return ref_match_result
                else:
                    logger.info("v2: best combined %.4f < %.2f, falling to ensemble",
                                best_score, effective_threshold)

    # -----------------------------------------------------------------------
    # Step 5: Attack-based identification
    # Try attack OCR when: (a) name OCR failed entirely, or (b) the OCR-based
    # candidate match scored low (< 0.60), suggesting the OCR name may be wrong.
    # Attack OCR has 92% recall — much more reliable than DINOv2 global search.
    # -----------------------------------------------------------------------
    attack_result = None
    if not ocr_name or ref_match_result is not None:
        logger.info("v2 step5: no OCR name, trying attack-based identification")
        try:
            from cardprice.ml.attack_ocr import identify_by_attacks
            atk_results = identify_by_attacks(image_path)
            if atk_results:
                atk_candidate_ids = [cid for cid, _s in atk_results[:50]]

                # Era filtering: if page_era is known, prefer candidates
                # from the same era. Keep era-matched candidates first,
                # but fall back to all candidates if too few match.
                if page_era:
                    from cardprice.ml.page_context import _era_for_set, _extract_set_id
                    era_matched = [
                        cid for cid in atk_candidate_ids
                        if _era_for_set(_extract_set_id(cid)) == page_era
                    ]
                    # Use era filter when: enough candidates OR many total
                    # candidates (indistinguishable by DINOv2).
                    if era_matched and (len(era_matched) >= 3 or len(atk_candidate_ids) >= 20):
                        logger.info("v2 step5: era filter %s: %d/%d candidates",
                                    page_era, len(era_matched), len(atk_candidate_ids))
                        atk_candidate_ids = era_matched

                era_filtered = page_era and len(atk_candidate_ids) < 50
                combined_results = _score_candidates_combined(image_path, atk_candidate_ids)
                if combined_results:
                    best_cid, best_score, best_detail = combined_results[0]
                    # Boost confidence when era filtering significantly reduced
                    # the candidate set (strong prior from page context)
                    if era_filtered:
                        best_score = min(best_score + 0.10, 1.0)
                    # Penalize when many candidates share the same attacks
                    # (low discrimination — DINOv2 picks randomly among 50 Rattatas)
                    if len(atk_candidate_ids) >= 30 and len(combined_results) >= 2:
                        score_gap = combined_results[0][1] - combined_results[1][1]
                        if score_gap < 0.05:
                            best_score *= 0.85  # moderate penalty for low discrimination
                            logger.info("v2 step5: %d candidates, gap=%.3f -> penalty to %.3f",
                                        len(atk_candidate_ids), score_gap, best_score)
                    if best_score >= 0.35:
                        alt_list = [(cid, score) for cid, score, _ in combined_results[1:4]]
                        alt_str = ", ".join(f"{cid} ({s:.0%})" for cid, s in alt_list)
                        attack_names = best_detail.get("matched_attacks", [])
                        explanation = (
                            f"v2: attack OCR -> {len(atk_candidate_ids)} candidates, "
                            f"combined={best_score:.3f} (dino={best_detail['dino_score']:.3f}, "
                            f"atk={best_detail['attack_score']:.3f})"
                        )
                        if attack_names:
                            explanation += f", attacks={attack_names}"
                        if alt_str:
                            explanation += f". Alts: {alt_str}"

                        attack_result = {
                            "card_id": best_cid,
                            "confidence": float(best_score),
                            "method": "v2_attack_fallback",
                            "explanation": explanation,
                            "raw_response": {
                                "ocr_name": ocr_name, "hp": hp_value,
                                "color_type": color_type,
                                "attack_candidates": atk_results[:5],
                                "combined_results": [(c, round(s, 4), d)
                                                     for c, s, d in combined_results[:5]],
                            },
                        }
                        logger.info("v2: attack fallback -> %s (combined=%.3f)",
                                    best_cid, best_score)
        except Exception as e:
            logger.warning("v2 step5: attack fallback failed: %s", e)

    # -----------------------------------------------------------------------
    # Step 6: Ensemble fallback (last resort)
    # -----------------------------------------------------------------------
    logger.info(
        "v2 step6: running ensemble (ocr_name=%r, candidates=%d)",
        ocr_name, len(candidates),
    )
    fallback = identify_card_ensemble(image_path, session=session)
    fallback_conf = fallback.get("confidence", 0.0)

    # Pick best among ref_match_result (if pending), attack_result, and ensemble.
    # When page_era is known, give era-matched results a 0.10 bonus so they
    # beat ensemble results from wrong eras.
    best_alt = None
    best_alt_conf = fallback_conf
    if page_era:
        from cardprice.ml.page_context import _era_for_set, _extract_set_id
        fallback_cid = fallback.get("card_id", "")
        fallback_era = _era_for_set(_extract_set_id(fallback_cid)) if fallback_cid else None
        if fallback_era != page_era:
            best_alt_conf -= 0.10  # penalize wrong-era ensemble result
    for candidate in [ref_match_result, attack_result]:
        if candidate and candidate["confidence"] > best_alt_conf:
            best_alt = candidate
            best_alt_conf = candidate["confidence"]
    if best_alt:
        logger.info("v2: %s (%.3f) > ensemble (%.3f)",
                     best_alt["method"], best_alt_conf, fallback_conf)
        _cache_store(cache_key, best_alt)
        return best_alt

    fallback["method"] = f"v2_fallback({fallback.get('method', 'ensemble')})"
    fallback["explanation"] = (
        f"v2 fallback: OCR name={ocr_name!r} yielded {len(candidates)} candidates "
        f"but DINOv2 ref-match was insufficient. "
        + (fallback.get("explanation") or "")
    )
    # Preserve v2 signal info in raw_response
    raw = fallback.get("raw_response", {})
    raw["v2_signals"] = {
        "ocr_name": ocr_name,
        "ocr_confidence": ocr_conf,
        "hp": hp_value,
        "color_type": color_type,
        "color_confidence": color_conf,
        "n_candidates": len(candidates),
    }
    fallback["raw_response"] = raw

    _cache_store(cache_key, fallback)
    return fallback


def identify_page_v2(card_image_paths, session=None):
    """V2 page identification: runs identify_card_v2 on each card, then
    applies page context reranking for low-confidence results.

    Pipeline:
        1. Run identify_card_v2 for each card in parallel.
        2. Build page context from high-confidence results (set/era inference).
        3. For low-confidence cards, re-run with page context boosting
           candidates from the inferred set.

    Args:
        card_image_paths: List of paths to individual card segment images.
        session: Optional SQLAlchemy DB session.

    Returns:
        List of result dicts (same format as identify_card_v2), one per card.
    """
    from cardprice.ml.page_context import identify_page_context

    if not card_image_paths:
        return []

    n_cards = len(card_image_paths)
    logger.info("identify_page_v2: processing %d cards", n_cards)

    # -----------------------------------------------------------------------
    # Pass 1: Run identify_card_v2 sequentially (PaddleOCR is not thread-safe)
    # -----------------------------------------------------------------------
    results = [None] * n_cards

    for i, path in enumerate(card_image_paths):
        try:
            results[i] = identify_card_v2(str(path), session=session)
        except Exception as e:
            logger.warning("identify_page_v2: card %d failed: %s", i, e)
            results[i] = {
                "card_id": None, "confidence": 0.0,
                "method": "v2_error",
                "explanation": f"identify_card_v2 failed: {e}",
                "raw_response": {},
            }

    # -----------------------------------------------------------------------
    # Pass 2: Page context reranking
    # -----------------------------------------------------------------------
    RERUN_THRESHOLD = 0.65  # re-run cards below this confidence

    ctx = identify_page_context(results)
    logger.info(
        "identify_page_v2: page context: sets=%s, era=%s, confidence=%.2f",
        ctx.get("likely_sets", [])[:3], ctx.get("era"), ctx.get("confidence", 0),
    )

    # Only apply page context if it's strong enough
    if not ctx.get("likely_sets") or ctx.get("confidence", 0) < 0.50:
        logger.info("identify_page_v2: page context too weak, skipping pass 2")
        return results

    ctx_sets = set(ctx.get("likely_sets", []))

    for i, (path, result) in enumerate(zip(card_image_paths, results)):
        if result["confidence"] >= RERUN_THRESHOLD:
            continue

        # Build leave-one-out context (exclude current card)
        loo_results = results[:i] + results[i + 1:]
        loo_ctx = identify_page_context(loo_results)
        if not loo_ctx.get("likely_sets") or loo_ctx.get("confidence", 0) < 0.40:
            continue

        loo_sets = set(loo_ctx.get("likely_sets", []))

        logger.info(
            "identify_page_v2 pass2: re-examining card %d (conf=%.2f, method=%s) "
            "with page context sets=%s",
            i, result["confidence"], result.get("method"), list(loo_sets)[:3],
        )

        # Strategy: if v2 found candidates via OCR, check if any are in the
        # page context set and re-score with a set bonus.
        raw = result.get("raw_response", {})
        dino_ref_results = raw.get("dino_ref_results", [])
        v2_signals = raw.get("v2_signals", {})

        # Check if the current best is already from the page's set
        current_set = _extract_set_from_card_id(result.get("card_id"))
        if current_set in loo_sets:
            # Already in the right set, just boost confidence slightly
            result["confidence"] = min(result["confidence"] + 0.05, 1.0)
            result["explanation"] = (
                (result.get("explanation") or "") + " (page context confirms set)"
            )
            continue

        # Look through DINOv2 ref results for a candidate from the page set
        reranked = False
        for cid, score in dino_ref_results:
            cand_set = _extract_set_from_card_id(cid)
            if cand_set in loo_sets and score >= _V2_FALLBACK_CONFIDENCE:
                # Found a candidate from the page's set with acceptable score
                old_cid = result.get("card_id")
                result["card_id"] = cid
                result["confidence"] = float(score) + 0.10  # page context bonus
                result["method"] = "v2_page_context"
                result["explanation"] = (
                    f"v2 page context rerank: {old_cid} -> {cid} "
                    f"(set {cand_set} matches page context, score={score:.3f}+0.10)"
                )
                reranked = True
                logger.info(
                    "identify_page_v2 pass2: card %d reranked %s -> %s (page context)",
                    i, old_cid, cid,
                )
                break

        if not reranked:
            # Try re-running with the page context's set as a strong prior.
            # Re-query DB with the name (if we had one) filtered to page context sets.
            ocr_name = v2_signals.get("ocr_name") or raw.get("ocr_name")
            if ocr_name:
                # Get candidates specifically from page context sets
                all_candidates = _get_candidates_from_db(
                    name=ocr_name, hp=v2_signals.get("hp"),
                    card_type=None, session=session,
                )
                # Filter to page context sets
                set_filtered = [
                    cid for cid in all_candidates
                    if _extract_set_from_card_id(cid) in loo_sets
                ]
                if set_filtered:
                    dino_set_results = _dino_dot_product_against_refs(
                        str(path), set_filtered,
                    )
                    if dino_set_results and dino_set_results[0][1] >= _V2_FALLBACK_CONFIDENCE:
                        best_cid, best_score = dino_set_results[0]
                        old_cid = result.get("card_id")
                        result["card_id"] = best_cid
                        result["confidence"] = float(best_score) + 0.10
                        result["method"] = "v2_page_context_requery"
                        result["explanation"] = (
                            f"v2 page context requery: {old_cid} -> {best_cid} "
                            f"(filtered to sets {list(loo_sets)[:3]}, "
                            f"score={best_score:.3f}+0.10)"
                        )
                        logger.info(
                            "identify_page_v2 pass2: card %d requeried %s -> %s",
                            i, old_cid, best_cid,
                        )

    # -----------------------------------------------------------------------
    # Pass 3: Re-run fallback cards with era context
    # Only re-run cards that used attack_fallback or ensemble_fallback
    # AND have low confidence. Don't touch high-confidence results.
    # -----------------------------------------------------------------------
    from cardprice.ml.page_context import _era_for_set, _extract_set_id
    page_era = ctx.get("era")
    if page_era and ctx.get("confidence", 0) >= 0.50:
        for i, (path, result) in enumerate(zip(card_image_paths, results)):
            method = result.get("method", "")
            # Only re-run fallback/low-quality results with low confidence
            if "fallback" not in method and "page_context" not in method:
                continue
            if result["confidence"] >= 0.80:
                continue

            # Build leave-one-out era context
            loo_results = results[:i] + results[i + 1:]
            loo_ctx = identify_page_context(loo_results)
            loo_era = loo_ctx.get("era")
            if not loo_era or loo_ctx.get("confidence", 0) < 0.40:
                continue

            logger.info(
                "identify_page_v2 pass3: re-running card %d (method=%s, conf=%.2f) "
                "with page_era=%s",
                i, method, result["confidence"], loo_era,
            )
            _scan_cache.clear()  # force re-run without cache
            rerun = identify_card_v2(str(path), session=session, page_era=loo_era)

            # Accept re-run if: (a) confidence improved, OR (b) the re-run
            # result is from the correct era and original wasn't.
            rerun_era = _era_for_set(_extract_set_id(rerun.get("card_id", ""))) if rerun.get("card_id") else None
            orig_era = _era_for_set(_extract_set_id(result.get("card_id", ""))) if result.get("card_id") else None
            era_improved = rerun_era == loo_era and orig_era != loo_era
            conf_improved = rerun.get("confidence", 0) > result["confidence"]
            if conf_improved or (era_improved and rerun.get("confidence", 0) >= 0.40):
                old_cid = result.get("card_id")
                results[i] = rerun
                rerun["explanation"] = (
                    (rerun.get("explanation") or "")
                    + f" (pass3: era={loo_era}, was {old_cid})"
                )
                logger.info(
                    "identify_page_v2 pass3: card %d improved %s -> %s (era=%s)",
                    i, old_cid, rerun.get("card_id"), loo_era,
                )

    # Summary logging
    methods = [r.get("method", "?") for r in results if r]
    confidences = [r.get("confidence", 0) for r in results if r]
    v2_count = sum(1 for m in methods if m and m.startswith("v2"))
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    logger.info(
        "identify_page_v2: %d/%d v2-matched, avg confidence=%.3f",
        v2_count, n_cards, avg_conf,
    )

    return results


def _extract_set_from_card_id(card_id) -> str:
    """Extract set ID from a card_id like 'base1-4/normal' -> 'base1'.

    Returns empty string if card_id is None or malformed.
    """
    if not card_id:
        return ""
    base = card_id.split("/")[0]  # "base1-4"
    parts = base.rsplit("-", 1)
    return parts[0] if len(parts) == 2 else base
