"""Fast OCR-only card identification for close-range scans.

At close range (~15cm), card text is large and clear. This module
provides a fast identification path that uses ONLY OCR signals
(name, HP, card number) without DINOv2 visual matching.

The key insight: name + HP + card_number is unique for 94.7% of cards
in the database. Even name + HP alone resolves 13.5% of cards to a
single candidate. For the remaining cases, we fall back to the full
identify_card_v2 pipeline.

Timing target: <500ms per card for unique OCR matches (vs 3-5s full pipeline).

Usage:
    from cardprice.ml.fast_identify import fast_identify
    result = fast_identify("path/to/card.png")
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded lookup table: {(name_lower, hp_int): [card_id, ...]}
_name_hp_index: dict[tuple[str, int | None], list[str]] | None = None

# Lazy-loaded lookup table: {card_number_stripped: [card_id, ...]}
_card_number_index: dict[str, list[str]] | None = None


def _build_name_hp_index(session=None):
    """Build the in-memory name+HP -> card_ids index from the database.

    This is loaded once and cached. The index maps (lowercase_name, hp)
    tuples to lists of card_ids. HP=None is stored as a separate key
    for cards without HP (trainers, energy).
    """
    global _name_hp_index
    if _name_hp_index is not None:
        return _name_hp_index

    own_session = session is None
    if own_session:
        from cardprice.db.session import SessionLocal
        session = SessionLocal()

    try:
        from sqlalchemy import text as sa_text
        rows = session.execute(
            sa_text("SELECT card_id, LOWER(name), hp FROM dim_cards")
        ).fetchall()

        _name_hp_index = {}
        for card_id, name_lower, hp in rows:
            key = (name_lower, int(hp) if hp is not None else None)
            _name_hp_index.setdefault(key, []).append(card_id)

            # Also index by base name (without suffix like " ex", " V", etc.)
            # so OCR that misses the suffix still works
            base = re.sub(
                r'\s+(ex|gx|v|vstar|vmax|lv\.\w+|δ|delta|star)$',
                '', name_lower, flags=re.IGNORECASE,
            ).strip()
            if base != name_lower:
                base_key = (base, int(hp) if hp is not None else None)
                _name_hp_index.setdefault(base_key, []).append(card_id)

        logger.info("Built name+HP index: %d entries from %d cards", len(_name_hp_index), len(rows))
    finally:
        if own_session and session:
            session.close()

    return _name_hp_index


def _build_card_number_index(session=None):
    """Build the in-memory card_number -> card_ids index."""
    global _card_number_index
    if _card_number_index is not None:
        return _card_number_index

    own_session = session is None
    if own_session:
        from cardprice.db.session import SessionLocal
        session = SessionLocal()

    try:
        from sqlalchemy import text as sa_text
        rows = session.execute(
            sa_text("SELECT card_id, card_number FROM dim_cards WHERE card_number IS NOT NULL")
        ).fetchall()

        _card_number_index = {}
        for card_id, card_number in rows:
            # Strip leading zeros for matching
            num_stripped = str(card_number).lstrip('0') or '0'
            _card_number_index.setdefault(num_stripped, []).append(card_id)

        logger.info("Built card_number index: %d entries", len(_card_number_index))
    finally:
        if own_session and session:
            session.close()

    return _card_number_index


def _ocr_name_hp_cardnum(image_path: str, skip_card_number: bool = False) -> dict:
    """Run OCR on a card image to extract name, HP, and optionally card number.

    Performs targeted OCR passes:
    1. Top 30% of card: name + HP (single RapidOCR pass)
    2. Bottom 15% of card: card number (e.g. "205/165") -- skipped if skip_card_number=True

    Returns dict with keys: ocr_name, ocr_conf, hp_value, card_number, set_total, timings.
    """
    import cv2
    import re
    from cardprice.ml.ocr_matcher import get_rapid_engine

    timings = {}
    result = {
        "ocr_name": None,
        "ocr_conf": 0.0,
        "hp_value": None,
        "card_number": None,
        "set_total": None,
        "timings": timings,
    }

    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("fast_identify: could not read image: %s", image_path)
        return result

    h, w = img.shape[:2]
    engine = get_rapid_engine()

    # --- Pass 1: Name + HP from top 30% ---
    t0 = time.perf_counter()

    # Crop top portion and upscale 2x
    top_crop = img[0:int(h * 0.30), :]
    top_up = cv2.resize(top_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Unsharp mask for sharper text
    blur = cv2.GaussianBlur(top_up, (0, 0), 2.0)
    top_up = cv2.addWeighted(top_up, 1.5, blur, -0.5, 0)

    try:
        ocr_result, _ = engine(top_up)
    except Exception as e:
        logger.warning("fast_identify: top OCR failed: %s", e)
        ocr_result = None

    timings["name_hp_ocr_ms"] = (time.perf_counter() - t0) * 1000

    if ocr_result:
        # Split detections into name candidates (left/center) and HP candidates (right)
        top_h, top_w = top_up.shape[:2]
        name_fragments = []
        hp_fragments = []

        for box, text, conf in ocr_result:
            text = text.strip()
            if not text:
                continue
            # Compute x center as fraction of width
            xs = [pt[0] for pt in box]
            x_center = sum(xs) / len(xs) / top_w

            # HP is usually on the right side (x > 0.5) and contains digits
            if x_center > 0.50 and re.search(r'\d', text):
                hp_fragments.append((text, float(conf)))
            else:
                name_fragments.append((text, float(conf)))

        # Extract HP
        if hp_fragments:
            from cardprice.ml.hp_detector import _parse_hp_from_texts
            result["hp_value"] = _parse_hp_from_texts(hp_fragments)

        # Match name against known Pokemon names
        if name_fragments:
            from cardprice.ml.ocr_matcher import _load_unique_pokemon_names
            from rapidfuzz import fuzz, process

            known_names = _load_unique_pokemon_names()
            known_names_lower = list(set(n.lower() for n in known_names))

            # Filter to name-like fragments (skip "BASIC", "STAGE 1", etc.)
            _NON_NAME = {"basic", "stage", "stage 1", "stage 2", "trainer",
                         "supporter", "pokemon", "item", "energy", "stadium",
                         "tool", "hp"}

            best_name = None
            best_conf = 0.0
            best_raw = None

            for raw_text, conf in name_fragments:
                cleaned = raw_text.strip()
                if cleaned.lower() in _NON_NAME:
                    continue
                if len(cleaned) < 2:
                    continue

                # Clean OCR artifacts
                cleaned = re.sub(r'^(BASIC|Stage\s*[12I])\s+', '', cleaned, flags=re.IGNORECASE).strip()

                # Try fuzzy matching
                match = process.extractOne(
                    cleaned.lower(),
                    known_names_lower,
                    scorer=fuzz.ratio,
                    score_cutoff=60.0,
                )

                if match:
                    matched_name, score, _ = match
                    # Find the properly-cased version
                    for kn in known_names:
                        if kn.lower() == matched_name:
                            matched_name = kn
                            break
                    norm_conf = score / 100.0
                    if norm_conf > best_conf:
                        best_conf = norm_conf
                        best_name = matched_name
                        best_raw = raw_text

            if best_name:
                result["ocr_name"] = best_name
                result["ocr_conf"] = best_conf
                result["ocr_raw"] = best_raw

    # --- Pass 2: Card number from bottom 15% (optional) ---
    if not skip_card_number:
        t1 = time.perf_counter()

        bottom_crop = img[int(h * 0.85):, :]
        bottom_up = cv2.resize(bottom_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        try:
            ocr_bottom, _ = engine(bottom_up)
        except Exception as e:
            logger.warning("fast_identify: bottom OCR failed: %s", e)
            ocr_bottom = None

        timings["card_num_ocr_ms"] = (time.perf_counter() - t1) * 1000

        if ocr_bottom:
            for box, text, conf in ocr_bottom:
                m = re.search(r'(\d{1,4})\s*/\s*(\d{1,4})', text)
                if m:
                    result["card_number"] = m.group(1).lstrip('0') or '0'
                    result["set_total"] = m.group(2).lstrip('0') or '0'
                    break

    return result


def fast_identify(image_path, session=None, detect_variants=True):
    """Fast identification using OCR only, no DINOv2.

    For close-range scans where text is clear. Returns same format as
    identify_card_v2 but much faster for unique matches.

    Pipeline:
    1. Run RapidOCR on top 30% of card (name + HP area)
    2. Fuzzy match name against known Pokemon names
    3. If name + HP gives unique card -> return immediately (no DINOv2)
    4. If 2+ candidates -> try card number OCR (bottom 15%)
    5. If name + HP + card_number is unique -> return
    6. If still ambiguous -> fall back to identify_card_v2

    Args:
        image_path: Path to the card image.
        session: Optional SQLAlchemy session.
        detect_variants: Whether to run variant detection (default True).

    Returns:
        Dict with keys: card_id, confidence, method, explanation, raw_response.
        Same format as identify_card_v2.
    """
    t_start = time.perf_counter()
    image_path = str(image_path)

    # Build indexes on first call
    name_hp_idx = _build_name_hp_index(session)
    cardnum_idx = _build_card_number_index(session)

    # Step 1-2: Run OCR (name + HP + card number in two passes)
    ocr = _ocr_name_hp_cardnum(image_path)
    ocr_name = ocr.get("ocr_name")
    ocr_conf = ocr.get("ocr_conf", 0.0)
    hp_value = ocr.get("hp_value")
    card_number = ocr.get("card_number")
    set_total = ocr.get("set_total")

    logger.info(
        "fast_identify: name=%r (conf=%.2f), hp=%s, card_num=%s/%s [%s]",
        ocr_name, ocr_conf, hp_value, card_number, set_total,
        Path(image_path).name,
    )

    # Need at least a name to proceed
    if not ocr_name or ocr_conf < 0.60:
        logger.info("fast_identify: no reliable name, falling back to full pipeline")
        return _fallback_to_v2(image_path, session, detect_variants)

    # Step 3: Look up name + HP in index
    name_lower = ocr_name.lower()
    key_with_hp = (name_lower, hp_value) if hp_value else None

    candidates = []
    if key_with_hp and key_with_hp in name_hp_idx:
        # Exact name + HP match
        candidates = list(name_hp_idx[key_with_hp])
    elif hp_value:
        # HP was detected but exact (name, hp) combo not found.
        # Gather all cards with this name (any HP).
        for (n, h), cids in name_hp_idx.items():
            if n == name_lower:
                candidates.extend(cids)
    else:
        # No HP detected, gather all cards with this name
        for (n, h), cids in name_hp_idx.items():
            if n == name_lower:
                candidates.extend(cids)

    # Deduplicate
    candidates = list(dict.fromkeys(candidates))

    logger.info("fast_identify: %d candidates for name=%r hp=%s", len(candidates), ocr_name, hp_value)

    if not candidates:
        logger.info("fast_identify: no candidates found, falling back to full pipeline")
        return _fallback_to_v2(image_path, session, detect_variants)

    # Step 3a: Validate unique name+HP match against card number if available
    if len(candidates) == 1 and card_number:
        # Verify the single candidate's card number matches OCR
        cardnum_cards = set(cardnum_idx.get(card_number, []))
        if candidates[0] in cardnum_cards:
            # Card number confirms the match -- high confidence
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            result = _build_result(
                card_id=candidates[0],
                confidence=max(ocr_conf, 0.90),
                method="fast_name_hp_cardnum_confirmed",
                ocr_name=ocr_name,
                ocr_conf=ocr_conf,
                hp=hp_value,
                card_number=card_number,
                n_candidates=1,
                elapsed_ms=elapsed_ms,
            )
            _apply_variants(result, image_path, detect_variants)
            return result
        elif cardnum_cards:
            # Card number contradicts the single candidate -- OCR name might
            # be slightly off (e.g., "Croconaw" vs "Croconaw delta").
            # Try finding a card with matching card_number among ALL cards
            # with a similar name.
            logger.info(
                "fast_identify: card_number %s contradicts unique candidate %s, "
                "searching broader",
                card_number, candidates[0],
            )
            # Expand search: all cards with names containing the OCR name
            broader = []
            for (n, h), cids in name_hp_idx.items():
                if name_lower in n or n in name_lower:
                    broader.extend(cids)
            broader = list(dict.fromkeys(broader))
            num_matched = [c for c in broader if c in cardnum_cards]
            if len(num_matched) == 1:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                result = _build_result(
                    card_id=num_matched[0],
                    confidence=max(ocr_conf, 0.85),
                    method="fast_name_hp_cardnum_corrected",
                    ocr_name=ocr_name,
                    ocr_conf=ocr_conf,
                    hp=hp_value,
                    card_number=card_number,
                    n_candidates=len(broader),
                    elapsed_ms=elapsed_ms,
                )
                _apply_variants(result, image_path, detect_variants)
                return result
            # Still ambiguous or no match -- fall through to fallback
            candidates = num_matched if num_matched else broader

    # Step 3b: Unique match by name + HP (no card number to verify)
    if len(candidates) == 1:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        result = _build_result(
            card_id=candidates[0],
            confidence=max(ocr_conf, 0.80),
            method="fast_name_hp",
            ocr_name=ocr_name,
            ocr_conf=ocr_conf,
            hp=hp_value,
            card_number=card_number,
            n_candidates=1,
            elapsed_ms=elapsed_ms,
        )
        _apply_variants(result, image_path, detect_variants)
        return result

    # Step 4: Try card number to disambiguate among multiple candidates
    if card_number and len(candidates) >= 2:
        # Find which candidates have this card number
        cardnum_cards = set(cardnum_idx.get(card_number, []))
        num_matched = [c for c in candidates if c in cardnum_cards]

        if len(num_matched) == 1:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            result = _build_result(
                card_id=num_matched[0],
                confidence=max(ocr_conf, 0.85),
                method="fast_name_hp_cardnum",
                ocr_name=ocr_name,
                ocr_conf=ocr_conf,
                hp=hp_value,
                card_number=card_number,
                set_total=set_total,
                n_candidates=len(candidates),
                elapsed_ms=elapsed_ms,
            )
            _apply_variants(result, image_path, detect_variants)
            return result
        elif len(num_matched) >= 2:
            # Narrowed but not unique -- still better
            candidates = num_matched
            logger.info("fast_identify: card_number %s narrowed to %d candidates", card_number, len(candidates))

    # Step 5: Still ambiguous -- try without variant suffixes
    # Many "duplicates" are just normal/holofoil variants of the same base card.
    base_ids = set()
    base_to_cids = {}
    for cid in candidates:
        base = cid.split("/")[0]
        base_ids.add(base)
        base_to_cids.setdefault(base, []).append(cid)

    if len(base_ids) == 1:
        # All candidates are variants of the same base card.
        # Pick the first one (variant detection will handle the rest).
        base = list(base_ids)[0]
        best_cid = base_to_cids[base][0]
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        result = _build_result(
            card_id=best_cid,
            confidence=max(ocr_conf, 0.80),
            method="fast_name_hp_single_base",
            ocr_name=ocr_name,
            ocr_conf=ocr_conf,
            hp=hp_value,
            card_number=card_number,
            n_candidates=len(candidates),
            n_base_ids=1,
            elapsed_ms=elapsed_ms,
        )
        _apply_variants(result, image_path, detect_variants)
        return result

    # Step 6: Multiple distinct base cards -- need visual disambiguation.
    # Fall back to identify_card_v2 with precomputed OCR to save time.
    logger.info(
        "fast_identify: %d base cards remain after OCR, falling back to v2 "
        "(name=%r, hp=%s, card_num=%s)",
        len(base_ids), ocr_name, hp_value, card_number,
    )
    return _fallback_to_v2(
        image_path, session, detect_variants,
        precomputed_ocr={
            "ocr_name": ocr_name,
            "ocr_conf": ocr_conf,
            "ocr_raw": ocr.get("ocr_raw"),
            "hp_value": hp_value,
            "ocr_card_num": card_number,
            "ocr_set_total": set_total,
        },
    )


def _build_result(card_id, confidence, method, **kwargs):
    """Build a result dict in the same format as identify_card_v2."""
    elapsed = kwargs.pop("elapsed_ms", None)
    explanation_parts = [f"fast: {method}"]
    if kwargs.get("ocr_name"):
        explanation_parts.append(f"name={kwargs['ocr_name']!r}")
    if kwargs.get("hp"):
        explanation_parts.append(f"hp={kwargs['hp']}")
    if kwargs.get("card_number"):
        cn = kwargs["card_number"]
        st = kwargs.get("set_total", "?")
        explanation_parts.append(f"card_num={cn}/{st}")
    if elapsed is not None:
        explanation_parts.append(f"{elapsed:.0f}ms")

    raw = {k: v for k, v in kwargs.items()}
    if elapsed is not None:
        raw["elapsed_ms"] = elapsed

    return {
        "card_id": card_id,
        "confidence": float(confidence),
        "method": method,
        "explanation": ", ".join(explanation_parts),
        "raw_response": raw,
    }


def _apply_variants(result, image_path, detect_variants):
    """Apply variant detection to a result dict."""
    if not detect_variants:
        return
    try:
        from cardprice.ml import _apply_variant_detection
        _apply_variant_detection(result, image_path, detect_variants=detect_variants)
    except Exception as e:
        logger.warning("fast_identify: variant detection failed: %s", e)


def _fallback_to_v2(image_path, session, detect_variants, precomputed_ocr=None):
    """Fall back to the full identify_card_v2 pipeline."""
    from cardprice.ml import identify_card_v2
    return identify_card_v2(
        image_path,
        session=session,
        detect_variants=detect_variants,
        _precomputed_ocr=precomputed_ocr,
    )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def fast_identify_batch(image_paths, session=None, detect_variants=True):
    """Identify multiple cards using the fast OCR path.

    Returns list of result dicts, one per image_path.
    Prints a summary of fast vs fallback counts.
    """
    results = []
    fast_count = 0
    fallback_count = 0

    # Pre-build indexes
    _build_name_hp_index(session)
    _build_card_number_index(session)

    for path in image_paths:
        result = fast_identify(path, session=session, detect_variants=detect_variants)
        results.append(result)
        if result.get("method", "").startswith("fast_"):
            fast_count += 1
        else:
            fallback_count += 1

    logger.info(
        "fast_identify_batch: %d/%d fast, %d/%d fallback",
        fast_count, len(image_paths), fallback_count, len(image_paths),
    )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m cardprice.ml.fast_identify <image_path> [image_path ...]")
        print("       python -m cardprice.ml.fast_identify --eval")
        sys.exit(1)

    if sys.argv[1] == "--eval":
        # Run against ground truth
        with open("data/ground_truth.json") as f:
            gt = json.load(f)

        t_build = time.perf_counter()
        _build_name_hp_index()
        _build_card_number_index()
        print(f"Index build: {(time.perf_counter() - t_build) * 1000:.0f}ms\n")

        correct = 0
        total = 0
        fast_ok = 0
        fast_wrong = 0
        fast_total = 0
        fast_times = []

        for page_name, page_data in gt["pages"].items():
            page_dir = f"data/inbox/{page_name}"
            for card_key in sorted(k for k in page_data if k.startswith("card_")):
                card_data = page_data[card_key]
                expected_id = card_data.get("card_id")
                if not expected_id:
                    continue

                img_path = f"{page_dir}/{card_key}.png"
                total += 1
                t1 = time.perf_counter()
                result = fast_identify(img_path, detect_variants=False)
                elapsed = (time.perf_counter() - t1) * 1000

                got_id = result.get("card_id")
                method = result.get("method", "?")
                is_fast = method.startswith("fast_")

                expected_base = expected_id.split("/")[0]
                got_base = got_id.split("/")[0] if got_id else None
                match = expected_base == got_base

                if match:
                    correct += 1
                if is_fast:
                    fast_total += 1
                    fast_times.append(elapsed)
                    if match:
                        fast_ok += 1
                    else:
                        fast_wrong += 1

                status = "OK" if match else "WRONG"
                if not match or is_fast:
                    print(
                        f"{status:5s} [{elapsed:6.0f}ms] [{method:35s}] "
                        f"{page_name}/{card_key}: "
                        f"expected={expected_base}, got={got_base}"
                    )

        print(f"\n{'=' * 60}")
        print(f"Total: {total}")
        print(f"Correct: {correct}/{total} ({correct / total * 100:.1f}%)")
        print(
            f"Fast path: {fast_total}/{total} ({fast_total / total * 100:.1f}%) "
            f"-- {fast_ok} correct, {fast_wrong} wrong"
        )
        print(f"Fallback to v2: {total - fast_total}/{total}")
        if fast_times:
            fast_times.sort()
            print(
                f"\nFast path timing: "
                f"min={fast_times[0]:.0f}ms, "
                f"max={fast_times[-1]:.0f}ms, "
                f"median={fast_times[len(fast_times) // 2]:.0f}ms, "
                f"mean={sum(fast_times) / len(fast_times):.0f}ms"
            )
    else:
        _build_name_hp_index()
        _build_card_number_index()
        for path in sys.argv[1:]:
            print(f"\n{'=' * 60}")
            print(f"File: {path}")
            print(f"{'=' * 60}")
            t1 = time.perf_counter()
            result = fast_identify(path, detect_variants=False)
            elapsed = (time.perf_counter() - t1) * 1000
            print(f"  card_id:    {result.get('card_id')}")
            print(f"  confidence: {result.get('confidence', 0):.2f}")
            print(f"  method:     {result.get('method')}")
            print(f"  time:       {elapsed:.0f}ms")
            print(f"  explanation: {result.get('explanation')}")
