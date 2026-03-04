"""Detect the rarity symbol on a Pokemon card image.

Pokemon cards have a rarity symbol next to the set number at the bottom-right
of the card:
  - Circle   = Common
  - Diamond  = Uncommon
  - Star     = Rare (includes Holo Rare, Reverse Holo, etc.)

Detection strategy:
1. Crop the bottom-right corner of the card image.
2. Use HSV saturation to mask out the binder sleeve (high-saturation area).
3. Threshold the masked gray image to isolate dark ink.
4. Find contours and classify by shape descriptors:
   - Star:    low solidity (<0.63), 5+ polygon vertices
   - Circle:  high circularity (>0.65), high solidity (>0.85)
   - Diamond: 4 polygon vertices, high solidity (>0.80)
5. Pick the rightmost qualifying contour (closest to where the symbol sits).
6. Vote across multiple crop regions and thresholding methods.

The saturation mask is critical: it removes binder sleeve texture that would
otherwise produce many false circular (common) detections.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Rarity constants
RARITY_COMMON = "common"        # Circle symbol
RARITY_UNCOMMON = "uncommon"    # Diamond symbol
RARITY_RARE = "rare"            # Star symbol (includes holo rare)
RARITY_UNKNOWN = "unknown"      # Could not detect


def _classify_contour(c):
    """Classify a contour as a rarity symbol.

    Returns (rarity, confidence) or (None, 0.0).
    """
    area = cv2.contourArea(c)
    if area < 12:
        return None, 0.0

    x, y, bw, bh = cv2.boundingRect(c)
    if bw < 4 or bh < 4 or bw > 40 or bh > 40:
        return None, 0.0

    aspect = bw / max(bh, 1)
    if aspect < 0.4 or aspect > 2.5:
        return None, 0.0

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1)
    perimeter = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / max(perimeter ** 2, 1)
    approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
    n_vertices = len(approx)
    extent = area / max(bw * bh, 1)

    aspect_bonus = 1.0 - min(abs(aspect - 1.0), 1.0)

    # STAR: low solidity (concave indentations between star arms)
    if solidity < 0.63 and n_vertices >= 5:
        conf = (
            (0.63 - solidity) / 0.30 * 0.5
            + min(n_vertices, 10) / 10.0 * 0.3
            + aspect_bonus * 0.2
        )
        return RARITY_RARE, min(conf, 1.0)

    # CIRCLE: high circularity and high solidity, many polygon vertices.
    # Be strict: require circularity > 0.72 to avoid confusing card border
    # corners and text fragments as circles.
    if circularity > 0.72 and solidity > 0.88 and n_vertices >= 6:
        conf = (
            circularity * 0.4
            + solidity * 0.3
            + aspect_bonus * 0.2
            + 0.1
        )
        return RARITY_COMMON, min(conf, 1.0)

    # DIAMOND: exactly 4 polygon vertices, high solidity.
    # A diamond fills ~50% of its bounding box (extent ~0.50).
    # Require aspect ratio close to 1:1 (diamonds are symmetric).
    if n_vertices == 4 and solidity > 0.82 and 0.40 < extent < 0.60 and 0.6 < aspect < 1.5:
        conf = (
            solidity * 0.3
            + aspect_bonus * 0.4
            + max(0.0, 1.0 - abs(extent - 0.50) * 4) * 0.3
        )
        return RARITY_UNCOMMON, max(0.0, min(conf, 1.0))

    return None, 0.0


def _make_card_mask(crop_bgr):
    """Create a mask that keeps card interior and removes binder sleeve.

    Uses HSV saturation: binder sleeves are highly saturated (orange/purple),
    while card interior (where the rarity symbol sits) has lower saturation
    (white/gray/yellow text area).

    Returns a uint8 mask (255 = card, 0 = binder).
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    # Card regions: low saturation (text on white/gray) OR very bright (border)
    # Use a generous threshold to avoid masking the symbol itself
    card_mask = ((s_chan < 140) | (v_chan > 210)).astype(np.uint8) * 255

    # Gentle morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return card_mask


def _find_best_in_binary(binary, crop_h, crop_w):
    """Find the best rarity-symbol contour in a binary image.

    Returns (rarity, confidence, x_position) or (None, 0.0, 0).
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for c in contours:
        rarity, conf = _classify_contour(c)
        if rarity is not None:
            x = cv2.boundingRect(c)[0]
            candidates.append((rarity, conf, x))

    if not candidates:
        return None, 0.0, 0

    # Prefer rightmost candidate (rarity symbol is the rightmost mark
    # on the set-number line).  Weight: 70% rightness + 30% confidence.
    max_x = max(c[2] for c in candidates)
    best = max(
        candidates,
        key=lambda t: (t[2] / max(max_x, 1)) * 0.7 + t[1] * 0.3
    )
    return best


def _detect_in_crop(crop_bgr, use_mask=True):
    """Detect rarity in a single crop region.

    Returns list of (rarity, confidence) tuples -- one per thresholding
    method that found a candidate.
    """
    ch, cw = crop_bgr.shape[:2]
    if ch < 8 or cw < 8:
        return []

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Optionally mask out binder sleeve areas
    if use_mask:
        mask = _make_card_mask(crop_bgr)
        # Set masked-out areas to white so they don't produce dark contours
        gray = gray.copy()
        gray[mask == 0] = 255

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    results = []

    # Method 1: Otsu threshold (inverted)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    r, c, _ = _find_best_in_binary(binary, ch, cw)
    if r is not None:
        results.append((r, c))

    # Method 2: Fixed low threshold for dark ink
    _, binary2 = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)
    r, c, _ = _find_best_in_binary(binary2, ch, cw)
    if r is not None:
        results.append((r, c))

    # Method 3: Adaptive threshold
    if ch >= 15 and cw >= 15:
        binary3 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 3
        )
        r, c, _ = _find_best_in_binary(binary3, ch, cw)
        if r is not None:
            results.append((r, c))

    return results


def detect_rarity(image_path):
    """Detect the rarity symbol on a Pokemon card image.

    Args:
        image_path: Path to a card image (photo or reference).

    Returns:
        dict with keys:
            rarity: One of "common", "uncommon", "rare", "unknown"
            confidence: Float 0.0-1.0 (higher = more certain)
            details: Dict with vote counts and detection info
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("rarity_detector: could not read image %s", image_path)
        return {
            "rarity": RARITY_UNKNOWN,
            "confidence": 0.0,
            "details": {"error": "could not read image"},
        }

    h, w = img.shape[:2]

    # Multiple crop regions for robustness against varied card positioning.
    # Each region is (y_start_frac, y_end_frac, x_start_frac, x_end_frac).
    crop_regions = [
        (0.87, 0.96, 0.73, 0.95),   # Primary
        (0.85, 0.97, 0.70, 0.97),   # Wider
        (0.89, 0.95, 0.76, 0.93),   # Tighter
        (0.88, 0.94, 0.78, 0.92),   # Focused on set number line
    ]

    votes = []

    for ys, ye, xs, xe in crop_regions:
        y1, y2 = int(h * ys), int(h * ye)
        x1, x2 = int(w * xs), int(w * xe)
        crop = img[y1:y2, x1:x2]

        # Use binder-sleeve mask to suppress false detections from sleeve
        # texture.  Without masking, binder bumps register as circles and
        # overwhelm the vote with false "common" detections.
        results = _detect_in_crop(crop, use_mask=True)
        votes.extend(results)

    if not votes:
        logger.info(
            "rarity_detector: %s -> unknown (no symbol candidates)",
            Path(image_path).name,
        )
        return {
            "rarity": RARITY_UNKNOWN,
            "confidence": 0.0,
            "details": {"vote_counts": {}, "total_votes": 0},
        }

    # Aggregate votes
    vote_counts = {RARITY_COMMON: 0, RARITY_UNCOMMON: 0, RARITY_RARE: 0}
    vote_conf_sums = {RARITY_COMMON: 0.0, RARITY_UNCOMMON: 0.0, RARITY_RARE: 0.0}
    for r, c in votes:
        vote_counts[r] += 1
        vote_conf_sums[r] += c

    total_votes = len(votes)

    # Score: average confidence * vote fraction
    scores = {}
    for rarity in [RARITY_COMMON, RARITY_UNCOMMON, RARITY_RARE]:
        n = vote_counts[rarity]
        if n > 0:
            avg_conf = vote_conf_sums[rarity] / n
            vote_frac = n / total_votes
            scores[rarity] = avg_conf * vote_frac
        else:
            scores[rarity] = 0.0

    winner = max(scores, key=scores.get)
    win_score = scores[winner]

    details = {
        "vote_counts": vote_counts,
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "total_votes": total_votes,
    }

    logger.info(
        "rarity_detector: %s -> %s (conf=%.3f, votes=%s)",
        Path(image_path).name, winner, win_score, vote_counts,
    )

    return {"rarity": winner, "confidence": win_score, "details": details}


def detect_rarity_batch(image_paths):
    """Detect rarity for multiple card images.

    Args:
        image_paths: Iterable of paths to card images.

    Returns:
        List of result dicts (same format as detect_rarity).
    """
    return [detect_rarity(p) for p in image_paths]


def filter_candidates_by_rarity(detected_rarity, candidate_card_ids, session=None):
    """Filter ML candidate card_ids by detected rarity.

    Looks up each candidate's rarity in the database and removes those that
    do not match the detected rarity.  If the detection is "unknown" or the
    DB lookup fails, returns all candidates unchanged.

    Args:
        detected_rarity: One of "common", "uncommon", "rare", "unknown".
        candidate_card_ids: List of card_id strings.
        session: SQLAlchemy session (optional, will create one if needed).

    Returns:
        Filtered list of card_id strings.
    """
    if detected_rarity == RARITY_UNKNOWN or not candidate_card_ids:
        return candidate_card_ids

    # Map our rarity names to pokemontcg.io rarity values
    rarity_map = {
        RARITY_COMMON: {"Common"},
        RARITY_UNCOMMON: {"Uncommon"},
        RARITY_RARE: {
            "Rare", "Rare Holo", "Rare Holo EX", "Rare Holo GX",
            "Rare Holo V", "Rare Holo VMAX", "Rare Holo VSTAR",
            "Rare Ultra", "Rare Secret", "Rare Shiny", "Rare Rainbow",
            "Rare Holo Star", "Rare Prime", "Rare ACE",
            "Rare BREAK", "Rare Prism Star",
            "Illustration Rare", "Special Illustration Rare",
            "Hyper Rare", "Double Rare", "Ultra Rare",
        },
    }

    allowed = rarity_map.get(detected_rarity)
    if allowed is None:
        return candidate_card_ids

    try:
        from sqlalchemy import text
        from cardprice.db.session import SessionLocal

        own_session = session is None
        if own_session:
            session = SessionLocal()

        try:
            # Query rarity for all candidate card_ids.
            # card_ids may have variant suffix (e.g. "base1-5/normal").
            # The dim_cards table stores the full card_id with variant.
            placeholders = ",".join(f":id{i}" for i in range(len(candidate_card_ids)))
            params = {f"id{i}": cid for i, cid in enumerate(candidate_card_ids)}
            rows = session.execute(
                text(f"SELECT card_id, rarity FROM dim_cards WHERE card_id IN ({placeholders})"),
                params,
            ).fetchall()
            rarity_lookup = {r[0]: r[1] for r in rows}

            filtered = []
            for cid in candidate_card_ids:
                card_rarity = rarity_lookup.get(cid)
                if card_rarity is None:
                    # Not found in DB - keep the candidate
                    filtered.append(cid)
                elif card_rarity in allowed:
                    filtered.append(cid)
                # else: rarity mismatch, drop this candidate

            if not filtered:
                logger.warning(
                    "rarity filter removed all %d candidates for rarity=%s, "
                    "returning unfiltered",
                    len(candidate_card_ids), detected_rarity,
                )
                return candidate_card_ids

            removed = len(candidate_card_ids) - len(filtered)
            if removed:
                logger.info(
                    "rarity filter: %s -> kept %d/%d candidates (removed %d non-%s)",
                    detected_rarity, len(filtered), len(candidate_card_ids),
                    removed, detected_rarity,
                )
            return filtered

        finally:
            if own_session:
                session.close()

    except Exception as e:
        logger.warning("rarity filter failed (returning unfiltered): %s", e)
        return candidate_card_ids


# ---------------------------------------------------------------------------
# CLI entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m cardprice.ml.rarity_detector <image_path> [...]")
        print("       python -m cardprice.ml.rarity_detector --test-dir <dir>")
        sys.exit(1)

    if sys.argv[1] == "--test-dir":
        import os
        d = sys.argv[2]
        paths = sorted(
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.endswith((".png", ".jpg", ".jpeg"))
        )
        for p in paths:
            result = detect_rarity(p)
            r = result["rarity"]
            c = result["confidence"]
            vc = result["details"].get("vote_counts", {})
            print(f"  {Path(p).name}: {r} (conf={c:.3f}, votes={vc})")
    else:
        for p in sys.argv[1:]:
            result = detect_rarity(p)
            print(
                f"{Path(p).name}: {result['rarity']} "
                f"(conf={result['confidence']:.3f})"
            )
            print(f"  details: {result['details']}")
