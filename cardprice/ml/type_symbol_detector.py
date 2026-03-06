"""Pokemon card energy type detection from the top-right energy symbol.

Pokemon cards have a small circular energy type symbol in the top-right
corner (next to the HP number).  The symbol's color is distinctive per type.

    Fire:       red/orange flame       Water:      blue water drop
    Grass:      green leaf             Lightning:  yellow lightning bolt
    Psychic:    purple eye             Fighting:   brown/orange fist
    Darkness:   dark crescent moon     Metal:      silver/gray gear
    Fairy:      pink (XY-SM only)      Dragon:     gold
    Colorless:  white/gray star

Implementation:
  1. Crop the top-right region of the card (right 40%, top 16%).
  2. Scan a grid of sample positions across the symbol search area.
  3. At each position, sample a small circular patch and compute its HSV.
  4. Measure distance to each type's calibrated HSV centroid.
  5. The type with the lowest distance across all positions wins.
  6. Return top-N candidates with confidence scores.

The grid-scan approach is robust to the fact that the energy symbol
position varies by card era (WOTC: ~x=88%, modern: ~x=95%) -- the
symbol is guaranteed to be somewhere in the search area, and its color
profile will produce the best (lowest distance) match.

Works best on clean reference images (240x330 or 245x342).  On binder
page scans through plastic sleeves, the orange tint can reduce accuracy.
For binder scans, combine with the frame-based color_detector.py.

Pure OpenCV, no ML models.  ~10ms per card.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HSV type profiles calibrated from reference card images
# ---------------------------------------------------------------------------
# Each profile: (H_center, H_spread, S_center, S_spread, V_center, V_spread)
# H is OpenCV hue [0, 179]; S, V are [0, 255].
# The distance metric is sum of squared (value - center) / spread terms.
#
# Calibrated by pixel-level analysis of energy symbols on ~100 reference
# images per type from data/card_images/ (20,000+ card collection).
#
# Key discrimination challenges and how they're resolved:
#   Lightning vs Fighting: Lightning H=24, Fighting H=16 (6 units apart)
#     -> Tight H spreads (3-4) make this work. V also helps (Lightning brighter).
#   Colorless vs Metal: Both low sat. Colorless has V>200, Metal V~170.
#   Psychic vs Fairy: Psychic H=148, Fairy H=163 (15 units apart).
#     -> S differs: Psychic S=125, Fairy S=110.
#   Fire vs Fighting: Fire H=5, Fighting H=16 (11 units apart).
#     -> Tight spreads resolve this well on clean images.

_TYPE_PROFILES = {
    #                  H_c  H_sp  S_c  S_sp  V_c  V_sp
    "Fire":         (   5,    7,  170,   35,  230,   30),
    "Water":        ( 100,   10,  170,   40,  220,   40),
    "Grass":        (  45,   12,  200,   45,  140,   50),
    "Lightning":    (  24,    3,  190,   30,  230,   20),
    "Psychic":      ( 148,   18,  125,   30,  135,   40),
    "Fighting":     (  16,    4,  185,   25,  220,   25),
    "Darkness":     (   0,   90,   30,   25,   40,   30),
    "Metal":        (  20,   30,   30,   15,  175,   25),
    "Fairy":        ( 163,   12,  110,   35,  190,   30),
    "Dragon":       (  25,   10,  130,   25,  175,   30),
    "Colorless":    (   0,   90,   18,   12,  215,   20),
}


# ---------------------------------------------------------------------------
# Search grid parameters
# ---------------------------------------------------------------------------
# The energy symbol sits in the name bar area.  Its exact x-position
# varies by era:
#   WOTC/Base:  ~85-92% of card width (before HP text)
#   e-series:   ~88-94%
#   EX-DP:      ~90-96%
#   Modern:     ~92-97% (after HP text)
#
# Y position is more consistent: 4-13% of card height.
#
# We scan a grid covering all possible positions.

_GRID_X_RANGE = (0.82, 0.99, 0.015)   # x: 82-99% in 1.5% steps
_GRID_Y_RANGE = (0.04, 0.14, 0.008)   # y: 4-14% in 0.8% steps


def _type_distance(
    h: float, s: float, v: float,
    h_c: float, h_sp: float,
    s_c: float, s_sp: float,
    v_c: float, v_sp: float,
) -> float:
    """Squared Mahalanobis-like distance from a type centroid.

    Handles hue wrap-around (H=0 and H=179 are adjacent).
    """
    dh = abs(h - h_c)
    if dh > 89.5:
        dh = 179.0 - dh
    return (dh / max(h_sp, 0.1)) ** 2 + \
           ((s - s_c) / max(s_sp, 0.1)) ** 2 + \
           ((v - v_c) / max(v_sp, 0.1)) ** 2


def _classify_hsv(
    h: float, s: float, v: float,
) -> List[Tuple[str, float]]:
    """Classify HSV into type predictions with confidence scores.

    Returns list of (type_name, confidence) sorted descending.
    Confidence is derived from inverse distance (closer = higher).
    """
    distances = {}
    for tname, (h_c, h_sp, s_c, s_sp, v_c, v_sp) in _TYPE_PROFILES.items():
        distances[tname] = _type_distance(h, s, v, h_c, h_sp, s_c, s_sp, v_c, v_sp)

    # Convert distances to scores: score = exp(-distance)
    scores = {}
    for tname, dist in distances.items():
        scores[tname] = float(np.exp(-dist * 0.5))

    # Normalize
    total = sum(scores.values())
    if total <= 0:
        return [("Colorless", 1.0)]

    results = [(name, score / total) for name, score in scores.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _grid_scan_symbol(img: np.ndarray) -> tuple:
    """Scan a grid of positions in the symbol area and find the best type match.

    Returns (best_type, best_distance, best_hsv, best_position) or None.
    """
    h, w = img.shape[:2]

    # Crop to search region first (performance: avoid HSV on full image)
    x_start_frac, x_end_frac, x_step_frac = _GRID_X_RANGE
    y_start_frac, y_end_frac, y_step_frac = _GRID_Y_RANGE

    crop_x1 = max(0, int(w * x_start_frac) - 5)
    crop_y1 = max(0, int(h * y_start_frac) - 5)
    crop_x2 = min(w, int(w * x_end_frac) + 5)
    crop_y2 = min(h, int(h * y_end_frac) + 5)

    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ch, cw = crop.shape[:2]

    # Sampling radius: ~1% of image width (2-3 px on 240w, ~10px on 1008w)
    r = max(1, int(w * 0.010))

    # Track the best match for each type
    best_per_type = {}  # type -> (distance, position, hsv_values)

    for x_pct in np.arange(x_start_frac, x_end_frac, x_step_frac):
        # Convert to crop-local coordinates
        sx = int(w * x_pct) - crop_x1
        if sx - r < 0 or sx + r >= cw:
            continue

        for y_pct in np.arange(y_start_frac, y_end_frac, y_step_frac):
            sy = int(h * y_pct) - crop_y1
            if sy - r < 0 or sy + r >= ch:
                continue

            # Sample a small circular patch
            mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.circle(mask, (sx, sy), r, 255, -1)
            pixels = hsv[mask > 0]

            if len(pixels) < 1:
                continue

            ph = float(pixels[:, 0].mean())
            ps = float(pixels[:, 1].mean())
            pv = float(pixels[:, 2].mean())

            # Find best type for this pixel
            for tname, (h_c, h_sp, s_c, s_sp, v_c, v_sp) in _TYPE_PROFILES.items():
                dist = _type_distance(ph, ps, pv, h_c, h_sp, s_c, s_sp, v_c, v_sp)

                if tname not in best_per_type or dist < best_per_type[tname][0]:
                    best_per_type[tname] = (dist, (float(x_pct), float(y_pct)), (ph, ps, pv))

    if not best_per_type:
        return None

    # Pick the type with the lowest distance overall
    best_type = min(best_per_type, key=lambda t: best_per_type[t][0])
    best_dist, best_pos, best_hsv = best_per_type[best_type]

    return best_type, best_dist, best_hsv, best_pos


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_type_from_symbol(
    image_path: str,
    top_n: int = 3,
) -> List[Tuple[str, float]]:
    """Detect Pokemon type from the energy symbol in the top-right corner.

    Scans a grid of positions in the top-right area of the card and
    classifies each using calibrated HSV profiles.  The best-matching
    type across all positions is returned.

    Parameters
    ----------
    image_path : str or Path
        Path to a card segment image.
    top_n : int
        Number of top predictions to return.

    Returns
    -------
    list of (type_name, confidence) tuples
        Sorted by confidence descending.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    ValueError
        If the image cannot be decoded.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    return detect_type_from_symbol_array(img, top_n=top_n, label=image_path.name)


def detect_type_from_symbol_array(
    img: np.ndarray,
    *,
    top_n: int = 3,
    label: str = "<array>",
) -> List[Tuple[str, float]]:
    """Detect type from an already-loaded BGR image array.

    Parameters
    ----------
    img : numpy.ndarray
        BGR image (as from cv2.imread).
    top_n : int
        Number of top predictions to return.
    label : str
        Label for logging.

    Returns
    -------
    list of (type_name, confidence)
    """
    result = _grid_scan_symbol(img)

    if result is None:
        logger.warning("Could not detect energy symbol for %s", label)
        return [("Colorless", 0.0)]

    best_type, best_dist, best_hsv, best_pos = result

    # Build full confidence distribution from the best HSV
    h_val, s_val, v_val = best_hsv
    predictions = _classify_hsv(h_val, s_val, v_val)

    logger.debug(
        "Symbol type for %s: HSV=(%.0f,%.0f,%.0f) pos=(%.0f%%,%.0f%%) "
        "dist=%.2f -> %s",
        label,
        h_val, s_val, v_val,
        best_pos[0] * 100, best_pos[1] * 100,
        best_dist,
        ", ".join(f"{n} ({c:.0%})" for n, c in predictions[:top_n]),
    )

    return predictions[:top_n]


def detect_type_from_symbol_with_debug(
    image_path: str,
    *,
    top_n: int = 3,
) -> dict:
    """Like detect_type_from_symbol but returns additional debug info.

    Returns dict with keys:
      - predictions: list of (type, confidence)
      - symbol_hsv: (H, S, V) of best match point, or None
      - match_position: (x_frac, y_frac), or None
      - match_distance: float
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    result = _grid_scan_symbol(img)

    if result is None:
        return {
            "predictions": [("Colorless", 0.0)],
            "symbol_hsv": None,
            "match_position": None,
            "match_distance": 999.0,
        }

    best_type, best_dist, best_hsv, best_pos = result
    h_val, s_val, v_val = best_hsv
    predictions = _classify_hsv(h_val, s_val, v_val)

    return {
        "predictions": predictions[:top_n],
        "symbol_hsv": (round(h_val, 1), round(s_val, 1), round(v_val, 1)),
        "match_position": (round(float(best_pos[0]), 3), round(float(best_pos[1]), 3)),
        "match_distance": round(best_dist, 3),
    }


# ---------------------------------------------------------------------------
# Ground truth for eval
# ---------------------------------------------------------------------------
_EVAL_GROUND_TRUTH = {
    # Page 1 (EX delta species + EX era)
    "page_20260228_174819_cards_v4/card_08.png": "Grass",
    "page_20260228_174819_cards_v4/card_07.png": "Psychic",
    "page_20260228_174819_cards_v4/card_06.png": "Water",
    "page_20260228_174819_cards_v4/card_05.png": "Colorless",
    "page_20260228_174819_cards_v4/card_04.png": "Colorless",
    "page_20260228_174819_cards_v4/card_03.png": "Psychic",
    "page_20260228_174819_cards_v4/card_02.png": "Psychic",
    "page_20260228_174819_cards_v4/card_01.png": "Colorless",
    "page_20260228_174819_cards_v4/card_00.png": "Lightning",
    # Page 2 (e-series)
    "page_20260228_195512_cards/card_00.png": "Psychic",
    "page_20260228_195512_cards/card_01.png": "Psychic",
    "page_20260228_195512_cards/card_02.png": "Psychic",
    "page_20260228_195512_cards/card_03.png": "Psychic",
    "page_20260228_195512_cards/card_04.png": "Psychic",
    "page_20260228_195512_cards/card_05.png": "Colorless",
    "page_20260228_195512_cards/card_06.png": "Colorless",
    "page_20260228_195512_cards/card_07.png": "Colorless",
    "page_20260228_195512_cards/card_08.png": "Colorless",
    # Page 3 (mixed eras)
    "page_20260228_202134_cards/card_00.png": None,
    "page_20260228_202134_cards/card_01.png": "Colorless",
    "page_20260228_202134_cards/card_02.png": "Colorless",
    "page_20260228_202134_cards/card_03.png": "Grass",
    "page_20260228_202134_cards/card_04.png": "Colorless",
    "page_20260228_202134_cards/card_05.png": "Lightning",
    "page_20260228_202134_cards/card_06.png": "Water",
    "page_20260228_202134_cards/card_07.png": "Water",
    "page_20260228_202134_cards/card_08.png": "Colorless",
}


def run_eval(data_root: str = "data/inbox") -> dict:
    """Run symbol type detection against eval ground truth."""
    root = Path(data_root)
    results = []
    correct = 0
    total = 0

    for rel_path, expected_type in _EVAL_GROUND_TRUTH.items():
        if expected_type is None:
            continue

        full_path = root / rel_path
        if not full_path.exists():
            results.append({
                "path": rel_path,
                "expected": expected_type,
                "predicted": "MISSING",
                "correct": False,
            })
            continue

        total += 1
        try:
            debug = detect_type_from_symbol_with_debug(str(full_path), top_n=3)
            predicted = debug["predictions"][0][0]
            conf = debug["predictions"][0][1]
            is_correct = predicted == expected_type

            if is_correct:
                correct += 1

            results.append({
                "path": rel_path,
                "expected": expected_type,
                "predicted": predicted,
                "confidence": conf,
                "symbol_hsv": debug["symbol_hsv"],
                "match_position": debug["match_position"],
                "match_distance": debug["match_distance"],
                "correct": is_correct,
            })
        except Exception as e:
            results.append({
                "path": rel_path,
                "expected": expected_type,
                "predicted": f"ERROR: {e}",
                "correct": False,
            })

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    args = sys.argv[1:]

    if not args or args[0] == "--eval":
        print("Running eval against binder segment ground truth...")
        print("=" * 95)
        eval_result = run_eval()

        for r in eval_result["results"]:
            marker = "OK" if r["correct"] else "MISS"
            conf_str = f"{r.get('confidence', 0):.0%}" if "confidence" in r else "?"
            hsv = r.get("symbol_hsv", "?")
            pos = r.get("match_position", "?")
            dist = r.get("match_distance", "?")
            print(
                f"[{marker:4s}] {r['path']:55s} "
                f"expected={r['expected']:12s} "
                f"got={r['predicted']:12s} "
                f"conf={conf_str:>4s} "
                f"hsv={hsv} pos={pos} dist={dist}"
            )

        print("=" * 95)
        print(
            f"Accuracy: {eval_result['correct']}/{eval_result['total']} "
            f"= {eval_result['accuracy']:.1%}"
        )

    elif args[0] == "--ref-eval":
        import json
        from collections import defaultdict

        with open("data/card_names.json") as f:
            data = json.load(f)
        type_lookup = {}
        for row in data:
            types = row[4] if len(row) > 4 else []
            if types:
                base_id = row[0].split("/")[0]
                type_lookup[base_id] = types[0]

        type_results = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})
        type_counts = defaultdict(int)
        max_per_type = 50

        ref_dir = Path("data/card_images/")
        for set_dir in sorted(ref_dir.iterdir()):
            if not set_dir.is_dir():
                continue
            for img_file in sorted(set_dir.iterdir()):
                if not img_file.name.endswith("_normal.png"):
                    continue
                base_id = img_file.stem.replace("_normal", "")
                expected = type_lookup.get(base_id)
                if not expected or type_counts[expected] >= max_per_type:
                    continue

                try:
                    preds = detect_type_from_symbol(str(img_file), top_n=3)
                    predicted = preds[0][0]
                    type_results[expected]["total"] += 1
                    if predicted == expected:
                        type_results[expected]["correct"] += 1
                    else:
                        type_results[expected]["errors"].append(
                            f"  {base_id}: got {predicted} ({preds[0][1]:.0%})"
                        )
                    type_counts[expected] += 1
                except Exception as e:
                    logger.warning("Error on %s: %s", img_file, e)

        print("Reference image accuracy by type:")
        print("=" * 60)
        total_c = 0
        total_a = 0
        for ptype in ["Fire", "Water", "Grass", "Lightning", "Psychic",
                       "Fighting", "Darkness", "Metal", "Fairy", "Dragon",
                       "Colorless"]:
            r = type_results[ptype]
            if r["total"] == 0:
                continue
            acc = r["correct"] / r["total"]
            total_c += r["correct"]
            total_a += r["total"]
            print(f"  {ptype:12s}: {r['correct']:2d}/{r['total']:2d} = {acc:.0%}")
            for err in r["errors"][:3]:
                print(err)

        if total_a > 0:
            print(f"\n  OVERALL:      {total_c}/{total_a} = {total_c / total_a:.0%}")

    else:
        for p in args:
            try:
                debug = detect_type_from_symbol_with_debug(p, top_n=5)
                top = debug["predictions"][0]
                alts = ", ".join(
                    f"{n} {c:.0%}" for n, c in debug["predictions"][1:]
                )
                print(
                    f"{Path(p).name:30s} -> {top[0]:12s} ({top[1]:.0%}) "
                    f"HSV={debug['symbol_hsv']} "
                    f"pos={debug['match_position']} "
                    f"dist={debug['match_distance']}"
                    + (f"   alts: {alts}" if alts else "")
                )
            except Exception as e:
                print(f"{Path(p).name:30s} -> ERROR: {e}")
