"""Unified card condition estimation combining surface, edge, and centering analysis.

Combines three existing detectors into a single function that returns
a TCGPlayer-style condition grade (NM/LP/MP/HP/DMG) without any labeled
training data.

Detectors used:
  - DINOv2 patch comparison (surface defects) — requires reference image
  - LAB+HSV edge whitening (border wear) — reference-free
  - HSV border symmetry (centering) — reference-free
  - Corner wear — derived from edge whitening data

Usage::

    from cardprice.ml.condition_estimator import estimate_condition

    result = estimate_condition("photo.jpg", card_id="base1-4")
    print(result["overall_grade"])   # "NM", "LP", "MP", "HP", or "DMG"
    print(result["overall_score"])   # 0-10
    print(result["sub_grades"])      # per-aspect scores
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Weights for combining sub-grades into overall score
WEIGHT_SURFACE = 0.40
WEIGHT_EDGES = 0.30
WEIGHT_CENTERING = 0.15
WEIGHT_CORNERS = 0.15

# Grade boundaries on 0-10 scale
_GRADE_THRESHOLDS = [
    (9.0, "NM"),
    (7.0, "LP"),
    (4.5, "MP"),
    (2.0, "HP"),
    (0.0, "DMG"),
]


def _score_to_grade(score: float) -> str:
    """Map a 0-10 score to a TCGPlayer condition grade."""
    for threshold, grade in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "DMG"


def _find_ref_image(card_id: str) -> Optional[str]:
    """Locate the reference image for a card_id.

    Checks data/card_images/{set_id}/{card_id}_normal.{ext} first,
    then data/ref_images/{card_id}.{ext} as fallback.
    """
    base = Path("data")

    # Primary: data/card_images/{set}/{card_id_with_variant}.png
    # card_id format: "sv8-162/normal" → set "sv8", file "sv8-162_normal.png"
    # Also handles "base1-4" (no variant) → file "base1-4.png"
    file_id = card_id.replace("/", "_")  # sv8-162/normal → sv8-162_normal
    parts = card_id.split("-", 1)
    if len(parts) == 2:
        set_id = parts[0]
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            # Try with variant suffix
            candidate = base / "card_images" / set_id / f"{file_id}{ext}"
            if candidate.is_file():
                return str(candidate)
            # Try without (bare card_id as filename)
            candidate = base / "card_images" / set_id / f"{card_id.split('/')[0]}{ext}"
            if candidate.is_file():
                return str(candidate)

    # Fallback: data/ref_images/{card_id}.ext
    ref_dir = base / "ref_images"
    if ref_dir.is_dir():
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = ref_dir / f"{card_id}{ext}"
            if candidate.is_file():
                return str(candidate)

    return None


def _surface_defect_score_to_10(defect_score: float) -> float:
    """Map surface detector's defect_score (0=mint, 1=damaged) to 0-10 scale.

    defect_score thresholds from surface_detector.py:
      0.00-0.02 -> NM  -> 10.0-9.0
      0.02-0.08 -> LP  -> 9.0-7.0
      0.08-0.20 -> MP  -> 7.0-4.5
      0.20-0.40 -> HP  -> 4.5-2.0
      0.40-1.00 -> DMG -> 2.0-0.0
    """
    if defect_score <= 0.0:
        return 10.0
    if defect_score <= 0.02:
        # 0.0 -> 10.0, 0.02 -> 9.0
        return 10.0 - (defect_score / 0.02) * 1.0
    if defect_score <= 0.08:
        # 0.02 -> 9.0, 0.08 -> 7.0
        return 9.0 - ((defect_score - 0.02) / 0.06) * 2.0
    if defect_score <= 0.20:
        # 0.08 -> 7.0, 0.20 -> 4.5
        return 7.0 - ((defect_score - 0.08) / 0.12) * 2.5
    if defect_score <= 0.40:
        # 0.20 -> 4.5, 0.40 -> 2.0
        return 4.5 - ((defect_score - 0.20) / 0.20) * 2.5
    # 0.40 -> 2.0, 1.0 -> 0.0
    return max(0.0, 2.0 - ((defect_score - 0.40) / 0.60) * 2.0)


def _edge_whitening_to_10(whitening_result: dict) -> float:
    """Map edge whitening overall_ratio to 0-10 scale.

    Thresholds from edge_whitening.py:
      0.000        -> Gem Mint  -> 10.0
      0.000-0.005  -> NM        -> 10.0-9.0
      0.005-0.02   -> LP        -> 9.0-7.0
      0.02-0.05    -> MP        -> 7.0-4.5
      > 0.05       -> HP        -> 4.5-2.0
    """
    ratio = whitening_result["overall_ratio"]

    if ratio <= 0.0:
        return 10.0
    if ratio <= 0.005:
        return 10.0 - (ratio / 0.005) * 1.0
    if ratio <= 0.02:
        return 9.0 - ((ratio - 0.005) / 0.015) * 2.0
    if ratio <= 0.05:
        return 7.0 - ((ratio - 0.02) / 0.03) * 2.5
    if ratio <= 0.15:
        return 4.5 - ((ratio - 0.05) / 0.10) * 2.5
    return max(0.0, 2.0 - ((ratio - 0.15) / 0.35) * 2.0)


def _centering_to_10(centering_result: dict) -> float:
    """Map centering detector score (1-10 PSA scale) to our 0-10 scale.

    The centering detector already returns a 1-10 score, so this is
    nearly a passthrough but we clamp to our range.
    """
    return max(0.0, min(10.0, centering_result["centering_score"]))


def _corner_score_from_edges(whitening_result: dict) -> float:
    """Derive corner condition from edge whitening data.

    Corners are where two edges meet. We use the two worst edge ratios
    as a proxy (same approach as server.py).

    Thresholds (slightly more lenient than edges):
      0.000       -> 10.0
      0.000-0.008 -> NM  -> 10.0-9.0
      0.008-0.03  -> LP  -> 9.0-7.0
      0.03-0.07   -> MP  -> 7.0-4.5
      > 0.07      -> HP  -> 4.5-2.0
    """
    edge_ratios = sorted(
        [whitening_result["edges"][s]["whitening_ratio"]
         for s in ("top", "bottom", "left", "right")],
        reverse=True,
    )
    # Average of two worst edges as corner proxy
    corner_ratio = (edge_ratios[0] + edge_ratios[1]) / 2

    if corner_ratio <= 0.0:
        return 10.0
    if corner_ratio <= 0.008:
        return 10.0 - (corner_ratio / 0.008) * 1.0
    if corner_ratio <= 0.03:
        return 9.0 - ((corner_ratio - 0.008) / 0.022) * 2.0
    if corner_ratio <= 0.07:
        return 7.0 - ((corner_ratio - 0.03) / 0.04) * 2.5
    if corner_ratio <= 0.20:
        return 4.5 - ((corner_ratio - 0.07) / 0.13) * 2.5
    return max(0.0, 2.0 - ((corner_ratio - 0.20) / 0.30) * 2.0)


def estimate_condition(
    image_path: str,
    card_id: str = None,
    ref_image_path: str = None,
) -> dict:
    """Estimate card condition by combining surface, edge, and centering analysis.

    If card_id is provided, looks up the reference image automatically.
    If ref_image_path is provided, uses that directly.
    If neither, only runs edge whitening + centering (no surface comparison).

    Parameters
    ----------
    image_path : str
        Path to the card photo to assess.
    card_id : str, optional
        Card identifier (e.g. "base1-4") for automatic reference lookup.
    ref_image_path : str, optional
        Explicit path to the pristine reference image.

    Returns
    -------
    dict with keys:
        overall_grade : str
            "NM", "LP", "MP", "HP", or "DMG"
        overall_score : float
            Weighted average on 0-10 scale.
        confidence : float
            0-1, higher when more sub-grades are available.
        sub_grades : dict
            surface, edges, centering, corners — each float 0-10 or None.
        defects : list
            Flagged defect patches from surface detector.
        details : dict
            Raw detector outputs for debugging.
    """
    image_path = str(Path(image_path).resolve())

    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    sub_grades = {
        "surface": None,
        "edges": None,
        "centering": None,
        "corners": None,
    }
    defects = []
    details = {}
    available_weights = {}

    # --- Resolve reference image ---
    ref_path = ref_image_path
    if ref_path is None and card_id is not None:
        ref_path = _find_ref_image(card_id)
        if ref_path:
            logger.info("Found reference image: %s", ref_path)
        else:
            logger.warning("No reference image found for card_id=%s", card_id)

    # --- Surface defect detection (requires reference) ---
    if ref_path and Path(ref_path).is_file():
        try:
            from cardprice.ml.surface_detector import detect_surface_defects
            surface_result = detect_surface_defects(image_path, ref_path)

            surface_score = _surface_defect_score_to_10(surface_result["defect_score"])
            sub_grades["surface"] = round(surface_score, 1)
            available_weights["surface"] = WEIGHT_SURFACE

            # Collect defect info
            defects = [
                {"row": r, "col": c, "similarity": round(sim, 3)}
                for r, c, sim in surface_result.get("defect_patches", [])
            ]

            # Store details (exclude numpy arrays for JSON serialization)
            details["surface"] = {
                "defect_score": surface_result["defect_score"],
                "defect_count": surface_result["defect_count"],
                "defect_ratio": surface_result["defect_ratio"],
                "mean_similarity": surface_result["mean_similarity"],
                "min_similarity": surface_result["min_similarity"],
                "patch_threshold": surface_result["patch_threshold"],
                "ref_image": ref_path,
            }
        except Exception as e:
            logger.warning("Surface defect detection failed: %s", e)
            details["surface"] = {"error": str(e)}

    # --- Edge whitening ---
    whitening_result = None
    try:
        from cardprice.ml.edge_whitening import measure_edge_whitening
        whitening_result = measure_edge_whitening(image_path)

        edge_score = _edge_whitening_to_10(whitening_result)
        sub_grades["edges"] = round(edge_score, 1)
        available_weights["edges"] = WEIGHT_EDGES

        details["edges"] = {
            "overall_ratio": whitening_result["overall_ratio"],
            "worst_edge": whitening_result["worst_edge"],
            "worst_ratio": whitening_result["worst_ratio"],
            "tcg_condition": whitening_result["tcg_condition"],
            "max_white_run": whitening_result["max_white_run"],
            "cluster_count": whitening_result["cluster_count"],
        }
    except Exception as e:
        logger.warning("Edge whitening detection failed: %s", e)
        details["edges"] = {"error": str(e)}

    # --- Corner wear (derived from edge data) ---
    if whitening_result is not None:
        try:
            corner_score = _corner_score_from_edges(whitening_result)
            sub_grades["corners"] = round(corner_score, 1)
            available_weights["corners"] = WEIGHT_CORNERS

            edge_ratios = sorted(
                [whitening_result["edges"][s]["whitening_ratio"]
                 for s in ("top", "bottom", "left", "right")],
                reverse=True,
            )
            details["corners"] = {
                "proxy_ratio": round((edge_ratios[0] + edge_ratios[1]) / 2, 6),
                "worst_two_edges": [
                    round(edge_ratios[0], 6),
                    round(edge_ratios[1], 6),
                ],
            }
        except Exception as e:
            logger.warning("Corner score derivation failed: %s", e)
            details["corners"] = {"error": str(e)}

    # --- Centering ---
    try:
        from cardprice.ml.centering_detector import measure_centering
        centering_result = measure_centering(image_path=image_path)

        centering_score = _centering_to_10(centering_result)
        sub_grades["centering"] = round(centering_score, 1)
        available_weights["centering"] = WEIGHT_CENTERING

        details["centering"] = {
            "front_lr": centering_result["front_lr"],
            "front_tb": centering_result["front_tb"],
            "centering_score": centering_result["centering_score"],
            "centering_confidence": centering_result["confidence"],
            "borders": centering_result["borders"],
        }
    except Exception as e:
        logger.warning("Centering detection failed: %s", e)
        details["centering"] = {"error": str(e)}

    # --- Compute overall score ---
    if available_weights:
        total_weight = sum(available_weights.values())
        weighted_sum = sum(
            sub_grades[key] * weight
            for key, weight in available_weights.items()
        )
        overall_score = round(weighted_sum / total_weight, 1)
    else:
        overall_score = 0.0

    # Confidence: proportion of total weight that was actually computed
    max_possible_weight = WEIGHT_SURFACE + WEIGHT_EDGES + WEIGHT_CENTERING + WEIGHT_CORNERS
    confidence = round(sum(available_weights.values()) / max_possible_weight, 2)

    overall_grade = _score_to_grade(overall_score)

    return {
        "overall_grade": overall_grade,
        "overall_score": overall_score,
        "confidence": confidence,
        "sub_grades": sub_grades,
        "defects": defects,
        "details": details,
    }
