"""Unified condition assessment pipeline for Pokemon card grading.

Combines all condition sub-modules into a single call:
  - Surface defect detection (DINOv2 patch comparison vs reference)
  - Edge whitening measurement (LAB+HSV border wear)
  - Centering measurement (HSV border symmetry)
  - Corner wear classification (EfficientNet-B0, requires checkpoint)

Each sub-module runs independently and failures are isolated -- if one
module errors out, the others still return results.  The overall grade
is a weighted combination of available sub-scores mapped to TCGPlayer
condition tiers (NM / LP / MP / HP / DMG).

Usage::

    from cardprice.ml.condition_assessor import assess_condition

    result = assess_condition("photo.jpg", card_id="base1-4/holofoil")
    print(result["overall_grade"])       # "LP"
    print(result["price_multiplier"])    # 0.8
    print(result["sub_scores"]["edges"]) # {score, grade, details}
"""

import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference image lookup
# ---------------------------------------------------------------------------

_CARD_IMAGES_DIR = Path("data/card_images")
_CARD_IMAGES_HIRES_DIR = Path("data/card_images_hires")


def _find_ref_image(card_id: str) -> Optional[str]:
    """Resolve a card_id to a reference image path.

    Card IDs have the format ``{set_id}-{num}/{variant}`` e.g.
    ``base1-4/holofoil``.  Reference images are stored as
    ``data/card_images/{set_id}/{set_id}-{num}_{variant}.png``.

    Tries hi-res directory first, then standard resolution.
    """
    if not card_id:
        return None

    parts = card_id.split("/", 1)
    base_id = parts[0]            # e.g. "base1-4"
    variant = parts[1] if len(parts) > 1 else "normal"

    # Derive set_id from base_id (everything before the last dash+number)
    dash_idx = base_id.rfind("-")
    if dash_idx <= 0:
        set_id = base_id
    else:
        set_id = base_id[:dash_idx]

    filename = f"{base_id}_{variant}.png"

    # Try hi-res first, then standard
    for images_dir in (_CARD_IMAGES_HIRES_DIR, _CARD_IMAGES_DIR):
        candidate = images_dir / set_id / filename
        if candidate.exists():
            return str(candidate)

    # Fallback: try "normal" variant if the requested variant wasn't found
    if variant != "normal":
        fallback_name = f"{base_id}_normal.png"
        for images_dir in (_CARD_IMAGES_HIRES_DIR, _CARD_IMAGES_DIR):
            candidate = images_dir / set_id / fallback_name
            if candidate.exists():
                logger.debug(
                    "Ref image for %s: variant '%s' not found, using 'normal'",
                    card_id, variant,
                )
                return str(candidate)

    logger.debug("No reference image found for card_id=%s", card_id)
    return None


# ---------------------------------------------------------------------------
# Sub-score grade mapping
# ---------------------------------------------------------------------------

# Numeric score thresholds -> TCGPlayer grade
# Each sub-module produces a 0-1 severity score (0=perfect, 1=worst).
_GRADE_ORDER = ["NM", "LP", "MP", "HP", "DMG"]
_GRADE_RANK = {g: i for i, g in enumerate(_GRADE_ORDER)}

_SEVERITY_TO_GRADE = [
    (0.05, "NM"),
    (0.15, "LP"),
    (0.35, "MP"),
    (0.60, "HP"),
    (1.01, "DMG"),
]


def _severity_to_grade(score: float) -> str:
    """Map a 0-1 severity score to a TCGPlayer condition grade."""
    for threshold, grade in _SEVERITY_TO_GRADE:
        if score < threshold:
            return grade
    return "DMG"


def _grade_to_severity(grade: str) -> float:
    """Map a grade string to a representative severity score (midpoint)."""
    mapping = {
        "NM": 0.02,
        "LP": 0.10,
        "MP": 0.25,
        "HP": 0.47,
        "DMG": 0.80,
    }
    return mapping.get(grade, 0.50)


# Corner classifier grades -> severity
_CORNER_GRADE_SEVERITY = {
    "Gem": 0.00,
    "Mint": 0.03,
    "Light": 0.15,
    "Moderate": 0.35,
    "Heavy": 0.70,
}


# ---------------------------------------------------------------------------
# Sub-module runners (each isolated with try/except)
# ---------------------------------------------------------------------------

def _run_surface(
    image_path: str,
    ref_image_path: str,
) -> Optional[dict]:
    """Run surface defect detection via DINOv2 patch comparison."""
    try:
        from cardprice.ml.surface_detector import (
            detect_surface_defects,
            estimate_condition,
        )

        result = detect_surface_defects(image_path, ref_image_path)
        condition = estimate_condition(result)

        return {
            "score": round(result["defect_score"], 4),
            "grade": condition["grade_abbrev"],
            "confidence": condition["confidence"],
            "details": {
                "defect_ratio": round(result["defect_ratio"], 4),
                "defect_count": result["defect_count"],
                "mean_similarity": round(result["mean_similarity"], 4),
                "min_similarity": round(result["min_similarity"], 4),
                "patch_threshold": result["patch_threshold"],
            },
        }
    except Exception as e:
        logger.warning("Surface detection failed: %s", e)
        return None


def _run_edges(image_path: str) -> Optional[dict]:
    """Run edge whitening measurement."""
    try:
        from cardprice.ml.edge_whitening import measure_edge_whitening

        result = measure_edge_whitening(image_path)

        # Map overall_ratio to a 0-1 severity score
        # Edge whitening thresholds: 0=NM, 0.005=LP, 0.02=MP, 0.05=HP
        ratio = result["overall_ratio"]
        if ratio <= 0.0:
            severity = 0.0
        elif ratio < 0.005:
            severity = ratio / 0.005 * 0.05   # 0 -> 0.05 (NM range)
        elif ratio < 0.02:
            severity = 0.05 + (ratio - 0.005) / 0.015 * 0.10  # 0.05 -> 0.15
        elif ratio < 0.05:
            severity = 0.15 + (ratio - 0.02) / 0.03 * 0.20    # 0.15 -> 0.35
        else:
            severity = min(0.35 + (ratio - 0.05) / 0.10 * 0.35, 1.0)

        return {
            "score": round(severity, 4),
            "grade": result["tcg_condition"],
            "details": {
                "overall_ratio": result["overall_ratio"],
                "worst_edge": result["worst_edge"],
                "worst_ratio": result["worst_ratio"],
                "max_white_run": result["max_white_run"],
                "cluster_count": result["cluster_count"],
                "per_edge": {
                    side: {
                        "whitening_ratio": info["whitening_ratio"],
                        "max_white_run": info["max_white_run"],
                    }
                    for side, info in result["edges"].items()
                },
            },
        }
    except Exception as e:
        logger.warning("Edge whitening detection failed: %s", e)
        return None


def _run_centering(image_path: str) -> Optional[dict]:
    """Run centering measurement (HSV border-based detector)."""
    try:
        from cardprice.ml.centering_detector import measure_centering

        result = measure_centering(image_path)

        # Extract LR and TB ratios as floats
        # front_lr is like "55/45" -- parse the bigger number
        lr_parts = result["front_lr"].split("/")
        tb_parts = result["front_tb"].split("/")
        lr_ratio = max(int(lr_parts[0]), int(lr_parts[1])) / 100.0
        tb_ratio = max(int(tb_parts[0]), int(tb_parts[1])) / 100.0

        centering_score = result["centering_score"]  # 1-10 PSA scale

        # Map PSA centering score to a grade string
        if centering_score >= 9.0:
            centering_grade = "NM"
        elif centering_score >= 8.0:
            centering_grade = "LP"
        elif centering_score >= 7.0:
            centering_grade = "MP"
        else:
            centering_grade = "HP"

        # Also derive a PSA-style centering grade number
        if centering_score >= 9.5:
            psa_centering = 10
        elif centering_score >= 8.5:
            psa_centering = 9
        elif centering_score >= 7.5:
            psa_centering = 8
        elif centering_score >= 6.5:
            psa_centering = 7
        else:
            psa_centering = 6

        return {
            "lr_ratio": lr_ratio,
            "tb_ratio": tb_ratio,
            "centering_score": centering_score,
            "psa_grade": psa_centering,
            "grade": centering_grade,
            "confidence": result["confidence"],
            "details": {
                "front_lr": result["front_lr"],
                "front_tb": result["front_tb"],
                "borders": result["borders"],
            },
        }
    except Exception as e:
        logger.warning("Centering detection failed: %s", e)
        return None


def _run_corners(image_path: str) -> Optional[dict]:
    """Run corner wear classification (requires trained checkpoint)."""
    try:
        from cardprice.ml.corner_classifier import DEFAULT_CHECKPOINT, grade_corners

        if not DEFAULT_CHECKPOINT.exists():
            logger.debug(
                "Corner classifier checkpoint not found at %s, skipping",
                DEFAULT_CHECKPOINT,
            )
            return None

        result = grade_corners(image_path)

        # Map corner grade to severity
        overall_grade = result["overall_grade"]
        severity = _CORNER_GRADE_SEVERITY.get(overall_grade, 0.50)

        # Map to TCG condition
        tcg_grade = _severity_to_grade(severity)

        return {
            "score": round(severity, 4),
            "grade": tcg_grade,
            "corner_grade": overall_grade,
            "confidence": result["overall_confidence"],
            "details": {
                "per_corner": result["corners"],
            },
        }
    except Exception as e:
        logger.warning("Corner classification failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Overall grade combination
# ---------------------------------------------------------------------------

# Weights for combining sub-scores into an overall severity.
# Surface and edges are the most important condition indicators.
# Centering is important for grading but less so for raw TCG condition.
# Corners are significant but only available with a trained model.
_WEIGHTS = {
    "surface": 0.35,
    "edges": 0.30,
    "centering": 0.15,
    "corners": 0.20,
}


def _combine_grades(sub_scores: dict) -> tuple[str, float]:
    """Combine available sub-scores into an overall grade.

    Uses a weighted average of severity scores from each available module.
    Weights are re-normalized to sum to 1.0 over the modules that succeeded.

    Returns (grade, confidence).
    """
    weighted_sum = 0.0
    weight_total = 0.0
    confidences = []

    for module_name, weight in _WEIGHTS.items():
        result = sub_scores.get(module_name)
        if result is None:
            continue

        # Get severity score -- either from "score" key or derived from grade
        if "score" in result:
            severity = result["score"]
        else:
            severity = _grade_to_severity(result.get("grade", "NM"))

        weighted_sum += severity * weight
        weight_total += weight

        if "confidence" in result:
            conf = result["confidence"]
            # Normalize string confidence to float
            if isinstance(conf, str):
                conf = {"high": 0.9, "low": 0.5}.get(conf, 0.7)
            confidences.append(conf)

    if weight_total == 0:
        return "NM", 0.0

    overall_severity = weighted_sum / weight_total
    overall_grade = _severity_to_grade(overall_severity)

    # Overall confidence: minimum of sub-module confidences, penalized
    # if fewer modules contributed
    if confidences:
        base_conf = min(confidences)
        # Penalty for missing modules: each missing module reduces confidence
        available_frac = weight_total / sum(_WEIGHTS.values())
        overall_confidence = round(base_conf * (0.5 + 0.5 * available_frac), 4)
    else:
        overall_confidence = 0.0

    return overall_grade, overall_confidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_condition(
    images: dict[str, str],
    card_id: Optional[str] = None,
    ref_image_path: Optional[str] = None,
) -> dict:
    """Run full condition assessment pipeline from multiple photos.

    Condition grading REQUIRES multiple photos taken at different angles.
    A single binder page scan is NOT sufficient for condition assessment.
    At minimum, front and back photos are required.

    Parameters
    ----------
    images : dict[str, str]
        Mapping of photo type to file path.  Required keys:
            "front" : Front face photo (straight on)
            "back"  : Back face photo
        Optional keys:
            "oblique" : Surface at ~30° angle (for holo scratches)
            "edge"    : Edge close-up (~75-80° near edge-on)
    card_id : str, optional
        Card identifier (e.g. ``"base1-4/holofoil"``).  Used to find the
        reference image for surface comparison and to look up pricing.
    ref_image_path : str, optional
        Explicit path to the pristine reference image.  If not provided,
        the reference is resolved from *card_id* via the card_images
        directory.

    Returns
    -------
    dict with keys:
        overall_grade : str
            TCGPlayer condition tier: "NM", "LP", "MP", "HP", or "DMG".
        overall_confidence : float
            Confidence in the overall grade (0.0 - 1.0).
        sub_scores : dict
            Per-module results:
                surface : dict or None
                    {score, grade, confidence, details}
                edges : dict or None
                    {score, grade, details}
                centering : dict or None
                    {lr_ratio, tb_ratio, centering_score, psa_grade,
                     grade, confidence, details}
                corners : dict or None
                    {score, grade, corner_grade, confidence, details}
        photos_used : list[str]
            Which photo types were provided.
        modules_run : list[str]
            Names of modules that produced results.
        modules_skipped : list[str]
            Names of modules that were skipped or failed.
        price_multiplier : float
            Condition multiplier relative to NM (e.g. 0.8 for LP).
        price_range : tuple[float, float]
            (ci_low, ci_high) multiplier confidence interval.

    Raises
    ------
    ValueError
        If required photos (front, back) are not provided.
    """
    # Enforce multi-photo requirement
    if not isinstance(images, dict):
        raise ValueError(
            "assess_condition requires a dict of {photo_type: path}. "
            "Condition grading cannot be done from a single image — "
            "multiple angles are required (at minimum: front + back)."
        )

    missing = []
    for required in ("front", "back"):
        if required not in images:
            missing.append(required)
    if missing:
        raise ValueError(
            f"Missing required photos: {missing}. "
            f"Condition grading requires at minimum front and back photos. "
            f"Provided: {list(images.keys())}"
        )

    front_path = str(images["front"])
    back_path = str(images["back"])
    oblique_path = str(images["oblique"]) if "oblique" in images else None
    edge_path = str(images["edge"]) if "edge" in images else None

    photos_used = list(images.keys())

    # Resolve reference image
    if ref_image_path is None and card_id:
        ref_image_path = _find_ref_image(card_id)

    # Run sub-modules
    sub_scores: dict[str, Optional[dict]] = {}

    # Surface detection: use front photo + reference image
    if ref_image_path and Path(ref_image_path).exists():
        sub_scores["surface"] = _run_surface(front_path, ref_image_path)
    else:
        sub_scores["surface"] = None
        if ref_image_path:
            logger.debug("Reference image not found: %s", ref_image_path)

    # Edge whitening: use BACK photo (whitening is most visible on the back)
    # Also check front if edge photo provided
    back_edges = _run_edges(back_path)
    if edge_path:
        edge_edges = _run_edges(edge_path)
        # Use the worse result between back and edge-closeup
        if edge_edges and back_edges:
            if edge_edges["score"] > back_edges["score"]:
                sub_scores["edges"] = edge_edges
            else:
                sub_scores["edges"] = back_edges
        else:
            sub_scores["edges"] = back_edges or edge_edges
    else:
        sub_scores["edges"] = back_edges

    # Centering: use front photo
    sub_scores["centering"] = _run_centering(front_path)

    # Corners: use edge photo if available, otherwise front
    corner_source = edge_path or front_path
    sub_scores["corners"] = _run_corners(corner_source)

    # Classify which modules ran vs skipped
    modules_run = [k for k, v in sub_scores.items() if v is not None]
    modules_skipped = [k for k, v in sub_scores.items() if v is None]

    # Combine into overall grade
    available_scores = {k: v for k, v in sub_scores.items() if v is not None}
    overall_grade, overall_confidence = _combine_grades(available_scores)

    # Look up price multiplier from condition_pricing
    try:
        from cardprice.models.condition_pricing import (
            CONDITION_MULTIPLIERS_WITH_CI,
        )
        mult_info = CONDITION_MULTIPLIERS_WITH_CI.get(
            overall_grade, (1.0, 0.9, 1.1)
        )
        price_multiplier = mult_info[0]
        price_range = (mult_info[1], mult_info[2])
    except Exception:
        # Inline fallback multipliers
        _fallback = {
            "NM":  (1.00, 0.90, 1.10),
            "LP":  (0.80, 0.70, 0.90),
            "MP":  (0.55, 0.40, 0.70),
            "HP":  (0.30, 0.20, 0.40),
            "DMG": (0.12, 0.05, 0.20),
        }
        info = _fallback.get(overall_grade, (1.0, 0.9, 1.1))
        price_multiplier = info[0]
        price_range = (info[1], info[2])

    logger.info(
        "Condition assessment: grade=%s conf=%.2f mult=%.2f "
        "modules_run=%s modules_skipped=%s",
        overall_grade, overall_confidence, price_multiplier,
        modules_run, modules_skipped,
    )

    return {
        "overall_grade": overall_grade,
        "overall_confidence": overall_confidence,
        "sub_scores": sub_scores,
        "modules_run": modules_run,
        "modules_skipped": modules_skipped,
        "price_multiplier": price_multiplier,
        "price_range": price_range,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m cardprice.ml.condition_assessor <image> [--card-id ID] [--ref PATH]")
        sys.exit(1)

    img = sys.argv[1]
    cid = None
    ref = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--card-id" and i + 1 < len(sys.argv):
            cid = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--ref" and i + 1 < len(sys.argv):
            ref = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    result = assess_condition(img, card_id=cid, ref_image_path=ref)

    # Make result JSON-serializable (remove numpy arrays etc.)
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(x) for x in obj]
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        return obj

    print(json.dumps(_sanitize(result), indent=2))
