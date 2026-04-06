"""Card segment image quality scorer.

Evaluates whether a card segment image is good enough to run through the
ML identification cascade, or should be re-captured.  Uses only OpenCV --
no deep learning dependencies.

Scored factors:
1. Blur (Laplacian variance) -- is the card in focus?
2. Glare (bright spot percentage) -- sleeve reflections / flash hotspots?
3. Contrast (histogram spread) -- is the image washed out or too dark?
4. Orientation -- portrait (correct) vs landscape (needs rotation)?
5. Completeness -- does the image contain a full card or just a partial?

Overall score is 0-1 with recommendation:
  "good"        -- high quality, run through cascade
  "try_matching" -- acceptable, may work but could be better
  "re_capture"  -- too poor, ask the user to retake
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard Pokemon card aspect ratio: 63mm x 88mm -> w/h ~0.716
CARD_ASPECT_RATIO = 63.0 / 88.0

# --- Thresholds (tuned on binder page segments) ---

# Blur: Laplacian variance.  Higher = sharper.
# < 50 is very blurry, 50-100 marginal, > 100 sharp.
BLUR_SHARP = 100.0
BLUR_MARGINAL = 50.0

# Glare: percentage of pixels above brightness threshold (0-1).
# > 5% means noticeable glare, > 15% is severe.
GLARE_OK = 0.03
GLARE_BAD = 0.12

# Contrast: standard deviation of grayscale histogram.
# < 25 is washed out / very dark, 25-40 marginal, > 40 good.
CONTRAST_GOOD = 40.0
CONTRAST_MARGINAL = 25.0

# Completeness: edge density around the border region.
# A complete card has visible edges (card border) on all 4 sides.
COMPLETENESS_GOOD = 0.65
COMPLETENESS_MARGINAL = 0.40

# Overall score thresholds for recommendation.
OVERALL_GOOD = 0.65
OVERALL_TRY = 0.40


def _score_blur(gray: np.ndarray) -> float:
    """Score image sharpness using Laplacian variance.

    Returns a 0-1 score where 1 is perfectly sharp.
    """
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var >= BLUR_SHARP:
        return 1.0
    elif laplacian_var <= BLUR_MARGINAL:
        # Linear ramp from 0 at variance=0 to 0.5 at BLUR_MARGINAL
        return 0.5 * (laplacian_var / BLUR_MARGINAL)
    else:
        # Linear ramp from 0.5 at BLUR_MARGINAL to 1.0 at BLUR_SHARP
        return 0.5 + 0.5 * (laplacian_var - BLUR_MARGINAL) / (BLUR_SHARP - BLUR_MARGINAL)


def _score_glare(gray: np.ndarray) -> float:
    """Score glare by measuring percentage of near-white pixels.

    Returns a 0-1 score where 1 means no glare.
    """
    # Count pixels above 240 (very bright -- likely glare/reflection)
    bright_pixels = np.sum(gray > 240)
    total_pixels = gray.size
    bright_frac = bright_pixels / total_pixels

    if bright_frac <= GLARE_OK:
        return 1.0
    elif bright_frac >= GLARE_BAD:
        return 0.0
    else:
        # Linear ramp from 1.0 at GLARE_OK to 0.0 at GLARE_BAD
        return 1.0 - (bright_frac - GLARE_OK) / (GLARE_BAD - GLARE_OK)


def _score_contrast(gray: np.ndarray) -> float:
    """Score contrast using the standard deviation of pixel intensities.

    Returns a 0-1 score where 1 means good contrast.
    """
    std = float(np.std(gray))
    if std >= CONTRAST_GOOD:
        return 1.0
    elif std <= CONTRAST_MARGINAL:
        return 0.5 * (std / CONTRAST_MARGINAL)
    else:
        return 0.5 + 0.5 * (std - CONTRAST_MARGINAL) / (CONTRAST_GOOD - CONTRAST_MARGINAL)


def _detect_orientation(h: int, w: int) -> str:
    """Determine if the image is portrait or landscape.

    Pokemon cards should be portrait (taller than wide).
    """
    if h >= w:
        return "portrait"
    else:
        return "landscape"


def _score_completeness(gray: np.ndarray) -> float:
    """Score whether the image contains a complete card.

    Checks for edge density in the border region on all four sides.
    A complete card will have visible edges (the card border) along
    all four sides.  A partial crop will be missing edges on one or
    more sides.

    Returns a 0-1 score where 1 means all four sides have card edges.
    """
    h, w = gray.shape[:2]
    border_frac = 0.08  # check the outer 8% on each side

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Define the four border strips
    border_t = int(h * border_frac)
    border_b = int(h * border_frac)
    border_l = int(w * border_frac)
    border_r = int(w * border_frac)

    strips = {
        "top": edges[:border_t, :],
        "bottom": edges[h - border_b:, :],
        "left": edges[:, :border_l],
        "right": edges[:, w - border_r:],
    }

    # Score each side by edge density
    side_scores = []
    for name, strip in strips.items():
        if strip.size == 0:
            side_scores.append(0.0)
            continue
        density = np.count_nonzero(strip) / strip.size
        # A good card border has density > ~0.05
        # Normalize: 0.02 = 0.0, 0.10 = 1.0
        score = np.clip((density - 0.02) / 0.08, 0.0, 1.0)
        side_scores.append(float(score))

    # Overall completeness is the average of all four sides.
    # If any side is very low, it drags down the average (likely partial crop).
    avg = float(np.mean(side_scores))

    # Also penalize if one side is much worse than the others (asymmetric crop)
    min_side = min(side_scores)
    if min_side < 0.2 and avg > 0.5:
        # One side is basically missing edges -- partial card
        avg = avg * 0.7

    return avg


def score_quality(image_path: str) -> dict:
    """Return quality metrics for a card segment image.

    Args:
        image_path: Path to a card segment image (PNG/JPG).

    Returns:
        Dictionary with keys:
            blur: float (0-1, higher is sharper)
            glare: float (0-1, higher is less glare)
            contrast: float (0-1, higher is better contrast)
            orientation: str ("portrait" or "landscape")
            completeness: float (0-1, higher means more complete card)
            overall: float (0-1, weighted combination)
            recommendation: str ("good", "try_matching", or "re_capture")

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be read by OpenCV.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Compute individual scores
    blur = _score_blur(gray)
    glare = _score_glare(gray)
    contrast = _score_contrast(gray)
    orientation = _detect_orientation(h, w)
    completeness = _score_completeness(gray)

    # Orientation penalty: landscape cards can still be matched (the cascade
    # tries rotations) but it's a signal that something may be off.
    orientation_penalty = 0.0 if orientation == "portrait" else 0.15

    # Weighted overall score.
    # Blur matters most (out-of-focus images rarely match well).
    # Glare is next (reflections confuse hash/embedding matchers).
    # Contrast and completeness are softer signals.
    overall = (
        blur * 0.35
        + glare * 0.25
        + contrast * 0.20
        + completeness * 0.20
        - orientation_penalty
    )
    overall = float(np.clip(overall, 0.0, 1.0))

    # Determine recommendation
    if overall >= OVERALL_GOOD:
        recommendation = "good"
    elif overall >= OVERALL_TRY:
        recommendation = "try_matching"
    else:
        recommendation = "re_capture"

    result = {
        "blur": round(blur, 3),
        "glare": round(glare, 3),
        "contrast": round(contrast, 3),
        "orientation": orientation,
        "completeness": round(completeness, 3),
        "overall": round(overall, 3),
        "recommendation": recommendation,
    }

    logger.info(
        "Quality score for %s: blur=%.2f glare=%.2f contrast=%.2f "
        "orient=%s complete=%.2f -> overall=%.2f (%s)",
        path.name, blur, glare, contrast, orientation,
        completeness, overall, recommendation,
    )

    return result


def score_quality_cv(image: np.ndarray) -> dict:
    """Score quality from an already-loaded OpenCV image (BGR ndarray).

    Same as score_quality() but skips the file read. Useful when the
    image is already in memory (e.g., from the segmenter).
    """
    if image is None or image.size == 0:
        raise ValueError("Empty or None image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    blur = _score_blur(gray)
    glare = _score_glare(gray)
    contrast = _score_contrast(gray)
    orientation = _detect_orientation(h, w)
    completeness = _score_completeness(gray)

    orientation_penalty = 0.0 if orientation == "portrait" else 0.15

    overall = (
        blur * 0.35
        + glare * 0.25
        + contrast * 0.20
        + completeness * 0.20
        - orientation_penalty
    )
    overall = float(np.clip(overall, 0.0, 1.0))

    if overall >= OVERALL_GOOD:
        recommendation = "good"
    elif overall >= OVERALL_TRY:
        recommendation = "try_matching"
    else:
        recommendation = "re_capture"

    return {
        "blur": round(blur, 3),
        "glare": round(glare, 3),
        "contrast": round(contrast, 3),
        "orientation": orientation,
        "completeness": round(completeness, 3),
        "overall": round(overall, 3),
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# CLI entry point for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        # Default: score all test binder segments
        project_root = Path(__file__).resolve().parent.parent.parent
        test_dirs = [
            project_root / "data" / "test_binder_segments",
            project_root / "data" / "test_binder_segments_rotated",
            project_root / "data" / "test_segments",
        ]
        paths = []
        for d in test_dirs:
            if d.exists():
                paths.extend(sorted(p for p in d.iterdir()
                                    if p.suffix.lower() in (".png", ".jpg", ".jpeg")
                                    and "debug" not in p.name))
        if not paths:
            print("Usage: python -m cardprice.ml.quality_scorer <image_path> [...]")
            sys.exit(1)
    else:
        paths = [Path(p) for p in sys.argv[1:]]

    print(f"\nScoring {len(paths)} images:\n")
    print(f"{'File':<50} {'Blur':>5} {'Glare':>5} {'Contr':>5} {'Compl':>5} "
          f"{'Orient':<10} {'Score':>5} {'Recommendation':<14}")
    print("-" * 110)

    for p in paths:
        try:
            result = score_quality(str(p))
            # Shorten path for display
            try:
                display = str(p.relative_to(Path(__file__).resolve().parent.parent.parent))
            except ValueError:
                display = str(p)
            print(f"{display:<50} {result['blur']:>5.2f} {result['glare']:>5.2f} "
                  f"{result['contrast']:>5.2f} {result['completeness']:>5.2f} "
                  f"{result['orientation']:<10} {result['overall']:>5.2f} "
                  f"{result['recommendation']:<14}")
        except Exception as e:
            print(f"{p}: ERROR -- {e}")

    print()
