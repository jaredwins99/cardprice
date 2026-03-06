"""Detect holographic card type: normal, holofoil, or reverse holofoil.

Reverse holo cards have a holographic foil pattern on the card BODY (border,
text box, name bar) while the artwork itself is NOT holographic.  Regular holo
cards have holographic foil on the artwork only.  Normal cards have no foil.

Detection approach:
    1. Split the card into artwork region (~top 40-65%) and body region
       (border strips, name bar, text box at bottom 35-60%).
    2. In each region, measure:
       a) Saturation variance -- holo foil creates rainbow shimmer with high
          saturation spread across many hue values.
       b) Hue spatial noise -- holographic surfaces produce rapid, noisy
          hue changes between adjacent pixels (prismatic micro-reflections)
          that differ from smooth gradients in printed artwork.
       c) Hue spread -- count of distinct hue bins with significant presence
          at high saturation.
    3. Classification logic:
       - Reverse holo: body has HIGH saturation variance, artwork has LOW.
       - Regular holo: artwork has HIGH saturation variance (harder from photo).
       - Normal: both regions have LOW saturation variance.

Limitations (documented for clarity):
    - SINGLE-PHOTO LIMITATION: Holographic effects are angle-dependent and
      may not be visible in a single binder page photo.  A card photographed
      at an angle that does not catch the light may appear normal even if it
      is holographic.  This is a fundamental limitation of single-image
      analysis.
    - BINDER SLEEVE EFFECTS: Binder sleeves add reflections and reduce
      contrast of holographic patterns, making detection harder.
    - LIGHTING DEPENDENCY: Detection works best under fluorescent or angled
      lighting that reveals prismatic reflections.  Even lighting (e.g.,
      diffused daylight) suppresses holo effects.
    - COLORFUL ARTWORK: Cards with very colorful artwork (rainbow trainers,
      Charizard, etc.) may trigger false positives on artwork hue variance.
      Spatial noise filtering mitigates this but does not eliminate it.
    - EX-ERA METALLIC FRAMES: Silver/metallic EX card frames have naturally
      higher saturation variance than standard borders, which can look like
      reverse holo to the detector.
    - CONFIDENCE INTERPRETATION: Confidence < 0.5 means the detection is
      unreliable.  A "normal" result at high confidence is much more
      trustworthy than a "holofoil" or "reverse_holofoil" result, because
      the absence of holo signal is easier to confirm than its presence
      from a single photo.

See also: variant_detector.py for the full variant detection pipeline
(including 1st edition, full art, gold, rainbow rare).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Region definitions (fractional coordinates on a standard Pokemon card)
# ---------------------------------------------------------------------------
# Artwork region: center area containing the illustration.
# Excludes the name bar and border to focus on the art itself.
ART_Y0, ART_Y1 = 0.12, 0.56
ART_X0, ART_X1 = 0.10, 0.90

# Body regions: border, name bar, and text box areas.
# These are the regions where reverse holo foil appears.
BODY_REGIONS = [
    # Name bar (above artwork)
    (0.03, 0.11, 0.10, 0.90),  # (y0, y1, x0, x1)
    # Text box (below artwork)
    (0.58, 0.92, 0.08, 0.92),
    # Left border strip
    (0.12, 0.92, 0.02, 0.10),
    # Right border strip
    (0.12, 0.92, 0.90, 0.98),
]

# ---------------------------------------------------------------------------
# Thresholds (calibrated against binder page segments and reference images)
# ---------------------------------------------------------------------------
# Minimum saturation to consider a pixel "colorful" (OpenCV S in [0,255])
MIN_SAT = 40

# Minimum value (brightness) -- very dark pixels have unreliable hue
MIN_VAL = 40

# Saturation standard deviation thresholds.
# Normal card body: sat_std ~15-30 (uniform border color).
# Reverse holo body: sat_std ~35-60+ (foil shimmer creates color patches).
BODY_SAT_STD_THRESHOLD = 33.0

# Hue standard deviation thresholds.
# Normal card body: hue_std ~8-20.
# Reverse holo body: hue_std ~25-45+.
BODY_HUE_STD_THRESHOLD = 22.0

# Hue spatial noise: Laplacian magnitude in non-edge regions.
# Digital art / normal print: typically 5-30.
# Real holo phone photo: typically 50-150+.
SPATIAL_NOISE_THRESHOLD = 35.0

# Combined holo score threshold (hue_spread * noise_factor).
# Below this, the region is classified as non-holographic.
HOLO_SCORE_THRESHOLD = 6.0

# If body holo score exceeds artwork holo score by this ratio,
# the card is reverse holo (foil on body, not art).
REVERSE_HOLO_RATIO = 1.25

# Minimum confidence for holo/reverse-holo classification.
# Below this, the result should be treated as unreliable.
MIN_HOLO_CONFIDENCE = 0.40


# ---------------------------------------------------------------------------
# Internal measurement functions
# ---------------------------------------------------------------------------

def _extract_region(img: np.ndarray, y0: float, y1: float,
                    x0: float, x1: float) -> np.ndarray:
    """Extract a sub-region using fractional coordinates."""
    h, w = img.shape[:2]
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _saturation_stats(region_bgr: np.ndarray) -> tuple[float, float]:
    """Measure saturation standard deviation and mean in a BGR region.

    Returns (sat_std, sat_mean).
    """
    if region_bgr.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    s_chan = hsv[:, :, 1].astype(np.float32)
    return float(np.std(s_chan)), float(np.mean(s_chan))


def _hue_stats(region_bgr: np.ndarray) -> tuple[float, float]:
    """Measure hue standard deviation and spread in a BGR region.

    Only considers pixels with sufficient saturation and brightness
    (low-saturation pixels have unreliable hue).

    Returns (hue_std, hue_spread) where hue_spread is the number of
    distinct hue bins (out of 36) with significant presence.
    """
    if region_bgr.size == 0:
        return 0.0, 0.0

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0].astype(np.float32)
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    # Only analyze colorful pixels
    mask = (s_chan >= MIN_SAT) & (v_chan >= MIN_VAL)
    hues = h_chan[mask]

    if len(hues) < 50:
        return 0.0, 0.0

    hue_std = float(np.std(hues))

    # Hue spread: number of hue bins with significant presence
    hist, _ = np.histogram(hues, bins=36, range=(0, 180))
    threshold = len(hues) * 0.01
    hue_spread = float(np.sum(hist > threshold))

    return hue_std, hue_spread


def _hue_spatial_noise(region_bgr: np.ndarray) -> float:
    """Measure non-edge hue variation (holographic noise signature).

    Holographic surfaces produce random color speckle -- adjacent pixels
    have different hues even in "flat" areas away from structural edges.
    Printed artwork concentrates color transitions at drawn edges.

    We compute the hue-channel Laplacian (high-frequency color changes),
    then mask out structural edges (Canny) and measure the remaining noise
    only in the flat regions.

    Returns:
        Mean absolute Laplacian of hue in non-edge, colorful regions.
        Normal print: ~5-30.  Real holo: ~50-150+.
    """
    if region_bgr.size == 0:
        return 0.0

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0].astype(np.float32)
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    # Only consider colorful pixels
    colorful_mask = (s_chan >= MIN_SAT) & (v_chan >= MIN_VAL)

    # Find structural edges and exclude them + a buffer zone
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    non_edge_mask = edge_dilated == 0

    combined_mask = colorful_mask & non_edge_mask

    # Hue Laplacian
    laplacian = cv2.Laplacian(h_chan, cv2.CV_32F, ksize=3)
    abs_lap = np.abs(laplacian)

    flat_region_lap = abs_lap[combined_mask]
    if len(flat_region_lap) < 30:
        return 0.0

    return float(np.mean(flat_region_lap))


def _region_holo_score(region_bgr: np.ndarray) -> dict:
    """Compute holographic metrics for a single region.

    Returns dict with keys:
        sat_std, sat_mean, hue_std, hue_spread, spatial_noise, combined_score
    """
    sat_std, sat_mean = _saturation_stats(region_bgr)
    hue_std, hue_spread = _hue_stats(region_bgr)
    spatial_noise = _hue_spatial_noise(region_bgr)

    # Combined score: hue spread weighted by spatial noise.
    # Both signals must be elevated for a high score.
    noise_factor = max(0.1, spatial_noise / SPATIAL_NOISE_THRESHOLD)
    combined_score = hue_spread * noise_factor

    return {
        "sat_std": sat_std,
        "sat_mean": sat_mean,
        "hue_std": hue_std,
        "hue_spread": hue_spread,
        "spatial_noise": spatial_noise,
        "combined_score": combined_score,
    }


def _combine_body_regions(img: np.ndarray) -> np.ndarray:
    """Extract and vertically stack all body region pixels."""
    parts = []
    for y0, y1, x0, x1 in BODY_REGIONS:
        region = _extract_region(img, y0, y1, x0, x1)
        if region.size > 0:
            parts.append(region)

    if not parts:
        return np.empty((0, 0, 3), dtype=np.uint8)

    # Resize all parts to the same width for stacking
    target_w = max(p.shape[1] for p in parts)
    resized = []
    for p in parts:
        if p.shape[1] != target_w:
            p = cv2.resize(p, (target_w, p.shape[0]))
        resized.append(p)

    return np.vstack(resized)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_holo_type(image_path: str) -> tuple[str, float]:
    """Detect if a card is normal, holo, or reverse holo.

    Analyzes saturation variance, hue distribution, and spatial hue noise
    in the artwork vs body regions of a card segment image.

    Parameters
    ----------
    image_path : str
        Path to a card segment image (PNG/JPG).

    Returns
    -------
    (type, confidence) where type is one of:
        "normal"           - no holographic elements detected
        "holofoil"         - artwork area shows holographic signal
        "reverse_holofoil" - body/border area shows holographic signal

    Confidence is in [0.0, 1.0].  Values below 0.5 indicate unreliable
    detection -- the holographic effect may not be visible in this photo.

    Limitations
    -----------
    - Single-photo analysis cannot reliably detect holographic effects
      that are not visible at the captured angle/lighting.
    - "normal" at high confidence is more trustworthy than "holofoil"
      or "reverse_holofoil", because absence of signal is easier to
      confirm than presence.
    - Binder sleeve reflections and warm lighting reduce accuracy.

    See module docstring for full limitations documentation.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not decode image: {path}")

    return detect_holo_type_from_array(img, label=path.name)


def detect_holo_type_from_array(
    img: np.ndarray,
    *,
    label: str = "<array>",
) -> tuple[str, float]:
    """Detect holo type from an already-loaded BGR image array.

    Parameters
    ----------
    img : numpy.ndarray
        BGR image (as from cv2.imread).
    label : str
        Label for logging.

    Returns
    -------
    (type, confidence) -- see detect_holo_type docstring.
    """
    h, w = img.shape[:2]
    if h < 50 or w < 50:
        logger.warning("Image too small for holo detection: %dx%d", w, h)
        return "normal", 0.0

    # Extract artwork and body regions
    art_region = _extract_region(img, ART_Y0, ART_Y1, ART_X0, ART_X1)
    body_region = _combine_body_regions(img)

    if art_region.size == 0 or body_region.size == 0:
        logger.warning("Could not extract regions from %s", label)
        return "normal", 0.0

    # Measure holographic signals in each region
    art_metrics = _region_holo_score(art_region)
    body_metrics = _region_holo_score(body_region)

    art_score = art_metrics["combined_score"]
    body_score = body_metrics["combined_score"]
    art_sat_std = art_metrics["sat_std"]
    body_sat_std = body_metrics["sat_std"]
    art_hue_std = art_metrics["hue_std"]
    body_hue_std = body_metrics["hue_std"]

    logger.debug(
        "Holo detection for %s: "
        "art(sat_std=%.1f, hue_std=%.1f, noise=%.1f, score=%.1f) "
        "body(sat_std=%.1f, hue_std=%.1f, noise=%.1f, score=%.1f)",
        label,
        art_sat_std, art_hue_std, art_metrics["spatial_noise"], art_score,
        body_sat_std, body_hue_std, body_metrics["spatial_noise"], body_score,
    )

    # --- Classification logic ---

    # Check if body shows holographic signal
    body_is_holo = (
        body_score >= HOLO_SCORE_THRESHOLD
        and body_sat_std >= BODY_SAT_STD_THRESHOLD
        and body_hue_std >= BODY_HUE_STD_THRESHOLD
    )

    # Check if artwork shows holographic signal
    art_is_holo = (
        art_score >= HOLO_SCORE_THRESHOLD
        and art_sat_std >= BODY_SAT_STD_THRESHOLD
    )

    if body_is_holo and not art_is_holo:
        # Body has holo signal, artwork does not -> reverse holo
        # Confidence based on how far body exceeds thresholds
        ratio = body_score / max(art_score, 0.1)
        conf = min(0.95, 0.50 + 0.10 * (ratio - REVERSE_HOLO_RATIO))
        conf = max(MIN_HOLO_CONFIDENCE, conf)
        logger.debug("Holo result for %s: reverse_holofoil (conf=%.2f, "
                      "body/art ratio=%.1f)", label, conf, ratio)
        return "reverse_holofoil", round(conf, 2)

    if body_is_holo and art_is_holo:
        # Both regions show signal -- compare magnitudes
        if body_score > art_score * REVERSE_HOLO_RATIO:
            # Body dominates -> reverse holo (the artwork signal may be
            # spillover from a bright body or colorful artwork)
            ratio = body_score / max(art_score, 0.1)
            conf = min(0.85, 0.45 + 0.08 * (ratio - REVERSE_HOLO_RATIO))
            conf = max(MIN_HOLO_CONFIDENCE, conf)
            logger.debug("Holo result for %s: reverse_holofoil (both hot, "
                          "body dominates, conf=%.2f)", label, conf)
            return "reverse_holofoil", round(conf, 2)
        elif art_score > body_score * REVERSE_HOLO_RATIO:
            # Artwork dominates -> regular holo
            ratio = art_score / max(body_score, 0.1)
            conf = min(0.85, 0.45 + 0.08 * (ratio - REVERSE_HOLO_RATIO))
            conf = max(MIN_HOLO_CONFIDENCE, conf)
            logger.debug("Holo result for %s: holofoil (both hot, "
                          "art dominates, conf=%.2f)", label, conf)
            return "holofoil", round(conf, 2)
        else:
            # Both roughly equal -- ambiguous. Could be a full-art holo
            # or a photo artifact. Default to holofoil with low confidence.
            conf = MIN_HOLO_CONFIDENCE
            logger.debug("Holo result for %s: holofoil (both hot, ambiguous, "
                          "conf=%.2f)", label, conf)
            return "holofoil", round(conf, 2)

    if art_is_holo and not body_is_holo:
        # Only artwork has holo signal -> regular holo
        ratio = art_score / max(body_score, 0.1)
        conf = min(0.90, 0.50 + 0.10 * (ratio - 1.0))
        conf = max(MIN_HOLO_CONFIDENCE, conf)
        logger.debug("Holo result for %s: holofoil (conf=%.2f, "
                      "art/body ratio=%.1f)", label, conf, ratio)
        return "holofoil", round(conf, 2)

    # Neither region shows holo signal -> normal
    # Confidence is HIGH for normal -- absence of signal is reliable.
    # Scale confidence based on how far below thresholds both scores are.
    max_score = max(art_score, body_score)
    if max_score < HOLO_SCORE_THRESHOLD * 0.3:
        conf = 0.95  # Very clearly not holo
    elif max_score < HOLO_SCORE_THRESHOLD * 0.6:
        conf = 0.85
    elif max_score < HOLO_SCORE_THRESHOLD:
        conf = 0.70  # Borderline -- some signal but below threshold
    else:
        conf = 0.55  # One metric passed but not all three

    logger.debug("Holo result for %s: normal (conf=%.2f, max_score=%.1f)",
                  label, conf, max_score)
    return "normal", round(conf, 2)


def detect_holo_type_detailed(image_path: str) -> dict:
    """Like detect_holo_type but returns detailed metrics for debugging.

    Returns dict with keys:
        type: str ("normal", "holofoil", "reverse_holofoil")
        confidence: float
        artwork_metrics: dict with sat_std, sat_mean, hue_std, hue_spread,
                         spatial_noise, combined_score
        body_metrics: dict (same keys)
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not decode image: {path}")

    h, w = img.shape[:2]
    if h < 50 or w < 50:
        return {
            "type": "normal",
            "confidence": 0.0,
            "artwork_metrics": {},
            "body_metrics": {},
        }

    art_region = _extract_region(img, ART_Y0, ART_Y1, ART_X0, ART_X1)
    body_region = _combine_body_regions(img)

    art_metrics = _region_holo_score(art_region)
    body_metrics = _region_holo_score(body_region)

    holo_type, confidence = detect_holo_type_from_array(img, label=path.name)

    return {
        "type": holo_type,
        "confidence": confidence,
        "artwork_metrics": art_metrics,
        "body_metrics": body_metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    args = sys.argv[1:]

    if not args:
        print("Usage: python -m cardprice.ml.holo_detector <image> [image ...]")
        print()
        print("Detects holographic card type from binder page segments.")
        print("Returns: normal | holofoil | reverse_holofoil")
        sys.exit(1)

    if args[0] == "--eval":
        # Run against eval dataset (all should be "normal")
        import json

        eval_path = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "binder_eval.json"
        if not eval_path.exists():
            print(f"Eval file not found: {eval_path}")
            sys.exit(1)

        with open(eval_path) as f:
            eval_data = json.load(f)

        project_root = Path(__file__).resolve().parent.parent.parent
        total = 0
        correct = 0
        errors = []

        for page in eval_data["pages"]:
            seg_dir = project_root / page["segments_dir"]
            for card in page["cards"]:
                if card.get("card_id") is None:
                    continue  # empty slot

                segment_path = seg_dir / card["segment"]
                if not segment_path.exists():
                    print(f"  SKIP  {segment_path} (not found)")
                    continue

                total += 1
                try:
                    result = detect_holo_type_detailed(str(segment_path))
                    holo_type = result["type"]
                    conf = result["confidence"]

                    # All eval cards are "normal" variant
                    expected = "normal"
                    is_correct = holo_type == expected

                    if is_correct:
                        correct += 1
                        marker = "OK"
                    else:
                        marker = "MISS"
                        errors.append((segment_path.name, holo_type, conf))

                    art = result["artwork_metrics"]
                    body = result["body_metrics"]
                    print(
                        f"  [{marker:4s}] {card['name']:25s} "
                        f"-> {holo_type:18s} ({conf:.0%})  "
                        f"art(sat={art.get('sat_std', 0):.1f} "
                        f"hue={art.get('hue_std', 0):.1f} "
                        f"noise={art.get('spatial_noise', 0):.1f} "
                        f"score={art.get('combined_score', 0):.1f}) "
                        f"body(sat={body.get('sat_std', 0):.1f} "
                        f"hue={body.get('hue_std', 0):.1f} "
                        f"noise={body.get('spatial_noise', 0):.1f} "
                        f"score={body.get('combined_score', 0):.1f})"
                    )
                except Exception as e:
                    print(f"  [ERR ] {card['name']:25s} -> {e}")

        print()
        print(f"Accuracy: {correct}/{total} = {correct/total:.1%}" if total else "No cards found")
        if errors:
            print(f"False positives: {errors}")
    else:
        for p in args:
            try:
                result = detect_holo_type_detailed(p)
                holo_type = result["type"]
                conf = result["confidence"]
                art = result["artwork_metrics"]
                body = result["body_metrics"]
                print(
                    f"{Path(p).name:30s} -> {holo_type:18s} ({conf:.0%})"
                )
                print(
                    f"  artwork: sat_std={art['sat_std']:.1f} "
                    f"hue_std={art['hue_std']:.1f} "
                    f"spread={art['hue_spread']:.0f} "
                    f"noise={art['spatial_noise']:.1f} "
                    f"score={art['combined_score']:.1f}"
                )
                print(
                    f"  body:    sat_std={body['sat_std']:.1f} "
                    f"hue_std={body['hue_std']:.1f} "
                    f"spread={body['hue_spread']:.0f} "
                    f"noise={body['spatial_noise']:.1f} "
                    f"score={body['combined_score']:.1f}"
                )
            except Exception as e:
                print(f"{Path(p).name:30s} -> ERROR: {e}")
