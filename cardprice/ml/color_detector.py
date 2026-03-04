"""Pokemon card type detection from background/border color analysis.

Uses K-means clustering on sampled border regions to find dominant card
frame colors, then classifies them using HSV-based rules calibrated from
actual binder page scans and reference card images.

Approach:
  1. Sample pixels from card frame regions (name bar, text box, borders)
  2. Filter out very dark pixels (binder sleeve bleed, shadows)
  3. K-means cluster the remaining pixels into groups
  4. Score each cluster as a "card background" candidate
  5. Classify the best candidate cluster in HSV space

Key challenges for binder page phone photos:
  - Binder sleeve adds orange/amber cast from page edges
  - Photos are desaturated compared to reference images
  - EX-era cards have silver metallic frames that override type color
  - Black text in sampled regions skews dark
  - White attack text boxes skew bright/desaturated

Pokemon card frame colors:
    Fire        = red/orange
    Water       = blue/teal
    Grass       = green
    Lightning   = yellow
    Psychic     = purple/lavender
    Fighting    = brown/tan (warmer/darker than Colorless)
    Darkness    = very dark/black
    Metal       = neutral gray/silver
    Dragon      = gold/amber
    Fairy       = pink
    Colorless   = warm cream/beige (high value, low sat)
"""

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _white_balance(img: np.ndarray) -> np.ndarray:
    """Correct warm color cast from binder page lighting.

    Uses a two-strategy approach:
      1. White-patch: find bright near-neutral pixels in the card text box
         and use them to estimate the illuminant color.
      2. Gray-world fallback: assume mean of channels should be equal.

    Key design decisions:
      - Only boosts the blue channel (warm cast = excess red/green, deficit blue).
        Never reduces red or green, which would desaturate purples/pinks.
      - Strength is proportional to the detected cast magnitude.
      - Very gentle on images that don't show a warm cast.
    """
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    img_f = img.astype(np.float32)

    gains = None  # BGR gains

    # --- Strategy 1: White patch from card text-box area ---
    text_region = img_f[int(h * 0.60):int(h * 0.80), int(w * 0.35):int(w * 0.65)]
    if text_region.size > 0:
        pixels = text_region.reshape(-1, 3)
        brightness = pixels.mean(axis=1)
        spread = pixels.max(axis=1) - pixels.min(axis=1)
        white_mask = (brightness > 130) & (spread < 90)
        white_pixels = pixels[white_mask]

        if len(white_pixels) > 50:
            white_mean = white_pixels.mean(axis=0)  # BGR
            # Detect warm cast: R > B significantly
            b_mean, g_mean, r_mean = white_mean[0], white_mean[1], white_mean[2]

            if r_mean > b_mean + 5:
                # There IS a warm cast. Compute how much to boost blue.
                # Target: make the white patch neutral (B == R).
                # But only boost blue — never reduce red (preserves purples).
                b_gain = r_mean / max(b_mean, 30)
                # Also slightly boost green if it's behind red
                g_gain = r_mean / max(g_mean, 30) if r_mean > g_mean + 3 else 1.0

                # Clamp: don't over-boost
                b_gain = min(b_gain, 1.50)
                g_gain = min(g_gain, 1.20)

                # Scale by confidence (how strong the cast is)
                cast_strength = (r_mean - b_mean) / r_mean  # 0-1
                # Apply proportionally: full correction when cast > 15%
                alpha = min(cast_strength / 0.15, 1.0) * 0.75

                gains = np.array([
                    1.0 + (b_gain - 1.0) * alpha,  # B
                    1.0 + (g_gain - 1.0) * alpha,  # G
                    1.0,                             # R: never reduce
                ], dtype=np.float32)

                logger.debug(
                    "White-balance (white patch): gains=(%.3f, %.3f, %.3f) "
                    "cast=%.1f%% from %d white pixels, "
                    "white_mean_bgr=(%d,%d,%d)",
                    gains[0], gains[1], gains[2],
                    cast_strength * 100, len(white_pixels),
                    int(b_mean), int(g_mean), int(r_mean),
                )

    # --- Strategy 2: Gray world fallback ---
    if gains is None:
        interior = img_f[int(h * 0.10):int(h * 0.90), int(w * 0.15):int(w * 0.85)]
        if interior.size == 0:
            return img

        channel_means = interior.reshape(-1, 3).mean(axis=0)  # BGR
        b_mean, g_mean, r_mean = channel_means

        if r_mean > b_mean + 5:
            b_gain = r_mean / max(b_mean, 20)
            g_gain = r_mean / max(g_mean, 20) if r_mean > g_mean + 3 else 1.0

            b_gain = min(b_gain, 1.40)
            g_gain = min(g_gain, 1.15)

            cast_strength = (r_mean - b_mean) / r_mean
            alpha = min(cast_strength / 0.20, 1.0) * 0.60  # more conservative

            gains = np.array([
                1.0 + (b_gain - 1.0) * alpha,
                1.0 + (g_gain - 1.0) * alpha,
                1.0,
            ], dtype=np.float32)

            logger.debug(
                "White-balance (gray world): gains=(%.3f, %.3f, %.3f) "
                "cast=%.1f%% channel_means_bgr=(%d,%d,%d)",
                gains[0], gains[1], gains[2],
                cast_strength * 100,
                int(b_mean), int(g_mean), int(r_mean),
            )
        else:
            # No warm cast detected
            return img

    # Apply gains (only boosts, never reduces)
    corrected = img_f * gains[np.newaxis, np.newaxis, :]
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


def _sample_regions(img: np.ndarray) -> np.ndarray:
    """Sample pixels from type-indicative card regions.

    Returns (N, 3) BGR pixel array.
    """
    h, w = img.shape[:2]
    regions = []

    # Region 1: Name bar (top strip above art)
    r = img[int(h * 0.04):int(h * 0.11), int(w * 0.30):int(w * 0.70)]
    if r.size > 0:
        regions.append(r)

    # Region 2: Text box background (below art)
    # This is the most type-indicative region on most cards
    r = img[int(h * 0.58):int(h * 0.82), int(w * 0.32):int(w * 0.68)]
    if r.size > 0:
        regions.append(r)

    # Region 3: Left inner border at text-box height
    r = img[int(h * 0.56):int(h * 0.83), int(w * 0.18):int(w * 0.30)]
    if r.size > 0:
        regions.append(r)

    # Region 4: Right inner border at text-box height
    r = img[int(h * 0.56):int(h * 0.83), int(w * 0.70):int(w * 0.82)]
    if r.size > 0:
        regions.append(r)

    if not regions:
        return np.empty((0, 3), dtype=np.uint8)

    return np.vstack([r.reshape(-1, 3) for r in regions])


def _find_dominant_colors(
    pixels_bgr: np.ndarray,
    k: int = 6,
) -> List[Tuple[np.ndarray, float]]:
    """Cluster pixels and return (centroid_bgr, fraction) for each cluster.

    Sorted by cluster size descending.
    """
    n = len(pixels_bgr)
    if n == 0:
        return []

    if n < k * 10:
        median = np.median(pixels_bgr.astype(np.float32), axis=0)
        return [(median, 1.0)]

    # Subsample if huge
    if n > 40000:
        idx = np.random.default_rng(42).choice(n, 40000, replace=False)
        pixels_bgr = pixels_bgr[idx]
        n = 40000

    data = pixels_bgr.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    labels_flat = labels.ravel()
    result = []
    for i in range(k):
        count = np.sum(labels_flat == i)
        if count > 0:
            result.append((centers[i], count / n))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def _bgr_to_hsv(bgr: np.ndarray) -> Tuple[float, float, float]:
    """Convert a single BGR float color to HSV."""
    pix = np.array([[bgr.astype(np.uint8)]], dtype=np.uint8)
    hsv = cv2.cvtColor(pix, cv2.COLOR_BGR2HSV)[0, 0]
    return float(hsv[0]), float(hsv[1]), float(hsv[2])


def _is_binder_orange(h: float, s: float, v: float) -> bool:
    """Check if a color looks like the orange binder page background."""
    # Orange binder: H=8-18, S=150+, V=150+
    return 5 <= h <= 22 and s >= 140 and v >= 140


def _is_text_or_noise(h: float, s: float, v: float) -> bool:
    """Check if color is likely black text or white text box."""
    if v < 40:
        return True  # very dark (text, shadows)
    if s < 15 and v > 220:
        return True  # nearly white (text box background)
    return False


def _classify_hsv(h: float, s: float, v: float) -> List[Tuple[str, float]]:
    """Classify a single HSV color into Pokemon type predictions.

    OpenCV HSV: H in [0,179], S in [0,255], V in [0,255].

    Rules are ordered so that more-specific conditions are checked first.
    The tricky boundaries:
      - Colorless (H=15-25, S=30-65, V=190+) vs Lightning (H=23-36, S=60+)
      - Psychic near-red (H=170-5, S=50-95) vs Fire (H=0-12, S=95+)
      - Fighting (H=13-22, S=85+) vs Colorless (similar hue, lower sat)
    """
    results = {}

    # Very dark -> Darkness type
    if v < 60:
        results["Darkness"] = 0.80
        results["Metal"] = 0.10
        results["Psychic"] = 0.10
        return _sort(results)

    # Very low saturation: Metal or Colorless
    # Binder photos are often dimmer than reference, so use a lower V
    # threshold. True Metal cards (silver EX frames) are typically V<120
    # with a very neutral hue, while Colorless in dim lighting sits V=110-170.
    if s < 20:
        if v < 80:
            results["Darkness"] = 0.45
            results["Metal"] = 0.40
            results["Colorless"] = 0.15
        elif v < 130:
            # Dim binder photos: Colorless cards often land here.
            # True Metal (silver EX) is rarer than Colorless in binder scans.
            results["Colorless"] = 0.45
            results["Metal"] = 0.40
            results["Water"] = 0.15
        else:
            results["Colorless"] = 0.55
            results["Metal"] = 0.35
            results["Water"] = 0.10
        return _sort(results)

    # --- Chromatic: classify by hue ---
    # Check specific hue ranges before fallback low-saturation rules

    # Hue near 0/180 boundary (red/pink/magenta)
    if h >= 170 or h <= 5:
        if s >= 95 and v >= 140:
            results["Fire"] = 0.80
            results["Psychic"] = 0.15
            results["Fighting"] = 0.05
        elif s >= 55 and v >= 180:
            # Moderate sat, high value, near-red -> likely Fire (Charizard-like)
            results["Fire"] = 0.55
            results["Psychic"] = 0.30
            results["Colorless"] = 0.15
        elif s >= 50:
            results["Psychic"] = 0.70
            results["Fire"] = 0.15
            results["Darkness"] = 0.15
        else:
            results["Colorless"] = 0.50
            results["Psychic"] = 0.30
            results["Metal"] = 0.20
        return _sort(results)

    # Orange-red (H 6-15) -- Fire and Fighting overlap here
    if 6 <= h <= 15:
        if s >= 90 and v >= 180:
            # Saturated warm + bright -> Fire (reference cards: H=11-13, S=95-135, V=230+)
            results["Fire"] = 0.70
            results["Fighting"] = 0.15
            results["Dragon"] = 0.15
        elif s >= 90:
            # Saturated warm, lower brightness -> Fighting
            results["Fighting"] = 0.50
            results["Fire"] = 0.30
            results["Dragon"] = 0.20
        elif s >= 50:
            # Moderate saturation
            results["Fighting"] = 0.45
            results["Fire"] = 0.30
            results["Colorless"] = 0.25
        else:
            results["Colorless"] = 0.50
            results["Fighting"] = 0.30
            results["Metal"] = 0.20
        return _sort(results)

    # Warm tan/brown (H 16-22) -- Colorless, Fighting, Dragon
    if 16 <= h <= 22:
        if s >= 130 and v >= 160:
            results["Dragon"] = 0.50
            results["Lightning"] = 0.30
            results["Fighting"] = 0.20
        elif s >= 90:
            results["Fighting"] = 0.55
            results["Dragon"] = 0.20
            results["Lightning"] = 0.15
            results["Colorless"] = 0.10
        elif s >= 50 and v >= 180:
            results["Colorless"] = 0.65
            results["Fighting"] = 0.20
            results["Lightning"] = 0.15
        elif s >= 50:
            results["Fighting"] = 0.40
            results["Colorless"] = 0.35
            results["Metal"] = 0.25
        else:
            results["Colorless"] = 0.50
            results["Metal"] = 0.30
            results["Fighting"] = 0.20
        return _sort(results)

    # Yellow (H 23-30) -- Lightning zone (pure yellow)
    if 23 <= h <= 30:
        if s >= 40:
            results["Lightning"] = 0.75
            results["Grass"] = 0.15
            results["Dragon"] = 0.10
        elif s >= 20:
            results["Lightning"] = 0.45
            results["Colorless"] = 0.40
            results["Metal"] = 0.15
        else:
            results["Colorless"] = 0.50
            results["Metal"] = 0.30
            results["Lightning"] = 0.20
        return _sort(results)

    # Yellow-green (H 31-44) -- Grass/Lightning overlap
    # Modern Grass cards (SV, SM, SWSH) have H=31-44 with high saturation
    # Lightning cards in this range tend to have lower saturation
    if 31 <= h <= 44:
        if s >= 110:
            # High saturation yellow-green -> Grass (modern green)
            results["Grass"] = 0.65
            results["Lightning"] = 0.20
            results["Dragon"] = 0.15
        elif s >= 40:
            # Moderate saturation -> ambiguous, slight Grass lean
            results["Grass"] = 0.40
            results["Lightning"] = 0.35
            results["Dragon"] = 0.25
        elif s >= 20:
            results["Lightning"] = 0.40
            results["Grass"] = 0.30
            results["Colorless"] = 0.30
        else:
            results["Colorless"] = 0.50
            results["Grass"] = 0.30
            results["Metal"] = 0.20
        return _sort(results)

    # Green (H 45-85)
    if 45 <= h <= 85:
        if s >= 30:
            results["Grass"] = 0.85
            results["Water"] = 0.10
            results["Dragon"] = 0.05
        else:
            results["Metal"] = 0.40
            results["Grass"] = 0.35
            results["Colorless"] = 0.25
        return _sort(results)

    # Teal/blue (H 86-110)
    if 86 <= h <= 110:
        if s >= 30:
            results["Water"] = 0.85
            results["Grass"] = 0.10
            results["Psychic"] = 0.05
        else:
            results["Metal"] = 0.50
            results["Water"] = 0.35
            results["Colorless"] = 0.15
        return _sort(results)

    # Blue-purple (H 111-155)
    if 111 <= h <= 155:
        if s >= 35:
            results["Psychic"] = 0.75
            results["Water"] = 0.15
            results["Darkness"] = 0.10
        else:
            results["Metal"] = 0.45
            results["Psychic"] = 0.35
            results["Water"] = 0.20
        return _sort(results)

    # Purple-magenta (H 156-169)
    if 156 <= h <= 169:
        if s >= 50:
            results["Psychic"] = 0.55
            results["Fairy"] = 0.25
            results["Darkness"] = 0.20
        else:
            results["Darkness"] = 0.45
            results["Metal"] = 0.30
            results["Psychic"] = 0.25
        return _sort(results)

    # Fallback
    results["Colorless"] = 0.40
    results["Metal"] = 0.35
    results["Fighting"] = 0.25
    return _sort(results)


def _sort(d: dict) -> List[Tuple[str, float]]:
    """Sort type->confidence dict into descending list."""
    return sorted(d.items(), key=lambda x: x[1], reverse=True)


def _select_best_cluster(
    clusters: List[Tuple[np.ndarray, float]],
) -> Tuple[np.ndarray, float]:
    """Select the cluster most likely to represent the card background color.

    Filters out:
      - Binder orange clusters (saturated warm orange from page edges)
      - Very dark clusters (text, shadows, binder sleeve)
      - Very white clusters (text boxes, glare)

    Among remaining, picks the largest by pixel fraction.
    If everything is filtered out, returns the overall largest cluster.
    """
    if not clusters:
        return np.array([180, 200, 220], dtype=np.float32), 0.0

    candidates = []
    for bgr, frac in clusters:
        h, s, v = _bgr_to_hsv(bgr)

        # Skip binder orange
        if _is_binder_orange(h, s, v):
            continue
        # Skip text/noise
        if _is_text_or_noise(h, s, v):
            continue
        # Skip very small clusters (likely noise)
        if frac < 0.05:
            continue

        candidates.append((bgr, frac, h, s, v))

    if not candidates:
        # Fallback: return largest cluster
        return clusters[0]

    # Among candidates, prefer the one with the most pixels
    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0]
    return best[0], best[1]


def _multi_cluster_vote(
    clusters: List[Tuple[np.ndarray, float]],
) -> List[Tuple[str, float]]:
    """Classify using weighted votes from ALL non-noise clusters.

    Each cluster votes for types based on its HSV, weighted by its pixel
    fraction. The combined votes are normalized to produce final predictions.

    This is more robust than single-cluster classification because even if
    the dominant cluster is ambiguous (e.g., warm-shifted by lighting), a
    secondary cluster with a more distinctive color can shift the prediction.
    """
    if not clusters:
        return [("Colorless", 0.50)]

    combined_votes: dict = {}
    total_weight = 0.0

    for bgr, frac in clusters:
        h, s, v = _bgr_to_hsv(bgr)

        # Skip binder orange
        if _is_binder_orange(h, s, v):
            continue
        # Skip pure text/noise
        if _is_text_or_noise(h, s, v):
            continue
        # Skip tiny clusters
        if frac < 0.03:
            continue

        # Get this cluster's type predictions
        preds = _classify_hsv(h, s, v)

        # Weight by cluster fraction
        for type_name, conf in preds:
            combined_votes[type_name] = combined_votes.get(type_name, 0) + conf * frac

        total_weight += frac

    if not combined_votes or total_weight == 0:
        # Fallback: classify the largest cluster directly
        if clusters:
            h, s, v = _bgr_to_hsv(clusters[0][0])
            return _classify_hsv(h, s, v)
        return [("Colorless", 0.50)]

    # Normalize
    results = []
    for type_name, score in combined_votes.items():
        results.append((type_name, score / total_weight))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def detect_color_type(
    image_path,
    *,
    top_n: int = 3,
) -> List[Tuple[str, float]]:
    """Detect Pokemon card type from border/frame color.

    Parameters
    ----------
    image_path : str or Path
        Path to a card segment image.
    top_n : int
        Number of top predictions to return.

    Returns
    -------
    list of (type_name, confidence)
        Sorted by confidence descending. Confidence is in [0, 1].
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    return detect_color_type_from_array(img, top_n=top_n, label=image_path.name)


def detect_color_type_from_array(
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
    # White-balance correction to remove warm amber cast from binder lighting
    img = _white_balance(img)

    pixels = _sample_regions(img)

    if len(pixels) == 0:
        logger.warning("No pixels sampled from %s", label)
        return [("Colorless", 0.0)]

    # Cluster into dominant colors
    clusters = _find_dominant_colors(pixels, k=6)

    # Use multi-cluster voting for more robust classification
    results = _multi_cluster_vote(clusters)

    # Also get the single best cluster for logging
    dominant, frac = _select_best_cluster(clusters)
    h, s, v = _bgr_to_hsv(dominant)

    logger.debug(
        "Color detection for %s: dominant BGR=(%.0f,%.0f,%.0f) "
        "HSV=(%.0f,%.0f,%.0f) frac=%.0f%% -> %s",
        label,
        dominant[0], dominant[1], dominant[2],
        h, s, v, frac * 100,
        ", ".join(f"{n} ({c:.0%})" for n, c in results[:top_n]),
    )

    return results[:top_n]


def detect_color_type_with_debug(
    image_path,
    *,
    top_n: int = 3,
) -> dict:
    """Like detect_color_type but returns additional debug info.

    Returns dict with keys:
      - predictions: list of (type, confidence)
      - dominant_bgr: the dominant BGR color found
      - dominant_hsv: the dominant color in HSV
      - pixel_count: number of pixels sampled
      - clusters: all cluster centroids with fractions
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    # White-balance correction
    img = _white_balance(img)

    pixels = _sample_regions(img)
    if len(pixels) == 0:
        return {
            "predictions": [("Colorless", 0.0)],
            "dominant_bgr": (0, 0, 0),
            "dominant_hsv": (0, 0, 0),
            "pixel_count": 0,
            "clusters": [],
        }

    clusters = _find_dominant_colors(pixels, k=6)
    results = _multi_cluster_vote(clusters)
    dominant, frac = _select_best_cluster(clusters)
    h, s, v = _bgr_to_hsv(dominant)

    cluster_info = []
    for bgr, cf in clusters:
        ch, cs, cv = _bgr_to_hsv(bgr)
        cluster_info.append({
            "bgr": tuple(int(x) for x in bgr),
            "hsv": (int(ch), int(cs), int(cv)),
            "fraction": round(cf, 3),
        })

    return {
        "predictions": results[:top_n],
        "dominant_bgr": tuple(int(x) for x in dominant),
        "dominant_hsv": (int(h), int(s), int(v)),
        "pixel_count": len(pixels),
        "clusters": cluster_info,
    }


# ---------------------------------------------------------------------------
# Ground truth for eval cards.
# Determined by visually inspecting reference card images to identify the
# actual frame color (not the Pokemon species type, since delta species
# and EX cards may differ).
# ---------------------------------------------------------------------------
_EVAL_GROUND_TRUTH = {
    # Page 1 (EX delta species + EX era)
    # EX cards have silver/metallic frames; text box shows type color
    "page_20260228_174819_cards_v4/card_08.png": "Metal",        # Flygon ex delta - silver EX frame
    "page_20260228_174819_cards_v4/card_07.png": "Metal",        # Jirachi ex - silver EX frame
    "page_20260228_174819_cards_v4/card_06.png": "Metal",        # Swampert ex - silver EX frame
    "page_20260228_174819_cards_v4/card_05.png": "Colorless",    # Wigglytuff ex - pinkish/Colorless EX
    "page_20260228_174819_cards_v4/card_04.png": "Metal",        # Delcatty ex - silver EX frame
    "page_20260228_174819_cards_v4/card_03.png": "Psychic",      # Vibrava delta - purple Psychic frame
    "page_20260228_174819_cards_v4/card_02.png": "Psychic",      # Trapinch delta - purple Psychic frame
    "page_20260228_174819_cards_v4/card_01.png": "Colorless",    # Skitty - tan Colorless frame
    "page_20260228_174819_cards_v4/card_00.png": "Lightning",    # Dragonair delta - yellow Lightning frame
    # Page 2 (e-series)
    "page_20260228_195512_cards/card_00.png": "Psychic",         # Natu
    "page_20260228_195512_cards/card_01.png": "Psychic",         # Xatu
    "page_20260228_195512_cards/card_02.png": "Psychic",         # Mr. Mime
    "page_20260228_195512_cards/card_03.png": "Psychic",         # Natu
    "page_20260228_195512_cards/card_04.png": "Psychic",         # Xatu
    "page_20260228_195512_cards/card_05.png": "Colorless",       # Rattata
    "page_20260228_195512_cards/card_06.png": "Colorless",       # Rattata
    "page_20260228_195512_cards/card_07.png": "Colorless",       # Raticate
    "page_20260228_195512_cards/card_08.png": "Colorless",       # Ditto
    # Page 3 (mixed eras)
    "page_20260228_202134_cards/card_00.png": None,              # Empty slot
    "page_20260228_202134_cards/card_01.png": "Colorless",       # Latios delta - tan Colorless
    "page_20260228_202134_cards/card_02.png": "Colorless",       # Latias ex - silver/Colorless EX
    "page_20260228_202134_cards/card_03.png": "Grass",           # Venusaur - green Grass frame
    "page_20260228_202134_cards/card_04.png": "Colorless",       # Flygon - tan Colorless frame
    "page_20260228_202134_cards/card_05.png": "Lightning",       # Raikou - yellow Lightning frame
    "page_20260228_202134_cards/card_06.png": "Water",           # Kingdra - blue Water frame
    "page_20260228_202134_cards/card_07.png": "Water",           # Suicune - blue Water frame
    "page_20260228_202134_cards/card_08.png": "Colorless",       # Staraptor - tan Colorless frame
}


def run_eval(data_root: str = "data/inbox") -> dict:
    """Run color detection against eval ground truth."""
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
            debug = detect_color_type_with_debug(full_path, top_n=3)
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
                "dominant_bgr": debug["dominant_bgr"],
                "dominant_hsv": debug["dominant_hsv"],
                "correct": is_correct,
                "clusters": debug["clusters"],
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
        print("Running eval against ground truth...")
        print("=" * 80)
        eval_result = run_eval()

        for r in eval_result["results"]:
            marker = "OK" if r["correct"] else "MISS"
            conf_str = f"{r.get('confidence', 0):.0%}" if "confidence" in r else "?"
            bgr = r.get("dominant_bgr", "?")
            hsv = r.get("dominant_hsv", "?")
            print(
                f"[{marker:4s}] {r['path']:55s} "
                f"expected={r['expected']:12s} "
                f"got={r['predicted']:12s} "
                f"conf={conf_str:>4s} "
                f"bgr={bgr} hsv={hsv}"
            )
            # Show clusters for misses
            if not r["correct"] and "clusters" in r:
                for i, cl in enumerate(r.get("clusters", [])[:4]):
                    print(
                        f"       cluster {i}: "
                        f"bgr={cl['bgr']} hsv={cl['hsv']} "
                        f"frac={cl['fraction']:.1%}"
                    )

        print("=" * 80)
        print(
            f"Accuracy: {eval_result['correct']}/{eval_result['total']} "
            f"= {eval_result['accuracy']:.1%}"
        )
    else:
        for p in args:
            try:
                debug = detect_color_type_with_debug(p, top_n=5)
                top = debug["predictions"][0]
                alts = ", ".join(
                    f"{n} {c:.0%}" for n, c in debug["predictions"][1:]
                )
                print(
                    f"{Path(p).name:30s} -> {top[0]:12s} ({top[1]:.0%}) "
                    f"BGR={debug['dominant_bgr']} HSV={debug['dominant_hsv']}"
                    + (f"   alts: {alts}" if alts else "")
                )
                for i, cl in enumerate(debug["clusters"][:4]):
                    print(
                        f"  cluster {i}: bgr={cl['bgr']} hsv={cl['hsv']} "
                        f"frac={cl['fraction']:.1%}"
                    )
            except Exception as e:
                print(f"{Path(p).name:30s} -> ERROR: {e}")
