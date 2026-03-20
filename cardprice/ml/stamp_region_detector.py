"""Production stamp detector: combined pixel + DINOv2 features on era-specific crop.

Detects EX-era set logo stamps (ex7-ex16) and WotC prerelease stamps on
binder-scanned Pokemon cards.  Uses the trained ensemble model
(``data/stamp_combined_classifier.pkl``) which combines:

  1. DINOv2 CLS features (PCA-reduced to 5-dim) via logistic regression
  2. Pixel features (11-dim: edge density, Laplacian, gold ratio, etc.)

When the ensemble model is available, both classifiers vote with equal
weight (probability averaging).  When it is unavailable, falls back to a
pixel-only heuristic.

Usage::

    from cardprice.ml.stamp_region_detector import detect_stamp_region

    result = detect_stamp_region("path/to/card.jpg", set_id="ex15")
    print(result["stamped"])       # True/False
    print(result["confidence"])    # 0.0-1.0
    print(result["method"])        # "ensemble", "pixel_only"
    print(result["details"])       # feature values for debugging
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_COMBINED_MODEL_PATH = _DATA_DIR / "stamp_combined_classifier.pkl"

# Era-specific stamp regions (normalized [x0, y0, x1, y1]).
_ERA_STAMP_REGIONS: dict[str, list[float]] = {
    "ex_era":          [0.55, 0.40, 0.92, 0.68],
    "wotc_prerelease": [0.50, 0.30, 0.90, 0.58],
    "default":         [0.50, 0.35, 0.92, 0.65],
}

# Control region: left side, same vertical band (no stamp expected).
_CONTROL_REGION: list[float] = [0.10, 0.45, 0.45, 0.70]

# Sets with stamped reverse holos (EX Team Rocket Returns - Power Keepers)
_STAMPED_SETS = frozenset({
    "ex7", "ex8", "ex9", "ex10", "ex11",
    "ex12", "ex13", "ex14", "ex15", "ex16",
})

# WotC-era sets (may have prerelease stamps)
_WOTC_SETS = frozenset({
    "base1", "base2", "base3", "base4", "base5", "base6",
    "basep", "gym1", "gym2",
    "neo1", "neo2", "neo3", "neo4",
})

# Cached ensemble model
_ensemble_model: Optional[dict] = None


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------

def _crop_region_cv2(img_bgr: np.ndarray, region: list[float]) -> np.ndarray:
    """Crop a normalized [x0, y0, x1, y1] region from a BGR numpy array."""
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = region
    return img_bgr[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def _crop_region_pil(image: Image.Image, region: list[float]) -> Image.Image:
    """Crop a normalized [x0, y0, x1, y1] region from a PIL Image."""
    w, h = image.size
    x0, y0, x1, y1 = region
    return image.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))


# ---------------------------------------------------------------------------
# Era / region resolution
# ---------------------------------------------------------------------------

def _resolve_era(card_id: Optional[str], set_id: Optional[str]) -> str:
    """Determine the stamp era key for region selection."""
    if set_id is None and card_id:
        bare = card_id.split("/")[0]
        set_id = bare.rsplit("-", 1)[0] if "-" in bare else bare

    if not set_id:
        return "default"
    if set_id in _STAMPED_SETS:
        return "ex_era"
    if set_id in _WOTC_SETS:
        return "wotc_prerelease"
    return "default"


def _get_stamp_region(era: str) -> list[float]:
    """Get the stamp crop region for a given era key."""
    return _ERA_STAMP_REGIONS.get(era, _ERA_STAMP_REGIONS["default"])


# ---------------------------------------------------------------------------
# Pixel feature extraction
# ---------------------------------------------------------------------------

def _compute_edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels that are Canny edges."""
    if gray.size == 0:
        return 0.0
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges > 0))


def _compute_laplacian_var(gray: np.ndarray) -> float:
    """Variance of Laplacian -- texture energy."""
    if gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))


def _compute_gold_pixel_ratio(bgr: np.ndarray) -> float:
    """Fraction of pixels matching gold color profile (HSV)."""
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 40, 120])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.mean(mask > 0))


def _compute_high_freq_energy(gray: np.ndarray) -> float:
    """High-frequency energy ratio via FFT (mask out center 20%)."""
    if gray.size == 0:
        return 0.0
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 5
    magnitude = np.abs(fshift)
    mask = np.ones_like(magnitude)
    mask[cy - r:cy + r, cx - r:cx + r] = 0
    high_freq = np.sum(magnitude * mask)
    total = np.sum(magnitude) + 1e-10
    return float(high_freq / total)


def _extract_pixel_features(
    img_bgr: np.ndarray, stamp_region: list[float]
) -> dict[str, float]:
    """Extract all pixel features from stamp and control regions.

    Returns a dict with the same 11 features expected by the trained
    pixel classifier, plus raw values for debugging.
    """
    stamp_crop = _crop_region_cv2(img_bgr, stamp_region)
    control_crop = _crop_region_cv2(img_bgr, _CONTROL_REGION)

    empty_result = {
        "edge_density": 0.0, "edge_density_ratio": 1.0,
        "gold_pixel_ratio": 0.0, "laplacian_var": 0.0,
        "laplacian_ratio": 1.0, "brightness_ratio": 1.0,
        "high_freq_energy": 0.0, "high_freq_ratio": 1.0,
        "mean_saturation": 0.0, "sat_std": 0.0,
        "gold_hue_fraction": 0.0,
    }
    if stamp_crop.size == 0 or control_crop.size == 0:
        return empty_result

    stamp_gray = cv2.cvtColor(stamp_crop, cv2.COLOR_BGR2GRAY)
    control_gray = cv2.cvtColor(control_crop, cv2.COLOR_BGR2GRAY)

    # Edge density
    stamp_edges = _compute_edge_density(stamp_gray)
    control_edges = _compute_edge_density(control_gray)
    edge_ratio = stamp_edges / (control_edges + 1e-10)

    # Laplacian variance
    stamp_lap = _compute_laplacian_var(stamp_gray)
    control_lap = _compute_laplacian_var(control_gray)
    lap_ratio = stamp_lap / (control_lap + 1e-10)

    # Brightness ratio
    stamp_brightness = float(np.mean(stamp_gray))
    control_brightness = float(np.mean(control_gray))
    brightness_ratio = stamp_brightness / (control_brightness + 1e-10)

    # HSV features on stamp region
    stamp_hsv = cv2.cvtColor(stamp_crop, cv2.COLOR_BGR2HSV)
    mean_sat = float(np.mean(stamp_hsv[:, :, 1]))
    sat_std = float(np.std(stamp_hsv[:, :, 1]))
    hue = stamp_hsv[:, :, 0]
    gold_hue_frac = float(np.mean((hue >= 15) & (hue <= 40)))

    # High-frequency energy
    stamp_hf = _compute_high_freq_energy(stamp_gray)
    control_hf = _compute_high_freq_energy(control_gray)
    hf_ratio = stamp_hf / (control_hf + 1e-10)

    # Gold pixel ratio
    gold_ratio = _compute_gold_pixel_ratio(stamp_crop)

    return {
        "edge_density": stamp_edges,
        "edge_density_ratio": edge_ratio,
        "gold_pixel_ratio": gold_ratio,
        "laplacian_var": stamp_lap,
        "laplacian_ratio": lap_ratio,
        "brightness_ratio": brightness_ratio,
        "high_freq_energy": stamp_hf,
        "high_freq_ratio": hf_ratio,
        "mean_saturation": mean_sat,
        "sat_std": sat_std,
        "gold_hue_fraction": gold_hue_frac,
    }


def _pixel_features_to_vector(
    features: dict[str, float],
    feature_names: list[str],
) -> np.ndarray:
    """Convert pixel features dict to a numpy vector in the order the
    trained classifier expects."""
    return np.array(
        [[features.get(name, 0.0) for name in feature_names]],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Ensemble model
# ---------------------------------------------------------------------------

def _load_ensemble_model() -> Optional[dict]:
    """Load the trained ensemble model (DINOv2 + pixel classifiers)."""
    global _ensemble_model
    if _ensemble_model is not None:
        return _ensemble_model

    if not _COMBINED_MODEL_PATH.exists():
        logger.debug(
            "Ensemble stamp model not found at %s", _COMBINED_MODEL_PATH,
        )
        return None

    logger.info("Loading ensemble stamp model from %s", _COMBINED_MODEL_PATH)
    with open(_COMBINED_MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    # Validate expected keys for the ensemble format
    model_type = data.get("model_type", "")
    if model_type == "ensemble_prob_avg" and "clf_dino" in data:
        _ensemble_model = data
        logger.info(
            "Ensemble stamp model loaded: %s (pixel_weight=%.2f, "
            "loo_cv=%.1f%%, transfer=%.1f%%)",
            model_type,
            data.get("weight_pixel", 0.5),
            data.get("metrics", {}).get("loo_cv_binder_acc", 0) * 100,
            data.get("metrics", {}).get("transfer_binder_acc", 0) * 100,
        )
        return _ensemble_model

    # Legacy format: single model with 'model' key
    if "model" in data:
        logger.info("Legacy stamp model format detected (feature_type=%s)",
                     data.get("feature_type", "unknown"))
        _ensemble_model = data
        return _ensemble_model

    logger.warning("Unknown stamp model format: keys=%s", list(data.keys()))
    return None


def _ensemble_predict(
    stamp_crop_pil: Image.Image,
    pixel_features: dict[str, float],
) -> Optional[tuple[bool, float, float, float, float]]:
    """Run the full ensemble (DINOv2 + pixel) classifier.

    Returns (is_stamped, confidence, combined_prob, dino_prob, pixel_prob)
    or None if the model is unavailable or prediction fails.
    """
    model_data = _load_ensemble_model()
    if model_data is None:
        return None

    model_type = model_data.get("model_type", "")

    if model_type == "ensemble_prob_avg":
        return _predict_ensemble_prob_avg(
            model_data, stamp_crop_pil, pixel_features
        )

    # Legacy single-model format
    if "model" in model_data:
        return _predict_legacy(model_data, stamp_crop_pil)

    return None


def _predict_ensemble_prob_avg(
    model_data: dict,
    stamp_crop_pil: Image.Image,
    pixel_features: dict[str, float],
) -> Optional[tuple[bool, float, float, float, float]]:
    """Predict using the ensemble_prob_avg model format.

    Pipeline:
      DINOv2: CLS (768) -> PCA (5) -> scaler_dino -> clf_dino -> prob_dino
      Pixel:  features (11) -> scaler_pixel -> clf_pixel -> prob_pixel
      Combined: weight_pixel * prob_pixel + (1 - weight_pixel) * prob_dino
    """
    try:
        # --- DINOv2 branch ---
        from cardprice.ml.stamp_classifier import _extract_features

        cls_token, _ = _extract_features(stamp_crop_pil)
        X_dino = cls_token.reshape(1, -1)

        pca = model_data.get("pca")
        if pca is not None:
            X_dino = pca.transform(X_dino)

        scaler_dino = model_data.get("scaler_dino")
        if scaler_dino is not None:
            X_dino = scaler_dino.transform(X_dino)

        clf_dino = model_data["clf_dino"]
        dino_proba = clf_dino.predict_proba(X_dino)[0]
        dino_prob = float(dino_proba[1])

        # --- Pixel branch ---
        feature_names = model_data["pixel_feature_names"]
        X_pixel = _pixel_features_to_vector(pixel_features, feature_names)

        scaler_pixel = model_data.get("scaler_pixel")
        if scaler_pixel is not None:
            X_pixel = scaler_pixel.transform(X_pixel)

        clf_pixel = model_data["clf_pixel"]
        pixel_proba = clf_pixel.predict_proba(X_pixel)[0]
        pixel_prob = float(pixel_proba[1])

        # --- Combine ---
        w_pixel = model_data.get("weight_pixel", 0.5)
        combined_prob = w_pixel * pixel_prob + (1.0 - w_pixel) * dino_prob
        is_stamped = combined_prob > 0.50

        # Confidence: how far from the decision boundary
        confidence = abs(combined_prob - 0.50) * 2.0
        confidence = min(confidence, 0.95)

        return is_stamped, confidence, combined_prob, dino_prob, pixel_prob

    except Exception as e:
        logger.warning("Ensemble prediction failed: %s", e)
        return None


def _predict_legacy(
    model_data: dict,
    stamp_crop_pil: Image.Image,
) -> Optional[tuple[bool, float, float, float, float]]:
    """Predict using the legacy single-model format."""
    try:
        from cardprice.ml.stamp_classifier import (
            _extract_features,
            _build_feature_vector,
        )

        cls_token, patch_tokens = _extract_features(stamp_crop_pil)
        feature_type = model_data.get("feature_type", "cls")

        if feature_type.startswith("dino_pca"):
            X = cls_token.reshape(1, -1)
            pca = model_data.get("pca")
            if pca is not None:
                X = pca.transform(X)
        elif feature_type.startswith("dino_cls_only"):
            X = cls_token.reshape(1, -1)
        else:
            X = _build_feature_vector(
                cls_token, patch_tokens, feature_type, model_data
            )

        scaler = model_data.get("scaler")
        if scaler is not None:
            is_scaled = (
                "scaled" in feature_type
                or model_data.get("metrics", {}).get("scaled", False)
            )
            if is_scaled:
                X = scaler.transform(X)

        clf = model_data["model"]
        model_type = model_data.get("model_type", "lr")

        if model_type in ("lr", "lr_combined"):
            proba = clf.predict_proba(X)[0]
            stamp_prob = float(proba[1])
            is_stamped = bool(clf.predict(X)[0] == 1)
            confidence = float(max(proba))
        elif model_type == "mlp":
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            clf.to(device)
            clf.eval()
            with torch.no_grad():
                logit = clf(
                    torch.tensor(X, dtype=torch.float32).to(device)
                ).squeeze()
                stamp_prob = float(torch.sigmoid(logit).cpu())
            is_stamped = stamp_prob > 0.5
            confidence = stamp_prob if is_stamped else 1.0 - stamp_prob
        else:
            return None

        return is_stamped, confidence, stamp_prob, stamp_prob, 0.0

    except Exception as e:
        logger.warning("Legacy model prediction failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Pixel-only fallback scoring
# ---------------------------------------------------------------------------

def _pixel_heuristic_score(features: dict[str, float]) -> float:
    """Heuristic stamp probability from pixel features (no trained model).

    Returns a rough probability in [0, 1].  Used when the trained ensemble
    model is unavailable.
    """
    score = 0.0

    edr = features.get("edge_density_ratio", 1.0)
    if edr > 1.5:
        score += 0.30
    elif edr > 1.2:
        score += 0.15

    lap = features.get("laplacian_ratio", 1.0)
    if lap > 1.3:
        score += 0.15
    elif lap > 1.1:
        score += 0.08

    br = features.get("brightness_ratio", 1.0)
    if abs(br - 1.0) > 0.10:
        score += 0.10

    svr = features.get("sat_std", 0.0)
    if svr > 50:
        score += 0.10

    gold = features.get("gold_pixel_ratio", 0.0)
    if gold > 0.5:
        score += 0.10
    elif gold > 0.1:
        score += 0.05

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def detect_stamp_region(
    image_path: str,
    card_id: Optional[str] = None,
    set_id: Optional[str] = None,
) -> dict:
    """Detect stamps using combined pixel + DINOv2 features on era-specific crop.

    Parameters
    ----------
    image_path : str
        Path to the card image (binder scan crop).
    card_id : str, optional
        Full card identifier (e.g. "ex15-44/normal").  Used for era lookup.
    set_id : str, optional
        Set identifier (e.g. "ex15").  Used as fallback if card_id is absent.

    Returns
    -------
    dict with keys:
        stamped : bool
            Whether the card appears to have a stamp overlay.
        confidence : float
            Classification confidence (0.0 to 1.0).
        method : str
            "ensemble" -- trained DINOv2 + pixel model
            "pixel_only" -- pixel heuristic fallback
        details : dict
            Feature values for debugging / logging.
    """
    # --- Load image ---
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # --- Determine era and crop region ---
    era = _resolve_era(card_id, set_id)
    stamp_region = _get_stamp_region(era)

    # --- Extract pixel features (always, ~5ms) ---
    pixel_features = _extract_pixel_features(img_bgr, stamp_region)

    # --- Try trained ensemble model ---
    ensemble_result = None
    dino_prob = None
    pixel_prob = None
    combined_prob = None

    try:
        pil_image = Image.open(image_path).convert("RGB")
        stamp_crop_pil = _crop_region_pil(pil_image, stamp_region)
        ensemble_result = _ensemble_predict(stamp_crop_pil, pixel_features)
    except Exception as e:
        logger.debug("Ensemble stamp detection failed: %s", e)

    if ensemble_result is not None:
        is_stamped, confidence, combined_prob, dino_prob, pixel_prob = ensemble_result
        method = "ensemble"
    else:
        # Pixel-only fallback
        heuristic_prob = _pixel_heuristic_score(pixel_features)
        is_stamped = heuristic_prob >= 0.40
        confidence = abs(heuristic_prob - 0.40) * 2.0
        confidence = min(confidence, 0.85)
        combined_prob = heuristic_prob
        method = "pixel_only"

    # --- Build details dict ---
    details = {
        "edge_density_ratio": round(pixel_features["edge_density_ratio"], 4),
        "stamp_crop_prob": round(dino_prob, 4) if dino_prob is not None else None,
        "pixel_score": round(pixel_prob or combined_prob, 4),
        "ensemble_score": round(combined_prob, 4) if combined_prob is not None else None,
        "laplacian_ratio": round(pixel_features["laplacian_ratio"], 4),
        "brightness_diff": round(
            (pixel_features["brightness_ratio"] - 1.0) * 100, 2
        ),
        "saturation_var_ratio": round(pixel_features.get("sat_std", 0), 2),
        "gold_pixel_ratio": round(pixel_features["gold_pixel_ratio"], 4),
        "era": era,
        "stamp_region": stamp_region,
    }

    logger.info(
        "Stamp detection: stamped=%s conf=%.3f method=%s era=%s "
        "edge_ratio=%.3f dino_prob=%s pixel_prob=%s combined=%.3f",
        is_stamped, confidence, method, era,
        pixel_features["edge_density_ratio"],
        f"{dino_prob:.3f}" if dino_prob is not None else "N/A",
        f"{pixel_prob:.3f}" if pixel_prob is not None else "N/A",
        combined_prob or 0.0,
    )

    return {
        "stamped": is_stamped,
        "confidence": round(confidence, 4),
        "method": method,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def detect_stamp_batch(
    image_paths: list[str],
    card_ids: Optional[list[str]] = None,
    set_ids: Optional[list[str]] = None,
) -> list[dict]:
    """Run stamp detection on a batch of images.

    Parameters
    ----------
    image_paths : list[str]
        Paths to card images.
    card_ids : list[str], optional
        Card identifiers, same length as image_paths.
    set_ids : list[str], optional
        Set identifiers, same length as image_paths.

    Returns
    -------
    list[dict] -- one result dict per image (same format as detect_stamp_region).
    """
    n = len(image_paths)
    card_ids = card_ids or [None] * n
    set_ids = set_ids or [None] * n

    results = []
    for path, cid, sid in zip(image_paths, card_ids, set_ids):
        try:
            result = detect_stamp_region(path, card_id=cid, set_id=sid)
        except Exception as e:
            logger.warning("Stamp detection failed for %s: %s", path, e)
            result = {
                "stamped": False,
                "confidence": 0.0,
                "method": "error",
                "details": {"error": str(e)},
            }
        results.append(result)
    return results
