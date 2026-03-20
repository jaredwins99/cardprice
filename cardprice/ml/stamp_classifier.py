"""Stamp classifier: detect EX-era stamp overlays on Pokemon cards.

Uses DINOv2 ViT-B/14 features fed through a trained logistic regression
classifier. The model is trained on synthetic stamped/clean pairs and
validated on real-world images.

Usage::

    from cardprice.ml.stamp_classifier import classify_stamp

    result = classify_stamp("path/to/card.jpg")
    print(result["stamped"])           # True/False
    print(result["confidence"])        # 0.0-1.0
    print(result["stamp_probability"]) # 0.0-1.0

    # Era-aware stamp region cropping (preferred when card_id is known):
    from cardprice.ml.stamp_classifier import classify_stamp_region

    result = classify_stamp_region("path/to/card.jpg", card_id="ex7-1/normal")
    # Same return format, but uses cropped stamp region for better accuracy.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Model paths -- prefer combined model (DINOv2 + edge density), fall back to basic
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_MODEL_PATH = _DATA_DIR / "stamp_classifier.pkl"
_COMBINED_MODEL_PATH = _DATA_DIR / "stamp_combined_classifier.pkl"

# Cached models
_classifier: Optional[dict] = None
_combined_classifier: Optional[dict] = None

# Era-specific stamp region coordinates (normalized [x0, y0, x1, y1])
# These define where the set logo stamp appears on the card artwork.
_ERA_STAMP_REGIONS: dict[str, list[float]] = {
    "ex_era": [0.55, 0.40, 0.92, 0.68],
    "wotc_base": [0.55, 0.40, 0.90, 0.65],
}

# Control region: same vertical band on the opposite side (for edge density ratio)
_CONTROL_REGION: list[float] = [0.10, 0.45, 0.45, 0.70]

# DINOv2 constants
GRID_SIZE = 16
EMBED_DIM = 768

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _get_model():
    """Get the cached DINOv2 model from dino_matcher (avoids loading twice)."""
    from cardprice.ml.dino_matcher import _load_model
    return _load_model()


def _load_classifier() -> dict:
    """Load the stamp classifier from disk, caching after first load."""
    global _classifier
    if _classifier is not None:
        return _classifier

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Stamp classifier model not found at {_MODEL_PATH}. "
            "Run 'python scripts/train_stamp_classifier.py' to train it."
        )

    logger.info("Loading stamp classifier from %s", _MODEL_PATH)
    with open(_MODEL_PATH, "rb") as f:
        _classifier = pickle.load(f)

    logger.info(
        "Stamp classifier loaded: feature_type=%s, val_acc=%.1f%%",
        _classifier["feature_type"],
        _classifier["metrics"]["val_acc"] * 100,
    )
    return _classifier


def _extract_features(image: Union[str, Path, Image.Image]) -> tuple[np.ndarray, np.ndarray]:
    """Extract CLS token and patch tokens from an image.

    Returns:
        cls_token: (768,) L2-normalized
        patch_tokens: (256, 768) L2-normalized
    """
    model, device = _get_model()

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Expected path or PIL Image, got {type(image)}")

    tensor = _transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_out = model(tensor)  # (1, 768)
        patch_out = model.get_intermediate_layers(tensor, n=1)
        patch_tokens = patch_out[0].squeeze(0)  # (256, 768)

    # L2-normalize CLS
    cls_np = cls_out.cpu().numpy().astype(np.float32).squeeze()
    norm = np.linalg.norm(cls_np)
    if norm > 0:
        cls_np /= norm

    # L2-normalize patches
    patches_np = patch_tokens.cpu().numpy().astype(np.float32)
    pnorms = np.linalg.norm(patches_np, axis=1, keepdims=True)
    pnorms[pnorms == 0] = 1.0
    patches_np /= pnorms

    return cls_np, patches_np


def _get_region_patches(patch_tokens: np.ndarray, region: str) -> np.ndarray:
    """Extract patches from a specific region. Input: (256, 768)."""
    grid = patch_tokens.reshape(GRID_SIZE, GRID_SIZE, EMBED_DIM)
    regions = {
        'br': (slice(8, None), slice(8, None)),
        'bl': (slice(8, None), slice(None, 8)),
        'tr': (slice(None, 8), slice(8, None)),
        'tl': (slice(None, 8), slice(None, 8)),
        'top': (slice(None, 8), slice(None)),
        'bottom': (slice(8, None), slice(None)),
        'all': (slice(None), slice(None)),
        'center': (slice(4, 12), slice(4, 12)),
        'bottom_center': (slice(10, None), slice(4, 12)),
    }
    if region not in regions:
        raise ValueError(f"Unknown region: {region}")
    r, c = regions[region]
    return grid[r, c, :].reshape(-1, EMBED_DIM)


def _patch_stats(patches: np.ndarray) -> np.ndarray:
    """Compute mean, std, max, min over patches. Input: (K, 768). Output: (768*4,)."""
    return np.concatenate([
        np.mean(patches, axis=0),
        np.std(patches, axis=0),
        np.max(patches, axis=0),
        np.min(patches, axis=0),
    ])


def _build_feature_vector(
    cls_token: np.ndarray, patch_tokens: np.ndarray, feature_type: str,
    classifier_data: dict,
) -> np.ndarray:
    """Build the feature vector based on the trained model's feature type.

    The feature_type may have a '_scaled' suffix, indicating a StandardScaler
    should be applied after building raw features.
    """
    # Strip _scaled suffix -- scaling is applied afterwards
    base_type = feature_type.replace("_scaled", "")

    if base_type == "cls":
        raw = cls_token.reshape(1, -1)

    elif base_type == "br_stats":
        br = _get_region_patches(patch_tokens, 'br')
        raw = _patch_stats(br).reshape(1, -1)

    elif base_type == "all_stats":
        all_patches = patch_tokens  # (256, 768)
        raw = _patch_stats(all_patches).reshape(1, -1)

    elif base_type == "cls_br_stats":
        br = _get_region_patches(patch_tokens, 'br')
        stats = _patch_stats(br)
        raw = np.concatenate([cls_token, stats]).reshape(1, -1)

    elif base_type == "cls_all_stats":
        stats = _patch_stats(patch_tokens)
        raw = np.concatenate([cls_token, stats]).reshape(1, -1)

    elif base_type == "multi_region_stats":
        parts = []
        for r in ['br', 'bl', 'tr', 'tl']:
            parts.append(_patch_stats(_get_region_patches(patch_tokens, r)))
        br_mean = np.mean(_get_region_patches(patch_tokens, 'br'), axis=0)
        bl_mean = np.mean(_get_region_patches(patch_tokens, 'bl'), axis=0)
        tr_mean = np.mean(_get_region_patches(patch_tokens, 'tr'), axis=0)
        tl_mean = np.mean(_get_region_patches(patch_tokens, 'tl'), axis=0)
        parts.extend([br_mean - bl_mean, br_mean - tr_mean, br_mean - tl_mean])
        raw = np.concatenate(parts).reshape(1, -1)

    elif base_type == "cls_multi_region":
        multi = _build_feature_vector(cls_token, patch_tokens, "multi_region_stats", classifier_data)
        raw = np.concatenate([cls_token.reshape(1, -1), multi], axis=1)

    elif base_type == "bottom_center_stats":
        bc = _get_region_patches(patch_tokens, 'bottom_center')
        raw = _patch_stats(bc).reshape(1, -1)

    elif base_type == "center_stats":
        center = _get_region_patches(patch_tokens, 'center')
        raw = _patch_stats(center).reshape(1, -1)

    elif base_type == "variance_ratio":
        br = _get_region_patches(patch_tokens, 'br')
        all_p = patch_tokens  # (256, 768)
        bc = _get_region_patches(patch_tokens, 'bottom_center')
        br_var = np.var(br, axis=0)
        all_var = np.maximum(np.var(all_p, axis=0), 1e-8)
        bc_var = np.var(bc, axis=0)
        raw = np.concatenate([
            _patch_stats(br),
            (br_var / all_var),
            (bc_var / all_var),
        ]).reshape(1, -1)

    elif base_type == "binder_robust":
        br = _get_region_patches(patch_tokens, 'br')
        bl = _get_region_patches(patch_tokens, 'bl')
        bc = _get_region_patches(patch_tokens, 'bottom_center')
        tr = _get_region_patches(patch_tokens, 'tr')
        all_p = patch_tokens
        br_mean = np.mean(br, axis=0)
        bl_mean = np.mean(bl, axis=0)
        tr_mean = np.mean(tr, axis=0)
        br_var = np.var(br, axis=0)
        bc_var = np.var(bc, axis=0)
        all_var = np.maximum(np.var(all_p, axis=0), 1e-8)
        raw = np.concatenate([
            _patch_stats(br),
            _patch_stats(bc),
            br_mean - bl_mean,
            br_mean - tr_mean,
            br_var / all_var,
            bc_var / all_var,
        ]).reshape(1, -1)

    elif base_type == "cls_binder_robust":
        binder = _build_feature_vector(cls_token, patch_tokens, "binder_robust", classifier_data)
        raw = np.concatenate([cls_token.reshape(1, -1), binder], axis=1)

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    # Apply scaler if present and feature type is scaled
    if "_scaled" in feature_type:
        scaler = classifier_data.get("scaler")
        if scaler is not None:
            raw = scaler.transform(raw)

    return raw


def classify_stamp(image: Union[str, Path, Image.Image]) -> dict:
    """Classify if a card image has an EX-era stamp overlay.

    Parameters
    ----------
    image : str, Path, or PIL.Image
        Card image to classify.

    Returns
    -------
    dict with keys:
        stamped : bool
            Whether the card appears to have a stamp overlay.
        confidence : float
            Classification confidence (0.0 to 1.0). Higher = more certain.
        stamp_probability : float
            Raw probability of being stamped (0.0 to 1.0).
    """
    classifier = _load_classifier()
    feature_type = classifier["feature_type"]

    # Extract DINOv2 features
    cls_token, patch_tokens = _extract_features(image)

    # Build feature vector
    X = _build_feature_vector(cls_token, patch_tokens, feature_type, classifier)

    # Predict
    is_stamped, confidence, stamp_prob = _predict_with_classifier(classifier, X)

    logger.info(
        "Stamp classification: stamped=%s, confidence=%.3f, stamp_prob=%.3f",
        is_stamped, confidence, stamp_prob,
    )

    return {
        "stamped": is_stamped,
        "confidence": confidence,
        "stamp_probability": stamp_prob,
    }


# ---------------------------------------------------------------------------
# Combined (region-cropped) classifier
# ---------------------------------------------------------------------------


def _load_combined_classifier() -> Optional[dict]:
    """Load the combined stamp classifier (DINOv2 + edge density) if available."""
    global _combined_classifier
    if _combined_classifier is not None:
        return _combined_classifier

    if not _COMBINED_MODEL_PATH.exists():
        logger.debug(
            "Combined stamp classifier not found at %s, will use basic classifier",
            _COMBINED_MODEL_PATH,
        )
        return None

    logger.info("Loading combined stamp classifier from %s", _COMBINED_MODEL_PATH)
    with open(_COMBINED_MODEL_PATH, "rb") as f:
        _combined_classifier = pickle.load(f)

    logger.info(
        "Combined stamp classifier loaded: feature_type=%s",
        _combined_classifier.get("feature_type", "unknown"),
    )
    return _combined_classifier


def _crop_region(image: Image.Image, region: list[float]) -> Image.Image:
    """Crop a normalized [x0, y0, x1, y1] region from a PIL Image."""
    w, h = image.size
    x0, y0, x1, y1 = region
    return image.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))


def _compute_edge_density_ratio(
    image: Image.Image, stamp_region: list[float]
) -> float:
    """Compute edge density ratio: stamp_region edges / control_region edges.

    A stamped card has more edge activity in the stamp region (text/logo)
    compared to a mirrored control region on the left side of the card.
    """
    try:
        import cv2
    except ImportError:
        logger.debug("cv2 not available for edge density computation")
        return 1.0

    img_np = np.array(image)
    h, w = img_np.shape[:2]

    # Crop stamp region
    sx0, sy0, sx1, sy1 = stamp_region
    stamp_crop = img_np[int(h * sy0):int(h * sy1), int(w * sx0):int(w * sx1)]

    # Crop control region (left side, same vertical band)
    cx0, cy0, cx1, cy1 = _CONTROL_REGION
    control_crop = img_np[int(h * cy0):int(h * cy1), int(w * cx0):int(w * cx1)]

    # Convert to grayscale and compute Canny edges
    stamp_gray = cv2.cvtColor(stamp_crop, cv2.COLOR_RGB2GRAY)
    control_gray = cv2.cvtColor(control_crop, cv2.COLOR_RGB2GRAY)

    stamp_edges = cv2.Canny(stamp_gray, 50, 150)
    control_edges = cv2.Canny(control_gray, 50, 150)

    stamp_density = float(np.mean(stamp_edges > 0))
    control_density = float(np.mean(control_edges > 0))

    ratio = stamp_density / (control_density + 1e-10)
    logger.debug(
        "Edge density: stamp=%.4f, control=%.4f, ratio=%.3f",
        stamp_density, control_density, ratio,
    )
    return ratio


def _get_stamp_region_for_card(
    card_id: Optional[str] = None, set_id: Optional[str] = None
) -> Optional[list[float]]:
    """Look up the era-specific stamp region for a card.

    Tries card_attributes first (O(1) lookup), falls back to set_id heuristic.

    Returns:
        Normalized [x0, y0, x1, y1] stamp region, or None if no region defined.
    """
    era_key = None

    # Try O(1) card_attributes lookup
    if card_id:
        try:
            from cardprice.ml.card_attributes import get_card_attrs
            attrs = get_card_attrs(card_id)
            if attrs is not None:
                era_key = attrs.era
        except Exception as e:
            logger.debug("card_attributes lookup failed: %s", e)

    # Fallback: infer era from set_id prefix
    if era_key is None and set_id:
        if set_id.startswith("ex"):
            era_key = "ex_era"
        elif set_id in (
            "base1", "base2", "base3", "base4", "base5", "base6",
            "basep", "gym1", "gym2", "neo1", "neo2", "neo3", "neo4",
        ):
            era_key = "wotc_base"

    if era_key is None:
        return None

    return _ERA_STAMP_REGIONS.get(era_key)


def _predict_with_classifier(
    classifier: dict, X: np.ndarray
) -> tuple[bool, float, float]:
    """Run prediction on a feature vector with a given classifier dict.

    Returns:
        (is_stamped, confidence, stamp_probability)
    """
    clf = classifier["model"]
    model_type = classifier.get("model_type", "lr")

    if model_type == "lr":
        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]
        stamp_prob = float(proba[1])
        is_stamped = bool(pred == 1)
        confidence = float(max(proba))
    elif model_type == "mlp":
        import torch as _torch
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        clf.to(device)
        clf.eval()
        with _torch.no_grad():
            logit = clf(_torch.tensor(X, dtype=_torch.float32).to(device)).squeeze()
            stamp_prob = float(_torch.sigmoid(logit).cpu())
        is_stamped = stamp_prob > 0.5
        confidence = stamp_prob if is_stamped else 1.0 - stamp_prob
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return is_stamped, confidence, stamp_prob


def classify_stamp_region(
    image: Union[str, Path, Image.Image],
    card_id: Optional[str] = None,
    set_id: Optional[str] = None,
) -> dict:
    """Classify stamp using era-specific region cropping for better accuracy.

    Crops the stamp region based on the card's era, extracts DINOv2 features
    from the crop (not the whole card), and optionally combines with edge
    density ratio. Falls back to whole-card ``classify_stamp()`` when no
    era-specific region is defined or the combined model is unavailable.

    Parameters
    ----------
    image : str, Path, or PIL.Image
        Card image to classify.
    card_id : str, optional
        Full card identifier (e.g. "ex7-1/normal"). Used for era lookup.
    set_id : str, optional
        Set identifier (e.g. "ex7"). Used as fallback if card_id is not given.

    Returns
    -------
    dict with keys:
        stamped : bool
        confidence : float (0.0 to 1.0)
        stamp_probability : float (0.0 to 1.0)
        method : str -- "stamp_region" or "stamp_whole_card"
    """
    # Resolve set_id from card_id if not provided
    if set_id is None and card_id:
        bare = card_id.split("/")[0]
        set_id = bare.rsplit("-", 1)[0] if "-" in bare else bare

    # Look up era-specific stamp region
    stamp_region = _get_stamp_region_for_card(card_id=card_id, set_id=set_id)

    if stamp_region is None:
        logger.debug(
            "No era-specific stamp region for card_id=%s set_id=%s, "
            "falling back to whole-card classify_stamp",
            card_id, set_id,
        )
        result = classify_stamp(image)
        result["method"] = "stamp_whole_card"
        return result

    # Load the image
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise TypeError(f"Expected path or PIL Image, got {type(image)}")

    # Crop the stamp region
    stamp_crop = _crop_region(pil_image, stamp_region)
    logger.debug(
        "Cropped stamp region %s -> %dx%d",
        stamp_region, stamp_crop.width, stamp_crop.height,
    )

    # Compute edge density ratio on the stamp region
    edge_density_ratio = _compute_edge_density_ratio(pil_image, stamp_region)

    # Try combined classifier first (DINOv2 crop features + edge density)
    combined = _load_combined_classifier()
    if combined is not None:
        try:
            cls_token, patch_tokens = _extract_features(stamp_crop)
            feature_type = combined["feature_type"]
            dino_features = _build_feature_vector(
                cls_token, patch_tokens, feature_type, combined
            )
            # Append edge density ratio as an extra feature
            edr_feature = np.array([[edge_density_ratio]], dtype=np.float32)
            X = np.concatenate([dino_features, edr_feature], axis=1)

            is_stamped, confidence, stamp_prob = _predict_with_classifier(combined, X)

            logger.info(
                "Stamp region classification (combined): stamped=%s, "
                "confidence=%.3f, stamp_prob=%.3f, edge_ratio=%.3f",
                is_stamped, confidence, stamp_prob, edge_density_ratio,
            )
            return {
                "stamped": is_stamped,
                "confidence": confidence,
                "stamp_probability": stamp_prob,
                "edge_density_ratio": edge_density_ratio,
                "method": "stamp_region_combined",
            }
        except Exception as e:
            logger.warning(
                "Combined classifier failed, falling back to basic crop: %s", e
            )

    # Fall back: use the basic classifier on crop features
    classifier = _load_classifier()
    cls_token, patch_tokens = _extract_features(stamp_crop)
    feature_type = classifier["feature_type"]
    X = _build_feature_vector(cls_token, patch_tokens, feature_type, classifier)

    is_stamped, confidence, stamp_prob = _predict_with_classifier(classifier, X)

    logger.info(
        "Stamp region classification (basic crop): stamped=%s, "
        "confidence=%.3f, stamp_prob=%.3f, edge_ratio=%.3f",
        is_stamped, confidence, stamp_prob, edge_density_ratio,
    )
    return {
        "stamped": is_stamped,
        "confidence": confidence,
        "stamp_probability": stamp_prob,
        "edge_density_ratio": edge_density_ratio,
        "method": "stamp_region",
    }
