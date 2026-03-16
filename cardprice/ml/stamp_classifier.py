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

# Model path
_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "stamp_classifier.pkl"

# Cached model
_classifier: Optional[dict] = None

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
    clf = classifier["model"]
    feature_type = classifier["feature_type"]
    model_type = classifier.get("model_type", "lr")

    # Extract DINOv2 features
    cls_token, patch_tokens = _extract_features(image)

    # Build feature vector
    X = _build_feature_vector(cls_token, patch_tokens, feature_type, classifier)

    # Predict based on model type
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

    logger.info(
        "Stamp classification: stamped=%s, confidence=%.3f, stamp_prob=%.3f",
        is_stamped, confidence, stamp_prob,
    )

    return {
        "stamped": is_stamped,
        "confidence": confidence,
        "stamp_probability": stamp_prob,
    }
