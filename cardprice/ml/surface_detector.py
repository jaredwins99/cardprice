"""Surface defect detection via DINOv2 patch-level comparison.

Extracts 256 patch tokens (16x16 grid of 768-dim vectors) from DINOv2 ViT-B/14
for both a query card image and its reference image.  Corresponding patches are
compared via cosine similarity to produce a 16x16 anomaly heatmap.  Low similarity
at a patch location indicates a potential surface defect (scratch, crease, stain,
whitening, etc.).

No training required -- works with a single reference image per card.

Performance: ~6ms patch extraction + ~0.7ms comparison = ~13ms/card (GPU).

Usage::

    from cardprice.ml.surface_detector import detect_surface_defects

    result = detect_surface_defects("photo.jpg", "data/card_images/base1/base1-4_normal.png")
    print(result["defect_score"])        # 0.0 (mint) to 1.0 (heavily damaged)
    print(result["anomaly_map"].shape)   # (16, 16) float32
    print(result["defect_patches"])      # list of (row, col, similarity) for flagged patches
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Reuse the same ImageNet normalization and 224x224 resize as dino_matcher.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# DINOv2 ViT-B/14 with 224x224 input: 224/14 = 16 patches per side -> 256 patches
GRID_SIZE = 16
NUM_PATCHES = GRID_SIZE * GRID_SIZE  # 256
EMBED_DIM = 768

# Default thresholds (tunable)
DEFAULT_PATCH_THRESHOLD = 0.85  # patches below this similarity are flagged
DEFAULT_DEFECT_RATIO_THRESHOLD = 0.10  # >10% flagged patches = "damaged"


# ---------------------------------------------------------------------------
# Patch token extraction
# ---------------------------------------------------------------------------

def _get_model():
    """Get the cached DINOv2 model from dino_matcher (avoids loading twice)."""
    from cardprice.ml.dino_matcher import _load_model
    return _load_model()


def _prepare_tensor(image: Union[str, Path, Image.Image]) -> torch.Tensor:
    """Load and transform an image to a (1, 3, 224, 224) tensor."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Expected path or PIL Image, got {type(image)}")
    return _transform(image).unsqueeze(0)


def extract_patch_tokens(
    image: Union[str, Path, Image.Image],
) -> np.ndarray:
    """Extract 256 patch tokens (16x16 grid of 768-dim) from DINOv2.

    Uses ``model.get_intermediate_layers()`` to retrieve the final layer's
    patch tokens (excluding CLS), then L2-normalizes each token.

    Parameters
    ----------
    image : str, Path, or PIL.Image
        Input card image.

    Returns
    -------
    np.ndarray
        Shape ``(256, 768)`` float32, L2-normalized patch embeddings.
        Patches are in raster order: row 0 left-to-right, then row 1, etc.
    """
    model, device = _get_model()
    tensor = _prepare_tensor(image).to(device)

    with torch.no_grad():
        # get_intermediate_layers returns list of tensors, one per requested layer.
        # Each tensor is (batch, num_patches, embed_dim) -- patch tokens only (no CLS).
        outputs = model.get_intermediate_layers(tensor, n=1)
        patch_tokens = outputs[0].squeeze(0)  # (256, 768)

    assert patch_tokens.shape == (NUM_PATCHES, EMBED_DIM), (
        f"Expected ({NUM_PATCHES}, {EMBED_DIM}), got {patch_tokens.shape}"
    )

    tokens = patch_tokens.cpu().numpy().astype(np.float32)

    # L2-normalize each patch token
    norms = np.linalg.norm(tokens, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tokens /= norms

    return tokens


def extract_patch_tokens_batch(
    images: list[Union[str, Path, Image.Image]],
) -> list[np.ndarray]:
    """Extract patch tokens for multiple images in a single GPU forward pass.

    Parameters
    ----------
    images : list
        List of image paths or PIL Images.

    Returns
    -------
    list[np.ndarray]
        List of (256, 768) arrays, one per image.
    """
    if not images:
        return []

    model, device = _get_model()

    tensors = []
    valid_indices = []
    for i, img in enumerate(images):
        try:
            tensors.append(_prepare_tensor(img).squeeze(0))
            valid_indices.append(i)
        except Exception:
            logger.warning("Failed to load image %s for patch extraction", img)

    if not tensors:
        return [np.zeros((NUM_PATCHES, EMBED_DIM), dtype=np.float32)] * len(images)

    batch = torch.stack(tensors).to(device)  # (N, 3, 224, 224)

    with torch.no_grad():
        outputs = model.get_intermediate_layers(batch, n=1)
        all_patches = outputs[0]  # (N, 256, 768)

    all_patches = all_patches.cpu().numpy().astype(np.float32)

    results = [np.zeros((NUM_PATCHES, EMBED_DIM), dtype=np.float32)] * len(images)
    for j, orig_i in enumerate(valid_indices):
        tokens = all_patches[j]
        norms = np.linalg.norm(tokens, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tokens /= norms
        results[orig_i] = tokens

    return results


# ---------------------------------------------------------------------------
# Patch comparison
# ---------------------------------------------------------------------------

def compare_patches(
    query_patches: np.ndarray,
    ref_patches: np.ndarray,
) -> np.ndarray:
    """Compare corresponding patches via cosine similarity.

    Parameters
    ----------
    query_patches : np.ndarray
        Shape ``(256, 768)`` L2-normalized patch tokens from the query image.
    ref_patches : np.ndarray
        Shape ``(256, 768)`` L2-normalized patch tokens from the reference image.

    Returns
    -------
    np.ndarray
        Shape ``(16, 16)`` float32 similarity map.  Values in [-1, 1] where
        1.0 = identical patch, lower values = more anomalous.
    """
    assert query_patches.shape == (NUM_PATCHES, EMBED_DIM)
    assert ref_patches.shape == (NUM_PATCHES, EMBED_DIM)

    # Element-wise dot product of corresponding patches (both L2-normed -> cosine)
    similarities = np.sum(query_patches * ref_patches, axis=1)  # (256,)
    return similarities.reshape(GRID_SIZE, GRID_SIZE)


# ---------------------------------------------------------------------------
# Defect detection (high-level API)
# ---------------------------------------------------------------------------

def detect_surface_defects(
    query_image: Union[str, Path, Image.Image],
    ref_image: Union[str, Path, Image.Image],
    patch_threshold: float = DEFAULT_PATCH_THRESHOLD,
    *,
    query_patches: Optional[np.ndarray] = None,
    ref_patches: Optional[np.ndarray] = None,
) -> dict:
    """Detect surface defects by comparing patch-level DINOv2 features.

    Parameters
    ----------
    query_image : str, Path, or PIL.Image
        The photographed card to assess.
    ref_image : str, Path, or PIL.Image
        The pristine reference image for the same card.
    patch_threshold : float
        Cosine similarity below which a patch is flagged as anomalous.
        Default 0.85.
    query_patches : np.ndarray, optional
        Pre-computed (256, 768) patch tokens for the query (skips extraction).
    ref_patches : np.ndarray, optional
        Pre-computed (256, 768) patch tokens for the reference (skips extraction).

    Returns
    -------
    dict with keys:
        anomaly_map : np.ndarray
            (16, 16) cosine similarity map.  Lower = more anomalous.
        defect_map : np.ndarray
            (16, 16) boolean mask where True = flagged defect patch.
        defect_patches : list[tuple[int, int, float]]
            List of (row, col, similarity) for each flagged patch, sorted by
            similarity ascending (worst first).
        defect_count : int
            Number of patches flagged as defective.
        defect_ratio : float
            Fraction of patches flagged (0.0 to 1.0).
        defect_score : float
            Overall defect severity score in [0.0, 1.0].
            0.0 = mint condition, 1.0 = heavily damaged.
            Computed as weighted combination of defect ratio and mean anomaly
            depth of flagged patches.
        mean_similarity : float
            Mean patch similarity across all 256 patches.
        min_similarity : float
            Minimum patch similarity (worst single patch).
        patch_threshold : float
            The threshold used for flagging.
    """
    # Extract patch tokens (or use pre-computed)
    if query_patches is None:
        query_patches = extract_patch_tokens(query_image)
    if ref_patches is None:
        ref_patches = extract_patch_tokens(ref_image)

    # Compare
    anomaly_map = compare_patches(query_patches, ref_patches)

    # Flag defective patches
    defect_map = anomaly_map < patch_threshold

    # Collect flagged patches
    defect_patches = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if defect_map[r, c]:
                defect_patches.append((r, c, float(anomaly_map[r, c])))
    defect_patches.sort(key=lambda x: x[2])  # worst first

    defect_count = len(defect_patches)
    defect_ratio = defect_count / NUM_PATCHES

    mean_similarity = float(np.mean(anomaly_map))
    min_similarity = float(np.min(anomaly_map))

    # Compute defect score: blend of ratio and depth
    if defect_count > 0:
        # Mean anomaly depth: how far below threshold the flagged patches are
        flagged_sims = np.array([p[2] for p in defect_patches])
        mean_depth = float(np.mean(patch_threshold - flagged_sims))
        # Normalize depth: a patch at similarity 0.0 has depth = threshold
        normalized_depth = min(mean_depth / patch_threshold, 1.0)

        # Blend: 60% ratio weight (how widespread), 40% depth weight (how severe)
        defect_score = min(0.6 * (defect_ratio / 0.5) + 0.4 * normalized_depth, 1.0)
        # defect_ratio / 0.5 means 50% flagged patches -> ratio component = 0.6
    else:
        defect_score = 0.0

    logger.info(
        "Surface defect detection: score=%.3f, %d/%d patches flagged (ratio=%.2f), "
        "mean_sim=%.3f, min_sim=%.3f",
        defect_score, defect_count, NUM_PATCHES, defect_ratio,
        mean_similarity, min_similarity,
    )

    return {
        "anomaly_map": anomaly_map,
        "defect_map": defect_map,
        "defect_patches": defect_patches,
        "defect_count": defect_count,
        "defect_ratio": defect_ratio,
        "defect_score": defect_score,
        "mean_similarity": mean_similarity,
        "min_similarity": min_similarity,
        "patch_threshold": patch_threshold,
    }


# ---------------------------------------------------------------------------
# Condition grade estimation
# ---------------------------------------------------------------------------

# TCGPlayer-style condition grades mapped from defect_score ranges
_GRADE_THRESHOLDS = [
    (0.02, "Near Mint", "NM"),
    (0.08, "Lightly Played", "LP"),
    (0.20, "Moderately Played", "MP"),
    (0.40, "Heavily Played", "HP"),
    (1.01, "Damaged", "DMG"),
]


def estimate_condition(defect_result: dict) -> dict:
    """Map a defect detection result to a TCGPlayer-style condition grade.

    Parameters
    ----------
    defect_result : dict
        Output from ``detect_surface_defects()``.

    Returns
    -------
    dict with keys:
        grade : str
            Full grade name (e.g. "Near Mint", "Lightly Played").
        grade_abbrev : str
            Abbreviated grade (e.g. "NM", "LP").
        defect_score : float
            The underlying defect score.
        confidence : str
            "high" if score is well within grade boundaries, "low" if borderline.
    """
    score = defect_result["defect_score"]

    grade = "Damaged"
    abbrev = "DMG"
    prev_threshold = 0.0
    next_threshold = 1.0

    for threshold, name, short in _GRADE_THRESHOLDS:
        if score < threshold:
            grade = name
            abbrev = short
            next_threshold = threshold
            break
        prev_threshold = threshold

    # Confidence: high if we're in the middle of the grade range, low if near boundary
    range_size = next_threshold - prev_threshold
    if range_size > 0:
        position_in_range = (score - prev_threshold) / range_size
        confidence = "high" if 0.15 < position_in_range < 0.85 else "low"
    else:
        confidence = "low"

    return {
        "grade": grade,
        "grade_abbrev": abbrev,
        "defect_score": score,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Visualization helper (optional, requires matplotlib)
# ---------------------------------------------------------------------------

def render_heatmap(
    anomaly_map: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Surface Anomaly Heatmap",
    patch_threshold: float = DEFAULT_PATCH_THRESHOLD,
) -> Optional[np.ndarray]:
    """Render the 16x16 anomaly map as a color heatmap image.

    Parameters
    ----------
    anomaly_map : np.ndarray
        (16, 16) similarity map from ``compare_patches()`` or ``detect_surface_defects()``.
    output_path : str or Path, optional
        If provided, save the heatmap to this file.
    title : str
        Title for the plot.
    patch_threshold : float
        Threshold line to draw on the colorbar.

    Returns
    -------
    np.ndarray or None
        (H, W, 3) uint8 RGB image of the heatmap, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        logger.warning("matplotlib not available -- cannot render heatmap")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Use a diverging colormap centered on the threshold
    vmin = max(float(np.min(anomaly_map)) - 0.05, 0.0)
    vmax = 1.0
    vcenter = patch_threshold

    if vmin < vcenter < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = None

    im = ax.imshow(
        anomaly_map,
        cmap="RdYlGn",
        norm=norm,
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Patch column")
    ax.set_ylabel("Patch row")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine similarity")

    # Mark defect threshold
    cbar.ax.axhline(y=patch_threshold, color="black", linewidth=1.5, linestyle="--")

    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=100, bbox_inches="tight")
        logger.info("Saved heatmap to %s", output_path)

    # Convert to numpy array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    result = buf.copy()

    plt.close(fig)
    return result
