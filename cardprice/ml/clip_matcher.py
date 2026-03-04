"""Card identification using CLIP embeddings.

Provides text-to-image and image-to-image matching for Pokemon cards
using OpenAI's CLIP (ViT-Large/14) model.
"""

import io
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from transformers import CLIPModel, CLIPProcessor

from cardprice.db.session import SessionLocal
from sqlalchemy import text

logger = logging.getLogger(__name__)

MODEL_NAME = "openai/clip-vit-large-patch14"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_model: Optional[CLIPModel] = None
_processor: Optional[CLIPProcessor] = None


def _get_model_and_processor() -> tuple[CLIPModel, CLIPProcessor]:
    """Lazy-load and cache the CLIP model and processor."""
    global _model, _processor
    if _model is None or _processor is None:
        logger.info("Loading CLIP model: %s", MODEL_NAME)
        _model = CLIPModel.from_pretrained(MODEL_NAME)
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _processor


def _extract_text_features(model, **inputs) -> torch.Tensor:
    """Extract text features from CLIP, handling transformers 5.x API change.

    In transformers >=5.0, get_text_features returns BaseModelOutputWithPooling
    instead of a plain tensor. The pooler_output is already projected through
    text_projection, so we return it directly.
    """
    out = model.get_text_features(**inputs)
    if isinstance(out, torch.Tensor):
        return out
    # transformers 5.x: pooler_output is already the projected embedding
    return out.pooler_output


def _extract_image_features(model, **inputs) -> torch.Tensor:
    """Extract image features from CLIP, handling transformers 5.x API change.

    In transformers >=5.0, get_image_features returns BaseModelOutputWithPooling
    instead of a plain tensor. The pooler_output is already projected through
    visual_projection, so we return it directly.
    """
    out = model.get_image_features(**inputs)
    if isinstance(out, torch.Tensor):
        return out
    # transformers 5.x: pooler_output is already the projected embedding
    return out.pooler_output


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b.

    Args:
        a: Query vector of shape (D,).
        b: Index matrix of shape (N, D).

    Returns:
        Similarity scores of shape (N,).
    """
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return b_norm @ a_norm


# ---------------------------------------------------------------------------
# Text index: build descriptions from DB, encode with CLIP text encoder
# ---------------------------------------------------------------------------

def build_text_index(session=None, output_path: Optional[str] = None) -> Path:
    """Build a CLIP text embedding index from all cards in dim_cards.

    For each card, constructs a description string:
        "{name} {set_name} Pokemon card {rarity} {card_number}"
    and encodes it with the CLIP text encoder.

    Args:
        session: SQLAlchemy session. Created from SessionLocal if None.
        output_path: Where to save the index. Defaults to data/clip_text_index.pkl.

    Returns:
        Path to the saved index file.
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True

    try:
        query = text("""
            SELECT c.card_id, c.name, s.name AS set_name,
                   c.rarity, c.card_number
            FROM dim_cards c
            JOIN dim_sets s ON c.set_id = s.set_id
            ORDER BY c.card_id
        """)
        rows = session.execute(query).fetchall()
    finally:
        if close_session:
            session.close()

    if not rows:
        raise ValueError("No cards found in dim_cards. Run the catalog loader first.")

    logger.info("Building text index for %d cards", len(rows))

    card_ids = []
    descriptions = []
    for row in rows:
        card_id, name, set_name, rarity, card_number = row
        card_ids.append(card_id)
        parts = [name or "", set_name or "", "Pokemon card"]
        if rarity:
            parts.append(rarity)
        if card_number:
            parts.append(card_number)
        descriptions.append(" ".join(parts))

    model, processor = _get_model_and_processor()

    # Encode in batches to avoid OOM
    batch_size = 64
    all_embeddings = []

    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = _extract_text_features(model, **inputs)
        # Normalize embeddings
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        all_embeddings.append(text_features.cpu().numpy())

        if (i // batch_size) % 50 == 0:
            logger.info("  Encoded %d / %d descriptions", min(i + batch_size, len(descriptions)), len(descriptions))

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    index = {
        "card_ids": card_ids,
        "embeddings": embeddings,
        "descriptions": descriptions,
        "model": MODEL_NAME,
    }

    save_path = Path(output_path) if output_path else DATA_DIR / "clip_text_index.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Text index saved to %s  (%d cards, embeddings shape %s)",
                save_path, len(card_ids), embeddings.shape)
    return save_path


# ---------------------------------------------------------------------------
# Image -> text matching
# ---------------------------------------------------------------------------

def identify_card(
    image_path: str,
    top_k: int = 5,
    index_path: Optional[str] = None,
) -> list[tuple[str, float]]:
    """Identify a card from a photo using CLIP image-to-text matching.

    Encodes the input image with the CLIP image encoder and compares
    against the precomputed text embedding index via cosine similarity.

    Args:
        image_path: Path to the card image.
        top_k: Number of top matches to return.
        index_path: Path to the text index pickle. Defaults to data/clip_text_index.pkl.

    Returns:
        List of (card_id, similarity_score) tuples, sorted descending.
    """
    idx_path = Path(index_path) if index_path else DATA_DIR / "clip_text_index.pkl"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Text index not found at {idx_path}. Run build_text_index() first."
        )

    with open(idx_path, "rb") as f:
        index = pickle.load(f)

    card_ids = index["card_ids"]
    embeddings = index["embeddings"]  # (N, D)

    model, processor = _get_model_and_processor()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = _extract_image_features(model, **inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    query = image_features.cpu().numpy().squeeze()  # (D,)

    scores = _cosine_similarity(query, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [(card_ids[i], float(scores[i])) for i in top_indices]
    return results


# ---------------------------------------------------------------------------
# Image index: encode reference card images with CLIP image encoder
# ---------------------------------------------------------------------------

def build_image_index(
    image_dir: str,
    output_path: str = "data/clip_image_index.pkl",
) -> Path:
    """Build a CLIP image embedding index from a directory of card images.

    Expects image filenames to encode the card_id (with '/' replaced by '__'),
    e.g. 'base1-4__holofoil.jpg' for card_id 'base1-4/holofoil'.

    Args:
        image_dir: Directory containing reference card images.
        output_path: Where to save the index.

    Returns:
        Path to the saved index file.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")

    logger.info("Building image index from %d images in %s", len(image_files), image_dir)

    model, processor = _get_model_and_processor()

    card_ids = []
    all_embeddings = []
    batch_size = 32
    batch_images = []
    batch_ids = []

    for img_path in image_files:
        # Derive card_id from relative path: e.g.
        # data/card_images/sv8/sv8-162_normal.png -> "sv8-162/normal"
        rel = img_path.relative_to(image_dir).with_suffix("")
        card_id = str(rel).replace(os.sep, "/")
        # The filename uses '_' to separate variant; replace last '_' with '/'
        last_under = card_id.rfind("_")
        if last_under != -1:
            card_id = card_id[:last_under] + "/" + card_id[last_under + 1:]
        card_ids.append(card_id)
        batch_ids.append(card_id)

        try:
            img = Image.open(img_path).convert("RGB")
            batch_images.append(img)
        except Exception as e:
            logger.warning("Failed to open %s: %s", img_path, e)
            card_ids.pop()
            batch_ids.pop()
            continue

        if len(batch_images) >= batch_size:
            inputs = processor(images=batch_images, return_tensors="pt")
            with torch.no_grad():
                feats = _extract_image_features(model, **inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().numpy())
            batch_images = []
            batch_ids = []
            logger.info("  Encoded %d / %d images", len(card_ids), len(image_files))

    # Process remaining batch
    if batch_images:
        inputs = processor(images=batch_images, return_tensors="pt")
        with torch.no_grad():
            feats = _extract_image_features(model, **inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    index = {
        "card_ids": card_ids,
        "embeddings": embeddings,
        "model": MODEL_NAME,
    }

    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Image index saved to %s  (%d images, embeddings shape %s)",
                save_path, len(card_ids), embeddings.shape)
    return save_path


# ---------------------------------------------------------------------------
# Image -> image matching
# ---------------------------------------------------------------------------

def identify_card_by_image(
    image_path: str,
    index_path: str = "data/clip_image_index.pkl",
    top_k: int = 5,
    *,
    preloaded_index: dict | None = None,
) -> list[tuple[str, float]]:
    """Identify a card from a photo using CLIP image-to-image matching.

    Encodes the input image with the CLIP image encoder and compares
    against a precomputed image embedding index via cosine similarity.

    Args:
        image_path: Path to the query card image.
        index_path: Path to the image index pickle.
        top_k: Number of top matches to return.
        preloaded_index: Pre-loaded index dict with 'card_ids' and 'embeddings'.
            If provided, *index_path* is ignored.

    Returns:
        List of (card_id, similarity_score) tuples, sorted descending.
    """
    if preloaded_index is not None:
        card_ids = preloaded_index["card_ids"]
        embeddings = preloaded_index["embeddings"]
    else:
        idx_path = Path(index_path)
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Image index not found at {idx_path}. Run build_image_index() first."
            )

        with open(idx_path, "rb") as f:
            index = pickle.load(f)

        card_ids = index["card_ids"]
        embeddings = index["embeddings"]  # (N, D)

    model, processor = _get_model_and_processor()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = _extract_image_features(model, **inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    query = image_features.cpu().numpy().squeeze()  # (D,)

    scores = _cosine_similarity(query, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(card_ids[i], float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Phone-photo augmentation pipeline
# ---------------------------------------------------------------------------

def _augment_rotation(img: Image.Image, angle: float) -> Image.Image:
    """Rotate image by a small angle, filling background with border color."""
    return img.rotate(angle, resample=Image.BICUBIC, expand=False,
                      fillcolor=(200, 200, 200))


def _augment_brightness_contrast(img: Image.Image,
                                  brightness: float = 1.0,
                                  contrast: float = 1.0) -> Image.Image:
    """Adjust brightness and contrast."""
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def _augment_blur(img: Image.Image, radius: float = 1.5) -> Image.Image:
    """Apply Gaussian blur to simulate phone defocus."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _augment_perspective(img: Image.Image, strength: float = 0.05) -> Image.Image:
    """Apply a random perspective warp using OpenCV.

    Args:
        img: Input PIL image.
        strength: Max fraction of image dimension to perturb corners.

    Returns:
        Warped PIL image.
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    margin_w = int(w * strength)
    margin_h = int(h * strength)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, margin_w), random.randint(0, margin_h)],
        [w - random.randint(0, margin_w), random.randint(0, margin_h)],
        [w - random.randint(0, margin_w), h - random.randint(0, margin_h)],
        [random.randint(0, margin_w), h - random.randint(0, margin_h)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, M, (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def _augment_jpeg(img: Image.Image, quality: int = 50) -> Image.Image:
    """Re-compress image as JPEG to introduce compression artifacts."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def generate_augmented_views(img: Image.Image,
                              seed: int = 42,
                              num_views: int = 5) -> list[Image.Image]:
    """Generate augmented views that simulate phone-photo conditions.

    Available views (selected by ``num_views``):
    1. Perspective warp + blur + JPEG (most impactful for binder photos)
    2. Rotation + brightness/contrast shift + JPEG
    3. Gaussian blur + JPEG compression
    4. Opposite rotation + contrast shift
    5. Kitchen sink (all combined, milder)

    The first 2 views cover the most common phone-photo distortions
    (perspective from camera angle, blur from motion/defocus, JPEG from
    phone compression). Use ``num_views=2`` for faster index builds with
    most of the benefit.

    Args:
        img: Clean reference card image (PIL).
        seed: Random seed for reproducibility.
        num_views: Number of augmented views to generate (1-5, default 5).

    Returns:
        List of up to ``num_views`` augmented PIL images.
    """
    rng = random.Random(seed)

    views = []

    # View 1: perspective + blur + JPEG (highest impact for binder pages)
    v = _augment_perspective(img, strength=rng.uniform(0.03, 0.07))
    v = _augment_blur(v, radius=rng.uniform(0.8, 2.0))
    v = _augment_jpeg(v, quality=rng.randint(45, 70))
    views.append(v)
    if len(views) >= num_views:
        return views

    # View 2: rotation + brightness/contrast + JPEG
    v = _augment_rotation(img, angle=rng.uniform(-5, 5))
    v = _augment_brightness_contrast(v, brightness=rng.uniform(0.85, 1.15),
                                      contrast=rng.uniform(0.85, 1.15))
    v = _augment_jpeg(v, quality=rng.randint(50, 75))
    views.append(v)
    if len(views) >= num_views:
        return views

    # View 3: blur + JPEG artifacts
    v = _augment_blur(img, radius=rng.uniform(1.0, 2.5))
    v = _augment_jpeg(v, quality=rng.randint(40, 70))
    views.append(v)
    if len(views) >= num_views:
        return views

    # View 4: opposite rotation + contrast
    v = _augment_rotation(img, angle=rng.uniform(-5, -2))
    v = _augment_brightness_contrast(v, contrast=rng.uniform(0.8, 1.2))
    views.append(v)
    if len(views) >= num_views:
        return views

    # View 5: kitchen sink (all combined, milder)
    v = _augment_rotation(img, angle=rng.uniform(-3, 3))
    v = _augment_blur(v, radius=rng.uniform(0.5, 1.5))
    v = _augment_perspective(v, strength=rng.uniform(0.02, 0.05))
    v = _augment_brightness_contrast(v, brightness=rng.uniform(0.9, 1.1),
                                      contrast=rng.uniform(0.9, 1.1))
    v = _augment_jpeg(v, quality=rng.randint(50, 80))
    views.append(v)

    return views


# ---------------------------------------------------------------------------
# Augmented image index: average clean + augmented embeddings per card
# ---------------------------------------------------------------------------

def _encode_batch(model, processor, pil_images: list, extract_fn) -> np.ndarray:
    """Encode a batch of PIL images with CLIP, returning normalized embeddings."""
    inputs = processor(images=pil_images, return_tensors="pt")
    with torch.no_grad():
        feats = extract_fn(model, **inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def build_augmented_image_index(
    image_dir: str,
    output_path: str = "data/clip_augmented_index.pkl",
    max_cards: int = 0,
    num_augmentations: int = 5,
    encode_batch_size: int = 32,
    chunk_size: int = 200,
) -> Path:
    """Build a CLIP image index using averaged clean + augmented embeddings.

    For each reference image, generates augmented views that simulate
    phone-photo conditions (rotation, blur, perspective, JPEG artifacts,
    brightness/contrast shifts). The final embedding per card is the
    L2-normalized average of the clean embedding and all augmented embeddings.

    This bridges the domain gap between clean digital reference images and
    noisy phone photos of binder pages.

    Processing is done in chunks of ``chunk_size`` cards to limit memory
    usage. Within each chunk, images from multiple cards are batched
    together for CLIP encoding (up to ``encode_batch_size`` per forward
    pass) to maximize throughput.

    Args:
        image_dir: Directory containing reference card images.
        output_path: Where to save the index pickle.
        max_cards: If > 0, only process this many cards (for testing).
        num_augmentations: Number of augmented views per card (default 5).
        encode_batch_size: Number of images per CLIP forward pass (default 32).
        chunk_size: Number of cards to buffer before encoding (default 200).

    Returns:
        Path to the saved index file.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")

    if max_cards > 0:
        image_files = image_files[:max_cards]

    views_per_card = 1 + num_augmentations
    total_images = len(image_files) * views_per_card
    logger.info("Building augmented image index from %d cards in %s "
                "(%d augmentations/card = %d total images, batch_size=%d, "
                "chunk_size=%d)",
                len(image_files), image_dir, num_augmentations,
                total_images, encode_batch_size, chunk_size)

    model, processor = _get_model_and_processor()

    card_ids: list[str] = []
    card_embeddings: list[np.ndarray] = []

    # Buffers for the current chunk
    chunk_card_ids: list[str] = []
    chunk_view_counts: list[int] = []
    chunk_images: list[Image.Image] = []

    def _flush_chunk():
        """Encode buffered chunk images, average per card, append results."""
        if not chunk_images:
            return
        # Encode all images in this chunk in large batches
        chunk_embs: list[np.ndarray] = []
        for bs in range(0, len(chunk_images), encode_batch_size):
            batch = chunk_images[bs:bs + encode_batch_size]
            embs = _encode_batch(model, processor, batch,
                                 _extract_image_features)
            chunk_embs.append(embs)

        flat = np.vstack(chunk_embs).astype(np.float32)

        # Average per card
        offset = 0
        for cid, count in zip(chunk_card_ids, chunk_view_counts):
            card_embs_slice = flat[offset:offset + count]
            avg = card_embs_slice.mean(axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-8)
            card_ids.append(cid)
            card_embeddings.append(avg)
            offset += count

        # Clear chunk buffers
        chunk_card_ids.clear()
        chunk_view_counts.clear()
        chunk_images.clear()

    for idx, img_path in enumerate(image_files):
        # Derive card_id (same logic as build_image_index)
        rel = img_path.relative_to(image_dir).with_suffix("")
        card_id = str(rel).replace(os.sep, "/")
        last_under = card_id.rfind("_")
        if last_under != -1:
            card_id = card_id[:last_under] + "/" + card_id[last_under + 1:]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to open %s: %s", img_path, e)
            continue

        views = [img]
        try:
            aug_views = generate_augmented_views(img, seed=idx)
            views.extend(aug_views[:num_augmentations])
        except Exception as e:
            logger.warning("Augmentation failed for %s: %s (using clean only)",
                           img_path, e)

        chunk_card_ids.append(card_id)
        chunk_view_counts.append(len(views))
        chunk_images.extend(views)

        # Flush chunk when we have accumulated enough cards
        if len(chunk_card_ids) >= chunk_size:
            _flush_chunk()
            logger.info("  Processed %d / %d cards", idx + 1, len(image_files))

    # Flush remaining
    _flush_chunk()
    logger.info("  Processed %d / %d cards (done)", len(image_files), len(image_files))

    embeddings = np.vstack(
        [e.reshape(1, -1) for e in card_embeddings]
    ).astype(np.float32)

    index = {
        "card_ids": card_ids,
        "embeddings": embeddings,
        "model": MODEL_NAME,
        "augmented": True,
        "num_augmentations": num_augmentations,
    }

    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Augmented image index saved to %s  (%d cards, embeddings shape %s)",
                save_path, len(card_ids), embeddings.shape)
    return save_path
