"""Card identification using CLIP embeddings.

Provides text-to-image and image-to-image matching for Pokemon cards
using OpenAI's CLIP (ViT-Large/14) model.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
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
    instead of a plain tensor. This helper extracts and projects the pooled output.
    """
    out = model.get_text_features(**inputs)
    if isinstance(out, torch.Tensor):
        return out
    # transformers 5.x: extract pooler_output and project
    pooled = out.pooler_output
    return model.text_projection(pooled)


def _extract_image_features(model, **inputs) -> torch.Tensor:
    """Extract image features from CLIP, handling transformers 5.x API change."""
    out = model.get_image_features(**inputs)
    if isinstance(out, torch.Tensor):
        return out
    pooled = out.pooler_output
    return model.visual_projection(pooled)


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
            feats = model.get_image_features(**inputs)
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
) -> list[tuple[str, float]]:
    """Identify a card from a photo using CLIP image-to-image matching.

    Encodes the input image with the CLIP image encoder and compares
    against a precomputed image embedding index via cosine similarity.

    Args:
        image_path: Path to the query card image.
        index_path: Path to the image index pickle.
        top_k: Number of top matches to return.

    Returns:
        List of (card_id, similarity_score) tuples, sorted descending.
    """
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

    results = [(card_ids[i], float(scores[i])) for i in top_indices]
    return results
