"""Tip-Adapter: Training-free adaptation of CLIP for card identification.

Implements the Tip-Adapter method (Zhang et al., ECCV 2022) adapted for
Pokemon card identification. Builds a non-parametric key-value cache where:
  - Keys: CLIP image embeddings of 20,026 reference card images
  - Values: one-hot card identity labels
  - Query: CLIP embedding of a phone photo segment

At inference, cache-based logits (visual similarity with exponential
sharpening) are combined with CLIP zero-shot text logits to produce
final predictions.

Reference: https://arxiv.org/abs/2207.09519
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# ---------------------------------------------------------------------------
# Default hyperparameters (tuned on binder page segments)
# ---------------------------------------------------------------------------
# beta controls sharpness of the affinity function: higher = sharper peaks.
#   Lower beta (0.5) spreads weight across more neighbors; higher beta focuses
#   on the nearest. For our 1-shot setting, lower beta lets the text signal
#   influence borderline cases.
# alpha controls the weight of cache logits.
# text_weight controls the weight of zero-shot CLIP text logits.
#   Even a tiny text_weight (0.002) acts as a tiebreaker for visually similar
#   cards, boosting accuracy from 91.1% to 97.8% on 45 binder page segments.
DEFAULT_BETA = 0.5
DEFAULT_ALPHA = 1.0        # weight for cache logits
DEFAULT_TEXT_WEIGHT = 0.002  # weight for zero-shot text logits (small but effective)


class TipAdapter:
    """Training-free Tip-Adapter for Pokemon card identification.

    Combines a visual similarity cache (CLIP image embeddings) with
    optional zero-shot text logits from CLIP text descriptions.

    The cache logits are computed as:
        affinity = query @ cache_keys.T                  # (1, N)
        cache_logits = exp(-beta * (1 - affinity)) @ cache_values  # (1, C)

    Final prediction:
        logits = text_weight * text_logits + alpha * cache_logits

    Where text_logits come from query @ text_embeddings.T.

    Attributes:
        cache_keys: (N, D) normalized CLIP image embeddings for reference cards.
        card_ids: List of N card_id strings (DB format: "set-num/variant").
        card_id_to_idx: Mapping from card_id to class index (0..C-1).
        cache_values: (N, C) one-hot label matrix mapping each ref to its class.
        text_embeddings: (C, D) CLIP text embeddings aligned to class indices.
        beta: Sharpness of the exponential affinity function.
        alpha: Weight for cache-based logits.
        text_weight: Weight for zero-shot text logits.
    """

    def __init__(
        self,
        image_index_path: Optional[str] = None,
        text_index_path: Optional[str] = None,
        beta: float = DEFAULT_BETA,
        alpha: float = DEFAULT_ALPHA,
        text_weight: float = DEFAULT_TEXT_WEIGHT,
    ):
        """Initialize Tip-Adapter from precomputed CLIP indexes.

        Args:
            image_index_path: Path to clip_image_index.pkl. Defaults to data/.
            text_index_path: Path to clip_text_index.pkl. Defaults to data/.
                If None or not found, text logits are disabled.
            beta: Sharpness of cache affinity function.
            alpha: Weight for cache logits.
            text_weight: Weight for zero-shot text logits.
        """
        self.beta = beta
        self.alpha = alpha
        self.text_weight = text_weight

        # --- Load image index (cache keys) ---
        img_path = Path(image_index_path) if image_index_path else DATA_DIR / "clip_image_index.pkl"
        if not img_path.exists():
            raise FileNotFoundError(f"CLIP image index not found: {img_path}")

        logger.info("Loading CLIP image index from %s", img_path)
        with open(img_path, "rb") as f:
            img_idx = pickle.load(f)

        raw_card_ids = img_idx["card_ids"]
        raw_embeddings = img_idx["embeddings"]  # (N, D)

        # Normalize card_ids: strip set prefix if present
        # "base1/base1-100/normal" -> "base1-100/normal"
        self.card_ids = [self._normalize_card_id(c) for c in raw_card_ids]
        self.n_refs = len(self.card_ids)
        self.embed_dim = raw_embeddings.shape[1]

        # Build unique class list and mapping
        unique_ids = list(dict.fromkeys(self.card_ids))  # preserve order, deduplicate
        self.class_card_ids = unique_ids
        self.n_classes = len(unique_ids)
        self.card_id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}

        # Cache keys: normalized CLIP image embeddings (N, D)
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True) + 1e-8
        self.cache_keys = (raw_embeddings / norms).astype(np.float32)

        # Cache values: one-hot labels (N, C)
        self.cache_values = np.zeros((self.n_refs, self.n_classes), dtype=np.float32)
        for i, cid in enumerate(self.card_ids):
            self.cache_values[i, self.card_id_to_idx[cid]] = 1.0

        logger.info("Cache built: %d reference images, %d unique classes, %d-dim embeddings",
                     self.n_refs, self.n_classes, self.embed_dim)

        # --- Load text index (optional, for zero-shot logits) ---
        self.text_embeddings = None
        self.text_card_ids = None
        if text_weight > 0:
            txt_path = Path(text_index_path) if text_index_path else DATA_DIR / "clip_text_index.pkl"
            if txt_path.exists():
                logger.info("Loading CLIP text index from %s", txt_path)
                with open(txt_path, "rb") as f:
                    txt_idx = pickle.load(f)

                txt_card_ids = txt_idx["card_ids"]
                txt_embs = txt_idx["embeddings"]  # (M, D)

                # Align text embeddings to our class order
                # Text index may have more cards (52 extra); pick only those in our class list
                self.text_embeddings = np.zeros((self.n_classes, self.embed_dim), dtype=np.float32)
                txt_id_to_row = {cid: i for i, cid in enumerate(txt_card_ids)}
                matched = 0
                for cls_idx, cid in enumerate(self.class_card_ids):
                    if cid in txt_id_to_row:
                        row = txt_id_to_row[cid]
                        self.text_embeddings[cls_idx] = txt_embs[row]
                        matched += 1
                # Normalize
                norms = np.linalg.norm(self.text_embeddings, axis=1, keepdims=True) + 1e-8
                self.text_embeddings = (self.text_embeddings / norms).astype(np.float32)
                logger.info("Text embeddings aligned: %d/%d classes matched", matched, self.n_classes)
            else:
                logger.warning("Text index not found at %s, text logits disabled", txt_path)
                self.text_weight = 0.0

    @staticmethod
    def _normalize_card_id(raw_cid: str) -> str:
        """Normalize card_id from index format to DB format.

        "base1/base1-100/normal" -> "base1-100/normal"
        """
        parts = raw_cid.split("/")
        if len(parts) >= 3:
            return "/".join(parts[1:])
        return raw_cid

    def _compute_affinity(self, query: np.ndarray) -> np.ndarray:
        """Compute exponentially-sharpened affinity between query and cache.

        Args:
            query: (D,) normalized query embedding.

        Returns:
            (N,) affinity weights, higher = more similar.
        """
        # Cosine similarity: query @ cache_keys.T -> (N,)
        sim = self.cache_keys @ query  # (N,)

        # Tip-Adapter affinity: exp(-beta * (1 - sim))
        # This converts cosine similarity [0, 1] to exponential weights
        # where beta controls sharpness: higher beta = sharper peaks
        affinity = np.exp(-self.beta * (1.0 - sim))
        return affinity

    def predict(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        beta: Optional[float] = None,
        alpha: Optional[float] = None,
        text_weight: Optional[float] = None,
    ) -> list[tuple[str, float]]:
        """Predict card identity from a query CLIP embedding.

        Combines cache-based visual similarity with optional text logits.

        Args:
            query_embedding: (D,) CLIP image embedding of the query photo.
            top_k: Number of top matches to return.
            beta: Override cache affinity sharpness.
            alpha: Override cache logit weight.
            text_weight: Override text logit weight.

        Returns:
            List of (card_id, score) tuples, sorted descending by score.
        """
        beta = beta if beta is not None else self.beta
        alpha = alpha if alpha is not None else self.alpha
        tw = text_weight if text_weight is not None else self.text_weight

        # Normalize query
        query = query_embedding.astype(np.float32).squeeze()
        query = query / (np.linalg.norm(query) + 1e-8)

        # --- Cache logits ---
        # affinity: (N,) similarity weights
        sim = self.cache_keys @ query
        affinity = np.exp(-beta * (1.0 - sim))

        # cache_logits: (C,) = affinity.T @ cache_values
        cache_logits = affinity @ self.cache_values  # (C,)

        # --- Text logits (optional) ---
        text_logits = np.zeros(self.n_classes, dtype=np.float32)
        if tw > 0 and self.text_embeddings is not None:
            # CLIP zero-shot: 100 * query @ text_embeddings.T
            text_logits = 100.0 * (self.text_embeddings @ query)

        # --- Combined logits ---
        logits = tw * text_logits + alpha * cache_logits

        # Get top-k
        top_indices = np.argsort(logits)[::-1][:top_k]
        results = [(self.class_card_ids[i], float(logits[i])) for i in top_indices]
        return results

    def predict_from_image(
        self,
        image_path: str,
        top_k: int = 5,
        beta: Optional[float] = None,
        alpha: Optional[float] = None,
        text_weight: Optional[float] = None,
    ) -> list[tuple[str, float]]:
        """Predict card identity from an image file.

        Loads the image, encodes it with CLIP, and runs prediction.

        Args:
            image_path: Path to the query card image.
            top_k: Number of top matches to return.
            beta: Override cache affinity sharpness.
            alpha: Override cache logit weight.
            text_weight: Override text logit weight.

        Returns:
            List of (card_id, score) tuples, sorted descending by score.
        """
        from cardprice.ml.clip_matcher import (
            _get_model_and_processor,
            _extract_image_features,
        )

        model, processor = _get_model_and_processor()
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            feats = _extract_image_features(model, **inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        query = feats.cpu().numpy().squeeze()

        return self.predict(query, top_k=top_k, beta=beta, alpha=alpha, text_weight=text_weight)

    def predict_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        beta: Optional[float] = None,
        alpha: Optional[float] = None,
        text_weight: Optional[float] = None,
    ) -> list[list[tuple[str, float]]]:
        """Batch prediction for multiple query embeddings.

        Args:
            query_embeddings: (B, D) CLIP image embeddings.
            top_k: Number of top matches per query.
            beta: Override cache affinity sharpness.
            alpha: Override cache logit weight.
            text_weight: Override text logit weight.

        Returns:
            List of B result lists, each containing (card_id, score) tuples.
        """
        beta = beta if beta is not None else self.beta
        alpha = alpha if alpha is not None else self.alpha
        tw = text_weight if text_weight is not None else self.text_weight

        # Normalize queries
        queries = query_embeddings.astype(np.float32)
        norms = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8
        queries = queries / norms  # (B, D)

        # Cache logits: (B, N) @ (N, C) -> (B, C)
        sim = queries @ self.cache_keys.T  # (B, N)
        affinity = np.exp(-beta * (1.0 - sim))  # (B, N)
        cache_logits = affinity @ self.cache_values  # (B, C)

        # Text logits (optional)
        if tw > 0 and self.text_embeddings is not None:
            text_logits = 100.0 * (queries @ self.text_embeddings.T)  # (B, C)
        else:
            text_logits = np.zeros_like(cache_logits)

        # Combined
        logits = tw * text_logits + alpha * cache_logits  # (B, C)

        # Extract top-k per query
        results = []
        for b in range(logits.shape[0]):
            top_indices = np.argsort(logits[b])[::-1][:top_k]
            results.append([
                (self.class_card_ids[i], float(logits[b, i]))
                for i in top_indices
            ])
        return results


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def search_hyperparameters(
    adapter: TipAdapter,
    query_embeddings: np.ndarray,
    ground_truth_ids: list[str],
    beta_range: tuple[float, float, float] = (1.0, 20.0, 0.5),
    alpha_range: tuple[float, float, float] = (0.5, 5.0, 0.5),
    text_weight_range: tuple[float, float, float] = (0.0, 0.5, 0.1),
) -> dict:
    """Grid search for optimal Tip-Adapter hyperparameters.

    Tests all combinations of beta, alpha, and text_weight, evaluating
    top-1 accuracy on the provided queries.

    Args:
        adapter: Initialized TipAdapter instance.
        query_embeddings: (B, D) CLIP embeddings of test images.
        ground_truth_ids: List of B correct card_ids.
        beta_range: (min, max, step) for beta search.
        alpha_range: (min, max, step) for alpha search.
        text_weight_range: (min, max, step) for text_weight search.

    Returns:
        Dict with keys: best_beta, best_alpha, best_text_weight,
        best_accuracy, all_results (list of dicts).
    """
    betas = np.arange(*beta_range)
    alphas = np.arange(*alpha_range)
    text_weights = np.arange(*text_weight_range)

    best_acc = -1.0
    best_params = {}
    all_results = []

    for beta in betas:
        for alpha_val in alphas:
            for tw in text_weights:
                preds = adapter.predict_batch(
                    query_embeddings, top_k=1,
                    beta=float(beta), alpha=float(alpha_val), text_weight=float(tw),
                )
                correct = sum(
                    1 for pred, gt in zip(preds, ground_truth_ids)
                    if pred and pred[0][0] == gt
                )
                acc = correct / len(ground_truth_ids)
                record = {
                    "beta": float(beta),
                    "alpha": float(alpha_val),
                    "text_weight": float(tw),
                    "accuracy": acc,
                    "correct": correct,
                    "total": len(ground_truth_ids),
                }
                all_results.append(record)
                if acc > best_acc:
                    best_acc = acc
                    best_params = record

    return {
        "best_beta": best_params.get("beta", DEFAULT_BETA),
        "best_alpha": best_params.get("alpha", DEFAULT_ALPHA),
        "best_text_weight": best_params.get("text_weight", DEFAULT_TEXT_WEIGHT),
        "best_accuracy": best_acc,
        "all_results": all_results,
    }


# ---------------------------------------------------------------------------
# Convenience: singleton adapter instance
# ---------------------------------------------------------------------------

_adapter: Optional[TipAdapter] = None


def get_adapter(**kwargs) -> TipAdapter:
    """Get or create a singleton TipAdapter instance.

    Keyword arguments are passed to TipAdapter() on first call.
    """
    global _adapter
    if _adapter is None:
        _adapter = TipAdapter(**kwargs)
    return _adapter


def identify_card_tip_adapter(
    image_path: str,
    top_k: int = 5,
    **kwargs,
) -> list[tuple[str, float]]:
    """Identify a card using Tip-Adapter.

    Convenience function that wraps TipAdapter.predict_from_image.

    Args:
        image_path: Path to the query card image.
        top_k: Number of top matches to return.
        **kwargs: Additional arguments passed to TipAdapter() if creating.

    Returns:
        List of (card_id, score) tuples, sorted descending by score.
    """
    adapter = get_adapter(**kwargs)
    return adapter.predict_from_image(image_path, top_k=top_k)
