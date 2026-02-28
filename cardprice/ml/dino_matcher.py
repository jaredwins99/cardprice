"""Card identification using DINOv2 embeddings and FAISS nearest-neighbor search.

Uses DINOv2 ViT-B/14 to extract visual embeddings from card images, builds a
FAISS index for fast cosine-similarity search, and provides a MatchPipeline
that combines visual matching with OCR fallback for robust card identification.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------------
_model: Optional[torch.nn.Module] = None
_device: Optional[torch.device] = None

# ImageNet normalization stats
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _load_model() -> tuple[torch.nn.Module, torch.device]:
    """Load DINOv2 ViT-B/14 and cache it globally.

    Returns the model and the device it lives on.
    """
    global _model, _device
    if _model is not None:
        return _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading DINOv2 ViT-B/14 on %s ...", _device)

    _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    _model.to(_device)
    _model.eval()

    logger.info("DINOv2 model loaded successfully.")
    return _model, _device


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embedding(image_path: str | Path) -> np.ndarray:
    """Extract a 768-dim L2-normalized DINOv2 CLS embedding from an image.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        768-dimensional float32 vector, L2-normalized.
    """
    model, device = _load_model()

    img = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        embedding = model(tensor)  # (1, 768)

    vec = embedding.cpu().numpy().astype(np.float32).squeeze()  # (768,)

    # L2-normalize so inner-product == cosine similarity
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec


# ---------------------------------------------------------------------------
# Reference index building
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_reference_index(
    image_dir: str = "data/card_images",
    index_path: str = "data/dino_index.faiss",
    mapping_path: str = "data/dino_card_ids.pkl",
) -> int:
    """Build a FAISS index from all card images in *image_dir*.

    Each image filename (without extension) is treated as the card_id.

    Parameters
    ----------
    image_dir : str
        Directory containing reference card images.
    index_path : str
        Where to save the FAISS index.
    mapping_path : str
        Where to save the card_id list (pickle).

    Returns
    -------
    int
        Number of images indexed.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Collect image paths
    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    logger.info("Found %d reference images in %s", len(image_files), image_dir)

    embeddings: list[np.ndarray] = []
    card_ids: list[str] = []
    failed = 0

    for i, img_path in enumerate(image_files):
        try:
            vec = extract_embedding(img_path)
            embeddings.append(vec)
            card_ids.append(img_path.stem)
        except Exception:
            logger.warning("Failed to process %s", img_path, exc_info=True)
            failed += 1
            continue

        if (i + 1) % 100 == 0:
            logger.info("Processed %d / %d images ...", i + 1, len(image_files))

    if not embeddings:
        raise RuntimeError("No embeddings could be extracted.")

    logger.info(
        "Extracted %d embeddings (%d failures). Building FAISS index ...",
        len(embeddings),
        failed,
    )

    # Build FAISS IndexFlatIP (inner-product == cosine on L2-normed vectors)
    matrix = np.stack(embeddings).astype(np.float32)  # (N, 768)
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(mapping_path) or ".", exist_ok=True)

    faiss.write_index(index, index_path)
    with open(mapping_path, "wb") as f:
        pickle.dump(card_ids, f)

    logger.info(
        "Saved FAISS index (%d vectors, dim=%d) to %s", index.ntotal, dim, index_path
    )
    logger.info("Saved card-ID mapping (%d entries) to %s", len(card_ids), mapping_path)

    return len(embeddings)


# ---------------------------------------------------------------------------
# Card identification (query time)
# ---------------------------------------------------------------------------

def identify_card(
    image_path: str | Path,
    index_path: str = "data/dino_index.faiss",
    mapping_path: str = "data/dino_card_ids.pkl",
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Identify a card by searching the FAISS index for nearest neighbors.

    Parameters
    ----------
    image_path : str or Path
        Path to the query card image.
    index_path : str
        Path to the saved FAISS index.
    mapping_path : str
        Path to the saved card-ID mapping.
    top_k : int
        Number of results to return.

    Returns
    -------
    list[tuple[str, float]]
        List of (card_id, cosine_similarity) sorted by descending similarity.
    """
    # Load index and mapping
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Card-ID mapping not found: {mapping_path}")

    index = faiss.read_index(index_path)
    with open(mapping_path, "rb") as f:
        card_ids: list[str] = pickle.load(f)

    # Extract query embedding
    query = extract_embedding(image_path).reshape(1, -1)

    # Search
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query, k)

    results: list[tuple[str, float]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append((card_ids[idx], float(score)))

    return results


# ---------------------------------------------------------------------------
# MatchPipeline — DINOv2 + OCR fallback
# ---------------------------------------------------------------------------

class MatchPipeline:
    """Combines DINOv2 visual matching with OCR fallback for card identification.

    Decision thresholds:
        >= 0.95  ->  accept the top DINOv2 match directly
        0.85-0.95 -> attempt OCR verification to confirm or override
        < 0.85   ->  flag for manual review

    Parameters
    ----------
    index_path : str
        Path to the FAISS index file.
    mapping_path : str
        Path to the card-ID mapping pickle.
    accept_threshold : float
        Minimum cosine similarity to auto-accept (default 0.95).
    review_threshold : float
        Minimum cosine similarity to attempt OCR fallback (default 0.85).
    top_k : int
        Number of FAISS results to retrieve (default 5).
    """

    def __init__(
        self,
        index_path: str = "data/dino_index.faiss",
        mapping_path: str = "data/dino_card_ids.pkl",
        accept_threshold: float = 0.95,
        review_threshold: float = 0.85,
        top_k: int = 5,
    ):
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.accept_threshold = accept_threshold
        self.review_threshold = review_threshold
        self.top_k = top_k

    # ------------------------------------------------------------------
    # OCR helper (lazy import to avoid hard dependency on pytesseract)
    # ------------------------------------------------------------------

    @staticmethod
    def _ocr_extract_text(image_path: str | Path) -> str:
        """Extract text from an image using Tesseract OCR.

        Returns an empty string if pytesseract is not installed.
        """
        try:
            import pytesseract
        except ImportError:
            logger.warning(
                "pytesseract not installed — OCR fallback unavailable. "
                "Install with: pip install pytesseract"
            )
            return ""

        try:
            img = Image.open(image_path).convert("RGB")
            return pytesseract.image_to_string(img).strip()
        except Exception:
            logger.warning("OCR failed for %s", image_path, exc_info=True)
            return ""

    @staticmethod
    def _ocr_verify(ocr_text: str, card_id: str) -> bool:
        """Check whether OCR text is consistent with the candidate card_id.

        A simple heuristic: if any significant token from the card_id appears
        in the OCR output (case-insensitive), treat it as a positive signal.
        """
        if not ocr_text:
            return False

        ocr_lower = ocr_text.lower()
        # card_id might look like "base1-4/holofoil" — split into tokens
        tokens = card_id.replace("/", "-").replace("_", "-").split("-")
        # Only check tokens with 3+ characters to avoid noise
        meaningful = [t for t in tokens if len(t) >= 3]
        if not meaningful:
            return False

        matches = sum(1 for t in meaningful if t.lower() in ocr_lower)
        return matches >= 1

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def match(self, image_path: str | Path) -> dict:
        """Run the full match pipeline on a single image.

        Returns
        -------
        dict with keys:
            card_id : str or None
                Identified card ID, or None if manual review is needed.
            similarity : float
                Cosine similarity of the top match.
            status : str
                One of "accepted", "ocr_verified", "ocr_failed", "manual_review".
            top_matches : list[tuple[str, float]]
                All top-k results from FAISS.
        """
        image_path = str(image_path)
        top_matches = identify_card(
            image_path,
            index_path=self.index_path,
            mapping_path=self.mapping_path,
            top_k=self.top_k,
        )

        if not top_matches:
            return {
                "card_id": None,
                "similarity": 0.0,
                "status": "manual_review",
                "top_matches": [],
            }

        best_id, best_score = top_matches[0]

        # High confidence — accept directly
        if best_score >= self.accept_threshold:
            logger.info(
                "Auto-accepted %s (similarity=%.4f)", best_id, best_score
            )
            return {
                "card_id": best_id,
                "similarity": best_score,
                "status": "accepted",
                "top_matches": top_matches,
            }

        # Medium confidence — try OCR verification
        if best_score >= self.review_threshold:
            ocr_text = self._ocr_extract_text(image_path)
            if self._ocr_verify(ocr_text, best_id):
                logger.info(
                    "OCR-verified %s (similarity=%.4f)", best_id, best_score
                )
                return {
                    "card_id": best_id,
                    "similarity": best_score,
                    "status": "ocr_verified",
                    "top_matches": top_matches,
                }
            else:
                logger.info(
                    "OCR could not verify %s (similarity=%.4f)", best_id, best_score
                )
                return {
                    "card_id": None,
                    "similarity": best_score,
                    "status": "ocr_failed",
                    "top_matches": top_matches,
                }

        # Low confidence — manual review
        logger.info(
            "Flagged for manual review (best=%s, similarity=%.4f)",
            best_id,
            best_score,
        )
        return {
            "card_id": None,
            "similarity": best_score,
            "status": "manual_review",
            "top_matches": top_matches,
        }
