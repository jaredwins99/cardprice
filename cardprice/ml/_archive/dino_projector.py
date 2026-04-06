"""Trainable projection head for DINOv2 to bridge the domain gap.

Freezes the DINOv2 ViT-B/14 backbone and trains a small MLP projection
(768 -> 512 -> 256) using InfoNCE contrastive loss on synthetically
augmented pairs.  The augmentation pipeline simulates phone-camera
artifacts (perspective distortion, glare, blur, color shifts, JPEG
compression) so that projected embeddings of phone photos land close
to their clean reference counterparts.

After training, the FAISS index is rebuilt with 256-dim projected
embeddings for faster, more accurate retrieval.

Usage:
    # Train
    python -m cardprice.ml.dino_projector train --epochs 30 --batch-size 32

    # Evaluate on binder_eval.json
    python -m cardprice.ml.dino_projector eval

    # Rebuild FAISS index with projected embeddings
    python -m cardprice.ml.dino_projector build-index
"""

import json
import logging
import os
import pickle
import random
import time
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_IMAGE_DIR = _PROJECT_ROOT / "data" / "card_images"
_EVAL_PATH = _PROJECT_ROOT / "data" / "eval" / "binder_eval.json"
_CHECKPOINT_DIR = _PROJECT_ROOT / "data" / "checkpoints"
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "dino_projector.pt"
_PROJECTED_INDEX_PATH = _PROJECT_ROOT / "data" / "dino_projected_index.faiss"
_PROJECTED_CARD_IDS_PATH = _PROJECT_ROOT / "data" / "dino_projected_card_ids.pkl"

# DINOv2 raw index paths (for baseline comparison)
_RAW_INDEX_PATH = _PROJECT_ROOT / "data" / "dino_index.faiss"
_RAW_CARD_IDS_PATH = _PROJECT_ROOT / "data" / "dino_card_ids.pkl"

# ImageNet normalization (same as dino_matcher.py)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---------------------------------------------------------------------------
# Projection Head
# ---------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """MLP projection: 768 -> 512 -> 256 with GELU activation.

    Designed to sit on top of frozen DINOv2 CLS embeddings.
    Output is L2-normalized for cosine-similarity retrieval.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512,
                 output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize.

        Parameters
        ----------
        x : (B, input_dim) tensor of DINOv2 CLS embeddings.

        Returns
        -------
        (B, output_dim) L2-normalized projected embeddings.
        """
        projected = self.projector(x)
        return F.normalize(projected, dim=-1)


# ---------------------------------------------------------------------------
# DINOv2 backbone wrapper (frozen, cached)
# ---------------------------------------------------------------------------

_backbone: Optional[nn.Module] = None
_device: Optional[torch.device] = None


def _get_backbone() -> tuple[nn.Module, torch.device]:
    """Load DINOv2 ViT-B/14 backbone (frozen, cached globally)."""
    global _backbone, _device
    if _backbone is not None:
        return _backbone, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading DINOv2 ViT-B/14 on %s ...", _device)

    _backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    _backbone.to(_device)
    _backbone.eval()
    for p in _backbone.parameters():
        p.requires_grad = False

    logger.info("DINOv2 backbone loaded and frozen.")
    return _backbone, _device


# ---------------------------------------------------------------------------
# Augmentation pipeline (simulates phone camera conditions)
# ---------------------------------------------------------------------------


def _build_phone_augmentation() -> A.Compose:
    """Build an albumentations pipeline that simulates phone photos of cards.

    Simulates: perspective distortion, rotation, blur, noise, lighting
    variation, color shifts, sleeve glare, and JPEG compression.
    """
    return A.Compose([
        # Geometric distortions (phone angle, slight misalignment)
        A.Perspective(scale=(0.02, 0.08), p=0.7),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomResizedCrop(size=(224, 224), scale=(0.70, 1.0),
                            ratio=(0.85, 1.15), p=1.0),

        # Optical artifacts (defocus, motion blur)
        A.MotionBlur(blur_limit=(3, 7), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),

        # Lighting variation (indoor, fluorescent, warm, etc.)
        A.RandomBrightnessContrast(brightness_limit=0.3,
                                   contrast_limit=0.3, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.3, hue=0.05, p=0.5),

        # Sleeve glare simulation (random bright spots)
        A.RandomSunFlare(
            flare_roi=(0.0, 0.0, 1.0, 1.0),
            src_radius=80, p=0.15,
        ),

        # JPEG compression (phone camera output)
        A.ImageCompression(quality_range=(50, 95), p=0.4),

        # Final normalization for DINOv2
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _build_anchor_transform() -> A.Compose:
    """Minimal augmentation for anchor (clean reference) images.

    Only resizes and normalizes -- no distortions.
    """
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset: generates (anchor, positive) pairs from reference images
# ---------------------------------------------------------------------------


class CardPairDataset(Dataset):
    """Generates augmented training pairs from reference card images.

    Each __getitem__ returns (anchor_tensor, positive_tensor) where:
    - anchor is the clean reference image (resize + normalize only)
    - positive is the same image with phone-camera augmentation applied
    """

    def __init__(self, image_dir: str | Path, pairs_per_image: int = 2):
        self.image_dir = Path(image_dir)
        self.pairs_per_image = pairs_per_image

        # Collect all reference images
        self.image_paths = sorted(
            p for p in self.image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in {self.image_dir}")

        self.anchor_transform = _build_anchor_transform()
        self.positive_transform = _build_phone_augmentation()

        logger.info("CardPairDataset: %d images, %d pairs/image = %d total",
                     len(self.image_paths), pairs_per_image,
                     len(self))

    def __len__(self) -> int:
        return len(self.image_paths) * self.pairs_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_idx = idx // self.pairs_per_image
        img_path = self.image_paths[img_idx]

        # Read image as RGB numpy array (albumentations expects HWC uint8)
        img = cv2.imread(str(img_path))
        if img is None:
            # Fallback: try PIL
            pil_img = Image.open(img_path).convert("RGB")
            img = np.array(pil_img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Anchor: clean resize + normalize
        anchor = self.anchor_transform(image=img)["image"]
        anchor = torch.from_numpy(anchor).permute(2, 0, 1).float()

        # Positive: phone-camera augmentation
        positive = self.positive_transform(image=img)["image"]
        positive = torch.from_numpy(positive).permute(2, 0, 1).float()

        return anchor, positive


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------


def info_nce_loss(anchors: torch.Tensor, positives: torch.Tensor,
                  temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE contrastive loss.

    Parameters
    ----------
    anchors : (B, D) L2-normalized embeddings of clean reference images.
    positives : (B, D) L2-normalized embeddings of augmented versions.
    temperature : float
        Temperature scaling factor.  Lower = sharper distribution.

    Returns
    -------
    Scalar loss tensor.
    """
    # Cosine similarity matrix: (B, B)
    logits = torch.mm(anchors, positives.T) / temperature

    # Positive pairs are on the diagonal
    labels = torch.arange(logits.shape[0], device=logits.device)

    # Symmetric loss
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)

    return (loss_a + loss_b) / 2


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    image_dir: str | Path = _IMAGE_DIR,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    temperature: float = 0.07,
    weight_decay: float = 1e-4,
    pairs_per_image: int = 2,
    checkpoint_path: str | Path = _CHECKPOINT_PATH,
    eval_every: int = 5,
    patience: int = 10,
    num_workers: int = 4,
) -> dict:
    """Train the projection head on synthetically augmented pairs.

    Parameters
    ----------
    image_dir : path
        Directory containing reference card images.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size (each sample is an anchor+positive pair).
    lr : float
        Learning rate for AdamW.
    temperature : float
        InfoNCE temperature parameter.
    weight_decay : float
        AdamW weight decay.
    pairs_per_image : int
        Number of augmented views per image per epoch.
    checkpoint_path : path
        Where to save the best model checkpoint.
    eval_every : int
        Run evaluation every N epochs.
    patience : int
        Early stopping patience (epochs without improvement).
    num_workers : int
        DataLoader workers.

    Returns
    -------
    dict with training metrics.
    """
    backbone, device = _get_backbone()

    # Initialize projection head
    proj = ProjectionHead().to(device)
    logger.info("ProjectionHead: %d trainable parameters",
                sum(p.numel() for p in proj.parameters()))

    # Dataset and dataloader
    dataset = CardPairDataset(image_dir, pairs_per_image=pairs_per_image)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # InfoNCE needs consistent batch sizes
        persistent_workers=num_workers > 0,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(proj.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Ensure checkpoint directory exists
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    best_eval_score = -1.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "eval_scores": []}

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        proj.train()
        epoch_loss = 0.0
        n_batches = 0

        for anchors_img, positives_img in loader:
            anchors_img = anchors_img.to(device, non_blocking=True)
            positives_img = positives_img.to(device, non_blocking=True)

            # Extract frozen DINOv2 features
            with torch.no_grad():
                anchor_feats = backbone(anchors_img)    # (B, 768)
                positive_feats = backbone(positives_img)  # (B, 768)

            # Project through the learnable head
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                anchor_proj = proj(anchor_feats)      # (B, 256)
                positive_proj = proj(positive_feats)   # (B, 256)
                loss = info_nce_loss(anchor_proj, positive_proj, temperature)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(proj.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        elapsed = time.time() - t_start
        logger.info(
            "Epoch %d/%d  loss=%.4f  lr=%.2e  elapsed=%.0fs",
            epoch, epochs, avg_loss,
            scheduler.get_last_lr()[0], elapsed,
        )

        # Periodic evaluation
        if epoch % eval_every == 0 or epoch == epochs:
            eval_result = evaluate(proj, backbone, device)
            score = eval_result.get("projected_top1_acc", 0.0)
            history["eval_scores"].append(
                {"epoch": epoch, "score": score, **eval_result})
            logger.info(
                "  Eval: projected top-1=%.1f%% top-5=%.1f%% "
                "mean_sim=%.4f  (raw top-1=%.1f%%)",
                eval_result.get("projected_top1_acc", 0) * 100,
                eval_result.get("projected_top5_acc", 0) * 100,
                eval_result.get("projected_mean_sim", 0),
                eval_result.get("raw_top1_acc", 0) * 100,
            )

            if score > best_eval_score:
                best_eval_score = score
                epochs_without_improvement = 0
                # Save best checkpoint
                torch.save({
                    "projection_head_state_dict": proj.state_dict(),
                    "epoch": epoch,
                    "loss": avg_loss,
                    "eval_score": score,
                    "config": {
                        "input_dim": 768,
                        "hidden_dim": 512,
                        "output_dim": 256,
                    },
                }, checkpoint_path)
                logger.info("  Saved best checkpoint (score=%.4f)", score)
            else:
                epochs_without_improvement += eval_every
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping after %d epochs without improvement.",
                        epochs_without_improvement)
                    break

    total_time = time.time() - t_start
    logger.info("Training complete in %.1f minutes.", total_time / 60)

    return {
        "total_time_s": total_time,
        "best_eval_score": best_eval_score,
        "final_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "epochs_completed": len(history["train_loss"]),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Evaluation: compare projected vs raw on binder_eval.json
# ---------------------------------------------------------------------------


def _load_eval_data() -> list[dict]:
    """Load evaluation entries from binder_eval.json.

    Returns a flat list of dicts with keys: segment_path, card_id, name.
    Entries with card_id=None (empty slots) are skipped.
    """
    if not _EVAL_PATH.exists():
        logger.warning("Eval dataset not found: %s", _EVAL_PATH)
        return []

    with open(_EVAL_PATH) as f:
        data = json.load(f)

    entries = []
    for page in data.get("pages", []):
        segments_dir = _PROJECT_ROOT / page["segments_dir"]
        for card in page.get("cards", []):
            if card.get("card_id") is None:
                continue
            seg_path = segments_dir / card["segment"]
            if seg_path.exists():
                entries.append({
                    "segment_path": str(seg_path),
                    "card_id": card["card_id"],
                    "name": card.get("name", ""),
                })
            else:
                logger.warning("Eval segment not found: %s", seg_path)

    logger.info("Loaded %d eval entries from %s", len(entries), _EVAL_PATH)
    return entries


def _card_id_from_index(raw_cid: str) -> str:
    """Convert index-format card_id to DB format.

    Index stores "set/set-num/variant" -> we want "set-num/variant".
    """
    parts = raw_cid.split("/")
    if len(parts) >= 3:
        return "/".join(parts[1:])
    return raw_cid


def evaluate(
    proj: Optional[ProjectionHead] = None,
    backbone: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    checkpoint_path: str | Path = _CHECKPOINT_PATH,
) -> dict:
    """Evaluate projected vs raw DINOv2 embeddings on binder_eval.json.

    For each eval segment:
    1. Extract DINOv2 embedding
    2. Search the raw FAISS index (768-dim) -> raw result
    3. Project through the trained head and search projected index -> projected result
    4. Compare top-1 and top-5 accuracy

    If no projected FAISS index exists yet, compares only raw cosine
    similarities (projected search is skipped).

    Parameters
    ----------
    proj : ProjectionHead, optional
        Pre-loaded projection head.  If None, loads from checkpoint.
    backbone : nn.Module, optional
        Pre-loaded DINOv2 backbone.  If None, loads from cache.
    device : torch.device, optional
    checkpoint_path : path
        Path to projection head checkpoint.

    Returns
    -------
    dict with accuracy metrics.
    """
    eval_entries = _load_eval_data()
    if not eval_entries:
        return {"error": "No eval data available"}

    # Load backbone
    if backbone is None or device is None:
        backbone, device = _get_backbone()

    # Load projection head
    if proj is None:
        proj = load_projection_head(checkpoint_path, device)
        if proj is None:
            logger.warning("No trained projection head found at %s", checkpoint_path)
            # Still evaluate raw DINOv2 as baseline
            proj = None

    # Load raw FAISS index for baseline comparison
    raw_index = None
    raw_card_ids = None
    if _RAW_INDEX_PATH.exists() and _RAW_CARD_IDS_PATH.exists():
        raw_index = faiss.read_index(str(_RAW_INDEX_PATH))
        with open(_RAW_CARD_IDS_PATH, "rb") as f:
            raw_card_ids = pickle.load(f)

    # Load projected FAISS index if available
    proj_index = None
    proj_card_ids = None
    if _PROJECTED_INDEX_PATH.exists() and _PROJECTED_CARD_IDS_PATH.exists():
        proj_index = faiss.read_index(str(_PROJECTED_INDEX_PATH))
        with open(_PROJECTED_CARD_IDS_PATH, "rb") as f:
            proj_card_ids = pickle.load(f)

    # DINOv2 transform (same as dino_matcher.py)
    dino_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    # Evaluate each segment
    raw_top1_correct = 0
    raw_top5_correct = 0
    raw_similarities = []
    proj_top1_correct = 0
    proj_top5_correct = 0
    proj_similarities = []
    details = []
    total = 0

    for entry in eval_entries:
        seg_path = entry["segment_path"]
        true_id = entry["card_id"]
        total += 1

        try:
            img = Image.open(seg_path).convert("RGB")
            tensor = dino_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = backbone(tensor)  # (1, 768)

            detail = {"segment": seg_path, "true_id": true_id,
                       "name": entry["name"]}

            # --- Raw DINOv2 search ---
            if raw_index is not None:
                raw_vec = feat.cpu().numpy().astype(np.float32)
                raw_norm = np.linalg.norm(raw_vec)
                if raw_norm > 0:
                    raw_vec /= raw_norm
                scores, indices = raw_index.search(raw_vec, 5)
                raw_matches = [
                    (_card_id_from_index(raw_card_ids[idx]), float(s))
                    for s, idx in zip(scores[0], indices[0]) if idx >= 0
                ]
                if raw_matches:
                    raw_similarities.append(raw_matches[0][1])
                    raw_top1_ids = [raw_matches[0][0]]
                    raw_top5_ids = [m[0] for m in raw_matches[:5]]
                    if true_id in raw_top1_ids:
                        raw_top1_correct += 1
                    if true_id in raw_top5_ids:
                        raw_top5_correct += 1
                    detail["raw_top1"] = raw_matches[0]
                    detail["raw_top5"] = raw_matches[:5]

            # --- Projected search ---
            if proj is not None and proj_index is not None:
                with torch.no_grad():
                    proj_vec = proj(feat).cpu().numpy().astype(np.float32)
                scores, indices = proj_index.search(proj_vec, 5)
                proj_matches = [
                    (_card_id_from_index(proj_card_ids[idx]), float(s))
                    for s, idx in zip(scores[0], indices[0]) if idx >= 0
                ]
                if proj_matches:
                    proj_similarities.append(proj_matches[0][1])
                    proj_top1_ids = [proj_matches[0][0]]
                    proj_top5_ids = [m[0] for m in proj_matches[:5]]
                    if true_id in proj_top1_ids:
                        proj_top1_correct += 1
                    if true_id in proj_top5_ids:
                        proj_top5_correct += 1
                    detail["proj_top1"] = proj_matches[0]
                    detail["proj_top5"] = proj_matches[:5]
            elif proj is not None:
                # No projected index yet; just show projected similarity
                # against the raw embedding to demonstrate the projection works
                with torch.no_grad():
                    proj_vec = proj(feat)  # (1, 256)
                detail["proj_embedding_norm"] = float(
                    torch.norm(proj_vec).item())

            details.append(detail)

        except Exception as e:
            logger.warning("Eval failed for %s: %s", seg_path, e)

    result = {
        "total": total,
        "raw_top1_acc": raw_top1_correct / max(total, 1),
        "raw_top5_acc": raw_top5_correct / max(total, 1),
        "raw_mean_sim": float(np.mean(raw_similarities))
            if raw_similarities else 0.0,
        "projected_top1_acc": proj_top1_correct / max(total, 1),
        "projected_top5_acc": proj_top5_correct / max(total, 1),
        "projected_mean_sim": float(np.mean(proj_similarities))
            if proj_similarities else 0.0,
        "details": details,
    }

    return result


# ---------------------------------------------------------------------------
# Load / save projection head
# ---------------------------------------------------------------------------


def load_projection_head(
    checkpoint_path: str | Path = _CHECKPOINT_PATH,
    device: Optional[torch.device] = None,
) -> Optional[ProjectionHead]:
    """Load a trained ProjectionHead from a checkpoint file.

    Returns None if the checkpoint does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ckpt.get("config", {})

    proj = ProjectionHead(
        input_dim=config.get("input_dim", 768),
        hidden_dim=config.get("hidden_dim", 512),
        output_dim=config.get("output_dim", 256),
    )
    proj.load_state_dict(ckpt["projection_head_state_dict"])
    proj.to(device)
    proj.eval()

    logger.info(
        "Loaded projection head from %s (epoch=%d, score=%.4f)",
        checkpoint_path, ckpt.get("epoch", -1), ckpt.get("eval_score", -1),
    )
    return proj


# ---------------------------------------------------------------------------
# Index building with projected embeddings
# ---------------------------------------------------------------------------


def build_projected_index(
    image_dir: str | Path = _IMAGE_DIR,
    checkpoint_path: str | Path = _CHECKPOINT_PATH,
    index_path: str | Path = _PROJECTED_INDEX_PATH,
    mapping_path: str | Path = _PROJECTED_CARD_IDS_PATH,
    batch_size: int = 64,
) -> int:
    """Build a FAISS index using projected 256-dim embeddings.

    Loads the frozen DINOv2 backbone and trained projection head,
    processes all reference images, and saves the index.

    Parameters
    ----------
    image_dir : path
        Directory containing reference card images.
    checkpoint_path : path
        Path to trained ProjectionHead checkpoint.
    index_path : path
        Where to save the projected FAISS index.
    mapping_path : path
        Where to save the card-ID mapping (pickle).
    batch_size : int
        Number of images to process at once on GPU.

    Returns
    -------
    int : number of images indexed.
    """
    backbone, device = _get_backbone()
    proj = load_projection_head(checkpoint_path, device)
    if proj is None:
        raise FileNotFoundError(
            f"No trained projection head at {checkpoint_path}. "
            "Run training first.")
    proj.eval()

    image_dir = Path(image_dir)
    image_files = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    logger.info("Building projected index from %d images ...",
                len(image_files))

    # DINOv2 transform
    dino_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    all_embeddings = []
    card_ids = []
    failed = 0

    # Process in batches for GPU efficiency
    batch_tensors = []
    batch_paths = []

    for i, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = dino_transform(img)
            batch_tensors.append(tensor)
            batch_paths.append(img_path)
        except Exception:
            logger.warning("Failed to load %s", img_path, exc_info=True)
            failed += 1
            continue

        # Process batch when full or at end
        if len(batch_tensors) >= batch_size or i == len(image_files) - 1:
            batch = torch.stack(batch_tensors).to(device)

            with torch.no_grad():
                feats = backbone(batch)       # (B, 768)
                projected = proj(feats)        # (B, 256)

            proj_np = projected.cpu().numpy().astype(np.float32)
            all_embeddings.append(proj_np)

            # Derive card_ids (same logic as dino_matcher.py)
            for bp in batch_paths:
                rel = bp.relative_to(image_dir).with_suffix("")
                cid = str(rel).replace(os.sep, "/")
                last_under = cid.rfind("_")
                if last_under != -1:
                    cid = cid[:last_under] + "/" + cid[last_under + 1:]
                card_ids.append(cid)

            batch_tensors.clear()
            batch_paths.clear()

            if (i + 1) % 1000 == 0 or i == len(image_files) - 1:
                logger.info("Processed %d / %d images ...",
                            i + 1, len(image_files))

    if not all_embeddings:
        raise RuntimeError("No embeddings extracted.")

    matrix = np.concatenate(all_embeddings, axis=0)  # (N, 256)
    dim = matrix.shape[1]

    # Build FAISS IndexFlatIP (inner product == cosine on L2-normed vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    # Save
    os.makedirs(os.path.dirname(str(index_path)) or ".", exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(mapping_path, "wb") as f:
        pickle.dump(card_ids, f)

    logger.info(
        "Saved projected FAISS index (%d vectors, dim=%d) to %s",
        index.ntotal, dim, index_path,
    )
    return len(card_ids)


# ---------------------------------------------------------------------------
# Project a single image (for integration with the cascade)
# ---------------------------------------------------------------------------

_proj_head: Optional[ProjectionHead] = None


def project_embedding(
    raw_embedding: np.ndarray,
    checkpoint_path: str | Path = _CHECKPOINT_PATH,
) -> np.ndarray:
    """Project a raw 768-dim DINOv2 embedding to 256-dim.

    Loads the projection head on first call and caches it.

    Parameters
    ----------
    raw_embedding : (768,) float32 array, L2-normalized.

    Returns
    -------
    (256,) float32 array, L2-normalized.
    """
    global _proj_head
    if _proj_head is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _proj_head = load_projection_head(checkpoint_path, device)
        if _proj_head is None:
            raise FileNotFoundError(
                f"No trained projection head at {checkpoint_path}")

    device = next(_proj_head.parameters()).device
    tensor = torch.from_numpy(raw_embedding).float().unsqueeze(0).to(device)

    with torch.no_grad():
        projected = _proj_head(tensor)

    return projected.cpu().numpy().astype(np.float32).squeeze()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="DINOv2 projection head training and evaluation")
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train projection head")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--temperature", type=float, default=0.07)
    p_train.add_argument("--pairs-per-image", type=int, default=2)
    p_train.add_argument("--patience", type=int, default=10)
    p_train.add_argument("--num-workers", type=int, default=4)
    p_train.add_argument("--image-dir", type=str,
                         default=str(_IMAGE_DIR))
    p_train.add_argument("--checkpoint", type=str,
                         default=str(_CHECKPOINT_PATH))

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate on binder_eval.json")
    p_eval.add_argument("--checkpoint", type=str,
                        default=str(_CHECKPOINT_PATH))

    # build-index
    p_idx = sub.add_parser("build-index",
                           help="Build projected FAISS index")
    p_idx.add_argument("--checkpoint", type=str,
                       default=str(_CHECKPOINT_PATH))
    p_idx.add_argument("--batch-size", type=int, default=64)
    p_idx.add_argument("--image-dir", type=str,
                       default=str(_IMAGE_DIR))

    args = parser.parse_args()

    if args.command == "train":
        result = train(
            image_dir=args.image_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            pairs_per_image=args.pairs_per_image,
            patience=args.patience,
            num_workers=args.num_workers,
            checkpoint_path=args.checkpoint,
        )
        print(f"\nTraining complete in {result['total_time_s']:.0f}s")
        print(f"Best eval score: {result['best_eval_score']:.4f}")
        print(f"Final loss: {result['final_loss']:.4f}")

    elif args.command == "eval":
        result = evaluate(checkpoint_path=args.checkpoint)
        print(f"\n{'='*60}")
        print(f"Evaluation Results ({result['total']} segments)")
        print(f"{'='*60}")
        print(f"Raw DINOv2:   top-1={result['raw_top1_acc']:.1%}  "
              f"top-5={result['raw_top5_acc']:.1%}  "
              f"mean_sim={result['raw_mean_sim']:.4f}")
        print(f"Projected:    top-1={result['projected_top1_acc']:.1%}  "
              f"top-5={result['projected_top5_acc']:.1%}  "
              f"mean_sim={result['projected_mean_sim']:.4f}")
        print()
        for d in result.get("details", []):
            raw_top1 = d.get("raw_top1", ("?", 0))
            proj_top1 = d.get("proj_top1", ("?", 0))
            match_raw = "OK" if raw_top1[0] == d["true_id"] else "MISS"
            match_proj = "OK" if proj_top1[0] == d["true_id"] else "MISS"
            print(f"  {d['name']:25s}  true={d['true_id']:20s}  "
                  f"raw={raw_top1[0]:20s} ({raw_top1[1]:.3f} {match_raw})  "
                  f"proj={proj_top1[0]:20s} ({proj_top1[1]:.3f} {match_proj})")

    elif args.command == "build-index":
        n = build_projected_index(
            image_dir=args.image_dir,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
        )
        print(f"\nBuilt projected FAISS index with {n} vectors.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
