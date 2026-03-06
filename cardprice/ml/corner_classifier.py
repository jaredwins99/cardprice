"""Corner wear classifier using EfficientNet-B0 for Pokemon card grading.

Extracts 4 corner ROIs from a card image and classifies each into one of
5 wear grades.  The overall corner grade is the worst (highest wear) of
the 4 corners, following the PSA grading standard where a single bad
corner drags down the whole card.

Architecture:
  - EfficientNet-B0 backbone (frozen, ImageNet pre-trained, 5.3M params)
  - Custom head: AdaptiveAvgPool -> Dropout(0.3) -> Linear(1280, 256)
                  -> ReLU -> Dropout(0.2) -> Linear(256, 5)
  - Input: 224x224 corner ROI (resized from 160x173 native extraction)
  - Output: 5-class softmax (Gem, Mint, Light, Moderate, Heavy)

Corner ROI extraction:
  Standard Pokemon card: 63mm x 88mm.  Each corner ROI covers ~10mm square,
  which maps to roughly 160x173 pixels on a 1008x1408 card image (the
  standard segment size produced by card_segmenter).  The ROIs include the
  rounded corner plus adjacent border and a sliver of the artwork frame,
  which is the exact region BGS/PSA graders inspect.

Training data source:
  eBay BGS slab photos where per-corner sub-grades are printed on the label
  (e.g., "Corners: 9.5").  A scraper + label OCR pipeline extracts these
  sub-grades to create labeled training pairs.

Usage:
    from cardprice.ml.corner_classifier import (
        extract_corner_rois,
        CornerClassifier,
        grade_corners,
    )

    # Quick inference
    result = grade_corners("path/to/card_segment.png")
    # result = {
    #     "overall_grade": "Mint",
    #     "overall_confidence": 0.87,
    #     "corners": {
    #         "top_left":     {"grade": "Gem",  "confidence": 0.95},
    #         "top_right":    {"grade": "Mint", "confidence": 0.87},
    #         "bottom_left":  {"grade": "Gem",  "confidence": 0.92},
    #         "bottom_right": {"grade": "Mint", "confidence": 0.89},
    #     },
    # }

    # Training
    from corner_classifier import CornerWearDataset, train
    train(data_dir="data/corner_training", epochs=30, batch_size=32)
"""

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Grade classes ordered from best to worst
GRADES = ["Gem", "Mint", "Light", "Moderate", "Heavy"]
GRADE_TO_IDX = {g: i for i, g in enumerate(GRADES)}
NUM_CLASSES = len(GRADES)

# Standard card segment dimensions from card_segmenter (pixels)
CARD_W = 1008
CARD_H = 1408  # 63:88 ratio

# Corner ROI size in pixels at standard card dimensions.
# ~10mm on a 63mm-wide card => 10/63 * 1008 ≈ 160 px wide
# ~10mm on an 88mm-tall card => 10/88 * 1408 ≈ 160 px tall
# We use slightly taller to capture the border-to-art transition.
ROI_W = 160
ROI_H = 173

# EfficientNet-B0 expected input size
INPUT_SIZE = 224

# Model checkpoint path
DEFAULT_CHECKPOINT = Path("data/checkpoints/corner_classifier.pt")

# ImageNet normalization (required for pre-trained EfficientNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Corner names
CORNER_NAMES = ["top_left", "top_right", "bottom_left", "bottom_right"]

# ---------------------------------------------------------------------------
# Corner ROI extraction
# ---------------------------------------------------------------------------


def extract_corner_rois(
    image: Union[str, Path, np.ndarray],
    roi_w: int = ROI_W,
    roi_h: int = ROI_H,
) -> dict[str, np.ndarray]:
    """Extract 4 corner ROI crops from a card image.

    Parameters
    ----------
    image : path or BGR ndarray
        Card segment image.  If a path, read from disk.
    roi_w, roi_h : int
        ROI dimensions in pixels (at the image's native resolution).
        Defaults are calibrated for 1008x1408 segments.  For other
        resolutions, the ROI is proportionally scaled.

    Returns
    -------
    dict mapping corner name -> BGR ndarray (roi_h x roi_w x 3)
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        img = image

    h, w = img.shape[:2]

    # Scale ROI proportionally if image differs from standard dimensions
    scale_x = w / CARD_W
    scale_y = h / CARD_H
    rw = max(int(roi_w * scale_x), 16)
    rh = max(int(roi_h * scale_y), 16)

    corners = {
        "top_left":     img[0:rh, 0:rw],
        "top_right":    img[0:rh, w - rw:w],
        "bottom_left":  img[h - rh:h, 0:rw],
        "bottom_right": img[h - rh:h, w - rw:w],
    }

    return corners


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CornerClassifier(nn.Module):
    """EfficientNet-B0 with frozen backbone and custom 5-class head.

    The backbone convolutional layers are frozen (no gradient) so only the
    classifier head trains.  This is fast, needs little data (~500 images
    per class), and avoids overfitting.

    Fine-tuning the last few backbone blocks can be enabled later via
    ``unfreeze_backbone(num_blocks)`` once more training data is available.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()

        # Load pre-trained EfficientNet-B0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        base = efficientnet_b0(weights=weights)

        # Backbone = everything except the final classifier
        self.features = base.features        # Conv layers
        self.avgpool = base.avgpool          # AdaptiveAvgPool2d(1)

        # Freeze all backbone parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # Custom classifier head
        # EfficientNet-B0 features output: 1280 channels
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

        # Inference transforms (same normalization as training)
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.  x shape: (B, 3, 224, 224)."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def unfreeze_backbone(self, num_blocks: int = 2):
        """Unfreeze the last N blocks of the backbone for fine-tuning.

        EfficientNet-B0 has 9 blocks (features[0] through features[8]).
        Unfreezing the last 2 blocks is a good starting point when you
        have 2k+ labeled images.
        """
        total_blocks = len(self.features)
        for i in range(max(0, total_blocks - num_blocks), total_blocks):
            for param in self.features[i].parameters():
                param.requires_grad = True
        logger.info("Unfroze backbone blocks %d-%d", total_blocks - num_blocks, total_blocks - 1)

    def preprocess(self, bgr_image: np.ndarray) -> torch.Tensor:
        """Convert a BGR corner ROI to a normalized tensor.

        Parameters
        ----------
        bgr_image : np.ndarray
            Corner crop in BGR format (OpenCV convention).

        Returns
        -------
        torch.Tensor shape (1, 3, 224, 224)
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb)
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def predict(
        self, corner_image: np.ndarray, device: Optional[torch.device] = None,
    ) -> tuple[str, float]:
        """Classify a single corner ROI.

        Parameters
        ----------
        corner_image : np.ndarray
            BGR corner crop from ``extract_corner_rois``.
        device : torch.device, optional
            Run inference on this device.  Defaults to CPU.

        Returns
        -------
        (grade_name, confidence) : (str, float)
            e.g. ("Mint", 0.93)
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        x = self.preprocess(corner_image).to(device)
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        idx = probs.argmax().item()
        return GRADES[idx], probs[idx].item()


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

# Module-level singleton (lazy loaded)
_model: Optional[CornerClassifier] = None
_device: Optional[torch.device] = None


def _get_model(
    checkpoint: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> tuple[CornerClassifier, torch.device]:
    """Load or return cached model singleton."""
    global _model, _device

    if _model is not None:
        return _model, _device

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _device = device

    model = CornerClassifier()

    ckpt_path = checkpoint or DEFAULT_CHECKPOINT
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded corner classifier from %s (epoch %d)", ckpt_path, state.get("epoch", -1))
    else:
        logger.warning(
            "No checkpoint at %s — model has random classifier head. "
            "Run training before using predictions.",
            ckpt_path,
        )

    model.to(device)
    model.eval()
    _model = model
    return _model, _device


def grade_corners(
    image: Union[str, Path, np.ndarray],
    checkpoint: Optional[Path] = None,
) -> dict:
    """Grade all 4 corners of a card image.

    Parameters
    ----------
    image : path or BGR ndarray
        Full card segment image.
    checkpoint : Path, optional
        Path to model checkpoint.  Uses default if not given.

    Returns
    -------
    dict with keys:
        overall_grade : str       -- worst corner grade
        overall_confidence : float -- confidence of the worst corner
        corners : dict            -- per-corner {grade, confidence}
    """
    model, device = _get_model(checkpoint)
    rois = extract_corner_rois(image)

    results = {}
    worst_idx = 0
    worst_conf = 1.0

    for name in CORNER_NAMES:
        grade, conf = model.predict(rois[name], device=device)
        results[name] = {"grade": grade, "confidence": round(conf, 4)}
        grade_idx = GRADE_TO_IDX[grade]
        if grade_idx > worst_idx or (grade_idx == worst_idx and conf > worst_conf):
            worst_idx = grade_idx
            worst_conf = conf

    return {
        "overall_grade": GRADES[worst_idx],
        "overall_confidence": round(worst_conf, 4),
        "corners": results,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class CornerWearDataset(torch.utils.data.Dataset):
    """Dataset for corner wear training images.

    Expected directory layout::

        data_dir/
            Gem/
                img_001_tl.png
                img_001_tr.png
                ...
            Mint/
                ...
            Light/
                ...
            Moderate/
                ...
            Heavy/
                ...

    Each image is a corner ROI crop (any resolution — resized to 224x224).
    File names don't matter; the parent folder name is the label.
    """

    def __init__(self, data_dir: Union[str, Path], augment: bool = True):
        self.data_dir = Path(data_dir)
        self.samples: list[tuple[Path, int]] = []
        self.augment = augment

        for grade in GRADES:
            grade_dir = self.data_dir / grade
            if not grade_dir.is_dir():
                logger.warning("Missing grade directory: %s", grade_dir)
                continue
            idx = GRADE_TO_IDX[grade]
            for img_path in sorted(grade_dir.glob("*.png")):
                self.samples.append((img_path, idx))
            for img_path in sorted(grade_dir.glob("*.jpg")):
                self.samples.append((img_path, idx))

        logger.info(
            "CornerWearDataset: %d samples from %s (%s)",
            len(self.samples),
            self.data_dir,
            {g: sum(1 for _, i in self.samples if i == GRADE_TO_IDX[g]) for g in GRADES},
        )

        # Training augmentations — deliberately mild because corner wear
        # features (edge fraying, whitening) are small and spatially precise.
        # Heavy geometric augmentation would destroy the signal.
        self._train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self._val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Cannot read: {path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transform = self._train_transform if self.augment else self._val_transform
        tensor = transform(rgb)
        return tensor, label


def train(
    data_dir: Union[str, Path] = "data/corner_training",
    checkpoint_dir: Union[str, Path] = "data/checkpoints",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.15,
    unfreeze_after: int = 15,
    unfreeze_blocks: int = 2,
    device: Optional[torch.device] = None,
):
    """Train the corner wear classifier.

    Two-phase training:
      Phase 1 (epochs 0..unfreeze_after-1): Only the classifier head trains.
      Phase 2 (epochs unfreeze_after..end): Last N backbone blocks unfrozen,
              learning rate reduced 10x for fine-tuning.

    Parameters
    ----------
    data_dir : path
        Root of the training data (see CornerWearDataset layout).
    checkpoint_dir : path
        Where to save checkpoints.
    epochs : int
        Total training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Initial learning rate (for head-only phase).
    val_split : float
        Fraction of data held out for validation.
    unfreeze_after : int
        Epoch at which to unfreeze backbone blocks.
    unfreeze_blocks : int
        Number of backbone blocks to unfreeze in phase 2.
    device : torch.device, optional
        Training device.  Auto-detects GPU if available.
    """
    from torch.utils.data import DataLoader, random_split

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    # --- Data ---
    full_dataset = CornerWearDataset(data_dir, augment=True)
    if len(full_dataset) == 0:
        raise RuntimeError(f"No training images found in {data_dir}")

    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # Disable augmentation for validation subset
    val_ds.dataset = CornerWearDataset(data_dir, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # --- Model ---
    model = CornerClassifier()
    model.to(device)

    # Class weights for imbalanced data — Gem/Mint are much more common
    # than Heavy in real-world grading.  Compute from dataset distribution.
    class_counts = torch.zeros(NUM_CLASSES)
    for _, label in full_dataset.samples:
        class_counts[label] += 1
    # Inverse frequency weighting, clamped to avoid explosion on rare classes
    class_weights = (class_counts.sum() / (NUM_CLASSES * class_counts.clamp(min=1))).to(device)
    logger.info("Class weights: %s", {GRADES[i]: f"{class_weights[i]:.2f}" for i in range(NUM_CLASSES)})

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer — only head parameters initially
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Training loop ---
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Phase 2: unfreeze backbone blocks
        if epoch == unfreeze_after:
            logger.info("Phase 2: unfreezing last %d backbone blocks", unfreeze_blocks)
            model.unfreeze_backbone(unfreeze_blocks)
            # Add backbone params to optimizer with lower LR
            optimizer.add_param_group({
                "params": [p for p in model.features.parameters() if p.requires_grad],
                "lr": lr * 0.1,
            })

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        logger.info(
            "Epoch %2d/%d  train_loss=%.4f  train_acc=%.3f  val_loss=%.4f  val_acc=%.3f  lr=%.1e",
            epoch + 1, epochs,
            train_loss / max(train_total, 1), train_acc,
            val_loss / max(val_total, 1), val_acc,
            optimizer.param_groups[0]["lr"],
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_weights": class_weights.cpu(),
            }
            torch.save(ckpt, checkpoint_dir / "corner_classifier.pt")
            logger.info("  -> Saved best checkpoint (val_acc=%.3f)", val_acc)

    logger.info("Training complete.  Best val_acc=%.3f", best_val_acc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Corner wear classifier")
    sub = parser.add_subparsers(dest="cmd")

    # --- predict ---
    p_pred = sub.add_parser("predict", help="Grade corners of a card image")
    p_pred.add_argument("image", help="Path to card segment image")
    p_pred.add_argument("--checkpoint", type=Path, default=None)

    # --- train ---
    p_train = sub.add_parser("train", help="Train the classifier")
    p_train.add_argument("--data-dir", default="data/corner_training")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)

    # --- extract ---
    p_extract = sub.add_parser("extract", help="Extract and save corner ROIs")
    p_extract.add_argument("image", help="Path to card segment image")
    p_extract.add_argument("--output-dir", default="data/corner_rois")

    args = parser.parse_args()

    if args.cmd == "predict":
        result = grade_corners(args.image, checkpoint=args.checkpoint)
        print(json.dumps(result, indent=2))

    elif args.cmd == "train":
        train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    elif args.cmd == "extract":
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rois = extract_corner_rois(args.image)
        stem = Path(args.image).stem
        for name, roi in rois.items():
            path = out / f"{stem}_{name}.png"
            cv2.imwrite(str(path), roi)
            print(f"Saved {path} ({roi.shape[1]}x{roi.shape[0]})")

    else:
        parser.print_help()
