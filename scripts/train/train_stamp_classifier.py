#!/usr/bin/env python3
"""Train a binary stamp classifier: stamped vs non-stamped card images.

Architecture: DINOv2 ViT-B/14 features -> logistic regression.

Data:
  - Synthetic (training): data/condition_training/stamps/labels.jsonl (400 images)
  - Real reference photos (training+val): data/condition_training/stamps_real/sources.jsonl
  - Binder scans (training+LOO val): data/condition_training/stamps_real/binder_ground_truth.jsonl

The binder scan data is augmented with brightness/contrast/color jitter to simulate
the domain gap caused by binder sleeve reflections.

Saves best model to data/stamp_classifier.pkl.
"""

import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "condition_training" / "stamps"
REAL_DIR = PROJECT_ROOT / "data" / "condition_training" / "stamps_real"
INBOX_DIR = PROJECT_ROOT / "data" / "inbox"
BINDER_GT_PATH = REAL_DIR / "binder_ground_truth.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "stamp_classifier.pkl"

GRID_SIZE = 16
NUM_PATCHES = GRID_SIZE * GRID_SIZE
EMBED_DIM = 768

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading DINOv2 ViT-B/14 on %s ...", device)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.to(device)
    model.eval()
    return model, device


def extract_features_batch(model, device, image_paths, batch_size=32):
    """Extract CLS tokens and patch tokens. Returns (N,768) and (N,256,768)."""
    all_cls = []
    all_patches = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        tensors = [_transform(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            cls_out = model(batch)
            patch_out = model.get_intermediate_layers(batch, n=1)
            patch_tokens = patch_out[0]

        cls_np = cls_out.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(cls_np, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        cls_np /= norms
        all_cls.append(cls_np)

        patches_np = patch_tokens.cpu().numpy().astype(np.float32)
        for i in range(patches_np.shape[0]):
            pnorms = np.linalg.norm(patches_np[i], axis=1, keepdims=True)
            pnorms[pnorms == 0] = 1.0
            patches_np[i] /= pnorms
        all_patches.append(patches_np)

        logger.info("  Extracted %d/%d", min(start + batch_size, len(image_paths)), len(image_paths))

    return np.concatenate(all_cls), np.concatenate(all_patches)


def extract_features_from_pil_images(model, device, pil_images, batch_size=32):
    """Extract features from PIL Image objects (for augmented images)."""
    all_cls = []
    all_patches = []

    for start in range(0, len(pil_images), batch_size):
        batch_imgs = pil_images[start:start + batch_size]
        tensors = [_transform(img) for img in batch_imgs]
        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            cls_out = model(batch)
            patch_out = model.get_intermediate_layers(batch, n=1)
            patch_tokens = patch_out[0]

        cls_np = cls_out.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(cls_np, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        cls_np /= norms
        all_cls.append(cls_np)

        patches_np = patch_tokens.cpu().numpy().astype(np.float32)
        for i in range(patches_np.shape[0]):
            pnorms = np.linalg.norm(patches_np[i], axis=1, keepdims=True)
            pnorms[pnorms == 0] = 1.0
            patches_np[i] /= pnorms
        all_patches.append(patches_np)

    return np.concatenate(all_cls), np.concatenate(all_patches)


def augment_binder_image(img, rng):
    """Apply binder-sleeve-style augmentation to a PIL image.

    Simulates the visual artifacts of photographing cards through binder sleeves:
    - Brightness shifts (sleeve reduces/adds light)
    - Contrast reduction (sleeve plastic diffuses)
    - Color temperature shifts (warm/cool from lighting + sleeve)
    - Slight saturation changes
    """
    img = img.copy()

    # Brightness jitter: sleeves can make cards brighter (glare) or darker
    brightness_factor = rng.uniform(0.7, 1.4)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Contrast reduction: sleeve plastic diffuses light
    contrast_factor = rng.uniform(0.7, 1.2)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # Color temperature shift: simulate warm/cool lighting through plastic
    arr = np.array(img, dtype=np.float32)
    # Shift R/G/B channels independently
    r_shift = rng.uniform(-15, 15)
    g_shift = rng.uniform(-10, 10)
    b_shift = rng.uniform(-15, 15)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + r_shift, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + g_shift, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + b_shift, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))

    # Saturation shift
    saturation_factor = rng.uniform(0.8, 1.3)
    img = ImageEnhance.Color(img).enhance(saturation_factor)

    return img


def get_region(patches, region):
    """Get patches from a region. patches: (N,256,768) or (256,768)."""
    single = patches.ndim == 2
    if single:
        patches = patches[np.newaxis]
    N = patches.shape[0]
    grid = patches.reshape(N, GRID_SIZE, GRID_SIZE, EMBED_DIM)
    if region == 'br':
        out = grid[:, 8:, 8:, :].reshape(N, -1, EMBED_DIM)
    elif region == 'bl':
        out = grid[:, 8:, :8, :].reshape(N, -1, EMBED_DIM)
    elif region == 'tr':
        out = grid[:, :8, 8:, :].reshape(N, -1, EMBED_DIM)
    elif region == 'tl':
        out = grid[:, :8, :8, :].reshape(N, -1, EMBED_DIM)
    elif region == 'bottom':
        out = grid[:, 8:, :, :].reshape(N, -1, EMBED_DIM)
    elif region == 'top':
        out = grid[:, :8, :, :].reshape(N, -1, EMBED_DIM)
    elif region == 'all':
        out = patches.reshape(N, -1, EMBED_DIM)
    elif region == 'center':
        # Central 8x8 region -- avoids edge artifacts from sleeve borders
        out = grid[:, 4:12, 4:12, :].reshape(N, -1, EMBED_DIM)
    elif region == 'bottom_center':
        # Bottom-center: where stamps appear, avoiding sleeve edge glare
        out = grid[:, 10:, 4:12, :].reshape(N, -1, EMBED_DIM)
    else:
        raise ValueError(region)
    return out[0] if single else out


def patch_stats(patches):
    """Mean, std, max, min over patches. (N,K,768) -> (N, 768*4)."""
    return np.hstack([
        np.mean(patches, axis=1),
        np.std(patches, axis=1),
        np.max(patches, axis=1),
        np.min(patches, axis=1),
    ])


def patch_variance_ratio(patches_region, patches_all):
    """Ratio of regional variance to global variance -- detects localized texture.
    Stamps add localized high-frequency texture that sleeves don't.
    Returns (N, 768).
    """
    region_var = np.var(patches_region, axis=1)  # (N, 768)
    all_var = np.var(patches_all, axis=1)  # (N, 768)
    # Avoid division by zero
    all_var = np.maximum(all_var, 1e-8)
    return region_var / all_var


def build_features(cls_tokens, patch_tokens, feature_type):
    """Build feature matrix for a given feature type."""
    N = cls_tokens.shape[0]

    if feature_type == "cls":
        return cls_tokens

    elif feature_type == "br_stats":
        return patch_stats(get_region(patch_tokens, 'br'))

    elif feature_type == "multi_region_stats":
        # Stats from 4 quadrants + full card
        parts = []
        for r in ['br', 'bl', 'tr', 'tl']:
            parts.append(patch_stats(get_region(patch_tokens, r)))
        # Also cross-region differences
        br_mean = np.mean(get_region(patch_tokens, 'br'), axis=1)
        bl_mean = np.mean(get_region(patch_tokens, 'bl'), axis=1)
        tr_mean = np.mean(get_region(patch_tokens, 'tr'), axis=1)
        tl_mean = np.mean(get_region(patch_tokens, 'tl'), axis=1)
        parts.append(br_mean - bl_mean)
        parts.append(br_mean - tr_mean)
        parts.append(br_mean - tl_mean)
        return np.hstack(parts)

    elif feature_type == "all_stats":
        return patch_stats(get_region(patch_tokens, 'all'))

    elif feature_type == "cls_br_stats":
        return np.hstack([cls_tokens, patch_stats(get_region(patch_tokens, 'br'))])

    elif feature_type == "cls_all_stats":
        return np.hstack([cls_tokens, patch_stats(get_region(patch_tokens, 'all'))])

    elif feature_type == "cls_multi_region":
        multi = build_features(cls_tokens, patch_tokens, "multi_region_stats")
        return np.hstack([cls_tokens, multi])

    elif feature_type == "bottom_center_stats":
        # Binder-optimized: focus on stamp region, avoid sleeve edge glare
        return patch_stats(get_region(patch_tokens, 'bottom_center'))

    elif feature_type == "center_stats":
        # Center region avoids sleeve edge artifacts
        return patch_stats(get_region(patch_tokens, 'center'))

    elif feature_type == "variance_ratio":
        # Variance ratio: stamps create localized texture, sleeves don't
        br = get_region(patch_tokens, 'br')
        all_p = get_region(patch_tokens, 'all')
        bc = get_region(patch_tokens, 'bottom_center')
        parts = [
            patch_stats(br),
            patch_variance_ratio(br, all_p),
            patch_variance_ratio(bc, all_p),
        ]
        return np.hstack(parts)

    elif feature_type == "binder_robust":
        # Designed for binder scans: cross-region differences normalize out
        # global lighting/sleeve effects. Variance ratios detect localized stamp
        # texture vs diffuse sleeve reflections.
        parts = []
        br = get_region(patch_tokens, 'br')
        bl = get_region(patch_tokens, 'bl')
        bc = get_region(patch_tokens, 'bottom_center')
        all_p = get_region(patch_tokens, 'all')

        # Bottom-right stats (primary stamp location)
        parts.append(patch_stats(br))
        # Bottom-center stats (stamp without edge glare)
        parts.append(patch_stats(bc))

        # Cross-region differences: cancel out sleeve-wide effects
        br_mean = np.mean(br, axis=1)
        bl_mean = np.mean(bl, axis=1)
        tr_mean = np.mean(get_region(patch_tokens, 'tr'), axis=1)
        parts.append(br_mean - bl_mean)  # Stamp asymmetry
        parts.append(br_mean - tr_mean)  # Vertical asymmetry

        # Variance ratios: stamps = localized texture, sleeves = global
        parts.append(patch_variance_ratio(br, all_p))
        parts.append(patch_variance_ratio(bc, all_p))

        return np.hstack(parts)

    elif feature_type == "cls_binder_robust":
        binder = build_features(cls_tokens, patch_tokens, "binder_robust")
        return np.hstack([cls_tokens, binder])

    else:
        raise ValueError(feature_type)


def load_dataset(jsonl_path, base_dir):
    entries = [json.loads(line) for line in open(jsonl_path)]
    paths, labels = [], []
    for e in entries:
        img_path = base_dir / e["image"]
        if img_path.exists():
            paths.append(img_path)
            labels.append(1 if e["stamped"] else 0)
        else:
            logger.warning("Missing: %s", img_path)
    return paths, np.array(labels, dtype=np.int32)


def load_binder_dataset():
    """Load binder ground truth. Images are in data/inbox/."""
    entries = [json.loads(line) for line in open(BINDER_GT_PATH)]
    paths, labels = [], []
    for e in entries:
        img_path = INBOX_DIR / e["image"]
        if img_path.exists():
            paths.append(img_path)
            labels.append(1 if e["stamped"] else 0)
        else:
            logger.warning("Missing binder image: %s", img_path)
    return paths, np.array(labels, dtype=np.int32), entries


def evaluate(clf, X_val, y_val, val_paths, feature_name, C):
    val_pred = clf.predict(X_val)
    val_proba = clf.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {feature_name} (C={C})")
    print(f"{'='*60}")
    print(f"  Val accuracy:   {val_acc:.1%} ({sum(val_pred == y_val)}/{len(y_val)})")
    print(f"  Val F1:         {val_f1:.3f}")
    print(classification_report(y_val, val_pred, target_names=["clean", "stamped"],
                                zero_division=0))

    wrong = []
    for i, (pred, true, prob) in enumerate(zip(val_pred, y_val, val_proba)):
        status = "OK" if pred == true else "WRONG"
        name = val_paths[i].name if hasattr(val_paths[i], 'name') else str(val_paths[i])
        if pred != true:
            wrong.append(name)
        print(f"    [{status}] prob={prob:.3f} pred={pred} true={true} {name}")

    if wrong:
        print(f"\n  Misclassified ({len(wrong)}): {', '.join(wrong)}")

    return val_acc, val_f1


class StampMLP(nn.Module):
    """Small MLP for stamp classification."""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, val_paths, feature_name,
              hidden_dim=128, dropout=0.3, lr=1e-3, epochs=200, weight_decay=1e-2,
              quiet=False):
    """Train a small MLP and return best val model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    model = StampMLP(input_dim, hidden_dim, dropout).to(device)

    # Class weights for imbalanced data
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_va = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_f1 = 0
    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr).squeeze()
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_va).squeeze()
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                if val_probs.ndim == 0:
                    val_probs = np.array([val_probs.item()])
                val_pred = (val_probs > 0.5).astype(int)
                acc = accuracy_score(y_val, val_pred)
                f1 = f1_score(y_val, val_pred, zero_division=0)
                train_probs = torch.sigmoid(model(X_tr).squeeze()).cpu().numpy()
                train_pred = (train_probs > 0.5).astype(int)
                train_acc = accuracy_score(y_train, train_pred)

            if f1 > best_f1 or (f1 == best_f1 and acc > best_acc):
                best_f1 = f1
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 50 == 0 and not quiet:
                logger.info(
                    "  Epoch %d: loss=%.4f train_acc=%.1f%% val_acc=%.1f%% val_f1=%.3f",
                    epoch + 1, loss.item(), train_acc * 100, acc * 100, f1,
                )

    # Evaluate best model
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_logits = model(X_va).squeeze()
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        if val_probs.ndim == 0:
            val_probs = np.array([val_probs.item()])

    val_pred = (val_probs > 0.5).astype(int)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, zero_division=0)

    if not quiet:
        print(f"\n{'='*60}")
        print(f"  MLP: {feature_name} (hidden={hidden_dim}, drop={dropout}, wd={weight_decay})")
        print(f"{'='*60}")
        print(f"  Val accuracy: {val_acc:.1%}")
        print(f"  Val F1:       {val_f1:.3f}")
        print(classification_report(y_val, val_pred, target_names=["clean", "stamped"],
                                    zero_division=0))

        wrong = []
        for i, (pred, true, prob) in enumerate(zip(val_pred, y_val, val_probs)):
            status = "OK" if pred == true else "WRONG"
            name = val_paths[i].name if hasattr(val_paths[i], 'name') else str(val_paths[i])
            if pred != true:
                wrong.append(name)
            print(f"    [{status}] prob={prob:.3f} pred={pred} true={true} {name}")

        if wrong:
            print(f"\n  Misclassified ({len(wrong)}): {', '.join(wrong)}")

    return model, val_acc, val_f1


def binder_loo_cv(base_cls, base_patches, y_base,
                  binder_cls, binder_patches, y_binder, binder_paths,
                  feature_type, C_values, sample_weight_binder=5.0):
    """Leave-one-out cross-validation on binder samples.

    For each binder sample:
      - Train on: all base data + all other binder samples (+ augmented binder)
      - Test on: the held-out binder sample
    Returns accuracy on binder samples.
    """
    N_binder = len(y_binder)
    predictions = np.zeros(N_binder, dtype=int)
    probabilities = np.zeros(N_binder, dtype=float)

    best_loo_acc = 0
    best_C = C_values[0]

    for C in C_values:
        correct = 0
        preds = np.zeros(N_binder, dtype=int)
        probs = np.zeros(N_binder, dtype=float)

        for i in range(N_binder):
            # Leave out sample i
            mask = np.ones(N_binder, dtype=bool)
            mask[i] = False

            # Combine base + remaining binder
            train_cls = np.vstack([base_cls, binder_cls[mask]])
            train_patches = np.vstack([base_patches, binder_patches[mask]])
            y_train = np.concatenate([y_base, y_binder[mask]])

            # Build features
            X_train = build_features(train_cls, train_patches, feature_type)
            X_test = build_features(
                binder_cls[i:i+1], binder_patches[i:i+1], feature_type
            )

            # Sample weights: upweight binder samples so they matter more
            weights = np.ones(len(y_train))
            weights[len(y_base):] = sample_weight_binder

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(
                C=C, max_iter=2000, solver="lbfgs",
                class_weight="balanced", random_state=42,
            )
            clf.fit(X_train_s, y_train, sample_weight=weights)

            pred = clf.predict(X_test_s)[0]
            prob = clf.predict_proba(X_test_s)[0, 1]
            preds[i] = pred
            probs[i] = prob
            if pred == y_binder[i]:
                correct += 1

        loo_acc = correct / N_binder
        if loo_acc > best_loo_acc:
            best_loo_acc = loo_acc
            best_C = C
            predictions = preds.copy()
            probabilities = probs.copy()

    return best_loo_acc, best_C, predictions, probabilities


def main():
    t0 = time.time()

    # ===== Load all datasets =====
    synth_paths, y_synth = load_dataset(SYNTHETIC_DIR / "labels.jsonl", SYNTHETIC_DIR)
    real_paths, y_real = load_dataset(REAL_DIR / "sources.jsonl", REAL_DIR)
    binder_paths, y_binder, binder_entries = load_binder_dataset()

    logger.info("Synthetic: %d (%d stamped)", len(synth_paths), sum(y_synth))
    logger.info("Real reference photos: %d (%d stamped)", len(real_paths), sum(y_real))
    logger.info("Binder scans: %d (%d stamped)", len(binder_paths), sum(y_binder))

    # Split real reference photos 70/30 for training/validation
    if len(real_paths) > 10:
        real_train_paths, ref_val_paths, y_real_train, y_ref_val = train_test_split(
            real_paths, y_real, test_size=0.3, random_state=42, stratify=y_real
        )
    else:
        real_train_paths, y_real_train = real_paths, y_real
        ref_val_paths, y_ref_val = [], np.array([], dtype=np.int32)

    # Base training set: synthetic + real reference train split
    base_train_paths = synth_paths + list(real_train_paths)
    y_base_train = np.concatenate([y_synth, y_real_train])

    logger.info("Base training: %d (%d synth + %d real ref, %d stamped)",
                len(base_train_paths), len(synth_paths), len(real_train_paths),
                sum(y_base_train))
    logger.info("Reference validation: %d (%d stamped)",
                len(ref_val_paths), sum(y_ref_val) if len(y_ref_val) > 0 else 0)

    # ===== Load DINOv2 =====
    dino_model, dino_device = load_model()

    # ===== Extract features for all datasets =====
    logger.info("Extracting base training features (%d images)...", len(base_train_paths))
    base_cls, base_patches = extract_features_batch(dino_model, dino_device, base_train_paths)

    if len(ref_val_paths) > 0:
        logger.info("Extracting reference validation features (%d images)...", len(ref_val_paths))
        ref_val_cls, ref_val_patches = extract_features_batch(dino_model, dino_device, ref_val_paths)

    logger.info("Extracting binder scan features (%d images)...", len(binder_paths))
    binder_cls, binder_patches = extract_features_batch(dino_model, dino_device, binder_paths)

    # ===== Generate augmented binder features =====
    # Each binder image gets N_AUG augmented copies to expand the small dataset
    N_AUG = 5
    rng = np.random.RandomState(42)
    logger.info("Generating %d augmented copies per binder image...", N_AUG)

    aug_pil_images = []
    aug_labels = []
    for path, label in zip(binder_paths, y_binder):
        img = Image.open(path).convert("RGB")
        for _ in range(N_AUG):
            aug_img = augment_binder_image(img, rng)
            aug_pil_images.append(aug_img)
            aug_labels.append(label)

    logger.info("Extracting augmented binder features (%d images)...", len(aug_pil_images))
    aug_cls, aug_patches = extract_features_from_pil_images(
        dino_model, dino_device, aug_pil_images
    )
    y_aug = np.array(aug_labels, dtype=np.int32)

    # Free GPU memory from DINOv2
    del dino_model
    torch.cuda.empty_cache()

    # ===== Combined training data: base + all binder + augmented binder =====
    full_train_cls = np.vstack([base_cls, binder_cls, aug_cls])
    full_train_patches = np.vstack([base_patches, binder_patches, aug_patches])
    y_full_train = np.concatenate([y_base_train, y_binder, y_aug])

    logger.info("Full training set: %d (%d base + %d binder + %d augmented, %d stamped)",
                len(y_full_train), len(y_base_train), len(y_binder), len(y_aug),
                sum(y_full_train))

    # Sample weights: upweight binder data (real + augmented) relative to
    # synthetic/reference photos, since binder domain is what we care about
    sample_weights_full = np.ones(len(y_full_train))
    binder_start = len(y_base_train)
    sample_weights_full[binder_start:] = 5.0  # Binder samples worth 5x

    # ===== Feature types to try =====
    # LOO-safe: compact feature types that won't OOM or take forever
    loo_feature_types = [
        "cls",              # 768
        "br_stats",         # 3072
        "all_stats",        # 3072
        "bottom_center_stats",  # 3072
        "center_stats",     # 3072
        "variance_ratio",   # 4608
    ]
    # Full sweep: includes high-dim types (only for final training, not LOO)
    all_feature_types = loo_feature_types + [
        "cls_br_stats",         # 3840
        "cls_all_stats",        # 3840
        "multi_region_stats",   # 14592
        "cls_multi_region",     # 15360
        "binder_robust",        # 10752
        "cls_binder_robust",    # 11520
    ]

    C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]

    best_model = None
    best_metrics = None
    best_feature_type = None
    best_scaler = None
    best_model_type = None

    # =====================================================================
    #  PART 1: Leave-one-out CV on binder samples
    # =====================================================================
    # Memory-efficient: pre-build features once, then index into them for LOO
    print("\n" + "#" * 60)
    print("  BINDER LEAVE-ONE-OUT CROSS-VALIDATION")
    print("#" * 60)

    N_binder = len(y_binder)
    loo_results = []

    for ftype in loo_feature_types:
        logger.info("LOO CV: %s ...", ftype)

        # Pre-build features for all data sources
        X_base = build_features(base_cls, base_patches, ftype)
        X_binder_feat = build_features(binder_cls, binder_patches, ftype)
        X_aug_feat = build_features(aug_cls, aug_patches, ftype)

        best_C_for_ftype = C_VALUES[0]
        best_loo_acc_for_ftype = 0
        best_loo_preds = np.zeros(N_binder, dtype=int)
        best_loo_probs = np.zeros(N_binder, dtype=float)

        for C in C_VALUES:
            fold_correct = 0
            fold_preds = np.zeros(N_binder, dtype=int)
            fold_probs = np.zeros(N_binder, dtype=float)

            for i in range(N_binder):
                mask = np.ones(N_binder, dtype=bool)
                mask[i] = False

                # Augmented mask: exclude augmented copies of held-out sample
                aug_mask = np.ones(len(y_aug), dtype=bool)
                aug_mask[i * N_AUG:(i + 1) * N_AUG] = False

                # Concatenate pre-built features (no recomputation)
                X_train = np.vstack([X_base, X_binder_feat[mask], X_aug_feat[aug_mask]])
                y_fold = np.concatenate([y_base_train, y_binder[mask], y_aug[aug_mask]])
                X_test = X_binder_feat[i:i+1]

                # Sample weights
                weights = np.ones(len(y_fold))
                weights[len(y_base_train):] = 5.0

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                clf = LogisticRegression(
                    C=C, max_iter=2000, solver="liblinear",
                    class_weight="balanced", random_state=42,
                )
                clf.fit(X_train_s, y_fold, sample_weight=weights)

                pred = clf.predict(X_test_s)[0]
                prob = clf.predict_proba(X_test_s)[0, 1]
                fold_preds[i] = pred
                fold_probs[i] = prob
                if pred == y_binder[i]:
                    fold_correct += 1

            fold_acc = fold_correct / N_binder
            if fold_acc > best_loo_acc_for_ftype:
                best_loo_acc_for_ftype = fold_acc
                best_C_for_ftype = C
                best_loo_preds = fold_preds.copy()
                best_loo_probs = fold_probs.copy()

        loo_results.append((ftype, best_loo_acc_for_ftype, best_C_for_ftype,
                            best_loo_preds, best_loo_probs))

        # Free feature matrices
        del X_base, X_binder_feat, X_aug_feat

        if best_loo_acc_for_ftype >= 0.5:
            print(f"\n  {ftype} (C={best_C_for_ftype}): "
                  f"LOO acc={best_loo_acc_for_ftype:.1%} "
                  f"({int(best_loo_acc_for_ftype * N_binder)}/{N_binder})")
            wrong = []
            for i in range(N_binder):
                if best_loo_preds[i] != y_binder[i]:
                    parent = binder_paths[i].parent.name
                    name = binder_paths[i].name
                    card = binder_entries[i].get("card_name", "")
                    wrong.append(f"{card} ({parent}/{name})")
                    print(f"    [WRONG] prob={best_loo_probs[i]:.3f} "
                          f"pred={best_loo_preds[i]} true={y_binder[i]} "
                          f"{card} ({parent}/{name})")
            if wrong:
                print(f"    Misclassified: {', '.join(wrong)}")

    # Sort LOO results by accuracy
    loo_results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  {'='*50}")
    print(f"  LOO RANKING (binder scans):")
    for ftype, acc, C, _, _ in loo_results:
        print(f"    {acc:.1%}  {ftype} (C={C})")

    # =====================================================================
    #  PART 2: Train on full data, evaluate on reference photo val set
    # =====================================================================
    print("\n" + "#" * 60)
    print("  FULL TRAINING: LOGISTIC REGRESSION")
    print("#" * 60)

    for ftype in all_feature_types:
        X_train = build_features(full_train_cls, full_train_patches, ftype)

        if len(ref_val_paths) > 0:
            X_ref_val = build_features(ref_val_cls, ref_val_patches, ftype)

        for C in C_VALUES:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)

            clf = LogisticRegression(
                C=C, max_iter=2000, solver="lbfgs",
                class_weight="balanced", random_state=42,
            )
            clf.fit(X_tr_s, y_full_train, sample_weight=sample_weights_full)

            # Evaluate on reference photos
            ref_acc = ref_f1 = 0.0
            if len(ref_val_paths) > 0:
                X_rv_s = scaler.transform(X_ref_val)
                ref_pred = clf.predict(X_rv_s)
                ref_acc = accuracy_score(y_ref_val, ref_pred)
                ref_f1 = f1_score(y_ref_val, ref_pred, zero_division=0)

            # Evaluate on binder (in-sample, but still useful for sanity check)
            X_binder_feat = build_features(binder_cls, binder_patches, ftype)
            X_binder_s = scaler.transform(X_binder_feat)
            binder_pred = clf.predict(X_binder_s)
            binder_acc = accuracy_score(y_binder, binder_pred)
            binder_f1 = f1_score(y_binder, binder_pred, zero_division=0)

            # Find the LOO accuracy for this ftype
            loo_acc = 0
            for r in loo_results:
                if r[0] == ftype:
                    loo_acc = r[1]
                    break

            # Combined score: weighted average favoring binder LOO
            combined_f1 = 0.6 * binder_f1 + 0.4 * ref_f1
            # But use LOO accuracy as the primary binder metric
            combined_score = 0.5 * loo_acc + 0.3 * binder_f1 + 0.2 * ref_f1

            name = f"{ftype}_scaled"
            if combined_score >= 0.5:
                print(f"\n  {name} C={C}: ref_acc={ref_acc:.1%} ref_f1={ref_f1:.3f} "
                      f"binder_acc={binder_acc:.1%} binder_f1={binder_f1:.3f} "
                      f"LOO={loo_acc:.1%} combined={combined_score:.3f}")

            if (best_metrics is None
                or combined_score > best_metrics.get("combined_score", 0)
                or (combined_score == best_metrics.get("combined_score", 0)
                    and loo_acc > best_metrics.get("binder_loo_acc", 0))):
                best_model = clf
                best_metrics = {
                    "feature_name": name,
                    "ref_val_acc": ref_acc, "ref_val_f1": ref_f1,
                    "binder_acc": binder_acc, "binder_f1": binder_f1,
                    "binder_loo_acc": loo_acc,
                    "combined_score": combined_score,
                    "val_acc": binder_acc,  # Backward compat
                    "val_f1": binder_f1,    # Backward compat
                    "C": C,
                }
                best_feature_type = ftype + "_scaled"
                best_scaler = scaler
                best_model_type = "lr"

    # =====================================================================
    #  PART 3: MLP experiments on promising feature types
    # =====================================================================
    print("\n" + "#" * 60)
    print("  MLP EXPERIMENTS")
    print("#" * 60)

    # Use top LOO feature types + classic ones
    mlp_ftypes = set()
    for ftype, acc, _, _, _ in loo_results[:5]:
        mlp_ftypes.add(ftype)
    mlp_ftypes.update(["br_stats", "all_stats", "binder_robust", "variance_ratio"])

    for ftype in mlp_ftypes:
        X_train = build_features(full_train_cls, full_train_patches, ftype)

        if len(ref_val_paths) > 0:
            X_ref_val = build_features(ref_val_cls, ref_val_patches, ftype)
        X_binder_feat = build_features(binder_cls, binder_patches, ftype)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_binder_s = scaler.transform(X_binder_feat)

        # Use binder samples as validation for MLP early stopping
        for hidden, dropout, wd in [(64, 0.5, 0.1), (128, 0.3, 0.01), (32, 0.5, 0.1)]:
            mlp, binder_acc_mlp, binder_f1_mlp = train_mlp(
                X_tr_s, y_full_train, X_binder_s, y_binder, binder_paths,
                feature_name=f"{ftype}_scaled",
                hidden_dim=hidden, dropout=dropout, weight_decay=wd,
                epochs=300,
            )

            # Find LOO accuracy
            loo_acc = 0
            for r in loo_results:
                if r[0] == ftype:
                    loo_acc = r[1]
                    break

            ref_f1 = 0.0
            if len(ref_val_paths) > 0:
                X_rv_s = scaler.transform(X_ref_val)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                mlp.eval()
                with torch.no_grad():
                    ref_logits = mlp(torch.tensor(X_rv_s, dtype=torch.float32).to(device)).squeeze()
                    ref_probs = torch.sigmoid(ref_logits).cpu().numpy()
                    if ref_probs.ndim == 0:
                        ref_probs = np.array([ref_probs.item()])
                    ref_pred = (ref_probs > 0.5).astype(int)
                    ref_f1 = f1_score(y_ref_val, ref_pred, zero_division=0)

            combined_score = 0.5 * loo_acc + 0.3 * binder_f1_mlp + 0.2 * ref_f1

            if (combined_score > best_metrics.get("combined_score", 0)
                or (combined_score == best_metrics.get("combined_score", 0)
                    and loo_acc > best_metrics.get("binder_loo_acc", 0))):
                best_model = mlp
                best_metrics = {
                    "feature_name": f"MLP_{ftype}_scaled",
                    "binder_acc": binder_acc_mlp, "binder_f1": binder_f1_mlp,
                    "binder_loo_acc": loo_acc,
                    "ref_val_f1": ref_f1,
                    "combined_score": combined_score,
                    "val_acc": binder_acc_mlp,
                    "val_f1": binder_f1_mlp,
                    "C": 0,
                    "hidden_dim": hidden, "dropout": dropout,
                }
                best_feature_type = ftype + "_scaled"
                best_scaler = scaler
                best_model_type = "mlp"

    # =====================================================================
    #  Final report
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"  BEST MODEL: {best_metrics['feature_name']} (type={best_model_type})")
    print(f"  Binder LOO accuracy: {best_metrics.get('binder_loo_acc', 0):.1%}")
    print(f"  Binder in-sample accuracy: {best_metrics.get('binder_acc', 0):.1%}")
    print(f"  Reference val F1: {best_metrics.get('ref_val_f1', 0):.3f}")
    print(f"  Combined score: {best_metrics.get('combined_score', 0):.3f}")
    print(f"  Feature type: {best_feature_type}")
    print(f"{'='*60}")

    # Detailed binder evaluation
    base_ftype = best_feature_type.replace("_scaled", "")
    X_binder_final = build_features(binder_cls, binder_patches, base_ftype)
    if best_scaler is not None:
        X_binder_final = best_scaler.transform(X_binder_final)

    print(f"\n  Binder scan detailed results:")
    if best_model_type == "lr":
        binder_pred = best_model.predict(X_binder_final)
        binder_proba = best_model.predict_proba(X_binder_final)[:, 1]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_model.eval()
        with torch.no_grad():
            logits = best_model(torch.tensor(X_binder_final, dtype=torch.float32).to(device)).squeeze()
            binder_proba = torch.sigmoid(logits).cpu().numpy()
            if binder_proba.ndim == 0:
                binder_proba = np.array([binder_proba.item()])
            binder_pred = (binder_proba > 0.5).astype(int)

    for i in range(len(y_binder)):
        status = "OK" if binder_pred[i] == y_binder[i] else "WRONG"
        parent = binder_paths[i].parent.name
        name = binder_paths[i].name
        card_name = binder_entries[i].get("card_name", "")
        print(f"    [{status}] prob={binder_proba[i]:.3f} pred={binder_pred[i]} "
              f"true={y_binder[i]} {card_name} ({parent}/{name})")

    # Reference val detailed results
    if len(ref_val_paths) > 0 and best_model_type == "lr":
        X_ref_final = build_features(ref_val_cls, ref_val_patches, base_ftype)
        if best_scaler is not None:
            X_ref_final = best_scaler.transform(X_ref_final)
        evaluate(best_model, X_ref_final, y_ref_val, ref_val_paths,
                 best_metrics["feature_name"], best_metrics.get("C", 0))

    # ===== Save =====
    save_obj = {
        "model": best_model,
        "feature_type": best_feature_type,
        "model_type": best_model_type,
        "metrics": best_metrics,
    }
    if best_scaler is not None:
        save_obj["scaler"] = best_scaler
    if best_model_type == "mlp":
        save_obj["model_state"] = best_model.state_dict()
        save_obj["model_config"] = {
            "input_dim": X_binder_final.shape[1],
            "hidden_dim": best_metrics.get("hidden_dim", 128),
            "dropout": best_metrics.get("dropout", 0.3),
        }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(save_obj, f)
    logger.info("Saved model to %s", OUTPUT_PATH)

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
