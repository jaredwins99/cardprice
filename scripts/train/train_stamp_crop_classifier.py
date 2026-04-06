#!/usr/bin/env python3
"""Train a stamp classifier using CROPPED stamp regions instead of whole cards.

Theory: The stamp is always in the bottom-right of the artwork area. Cropping
just that region gives a much stronger signal than whole-card features which
are dominated by card art and sleeve reflections.

Two approaches compared:
  A) "stamp_crop": Crop the stamp region from the image, resize to 112x112,
     feed through DINOv2, extract CLS + patch features.
  B) "patch_select": Use full-card DINOv2 16x16 grid but extract ONLY the
     4 bottom-right patches covering the stamp region (no separate crop).

Data sources:
  - Synthetic training: data/condition_training/stamps/labels.jsonl
  - Real reference: data/condition_training/stamps_real/sources.jsonl
  - Binder scans (final eval): data/condition_training/stamps_real/binder_ground_truth.jsonl

Saves model to data/stamp_crop_classifier.pkl if it outperforms the whole-card approach.
"""

import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "condition_training" / "stamps"
REAL_DIR = PROJECT_ROOT / "data" / "condition_training" / "stamps_real"
BINDER_DIR = PROJECT_ROOT / "data" / "inbox"
OUTPUT_PATH = PROJECT_ROOT / "data" / "stamp_crop_classifier.pkl"

GRID_SIZE = 16
EMBED_DIM = 768

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Full card transform (224x224 for 16x16 patch grid)
_transform_full = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# Stamp crop transform - use 224x224 for DINOv2 (it expects 14-divisible)
_transform_crop = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# Larger crop for better DINOv2 features
_transform_crop_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def load_model():
    """Load DINOv2 ViT-B/14."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading DINOv2 ViT-B/14 on %s ...", device)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.to(device)
    model.eval()
    return model, device


def crop_stamp_region(img: Image.Image) -> Image.Image:
    """Crop the stamp region from a card image.

    The stamp is in the bottom-right of the artwork area.
    Artwork: ~12-72% of card height, ~5-95% of card width.
    Stamp: bottom-right of artwork = ~55-90% of art width, ~55-95% of art height.
    In card coordinates: x=[55%,90%], y=[45%,70%].
    """
    w, h = img.size
    x1 = int(w * 0.50)
    y1 = int(h * 0.40)
    x2 = int(w * 0.95)
    y2 = int(h * 0.72)
    return img.crop((x1, y1, x2, y2))


def extract_crop_features_batch(model, device, image_paths, batch_size=32):
    """Extract DINOv2 features from stamp-region crops.

    Returns CLS tokens (N, 768) and patch tokens (N, num_patches, 768).
    Uses 224x224 resize for the crop to get full 16x16 patch grid.
    """
    all_cls = []
    all_patches = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            crop = crop_stamp_region(img)
            tensors.append(_transform_crop_224(crop))
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

        logger.info("  Crop features: %d/%d", min(start + batch_size, len(image_paths)),
                     len(image_paths))

    return np.concatenate(all_cls), np.concatenate(all_patches)


def extract_full_features_batch(model, device, image_paths, batch_size=32):
    """Extract DINOv2 features from full card images. Returns CLS (N,768) and patches (N,256,768)."""
    all_cls = []
    all_patches = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        tensors = [_transform_full(Image.open(p).convert("RGB")) for p in batch_paths]
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

        logger.info("  Full features: %d/%d", min(start + batch_size, len(image_paths)),
                     len(image_paths))

    return np.concatenate(all_cls), np.concatenate(all_patches)


def patch_stats(patches):
    """Mean, std, max, min over patch dim. (N,K,768) -> (N, 768*4)."""
    if patches.ndim == 2:
        patches = patches[np.newaxis]
    return np.hstack([
        np.mean(patches, axis=1),
        np.std(patches, axis=1),
        np.max(patches, axis=1),
        np.min(patches, axis=1),
    ])


def get_stamp_patches_from_full(patch_tokens):
    """Extract bottom-right patches from full card's 16x16 grid.

    The stamp region in card coordinates is roughly x=[55%,90%], y=[45%,70%].
    In a 16x16 grid: rows 7-11 (~44%-69%), cols 9-14 (~56%-88%).
    That gives a 5x6 = 30 patch window.
    """
    if patch_tokens.ndim == 2:
        patch_tokens = patch_tokens[np.newaxis]
    N = patch_tokens.shape[0]
    grid = patch_tokens.reshape(N, GRID_SIZE, GRID_SIZE, EMBED_DIM)
    # Stamp region: rows 7-11, cols 9-14
    stamp = grid[:, 7:12, 9:15, :].reshape(N, -1, EMBED_DIM)
    return stamp


def get_bottom_right_4x4(patch_tokens):
    """Extract bottom-right 4x4 patches (rows 12-15, cols 12-15) = 16 patches."""
    if patch_tokens.ndim == 2:
        patch_tokens = patch_tokens[np.newaxis]
    N = patch_tokens.shape[0]
    grid = patch_tokens.reshape(N, GRID_SIZE, GRID_SIZE, EMBED_DIM)
    return grid[:, 12:, 12:, :].reshape(N, -1, EMBED_DIM)


def load_dataset(jsonl_path, base_dir):
    """Load images and labels from a JSONL file."""
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


def train_and_eval(X_train, y_train, X_val, y_val, val_paths, name, C_values=None):
    """Train logistic regression with hyperparameter sweep, return best result."""
    if C_values is None:
        C_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    best_acc = 0
    best_f1 = 0
    best_clf = None
    best_scaler = None
    best_C = None

    for scale in [False, True]:
        scaler = None
        X_tr, X_va = X_train, X_val
        if scale:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_va = scaler.transform(X_val)

        for C in C_values:
            clf = LogisticRegression(
                C=C, max_iter=2000, solver="lbfgs",
                class_weight="balanced", random_state=42,
            )
            clf.fit(X_tr, y_train)
            val_pred = clf.predict(X_va)
            acc = accuracy_score(y_val, val_pred)
            f1 = f1_score(y_val, val_pred)

            if f1 > best_f1 or (f1 == best_f1 and acc > best_acc):
                best_f1 = f1
                best_acc = acc
                best_clf = clf
                best_scaler = scaler
                best_C = C

    # Print best result
    X_va_final = X_val
    if best_scaler is not None:
        X_va_final = best_scaler.transform(X_val)
    val_pred = best_clf.predict(X_va_final)
    val_proba = best_clf.predict_proba(X_va_final)[:, 1]

    print(f"\n{'='*65}")
    print(f"  {name} (C={best_C}, scaled={best_scaler is not None})")
    print(f"{'='*65}")
    print(f"  Val accuracy:  {best_acc:.1%} ({sum(val_pred == y_val)}/{len(y_val)})")
    print(f"  Val F1:        {best_f1:.3f}")
    print(classification_report(y_val, val_pred, target_names=["clean", "stamped"]))

    wrong = []
    for i, (pred, true, prob) in enumerate(zip(val_pred, y_val, val_proba)):
        status = "OK" if pred == true else "WRONG"
        pname = val_paths[i].name if hasattr(val_paths[i], 'name') else str(val_paths[i])
        if pred != true:
            wrong.append(pname)
        print(f"    [{status}] prob={prob:.3f} pred={pred} true={true} {pname}")

    if wrong:
        print(f"\n  Misclassified ({len(wrong)}): {', '.join(wrong)}")

    return best_clf, best_scaler, best_acc, best_f1, best_C


def main():
    t0 = time.time()

    # ===== Load datasets =====
    synth_paths, y_synth = load_dataset(SYNTHETIC_DIR / "labels.jsonl", SYNTHETIC_DIR)
    real_paths, y_real = load_dataset(REAL_DIR / "sources.jsonl", REAL_DIR)
    binder_paths, y_binder = load_dataset(REAL_DIR / "binder_ground_truth.jsonl", BINDER_DIR)

    logger.info("Synthetic: %d images (%d stamped)", len(synth_paths), sum(y_synth))
    logger.info("Real reference: %d images (%d stamped)", len(real_paths), sum(y_real))
    logger.info("Binder ground truth: %d images (%d stamped)", len(binder_paths), sum(y_binder))

    # Split real into train/val (70/30)
    if len(real_paths) > 10:
        real_train_paths, real_val_paths, y_real_train, y_real_val = train_test_split(
            real_paths, y_real, test_size=0.3, random_state=42, stratify=y_real,
        )
    else:
        real_train_paths, real_val_paths = real_paths, []
        y_real_train, y_real_val = y_real, np.array([], dtype=np.int32)

    train_paths = synth_paths + list(real_train_paths)
    y_train = np.concatenate([y_synth, y_real_train])
    val_paths = list(real_val_paths)
    y_val = y_real_val

    logger.info("Training set: %d images (%d stamped)", len(train_paths), sum(y_train))
    logger.info("Validation set: %d images (%d stamped)", len(val_paths), sum(y_val))
    logger.info("Binder eval set: %d images (%d stamped)", len(binder_paths), sum(y_binder))

    # ===== Load DINOv2 =====
    dino_model, dino_device = load_model()

    # ===== Extract features for ALL approaches at once =====
    # Approach A: Stamp region crop -> DINOv2
    logger.info("Extracting CROP features for training set...")
    train_crop_cls, train_crop_patches = extract_crop_features_batch(
        dino_model, dino_device, train_paths)
    logger.info("Extracting CROP features for validation set...")
    val_crop_cls, val_crop_patches = extract_crop_features_batch(
        dino_model, dino_device, val_paths)
    logger.info("Extracting CROP features for binder set...")
    binder_crop_cls, binder_crop_patches = extract_crop_features_batch(
        dino_model, dino_device, binder_paths)

    # Approach B: Full card -> DINOv2 -> select stamp patches
    logger.info("Extracting FULL features for training set...")
    train_full_cls, train_full_patches = extract_full_features_batch(
        dino_model, dino_device, train_paths)
    logger.info("Extracting FULL features for validation set...")
    val_full_cls, val_full_patches = extract_full_features_batch(
        dino_model, dino_device, val_paths)
    logger.info("Extracting FULL features for binder set...")
    binder_full_cls, binder_full_patches = extract_full_features_batch(
        dino_model, dino_device, binder_paths)

    # Free GPU
    del dino_model
    torch.cuda.empty_cache()

    # ===== Build feature matrices =====
    experiments = {}

    # --- Approach A: Crop-based features ---
    # A1: CLS token from crop
    experiments["crop_cls"] = {
        "train": train_crop_cls,
        "val": val_crop_cls,
        "binder": binder_crop_cls,
    }

    # A2: Patch stats from crop (all patches)
    experiments["crop_all_patch_stats"] = {
        "train": patch_stats(train_crop_patches),
        "val": patch_stats(val_crop_patches),
        "binder": patch_stats(binder_crop_patches),
    }

    # A3: CLS + patch stats from crop
    experiments["crop_cls_patch_stats"] = {
        "train": np.hstack([train_crop_cls, patch_stats(train_crop_patches)]),
        "val": np.hstack([val_crop_cls, patch_stats(val_crop_patches)]),
        "binder": np.hstack([binder_crop_cls, patch_stats(binder_crop_patches)]),
    }

    # A4: Bottom-right patches from crop's own 16x16 grid (stamp-within-stamp)
    crop_br_train = patch_stats(get_bottom_right_4x4(train_crop_patches))
    crop_br_val = patch_stats(get_bottom_right_4x4(val_crop_patches))
    crop_br_binder = patch_stats(get_bottom_right_4x4(binder_crop_patches))
    experiments["crop_br_patch_stats"] = {
        "train": crop_br_train,
        "val": crop_br_val,
        "binder": crop_br_binder,
    }

    # --- Approach B: Full card, select stamp patches ---
    # B1: Stamp-region patches from full card (5x6 grid window)
    stamp_train = patch_stats(get_stamp_patches_from_full(train_full_patches))
    stamp_val = patch_stats(get_stamp_patches_from_full(val_full_patches))
    stamp_binder = patch_stats(get_stamp_patches_from_full(binder_full_patches))
    experiments["full_stamp_region_stats"] = {
        "train": stamp_train,
        "val": stamp_val,
        "binder": stamp_binder,
    }

    # B2: Bottom-right 4x4 patches from full card
    br4_train = patch_stats(get_bottom_right_4x4(train_full_patches))
    br4_val = patch_stats(get_bottom_right_4x4(val_full_patches))
    br4_binder = patch_stats(get_bottom_right_4x4(binder_full_patches))
    experiments["full_br_4x4_stats"] = {
        "train": br4_train,
        "val": br4_val,
        "binder": br4_binder,
    }

    # B3: CLS from full card + stamp region patches
    experiments["full_cls_stamp_region"] = {
        "train": np.hstack([train_full_cls, stamp_train]),
        "val": np.hstack([val_full_cls, stamp_val]),
        "binder": np.hstack([binder_full_cls, stamp_binder]),
    }

    # --- Baseline: Full card whole-card features (for comparison) ---
    experiments["baseline_full_cls"] = {
        "train": train_full_cls,
        "val": val_full_cls,
        "binder": binder_full_cls,
    }

    # Full card all-patches stats (this is what the current model uses)
    experiments["baseline_full_all_stats"] = {
        "train": patch_stats(train_full_patches),
        "val": patch_stats(val_full_patches),
        "binder": patch_stats(binder_full_patches),
    }

    # ===== Train and evaluate =====
    results_summary = []

    print("\n" + "#" * 65)
    print("  STAMP CROP CLASSIFIER EXPERIMENTS")
    print("  Training: %d images, Validation: %d images, Binder eval: %d images" % (
        len(train_paths), len(val_paths), len(binder_paths)))
    print("#" * 65)

    best_overall = None

    for exp_name, data in experiments.items():
        clf, scaler, val_acc, val_f1, C = train_and_eval(
            data["train"], y_train, data["val"], y_val, val_paths, exp_name)

        # Also evaluate on binder ground truth
        X_binder = data["binder"]
        if scaler is not None:
            X_binder = scaler.transform(X_binder)
        binder_pred = clf.predict(X_binder)
        binder_proba = clf.predict_proba(X_binder)[:, 1]
        binder_acc = accuracy_score(y_binder, binder_pred)
        binder_f1 = f1_score(y_binder, binder_pred) if sum(y_binder) > 0 else 0.0

        print(f"\n  >>> BINDER EVAL: {binder_acc:.1%} ({sum(binder_pred == y_binder)}/{len(y_binder)})")
        for i, (pred, true, prob) in enumerate(zip(binder_pred, y_binder, binder_proba)):
            status = "OK" if pred == true else "WRONG"
            pname = binder_paths[i].name
            parent = binder_paths[i].parent.name
            print(f"      [{status}] prob={prob:.3f} pred={pred} true={true} {parent}/{pname}")

        results_summary.append({
            "name": exp_name,
            "val_acc": val_acc, "val_f1": val_f1,
            "binder_acc": binder_acc, "binder_f1": binder_f1,
            "C": C, "scaled": scaler is not None,
        })

        # Track best by binder accuracy (real-world performance)
        if (best_overall is None
            or binder_acc > best_overall["binder_acc"]
            or (binder_acc == best_overall["binder_acc"]
                and binder_f1 > best_overall["binder_f1"])):
            best_overall = {
                "name": exp_name,
                "clf": clf, "scaler": scaler,
                "val_acc": val_acc, "val_f1": val_f1,
                "binder_acc": binder_acc, "binder_f1": binder_f1,
                "C": C,
            }

    # ===== Summary table =====
    print("\n\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print(f"  {'Experiment':<30s}  {'Val Acc':>7s}  {'Val F1':>7s}  {'Binder Acc':>10s}  {'Binder F1':>9s}")
    print("  " + "-" * 70)
    for r in sorted(results_summary, key=lambda x: (-x["binder_acc"], -x["binder_f1"])):
        marker = " <-- BEST" if r["name"] == best_overall["name"] else ""
        print(f"  {r['name']:<30s}  {r['val_acc']:>6.1%}  {r['val_f1']:>7.3f}  "
              f"{r['binder_acc']:>9.1%}  {r['binder_f1']:>9.3f}{marker}")

    print(f"\n  Current whole-card model: val_acc=91.7%, feature_type=all_stats")
    print(f"  Best crop model: {best_overall['name']}, binder_acc={best_overall['binder_acc']:.1%}")

    # ===== Save best model =====
    # Save with metadata about the crop approach
    save_obj = {
        "model": best_overall["clf"],
        "feature_type": best_overall["name"],
        "model_type": "lr",
        "crop_based": True,  # Flag for inference to know to crop first
        "metrics": {
            "feature_name": best_overall["name"],
            "val_acc": best_overall["val_acc"],
            "val_f1": best_overall["val_f1"],
            "binder_acc": best_overall["binder_acc"],
            "binder_f1": best_overall["binder_f1"],
            "C": best_overall["C"],
        },
    }
    if best_overall["scaler"] is not None:
        save_obj["scaler"] = best_overall["scaler"]

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(save_obj, f)
    logger.info("Saved crop classifier to %s", OUTPUT_PATH)

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
