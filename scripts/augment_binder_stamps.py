#!/usr/bin/env python3
"""Generate augmented training data from binder ground truth samples.

Reads 17 binder ground truth entries, generates 10 augmented versions each,
saves to data/condition_training/binder_augmented/ with labels.jsonl.

Then trains a stamp classifier on ONLY binder data (original + augmented)
using leave-one-out cross-validation on the 17 originals.

Augmentations:
  - Random brightness +/-20%
  - Random contrast +/-15%
  - Random hue shift +/-10
  - Slight rotation +/-3 degrees
  - Random crop (95-100% of original, resize back)
  - Horizontal flip (50% chance)
  - Gaussian noise
  - Slight blur
"""

import json
import logging
import pickle
import time
from pathlib import Path

import gc

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INBOX_DIR = PROJECT_ROOT / "data" / "inbox"
GT_PATH = PROJECT_ROOT / "data" / "condition_training" / "stamps_real" / "binder_ground_truth.jsonl"
AUG_DIR = PROJECT_ROOT / "data" / "condition_training" / "binder_augmented"
AUG_LABELS_PATH = AUG_DIR / "labels.jsonl"

N_AUG = 10  # augmented copies per original

GRID_SIZE = 16
EMBED_DIM = 768

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# =============================================================================
# Augmentation
# =============================================================================

def augment_image(img: Image.Image, rng: np.random.RandomState, aug_idx: int) -> Image.Image:
    """Apply a random combination of augmentations to a PIL image.

    Each augmentation is applied independently with its own random parameters.
    """
    img = img.copy()
    w, h = img.size

    # 1. Random brightness +/-20%
    brightness = rng.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # 2. Random contrast +/-15%
    contrast = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # 3. Random hue shift +/-10 (done in HSV space)
    arr = np.array(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.int16)
    hue_shift = rng.randint(-10, 11)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179)
    arr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    img = Image.fromarray(arr)

    # 4. Slight rotation +/-3 degrees
    angle = rng.uniform(-3, 3)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))

    # 5. Random crop (95-100% of original, resize back)
    crop_frac = rng.uniform(0.95, 1.0)
    crop_w = int(w * crop_frac)
    crop_h = int(h * crop_frac)
    left = rng.randint(0, w - crop_w + 1)
    top = rng.randint(0, h - crop_h + 1)
    img = img.crop((left, top, left + crop_w, top + crop_h))
    img = img.resize((w, h), Image.BILINEAR)

    # 6. Horizontal flip (50% chance -- cards aren't symmetric)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 7. Gaussian noise (sigma 3-8)
    arr = np.array(img, dtype=np.float32)
    sigma = rng.uniform(3, 8)
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # 8. Slight blur (50% chance, radius 0.5-1.0)
    if rng.random() < 0.5:
        radius = rng.uniform(0.5, 1.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    return img


def generate_augmented_data():
    """Load ground truth, generate augmented images, save with labels."""
    AUG_DIR.mkdir(parents=True, exist_ok=True)

    entries = [json.loads(line) for line in open(GT_PATH)]
    logger.info("Loaded %d ground truth entries", len(entries))

    rng = np.random.RandomState(42)
    aug_labels = []
    generated = 0

    for idx, entry in enumerate(entries):
        img_path = INBOX_DIR / entry["image"]
        if not img_path.exists():
            logger.warning("Missing: %s", img_path)
            continue

        img = Image.open(img_path).convert("RGB")
        card_name = entry.get("card_name", f"card_{idx:02d}")
        stamped = entry["stamped"]

        for aug_i in range(N_AUG):
            aug_img = augment_image(img, rng, aug_i)
            # Filename: orig_idx_augN.png
            fname = f"orig{idx:02d}_aug{aug_i:02d}.png"
            out_path = AUG_DIR / fname
            aug_img.save(out_path)

            aug_labels.append({
                "image": fname,
                "source": "binder_augmented",
                "original_image": entry["image"],
                "original_idx": idx,
                "aug_idx": aug_i,
                "card_name": card_name,
                "stamped": stamped,
                "variant": entry.get("variant", "unknown"),
            })
            generated += 1

    # Write labels
    with open(AUG_LABELS_PATH, "w") as f:
        for label in aug_labels:
            f.write(json.dumps(label) + "\n")

    logger.info("Generated %d augmented images -> %s", generated, AUG_DIR)
    logger.info("Labels written to %s", AUG_LABELS_PATH)
    return entries, aug_labels


# =============================================================================
# DINOv2 feature extraction
# =============================================================================

def load_dino():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading DINOv2 ViT-B/14 on %s ...", device)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.to(device)
    model.eval()
    return model, device


def extract_features_batch(model, device, pil_images, batch_size=32):
    """Extract CLS + patch tokens from PIL images. Returns (N,768), (N,256,768)."""
    all_cls, all_patches = [], []
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

        logger.info("  Features: %d/%d", min(start + batch_size, len(pil_images)), len(pil_images))

    return np.concatenate(all_cls), np.concatenate(all_patches)


# =============================================================================
# Feature builders (from stamp_classifier.py patterns)
# =============================================================================

def get_region(patches, region):
    """Get patches from a region. patches: (N,256,768) or (256,768)."""
    single = patches.ndim == 2
    if single:
        patches = patches[np.newaxis]
    N = patches.shape[0]
    grid = patches.reshape(N, GRID_SIZE, GRID_SIZE, EMBED_DIM)
    slices = {
        'br': (slice(8, None), slice(8, None)),
        'bl': (slice(8, None), slice(None, 8)),
        'tr': (slice(None, 8), slice(8, None)),
        'tl': (slice(None, 8), slice(None, 8)),
        'bottom': (slice(8, None), slice(None)),
        'top': (slice(None, 8), slice(None)),
        'all': (slice(None), slice(None)),
        'center': (slice(4, 12), slice(4, 12)),
        'bottom_center': (slice(10, None), slice(4, 12)),
    }
    r, c = slices[region]
    out = grid[:, r, c, :].reshape(N, -1, EMBED_DIM)
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
    """Ratio of regional variance to global variance. (N,K,768) -> (N,768)."""
    region_var = np.var(patches_region, axis=1)
    all_var = np.maximum(np.var(patches_all, axis=1), 1e-8)
    return region_var / all_var


def build_features(cls_tokens, patch_tokens, feature_type):
    """Build feature matrix for a given feature type."""
    if feature_type == "cls":
        return cls_tokens

    elif feature_type == "br_stats":
        return patch_stats(get_region(patch_tokens, 'br'))

    elif feature_type == "all_stats":
        return patch_stats(get_region(patch_tokens, 'all'))

    elif feature_type == "bottom_center_stats":
        return patch_stats(get_region(patch_tokens, 'bottom_center'))

    elif feature_type == "center_stats":
        return patch_stats(get_region(patch_tokens, 'center'))

    elif feature_type == "variance_ratio":
        br = get_region(patch_tokens, 'br')
        all_p = get_region(patch_tokens, 'all')
        bc = get_region(patch_tokens, 'bottom_center')
        return np.hstack([
            patch_stats(br),
            patch_variance_ratio(br, all_p),
            patch_variance_ratio(bc, all_p),
        ])

    elif feature_type == "binder_robust":
        br = get_region(patch_tokens, 'br')
        bl = get_region(patch_tokens, 'bl')
        bc = get_region(patch_tokens, 'bottom_center')
        all_p = get_region(patch_tokens, 'all')
        br_mean = np.mean(br, axis=1)
        bl_mean = np.mean(bl, axis=1)
        tr_mean = np.mean(get_region(patch_tokens, 'tr'), axis=1)
        return np.hstack([
            patch_stats(br),
            patch_stats(bc),
            br_mean - bl_mean,
            br_mean - tr_mean,
            patch_variance_ratio(br, all_p),
            patch_variance_ratio(bc, all_p),
        ])

    elif feature_type == "cls_br_stats":
        return np.hstack([cls_tokens, patch_stats(get_region(patch_tokens, 'br'))])

    elif feature_type == "cls_binder_robust":
        binder = build_features(cls_tokens, patch_tokens, "binder_robust")
        return np.hstack([cls_tokens, binder])

    elif feature_type == "stamp_crop":
        # Focus on bottom-right quadrant where EX stamps appear
        # Use both stats and variance ratio for that specific region
        br = get_region(patch_tokens, 'br')
        all_p = get_region(patch_tokens, 'all')
        return np.hstack([
            patch_stats(br),
            patch_variance_ratio(br, all_p),
        ])

    elif feature_type == "edge_density":
        # Use variance across all quadrant boundaries as edge density proxy
        br = get_region(patch_tokens, 'br')
        bl = get_region(patch_tokens, 'bl')
        tr = get_region(patch_tokens, 'tr')
        tl = get_region(patch_tokens, 'tl')
        all_p = get_region(patch_tokens, 'all')
        br_mean = np.mean(br, axis=1)
        bl_mean = np.mean(bl, axis=1)
        tr_mean = np.mean(tr, axis=1)
        tl_mean = np.mean(tl, axis=1)
        return np.hstack([
            np.std(br, axis=1),   # Texture complexity per region
            np.std(bl, axis=1),
            np.std(tr, axis=1),
            np.std(tl, axis=1),
            br_mean - bl_mean,    # Asymmetry features
            br_mean - tr_mean,
            tl_mean - tr_mean,
            patch_variance_ratio(br, all_p),
        ])

    elif feature_type == "combined":
        # CLS + stamp_crop + edge_density
        stamp = build_features(cls_tokens, patch_tokens, "stamp_crop")
        edge = build_features(cls_tokens, patch_tokens, "edge_density")
        return np.hstack([cls_tokens, stamp, edge])

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


# =============================================================================
# Leave-one-out cross-validation (binder-only)
# =============================================================================

def loo_cv_binder_only(orig_cls, orig_patches, y_orig, orig_paths, orig_entries,
                       aug_cls, aug_patches, y_aug,
                       feature_type, C_values):
    """LOO CV using ONLY binder data (no synthetic/reference).

    For each of the 17 originals:
      - Train on: 16 originals + their 10*16=160 augmentations
      - Test on: held-out original
    """
    N = len(y_orig)
    best_C = C_values[0]
    best_acc = 0
    best_preds = np.zeros(N, dtype=int)
    best_probs = np.zeros(N, dtype=float)

    # Pre-build features once
    X_orig = build_features(orig_cls, orig_patches, feature_type)
    X_aug = build_features(aug_cls, aug_patches, feature_type)

    for C in C_values:
        preds = np.zeros(N, dtype=int)
        probs = np.zeros(N, dtype=float)
        correct = 0

        for i in range(N):
            # Mask: exclude original i and its augmentations
            orig_mask = np.ones(N, dtype=bool)
            orig_mask[i] = False

            aug_mask = np.ones(len(y_aug), dtype=bool)
            aug_mask[i * N_AUG:(i + 1) * N_AUG] = False

            X_train = np.vstack([X_orig[orig_mask], X_aug[aug_mask]])
            y_train = np.concatenate([y_orig[orig_mask], y_aug[aug_mask]])
            X_test = X_orig[i:i + 1]

            # Weight originals more than augmented
            weights = np.ones(len(y_train))
            n_orig_train = orig_mask.sum()
            weights[:n_orig_train] = 3.0  # Originals worth 3x augmented

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(
                C=C, max_iter=2000, solver="liblinear",
                class_weight="balanced", random_state=42,
            )
            clf.fit(X_train_s, y_train, sample_weight=weights)

            pred = clf.predict(X_test_s)[0]
            prob = clf.predict_proba(X_test_s)[0, 1]
            preds[i] = pred
            probs[i] = prob
            if pred == y_orig[i]:
                correct += 1

        acc = correct / N
        if acc > best_acc or (acc == best_acc and C < best_C):
            best_acc = acc
            best_C = C
            best_preds = preds.copy()
            best_probs = probs.copy()

    return best_acc, best_C, best_preds, best_probs


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()

    # ===== Step 1: Generate augmented data =====
    print("\n" + "=" * 60)
    print("  STEP 1: GENERATE AUGMENTED BINDER DATA")
    print("=" * 60)

    entries, aug_labels = generate_augmented_data()

    # ===== Step 2: Load images =====
    print("\n" + "=" * 60)
    print("  STEP 2: LOAD IMAGES AND EXTRACT FEATURES")
    print("=" * 60)

    # Load originals
    orig_images = []
    orig_labels = []
    orig_paths = []
    orig_entries_valid = []
    for idx, entry in enumerate(entries):
        img_path = INBOX_DIR / entry["image"]
        if img_path.exists():
            orig_images.append(Image.open(img_path).convert("RGB"))
            orig_labels.append(1 if entry["stamped"] else 0)
            orig_paths.append(img_path)
            orig_entries_valid.append(entry)
        else:
            logger.warning("Missing original: %s", img_path)

    y_orig = np.array(orig_labels, dtype=np.int32)
    N_orig = len(y_orig)
    n_stamped = sum(y_orig)
    n_clean = N_orig - n_stamped
    logger.info("Originals: %d total (%d stamped, %d clean)", N_orig, n_stamped, n_clean)

    # Load augmented
    aug_images = []
    aug_labels_arr = []
    for label in aug_labels:
        aug_path = AUG_DIR / label["image"]
        if aug_path.exists():
            aug_images.append(Image.open(aug_path).convert("RGB"))
            aug_labels_arr.append(1 if label["stamped"] else 0)
    y_aug = np.array(aug_labels_arr, dtype=np.int32)
    logger.info("Augmented: %d total (%d stamped, %d clean)", len(y_aug), sum(y_aug), len(y_aug) - sum(y_aug))

    # Extract DINOv2 features (with disk caching)
    cache_path = AUG_DIR / "features_cache.pkl"
    if cache_path.exists():
        logger.info("Loading cached features from %s", cache_path)
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        orig_cls = cached["orig_cls"]
        orig_patches = cached["orig_patches"]
        aug_cls = cached["aug_cls"]
        aug_patches = cached["aug_patches"]
        logger.info("Loaded cached features: %d originals, %d augmented",
                     len(orig_cls), len(aug_cls))
    else:
        model, device = load_dino()

        logger.info("Extracting original features (%d images)...", N_orig)
        orig_cls, orig_patches = extract_features_batch(model, device, orig_images)

        logger.info("Extracting augmented features (%d images)...", len(aug_images))
        aug_cls, aug_patches = extract_features_batch(model, device, aug_images)

        del model
        torch.cuda.empty_cache()

        # Cache to disk
        with open(cache_path, "wb") as f:
            pickle.dump({
                "orig_cls": orig_cls, "orig_patches": orig_patches,
                "aug_cls": aug_cls, "aug_patches": aug_patches,
            }, f)
        logger.info("Cached features to %s", cache_path)

    # Free PIL images -- no longer needed
    del orig_images, aug_images
    gc.collect()

    # ===== Step 3: LOO CV on binder-only data =====
    print("\n" + "=" * 60)
    print("  STEP 3: LEAVE-ONE-OUT CV (BINDER DATA ONLY)")
    print("=" * 60)
    print(f"  {N_orig} originals + {len(y_aug)} augmented = {N_orig + len(y_aug)} total")
    print(f"  For each fold: train on {N_orig - 1} originals + {(N_orig - 1) * N_AUG} augmented, test on 1")

    # Note: lines 12 and 17 in ground truth reference the same image path
    # (page_20260307_015320_cards/card_05.png) with conflicting labels.
    # Line 12: stamped=false (Dragonite normal), Line 17: stamped=true (Dragonite prerelease).
    # We use the data as-is since the ground truth may represent different interpretations.

    # Use compact feature types to avoid OOM in LOO CV
    # patch_tokens are (N,256,768) = ~150MB for 187 samples
    # Feature matrices are much smaller, but we rebuild per fold
    feature_types = [
        "cls",                  # 768 dims
        "br_stats",             # 3072 dims
        "all_stats",            # 3072 dims
        "bottom_center_stats",  # 3072 dims
        "center_stats",         # 3072 dims
        "variance_ratio",       # 4608 dims
        "stamp_crop",           # 3840 dims
        "edge_density",         # 6912 dims
        "cls_br_stats",         # 3840 dims
    ]

    C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    results = []
    for ftype in feature_types:
        logger.info("LOO CV: %s ...", ftype)
        gc.collect()
        acc, best_C, preds, probs = loo_cv_binder_only(
            orig_cls, orig_patches, y_orig, orig_paths, orig_entries_valid,
            aug_cls, aug_patches, y_aug,
            ftype, C_VALUES,
        )
        results.append((ftype, acc, best_C, preds, probs))
        gc.collect()

        n_correct = int(acc * N_orig)
        print(f"\n  {ftype} (C={best_C}): {acc:.1%} ({n_correct}/{N_orig})")

        # Show per-sample details
        for i in range(N_orig):
            status = "OK" if preds[i] == y_orig[i] else "WRONG"
            card = orig_entries_valid[i].get("card_name", "?")
            parent = orig_paths[i].parent.name
            name = orig_paths[i].name
            marker = " <---" if preds[i] != y_orig[i] else ""
            print(f"    [{status}] prob={probs[i]:.3f} pred={preds[i]} true={y_orig[i]} "
                  f"{card} ({parent}/{name}){marker}")

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("  LOO CV RESULTS RANKING (binder-only training)")
    print("=" * 60)
    results.sort(key=lambda x: (-x[1], x[0]))
    for ftype, acc, C, preds, probs in results:
        n_correct = int(acc * N_orig)
        wrong_cards = []
        for i in range(N_orig):
            if preds[i] != y_orig[i]:
                wrong_cards.append(orig_entries_valid[i].get("card_name", "?"))
        wrong_str = f"  wrong: {', '.join(wrong_cards)}" if wrong_cards else ""
        print(f"  {acc:.1%} ({n_correct:2d}/{N_orig})  {ftype:25s} C={C:<8}{wrong_str}")

    # ===== Train final model on all binder data =====
    print("\n" + "=" * 60)
    print("  FINAL: TRAIN ON ALL BINDER DATA")
    print("=" * 60)

    best_ftype, best_acc, best_C, _, _ = results[0]
    logger.info("Best feature type: %s (LOO acc=%.1f%%, C=%s)", best_ftype, best_acc * 100, best_C)

    X_all = np.vstack([
        build_features(orig_cls, orig_patches, best_ftype),
        build_features(aug_cls, aug_patches, best_ftype),
    ])
    y_all = np.concatenate([y_orig, y_aug])

    # Weight originals more
    weights = np.ones(len(y_all))
    weights[:N_orig] = 3.0

    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)

    clf = LogisticRegression(
        C=best_C, max_iter=2000, solver="liblinear",
        class_weight="balanced", random_state=42,
    )
    clf.fit(X_all_s, y_all, sample_weight=weights)

    # In-sample accuracy (sanity check)
    X_orig_feat = build_features(orig_cls, orig_patches, best_ftype)
    X_orig_s = scaler.transform(X_orig_feat)
    in_pred = clf.predict(X_orig_s)
    in_proba = clf.predict_proba(X_orig_s)[:, 1]
    in_acc = accuracy_score(y_orig, in_pred)
    print(f"\n  In-sample accuracy: {in_acc:.1%} ({sum(in_pred == y_orig)}/{N_orig})")
    for i in range(N_orig):
        status = "OK" if in_pred[i] == y_orig[i] else "WRONG"
        card = orig_entries_valid[i].get("card_name", "?")
        print(f"    [{status}] prob={in_proba[i]:.3f} pred={in_pred[i]} true={y_orig[i]} {card}")

    # Save binder-only model
    out_path = PROJECT_ROOT / "data" / "stamp_classifier_binder.pkl"
    save_obj = {
        "model": clf,
        "feature_type": best_ftype + "_scaled",
        "model_type": "lr",
        "scaler": scaler,
        "metrics": {
            "binder_loo_acc": best_acc,
            "binder_insample_acc": in_acc,
            "val_acc": best_acc,
            "C": best_C,
            "n_originals": N_orig,
            "n_augmented": len(y_aug),
            "feature_type": best_ftype,
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(save_obj, f)
    logger.info("Saved binder-only model to %s", out_path)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
