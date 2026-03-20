#!/usr/bin/env python3
"""Train a 3-class variant classifier: NORMAL vs HOLOFOIL vs REVERSE_HOLO_STAMPED.

Uses hand-crafted texture features from card images + DINOv2 CLS tokens.
Evaluates with leave-one-out cross-validation.

Ground truth from binder_ground_truth.jsonl + user-specified labels.
"""

import json
import logging
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INBOX_DIR = PROJECT_ROOT / "data" / "inbox"
OUTPUT_PATH = PROJECT_ROOT / "data" / "variant_3class_classifier.pkl"

CLASS_NAMES = ["NORMAL", "HOLOFOIL", "REVERSE_HOLO_STAMPED"]
CLASS_MAP = {"NORMAL": 0, "HOLOFOIL": 1, "REVERSE_HOLO_STAMPED": 2}

# DINOv2 setup
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
#  Ground truth: hardcoded from binder scans
# =============================================================================
GROUND_TRUTH = [
    # page_20260305_094228_cards - EX Unseen Forces starters
    {"image": "page_20260305_094228_cards/card_00.png", "name": "Chikorita", "label": "REVERSE_HOLO_STAMPED"},
    {"image": "page_20260305_094228_cards/card_01.png", "name": "Bayleef", "label": "NORMAL"},
    {"image": "page_20260305_094228_cards/card_02.png", "name": "Meganium", "label": "REVERSE_HOLO_STAMPED"},
    {"image": "page_20260305_094228_cards/card_03.png", "name": "Totodile", "label": "NORMAL"},
    {"image": "page_20260305_094228_cards/card_04.png", "name": "Croconaw", "label": "NORMAL"},
    {"image": "page_20260305_094228_cards/card_05.png", "name": "Feraligatr", "label": "HOLOFOIL"},
    {"image": "page_20260305_094228_cards/card_06.png", "name": "Cyndaquil", "label": "NORMAL"},
    {"image": "page_20260305_094228_cards/card_07.png", "name": "Quilava", "label": "NORMAL"},
    {"image": "page_20260305_094228_cards/card_08.png", "name": "Typhlosion", "label": "HOLOFOIL"},
    # page_20260228_174819_cards - stamped cards
    {"image": "page_20260228_174819_cards/card_01.png", "name": "Skitty", "label": "REVERSE_HOLO_STAMPED"},
    {"image": "page_20260228_174819_cards/card_05.png", "name": "Vibrava", "label": "REVERSE_HOLO_STAMPED"},
    # Prerelease stamps (treated as REVERSE_HOLO_STAMPED - same visual region)
    {"image": "page_20260307_014406_cards/card_02.png", "name": "Misty's Seadra", "label": "REVERSE_HOLO_STAMPED"},
    {"image": "page_20260307_020047_cards/card_08.png", "name": "Aerodactyl", "label": "REVERSE_HOLO_STAMPED"},
    {"image": "page_20260307_015320_cards/card_05.png", "name": "Dragonite", "label": "REVERSE_HOLO_STAMPED"},
    # Dark Dragonite holo
    {"image": "page_20260307_015320_cards/card_02.png", "name": "Dark Dragonite", "label": "HOLOFOIL"},
]


def load_ground_truth():
    """Load ground truth, verify images exist."""
    samples = []
    for entry in GROUND_TRUTH:
        path = INBOX_DIR / entry["image"]
        if path.exists():
            samples.append({
                "path": path,
                "name": entry["name"],
                "label": CLASS_MAP[entry["label"]],
                "label_name": entry["label"],
            })
        else:
            logger.warning("Missing image: %s", path)
    return samples


# =============================================================================
#  Feature extraction: hand-crafted texture features
# =============================================================================

def extract_texture_features(img_path):
    """Extract texture features from multiple regions of a card image.

    Returns a feature dict with named features for interpretability.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    features = {}

    # Define regions (proportional)
    regions = {
        "border_top":    (0, 0, w, int(h * 0.08)),
        "border_bottom": (0, int(h * 0.92), w, h),
        "border_left":   (0, 0, int(w * 0.06), h),
        "border_right":  (int(w * 0.94), 0, w, h),
        "artwork":       (int(w * 0.10), int(h * 0.10), int(w * 0.90), int(h * 0.55)),
        "text_area":     (int(w * 0.10), int(h * 0.55), int(w * 0.90), int(h * 0.85)),
        "stamp_region":  (int(w * 0.50), int(h * 0.65), int(w * 0.95), int(h * 0.95)),
        "full":          (0, 0, w, h),
    }

    for rname, (x1, y1, x2, y2) in regions.items():
        roi_gray = gray[y1:y2, x1:x2]
        roi_hsv = hsv[y1:y2, x1:x2]
        roi_bgr = img[y1:y2, x1:x2]

        if roi_gray.size == 0:
            continue

        # --- Intensity statistics ---
        features[f"{rname}_mean"] = float(np.mean(roi_gray))
        features[f"{rname}_std"] = float(np.std(roi_gray))
        features[f"{rname}_var"] = float(np.var(roi_gray))

        # --- Gradient energy (Sobel) ---
        sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        features[f"{rname}_grad_energy"] = float(np.mean(grad_mag))
        features[f"{rname}_grad_std"] = float(np.std(grad_mag))
        features[f"{rname}_grad_max"] = float(np.percentile(grad_mag, 95))

        # --- Laplacian variance (focus/texture measure) ---
        lap = cv2.Laplacian(roi_gray, cv2.CV_64F)
        features[f"{rname}_laplacian_var"] = float(np.var(lap))
        features[f"{rname}_laplacian_mean"] = float(np.mean(np.abs(lap)))

        # --- HSV saturation/value stats ---
        features[f"{rname}_sat_mean"] = float(np.mean(roi_hsv[:, :, 1]))
        features[f"{rname}_sat_std"] = float(np.std(roi_hsv[:, :, 1]))
        features[f"{rname}_val_mean"] = float(np.mean(roi_hsv[:, :, 2]))
        features[f"{rname}_val_std"] = float(np.std(roi_hsv[:, :, 2]))

        # --- Color channel variance (holographic shimmer) ---
        for ci, cname in enumerate(["b", "g", "r"]):
            features[f"{rname}_{cname}_std"] = float(np.std(roi_bgr[:, :, ci]))

        # --- Edge density (Canny) ---
        edges = cv2.Canny(roi_gray, 50, 150)
        features[f"{rname}_edge_density"] = float(np.mean(edges > 0))

        # --- High-frequency energy (DCT-based) ---
        roi_f = np.float32(roi_gray)
        # Resize to fixed size for DCT
        roi_resized = cv2.resize(roi_f, (64, 64))
        dct = cv2.dct(roi_resized)
        # High-freq energy: sum of absolute values in bottom-right quadrant
        hf = np.abs(dct[32:, 32:])
        lf = np.abs(dct[:32, :32])
        features[f"{rname}_hf_energy"] = float(np.mean(hf))
        features[f"{rname}_lf_energy"] = float(np.mean(lf))
        features[f"{rname}_hf_ratio"] = float(np.mean(hf) / (np.mean(lf) + 1e-8))

    # --- Cross-region ratios (key discriminating features) ---
    # Holofoil: artwork shimmery, border matte
    if "artwork_grad_energy" in features and "border_top_grad_energy" in features:
        avg_border_grad = np.mean([
            features.get("border_top_grad_energy", 0),
            features.get("border_bottom_grad_energy", 0),
            features.get("border_left_grad_energy", 0),
            features.get("border_right_grad_energy", 0),
        ])
        features["artwork_vs_border_grad"] = features["artwork_grad_energy"] / (avg_border_grad + 1e-8)

    # Reverse holo: border shimmery, artwork less so
    if "border_top_var" in features and "artwork_var" in features:
        avg_border_var = np.mean([
            features.get("border_top_var", 0),
            features.get("border_bottom_var", 0),
            features.get("border_left_var", 0),
            features.get("border_right_var", 0),
        ])
        features["border_vs_artwork_var"] = avg_border_var / (features["artwork_var"] + 1e-8)

    # Stamp region edge excess
    if "stamp_region_edge_density" in features and "text_area_edge_density" in features:
        features["stamp_edge_excess"] = (
            features["stamp_region_edge_density"] - features["text_area_edge_density"]
        )

    # Saturation contrasts
    if "artwork_sat_mean" in features and "border_top_sat_mean" in features:
        avg_border_sat = np.mean([
            features.get("border_top_sat_mean", 0),
            features.get("border_bottom_sat_mean", 0),
        ])
        features["artwork_vs_border_sat"] = features["artwork_sat_mean"] / (avg_border_sat + 1e-8)

    return features


def extract_dino_features(model, device, img_path):
    """Extract DINOv2 CLS token and regional patch statistics."""
    img = Image.open(img_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_out = model(tensor)
        patch_out = model.get_intermediate_layers(tensor, n=1)
        patch_tokens = patch_out[0].squeeze(0)  # (256, 768)

    cls_np = cls_out.cpu().numpy().astype(np.float32).squeeze()
    norm = np.linalg.norm(cls_np)
    if norm > 0:
        cls_np /= norm

    patches_np = patch_tokens.cpu().numpy().astype(np.float32)
    pnorms = np.linalg.norm(patches_np, axis=1, keepdims=True)
    pnorms[pnorms == 0] = 1.0
    patches_np /= pnorms

    # Reshape to grid
    grid = patches_np.reshape(GRID_SIZE, GRID_SIZE, EMBED_DIM)

    # Regional statistics
    dino_feats = {}

    # Full card CLS
    dino_feats["cls"] = cls_np  # (768,)

    # Regional patch means and stds
    region_slices = {
        "top":    (slice(0, 8), slice(None)),
        "bottom": (slice(8, 16), slice(None)),
        "artwork": (slice(2, 9), slice(2, 14)),  # Approx artwork area
        "border_tb": (slice(None), slice(None)),  # Will compute manually
        "stamp":  (slice(10, 16), slice(8, 16)),  # Bottom-right
    }

    for rname, (rs, cs) in region_slices.items():
        if rname == "border_tb":
            # Top 2 rows + bottom 2 rows
            border_patches = np.vstack([grid[:2, :, :].reshape(-1, EMBED_DIM),
                                         grid[14:, :, :].reshape(-1, EMBED_DIM)])
            dino_feats[f"dino_{rname}_mean"] = np.mean(border_patches, axis=0)
            dino_feats[f"dino_{rname}_std"] = np.std(border_patches, axis=0)
        else:
            region_patches = grid[rs, cs, :].reshape(-1, EMBED_DIM)
            dino_feats[f"dino_{rname}_mean"] = np.mean(region_patches, axis=0)
            dino_feats[f"dino_{rname}_std"] = np.std(region_patches, axis=0)

    # Stamp crop: extract CLS from bottom-right 40% of image
    w_img, h_img = img.size
    stamp_crop = img.crop((int(w_img * 0.5), int(h_img * 0.6), w_img, h_img))
    stamp_tensor = _transform(stamp_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        stamp_cls = model(stamp_tensor).cpu().numpy().astype(np.float32).squeeze()
    snorm = np.linalg.norm(stamp_cls)
    if snorm > 0:
        stamp_cls /= snorm
    dino_feats["stamp_crop_cls"] = stamp_cls

    return dino_feats


def build_feature_vector(texture_feats, dino_feats):
    """Combine texture + DINOv2 features into a single vector."""
    parts = []

    # Texture features (sorted for reproducibility)
    tex_keys = sorted(texture_feats.keys())
    parts.append(np.array([texture_feats[k] for k in tex_keys], dtype=np.float32))

    # DINOv2 CLS
    parts.append(dino_feats["cls"])

    # DINOv2 stamp crop CLS
    parts.append(dino_feats["stamp_crop_cls"])

    # DINOv2 regional means (compact)
    for rname in ["top", "bottom", "artwork", "border_tb", "stamp"]:
        mean_key = f"dino_{rname}_mean"
        std_key = f"dino_{rname}_std"
        if mean_key in dino_feats:
            parts.append(dino_feats[mean_key])
        if std_key in dino_feats:
            parts.append(dino_feats[std_key])

    return np.concatenate(parts)


def build_texture_only_vector(texture_feats):
    """Build feature vector from texture features only (no DINOv2)."""
    tex_keys = sorted(texture_feats.keys())
    return np.array([texture_feats[k] for k in tex_keys], dtype=np.float32)


def build_dino_only_vector(dino_feats):
    """Build feature vector from DINOv2 features only."""
    parts = []
    parts.append(dino_feats["cls"])
    parts.append(dino_feats["stamp_crop_cls"])
    for rname in ["top", "bottom", "artwork", "border_tb", "stamp"]:
        mean_key = f"dino_{rname}_mean"
        std_key = f"dino_{rname}_std"
        if mean_key in dino_feats:
            parts.append(dino_feats[mean_key])
        if std_key in dino_feats:
            parts.append(dino_feats[std_key])
    return np.concatenate(parts)


def leave_one_out_cv(X, y, samples, clf_factory, name=""):
    """Leave-one-out cross-validation. Returns predictions and accuracy."""
    N = len(y)
    predictions = np.zeros(N, dtype=int)
    probabilities = np.zeros((N, 3), dtype=float)

    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False

        X_train, y_train = X[mask], y[mask]
        X_test = X[i:i+1]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = clf_factory()
        clf.fit(X_train_s, y_train)

        predictions[i] = clf.predict(X_test_s)[0]
        if hasattr(clf, "predict_proba"):
            probabilities[i] = clf.predict_proba(X_test_s)[0]

    acc = accuracy_score(y, predictions)
    print(f"\n{'='*60}")
    print(f"  LOO-CV: {name}")
    print(f"  Accuracy: {acc:.1%} ({sum(predictions == y)}/{N})")
    print(f"{'='*60}")

    # Confusion matrix
    cm = confusion_matrix(y, predictions, labels=[0, 1, 2])
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"  {'':>20s} {'NORMAL':>10s} {'HOLOFOIL':>10s} {'REV_STAMP':>10s}")
    for i_row, row_name in enumerate(CLASS_NAMES):
        row_str = "  ".join(f"{cm[i_row, j]:>10d}" for j in range(3))
        print(f"  {row_name:>20s} {row_str}")

    print(f"\n  Classification Report:")
    print(classification_report(y, predictions, target_names=CLASS_NAMES, zero_division=0))

    # Per-sample results
    print(f"  Per-sample results:")
    for i in range(N):
        status = "OK" if predictions[i] == y[i] else "WRONG"
        pred_name = CLASS_NAMES[predictions[i]]
        true_name = CLASS_NAMES[y[i]]
        probs_str = " ".join(f"{probabilities[i, j]:.2f}" for j in range(3))
        print(f"    [{status:>5s}] {samples[i]['name']:>20s}  "
              f"pred={pred_name:<22s} true={true_name:<22s} probs=[{probs_str}]")

    return predictions, acc, cm


def main():
    t0 = time.time()

    # Load ground truth
    samples = load_ground_truth()
    logger.info("Loaded %d samples", len(samples))

    y = np.array([s["label"] for s in samples], dtype=int)
    for label_name in CLASS_NAMES:
        count = sum(1 for s in samples if s["label_name"] == label_name)
        logger.info("  %s: %d", label_name, count)

    # =================================================================
    #  Extract texture features from all samples
    # =================================================================
    logger.info("Extracting texture features...")
    texture_feats_list = []
    for s in samples:
        feats = extract_texture_features(s["path"])
        texture_feats_list.append(feats)
        logger.info("  %s: %d texture features", s["name"], len(feats))

    # =================================================================
    #  Extract DINOv2 features
    # =================================================================
    logger.info("Loading DINOv2...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    dino_model.to(device)
    dino_model.eval()

    logger.info("Extracting DINOv2 features...")
    dino_feats_list = []
    for s in samples:
        feats = extract_dino_features(dino_model, device, s["path"])
        dino_feats_list.append(feats)
        logger.info("  %s: DINOv2 extracted", s["name"])

    # Free GPU memory
    del dino_model
    torch.cuda.empty_cache()

    # =================================================================
    #  Build feature matrices for different configurations
    # =================================================================
    X_texture = np.stack([build_texture_only_vector(tf) for tf in texture_feats_list])
    X_dino = np.stack([build_dino_only_vector(df) for df in dino_feats_list])
    X_combined = np.stack([
        build_feature_vector(tf, df)
        for tf, df in zip(texture_feats_list, dino_feats_list)
    ])

    logger.info("Feature dimensions: texture=%d, dino=%d, combined=%d",
                X_texture.shape[1], X_dino.shape[1], X_combined.shape[1])

    # =================================================================
    #  LOO-CV experiments
    # =================================================================
    best_acc = 0.0
    best_config = None

    configurations = [
        # (name, X, classifier_factory)
        ("Texture-only LogReg C=1",
         X_texture,
         lambda: LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs",
                                     random_state=42)),
        ("Texture-only LogReg C=10",
         X_texture,
         lambda: LogisticRegression(C=10.0, max_iter=5000, solver="lbfgs",
                                     random_state=42)),
        ("Texture-only LogReg C=0.1",
         X_texture,
         lambda: LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs",
                                     random_state=42)),
        ("Texture-only RF",
         X_texture,
         lambda: RandomForestClassifier(n_estimators=100, max_depth=5,
                                         random_state=42, class_weight="balanced")),
        ("DINOv2-only LogReg C=1",
         X_dino,
         lambda: LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs",
                                     random_state=42)),
        ("DINOv2-only LogReg C=0.1",
         X_dino,
         lambda: LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs",
                                     random_state=42)),
        ("DINOv2-only RF",
         X_dino,
         lambda: RandomForestClassifier(n_estimators=100, max_depth=5,
                                         random_state=42, class_weight="balanced")),
        ("Combined LogReg C=1",
         X_combined,
         lambda: LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs",
                                     random_state=42)),
        ("Combined LogReg C=0.1",
         X_combined,
         lambda: LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs",
                                     random_state=42)),
        ("Combined LogReg C=10",
         X_combined,
         lambda: LogisticRegression(C=10.0, max_iter=5000, solver="lbfgs",
                                     random_state=42)),
        ("Combined RF",
         X_combined,
         lambda: RandomForestClassifier(n_estimators=100, max_depth=5,
                                         random_state=42, class_weight="balanced")),
        ("Combined RF deep",
         X_combined,
         lambda: RandomForestClassifier(n_estimators=200, max_depth=8,
                                         random_state=42, class_weight="balanced")),
    ]

    results = []
    for name, X, clf_factory in configurations:
        preds, acc, cm = leave_one_out_cv(X, y, samples, clf_factory, name=name)
        results.append((name, acc, preds, cm, X, clf_factory))
        if acc > best_acc:
            best_acc = acc
            best_config = (name, X, clf_factory)

    # =================================================================
    #  Summary
    # =================================================================
    print(f"\n{'#'*60}")
    print(f"  SUMMARY")
    print(f"{'#'*60}")
    results.sort(key=lambda x: x[1], reverse=True)
    for name, acc, _, _, _, _ in results:
        print(f"  {acc:.1%}  {name}")

    # =================================================================
    #  Train final model on all data if accuracy > 70%
    # =================================================================
    if best_acc >= 0.70:
        best_name, best_X, best_clf_factory = best_config
        print(f"\n{'='*60}")
        print(f"  Training final model: {best_name} (LOO acc={best_acc:.1%})")
        print(f"{'='*60}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(best_X)
        clf = best_clf_factory()
        clf.fit(X_scaled, y)

        # Determine feature type for saving
        if best_X is X_texture:
            feat_type = "texture"
        elif best_X is X_dino:
            feat_type = "dino"
        else:
            feat_type = "combined"

        # Get texture feature keys for reproducibility
        tex_keys = sorted(texture_feats_list[0].keys())

        save_obj = {
            "model": clf,
            "scaler": scaler,
            "feature_type": feat_type,
            "texture_feature_keys": tex_keys,
            "class_names": CLASS_NAMES,
            "metrics": {
                "loo_accuracy": best_acc,
                "config_name": best_name,
                "n_samples": len(y),
                "n_features": best_X.shape[1],
            },
        }

        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump(save_obj, f)
        logger.info("Saved model to %s", OUTPUT_PATH)
    else:
        print(f"\n  Best accuracy {best_acc:.1%} < 70%, NOT saving model.")

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
