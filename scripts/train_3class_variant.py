#!/usr/bin/env python3
"""Train a 3-class variant classifier: NORMAL vs HOLOFOIL vs REVERSE_HOLO_STAMPED.

Strategy: DINOv2 patch token statistics from artwork vs text regions.
Patch tokens capture LOCAL texture patterns (foil shimmer, speckle)
better than CLS tokens (which capture semantic content).

Key idea: compute patch token VARIANCE within regions. Foil surfaces
create more heterogeneous patch embeddings (high variance) compared to
matte surfaces (low variance, more consistent patches).

Then PCA to reduce dimensionality to match sample count.
"""

import logging
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INBOX_DIR = PROJECT_ROOT / "data" / "inbox"
OUTPUT_PATH = PROJECT_ROOT / "data" / "variant_3class_classifier.pkl"

CLASS_NAMES = ["NORMAL", "HOLOFOIL", "REVERSE_HOLO_STAMPED"]
CLASS_MAP = {"NORMAL": 0, "HOLOFOIL": 1, "REVERSE_HOLO_STAMPED": 2}

GRID_SIZE = 16
EMBED_DIM = 768

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


GROUND_TRUTH = [
    # EX Dragon Frontiers page
    {"image": "page_20260305_094228_cards/card_00.png", "name": "Chikorita", "label": "REVERSE_HOLO_STAMPED"},
    {"image": "page_20260305_094228_cards/card_01.png", "name": "Bayleef", "label": "NORMAL"},
    {"image": "page_20260305_094228_cards/card_02.png", "name": "Meganium", "label": "REVERSE_HOLO_STAMPED"},
    {"image": "page_20260305_094228_cards/card_03.png", "name": "Totodile", "label": "NORMAL"},
    {"image": "page_20260305_094228_cards/card_04.png", "name": "Croconaw", "label": "NORMAL"},
    {"image": "page_20260305_094228_cards/card_05.png", "name": "Feraligatr", "label": "HOLOFOIL"},
    {"image": "page_20260305_094228_cards/card_06.png", "name": "Cyndaquil", "label": "NORMAL"},
    {"image": "page_20260305_094228_cards/card_07.png", "name": "Quilava", "label": "NORMAL"},
    {"image": "page_20260305_094228_cards/card_08.png", "name": "Typhlosion", "label": "HOLOFOIL"},
    # More EX-era cards
    {"image": "page_20260228_174819_cards/card_01.png", "name": "Skitty", "label": "REVERSE_HOLO_STAMPED"},
    {"image": "page_20260228_174819_cards/card_05.png", "name": "Vibrava", "label": "REVERSE_HOLO_STAMPED"},
    {"image": "page_20260228_174819_cards/card_04.png", "name": "Delcatty ex", "label": "HOLOFOIL"},
    {"image": "page_20260228_174819_cards/card_08.png", "name": "Flygon ex", "label": "HOLOFOIL"},
    # Prerelease stamps
    {"image": "page_20260307_014406_cards/card_02.png", "name": "Misty's Seadra", "label": "REVERSE_HOLO_STAMPED"},
    {"image": "page_20260307_020047_cards/card_08.png", "name": "Aerodactyl", "label": "REVERSE_HOLO_STAMPED"},
    {"image": "page_20260307_015320_cards/card_05.png", "name": "Dragonite", "label": "REVERSE_HOLO_STAMPED"},
    # Old-era holofoil
    {"image": "page_20260307_015320_cards/card_02.png", "name": "Dark Dragonite", "label": "HOLOFOIL"},
]


def load_ground_truth():
    samples = []
    for entry in GROUND_TRUTH:
        path = INBOX_DIR / entry["image"]
        if path.exists():
            samples.append({
                "path": path, "name": entry["name"],
                "label": CLASS_MAP[entry["label"]], "label_name": entry["label"],
            })
        else:
            logger.warning("Missing image: %s", path)
    return samples


def extract_dino_patch_features(model, device, img_path):
    """Extract DINOv2 CLS + patch statistics from regions."""
    img = Image.open(img_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_out = model(tensor)
        patch_out = model.get_intermediate_layers(tensor, n=1)
        patches = patch_out[0].squeeze(0)  # (256, 768)

    cls_np = cls_out.cpu().numpy().astype(np.float32).squeeze()
    norm = np.linalg.norm(cls_np)
    if norm > 0:
        cls_np /= norm

    patches_np = patches.cpu().numpy().astype(np.float32)
    pnorms = np.linalg.norm(patches_np, axis=1, keepdims=True)
    pnorms[pnorms == 0] = 1.0
    patches_np /= pnorms

    grid = patches_np.reshape(GRID_SIZE, GRID_SIZE, EMBED_DIM)

    # Regions in 16x16 patch grid
    # Artwork: rows 2-7 (top half, inside border)
    artwork_patches = grid[2:8, 2:14, :].reshape(-1, EMBED_DIM)
    # Text area: rows 9-14 (bottom half, inside border)
    text_patches = grid[9:14, 2:14, :].reshape(-1, EMBED_DIM)
    # Stamp: bottom-right
    stamp_patches = grid[10:15, 8:15, :].reshape(-1, EMBED_DIM)
    # Border: top 2 rows + bottom 2 rows
    border_patches = np.vstack([
        grid[0:2, :, :].reshape(-1, EMBED_DIM),
        grid[14:16, :, :].reshape(-1, EMBED_DIM),
    ])

    features = {}
    features["cls"] = cls_np

    for rname, rpatches in [("artwork", artwork_patches), ("text", text_patches),
                             ("stamp", stamp_patches), ("border", border_patches)]:
        features[f"{rname}_mean"] = np.mean(rpatches, axis=0)
        features[f"{rname}_std"] = np.std(rpatches, axis=0)
        # Scalar statistics (very compact)
        features[f"{rname}_var_scalar"] = float(np.mean(np.var(rpatches, axis=0)))
        features[f"{rname}_mean_norm"] = float(np.mean(np.linalg.norm(rpatches, axis=1)))

        # Pairwise distance stats (captures internal consistency)
        if len(rpatches) > 2:
            # Random subsample of pairs for efficiency
            n = len(rpatches)
            idx1 = np.random.RandomState(42).randint(0, n, min(50, n))
            idx2 = np.random.RandomState(43).randint(0, n, min(50, n))
            dists = np.linalg.norm(rpatches[idx1] - rpatches[idx2], axis=1)
            features[f"{rname}_pair_dist_mean"] = float(np.mean(dists))
            features[f"{rname}_pair_dist_std"] = float(np.std(dists))

    return features


def extract_handcrafted(img_path):
    """Compact hand-crafted features."""
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros(8, dtype=np.float32)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    art = gray[int(h*0.10):int(h*0.48), int(w*0.12):int(w*0.88)]
    txt = gray[int(h*0.52):int(h*0.88), int(w*0.12):int(w*0.88)]
    art_hsv = hsv[int(h*0.10):int(h*0.48), int(w*0.12):int(w*0.88)]
    txt_hsv = hsv[int(h*0.52):int(h*0.88), int(w*0.12):int(w*0.88)]
    stp = gray[int(h*0.55):int(h*0.85), int(w*0.50):int(w*0.88)]

    art_lap = float(np.var(cv2.Laplacian(art, cv2.CV_64F)))
    txt_lap = float(np.var(cv2.Laplacian(txt, cv2.CV_64F)))

    stp_edges = cv2.Canny(stp, 50, 150)
    txt_edges = cv2.Canny(txt, 50, 150)

    return np.array([
        art_lap,
        txt_lap,
        art_lap / (txt_lap + 1e-8),
        float(np.std(art_hsv[:,:,0])),  # art hue std
        float(np.std(txt_hsv[:,:,0])),  # txt hue std
        float(np.std(art_hsv[:,:,1])),  # art sat std
        float(np.std(txt_hsv[:,:,1])),  # txt sat std
        float(np.mean(stp_edges > 0) - np.mean(txt_edges > 0)),  # stamp edge excess
    ], dtype=np.float32)


def build_feature_matrices(dino_feats, hc_feats):
    """Build various feature matrix configurations."""
    N = len(dino_feats)

    # Scalar DINOv2 features (very compact: ~24 scalars)
    scalar_keys = [k for k in dino_feats[0] if k.endswith("_scalar") or
                   k.endswith("_mean_norm") or k.endswith("_dist_mean") or
                   k.endswith("_dist_std")]
    X_scalar = np.stack([[dino_feats[i][k] for k in scalar_keys] for i in range(N)])

    # Regional means (768-dim each)
    X_art_mean = np.stack([d["artwork_mean"] for d in dino_feats])
    X_txt_mean = np.stack([d["text_mean"] for d in dino_feats])
    X_brd_mean = np.stack([d["border_mean"] for d in dino_feats])
    X_stp_mean = np.stack([d["stamp_mean"] for d in dino_feats])

    # Regional stds (768-dim each)
    X_art_std = np.stack([d["artwork_std"] for d in dino_feats])
    X_txt_std = np.stack([d["text_std"] for d in dino_feats])

    # CLS token
    X_cls = np.stack([d["cls"] for d in dino_feats])

    # Differences (artwork - text, captures which region has more texture)
    X_mean_diff = X_art_mean - X_txt_mean
    X_std_diff = X_art_std - X_txt_std

    matrices = {
        "scalar": X_scalar,
        "scalar_hc": np.hstack([X_scalar, hc_feats]),
        "cls": X_cls,
        "art_mean": X_art_mean,
        "txt_mean": X_txt_mean,
        "mean_diff": X_mean_diff,
        "std_diff": X_std_diff,
        "art_std": X_art_std,
        "txt_std": X_txt_std,
        "art_txt_std": np.hstack([X_art_std, X_txt_std]),
        "art_txt_mean": np.hstack([X_art_mean, X_txt_mean]),
        "cls_scalar_hc": np.hstack([X_cls, X_scalar, hc_feats]),
        "all_std": np.hstack([X_art_std, X_txt_std, X_scalar]),
        "mean_std_diff": np.hstack([X_mean_diff, X_std_diff]),
        "handcrafted": hc_feats,
    }
    return matrices, scalar_keys


def leave_one_out_cv(X, y, samples, clf_factory, name="", verbose=False):
    N = len(y)
    preds = np.zeros(N, dtype=int)
    probs = np.zeros((N, 3), dtype=float)

    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        clf = clf_factory()
        clf.fit(X[mask], y[mask])
        preds[i] = clf.predict(X[i:i+1])[0]
        if hasattr(clf, "predict_proba"):
            probs[i] = clf.predict_proba(X[i:i+1])[0]

    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds, labels=[0, 1, 2])

    if verbose:
        print(f"\n{'='*60}")
        print(f"  LOO-CV: {name}")
        print(f"  Accuracy: {acc:.1%} ({sum(preds == y)}/{N})")
        print(f"{'='*60}")
        print(f"\n  Confusion Matrix:")
        print(f"  {'':>20s} {'NORMAL':>8s} {'HOLO':>8s} {'STAMPED':>8s}")
        for ir, rn in enumerate(CLASS_NAMES):
            row = "  ".join(f"{cm[ir,j]:>8d}" for j in range(3))
            print(f"  {rn:>20s} {row}")
        print(classification_report(y, preds, target_names=CLASS_NAMES, zero_division=0))
        for i in range(N):
            status = "OK" if preds[i] == y[i] else "XX"
            ps = " ".join(f"{probs[i,j]:.2f}" for j in range(3))
            print(f"    [{status:>2s}] {samples[i]['name']:>20s}  "
                  f"pred={CLASS_NAMES[preds[i]]:<22s} true={CLASS_NAMES[y[i]]:<22s} [{ps}]")

    return preds, acc, cm


def main():
    t0 = time.time()

    samples = load_ground_truth()
    y = np.array([s["label"] for s in samples], dtype=int)
    N = len(y)
    logger.info("Loaded %d samples: %s", N,
                {ln: sum(1 for s in samples if s["label_name"] == ln) for ln in CLASS_NAMES})

    # Extract DINOv2 features
    logger.info("Loading DINOv2...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    dino_model.to(device).eval()

    logger.info("Extracting DINOv2 patch features...")
    dino_feats = [extract_dino_patch_features(dino_model, device, s["path"]) for s in samples]

    del dino_model
    torch.cuda.empty_cache()

    # Hand-crafted features
    logger.info("Extracting hand-crafted features...")
    hc_feats = np.stack([extract_handcrafted(s["path"]) for s in samples])

    # Build feature matrices
    matrices, scalar_keys = build_feature_matrices(dino_feats, hc_feats)

    # Print scalar DINOv2 features for interpretability
    print(f"\n  DINOv2 scalar features:")
    for ki, kn in enumerate(scalar_keys):
        print(f"\n  {kn}:")
        for cls in CLASS_NAMES:
            for i, s in enumerate(samples):
                if s["label_name"] == cls:
                    print(f"    {s['name']:>20s} ({cls[:8]:>8s}): {matrices['scalar'][i, ki]:>10.6f}")

    # LOO-CV sweep
    best_acc = 0.0
    results = []

    pca_dims = [3, 5, 8, 10]

    for feat_name, X_feat in matrices.items():
        ndim = X_feat.shape[1]

        # With PCA
        for pca_n in pca_dims:
            if pca_n >= N - 1 or ndim <= pca_n:
                continue

            for clf_name, clf_f in [
                ("lr01", lambda: LogisticRegression(C=0.1, max_iter=5000, random_state=42)),
                ("lr1", lambda: LogisticRegression(C=1.0, max_iter=5000, random_state=42)),
                ("lr10", lambda: LogisticRegression(C=10.0, max_iter=5000, random_state=42)),
                ("knn3", lambda: KNeighborsClassifier(n_neighbors=3, weights="distance")),
                ("svm1", lambda: SVC(C=1.0, kernel="rbf", probability=True, random_state=42, class_weight="balanced")),
                ("svm10", lambda: SVC(C=10.0, kernel="rbf", probability=True, random_state=42, class_weight="balanced")),
                ("rf2", lambda: RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42, class_weight="balanced")),
                ("rf3", lambda: RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, class_weight="balanced")),
            ]:
                name = f"{feat_name}_pca{pca_n}_{clf_name}"
                factory = lambda cf=clf_f, pn=pca_n: Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=pn, random_state=42)),
                    ("clf", cf()),
                ])
                preds, acc, cm = leave_one_out_cv(X_feat, y, samples, factory, name=name)
                results.append((name, acc, preds, cm, feat_name))
                if acc > best_acc:
                    best_acc = acc

        # Without PCA for small feature sets
        if ndim <= 15:
            for clf_name, clf_f in [
                ("lr01", lambda: LogisticRegression(C=0.1, max_iter=5000, random_state=42)),
                ("lr1", lambda: LogisticRegression(C=1.0, max_iter=5000, random_state=42)),
                ("knn3", lambda: KNeighborsClassifier(n_neighbors=3, weights="distance")),
                ("knn5", lambda: KNeighborsClassifier(n_neighbors=5, weights="distance")),
                ("rf2", lambda: RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42, class_weight="balanced")),
                ("svm1", lambda: SVC(C=1.0, kernel="rbf", probability=True, random_state=42, class_weight="balanced")),
            ]:
                name = f"{feat_name}_nopca_{clf_name}"
                factory = lambda cf=clf_f: Pipeline([("scaler", StandardScaler()), ("clf", cf())])
                preds, acc, cm = leave_one_out_cv(X_feat, y, samples, factory, name=name)
                results.append((name, acc, preds, cm, feat_name))
                if acc > best_acc:
                    best_acc = acc

    # Sort and show top results
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'#'*60}")
    print(f"  TOP 30 RESULTS (out of {len(results)})")
    print(f"{'#'*60}")
    for name, acc, _, _, _ in results[:30]:
        print(f"  {acc:.1%}  {name}")

    # Detailed results for top 3
    for rank in range(min(3, len(results))):
        name, acc, preds, cm, fkey = results[rank]
        print(f"\n{'='*60}")
        print(f"  #{rank+1}: {name} ({acc:.1%})")
        print(f"{'='*60}")
        print(f"\n  Confusion Matrix:")
        print(f"  {'':>20s} {'NORMAL':>8s} {'HOLO':>8s} {'STAMPED':>8s}")
        for ir, rn in enumerate(CLASS_NAMES):
            row = "  ".join(f"{cm[ir,j]:>8d}" for j in range(3))
            print(f"  {rn:>20s} {row}")
        print(classification_report(y, preds, target_names=CLASS_NAMES, zero_division=0))
        for i in range(N):
            status = "OK" if preds[i] == y[i] else "XX"
            print(f"    [{status:>2s}] {samples[i]['name']:>20s}  "
                  f"pred={CLASS_NAMES[preds[i]]:<22s} true={CLASS_NAMES[y[i]]:<22s}")

    # Save if > 70%
    best_result = results[0] if results else None
    if best_result and best_result[1] >= 0.70:
        best_name, best_acc_f, _, _, best_fkey = best_result
        X_final = matrices[best_fkey]

        # Parse config
        pca_n = 0
        for d in pca_dims:
            if f"pca{d}" in best_name:
                pca_n = d
                break

        clf_map = {
            "lr01": lambda: LogisticRegression(C=0.1, max_iter=5000, random_state=42),
            "lr1": lambda: LogisticRegression(C=1.0, max_iter=5000, random_state=42),
            "lr10": lambda: LogisticRegression(C=10.0, max_iter=5000, random_state=42),
            "knn3": lambda: KNeighborsClassifier(n_neighbors=3, weights="distance"),
            "knn5": lambda: KNeighborsClassifier(n_neighbors=5, weights="distance"),
            "svm1": lambda: SVC(C=1.0, kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
            "svm10": lambda: SVC(C=10.0, kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
            "rf2": lambda: RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42, class_weight="balanced"),
            "rf3": lambda: RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, class_weight="balanced"),
        }
        clf_key = best_name.split("_")[-1]
        clf_f = clf_map.get(clf_key, lambda: LogisticRegression(C=1.0, max_iter=5000, random_state=42))

        steps = [("scaler", StandardScaler())]
        if pca_n > 0:
            steps.append(("pca", PCA(n_components=pca_n, random_state=42)))
        steps.append(("clf", clf_f()))
        pipeline = Pipeline(steps)
        pipeline.fit(X_final, y)

        print(f"\n  Saved final model: {best_name} (LOO acc={best_acc_f:.1%})")
        save_obj = {
            "model": pipeline, "feature_key": best_fkey,
            "class_names": CLASS_NAMES,
            "metrics": {"loo_accuracy": best_acc_f, "config_name": best_name, "n_samples": N},
        }
        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump(save_obj, f)
        logger.info("Saved model to %s", OUTPUT_PATH)
    else:
        best_acc_val = best_result[1] if best_result else 0
        print(f"\n  Best accuracy {best_acc_val:.1%} < 70%, NOT saving model.")

    logger.info("Total time: %.1f seconds", time.time() - t0)


if __name__ == "__main__":
    main()
