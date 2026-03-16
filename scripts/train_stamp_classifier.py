#!/usr/bin/env python3
"""Train a binary stamp classifier: stamped vs non-stamped card images.

Architecture: DINOv2 ViT-B/14 features -> logistic regression.

Data:
  - Synthetic (training): data/condition_training/stamps/labels.jsonl (400 images)
  - Real (validation): data/condition_training/stamps_real/sources.jsonl (41 images)

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
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "condition_training" / "stamps"
REAL_DIR = PROJECT_ROOT / "data" / "condition_training" / "stamps_real"
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


def evaluate(clf, X_val, y_val, val_paths, feature_name, C):
    val_pred = clf.predict(X_val)
    val_proba = clf.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred)

    print(f"\n{'='*60}")
    print(f"  {feature_name} (C={C})")
    print(f"{'='*60}")
    print(f"  Val accuracy:   {val_acc:.1%} ({sum(val_pred == y_val)}/{len(y_val)})")
    print(f"  Val F1:         {val_f1:.3f}")
    print(classification_report(y_val, val_pred, target_names=["clean", "stamped"]))

    wrong = []
    for i, (pred, true, prob) in enumerate(zip(val_pred, y_val, val_proba)):
        status = "OK" if pred == true else "WRONG"
        name = val_paths[i].name
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
              hidden_dim=128, dropout=0.3, lr=1e-3, epochs=200, weight_decay=1e-2):
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
                val_pred = (val_probs > 0.5).astype(int)
                acc = accuracy_score(y_val, val_pred)
                f1 = f1_score(y_val, val_pred)
                train_probs = torch.sigmoid(model(X_tr).squeeze()).cpu().numpy()
                train_pred = (train_probs > 0.5).astype(int)
                train_acc = accuracy_score(y_train, train_pred)

            if f1 > best_f1 or (f1 == best_f1 and acc > best_acc):
                best_f1 = f1
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 50 == 0:
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

    val_pred = (val_probs > 0.5).astype(int)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred)

    print(f"\n{'='*60}")
    print(f"  MLP: {feature_name} (hidden={hidden_dim}, drop={dropout}, wd={weight_decay})")
    print(f"{'='*60}")
    print(f"  Val accuracy: {val_acc:.1%}")
    print(f"  Val F1:       {val_f1:.3f}")
    print(classification_report(y_val, val_pred, target_names=["clean", "stamped"]))

    wrong = []
    for i, (pred, true, prob) in enumerate(zip(val_pred, y_val, val_probs)):
        status = "OK" if pred == true else "WRONG"
        name = val_paths[i].name
        if pred != true:
            wrong.append(name)
        print(f"    [{status}] prob={prob:.3f} pred={pred} true={true} {name}")

    if wrong:
        print(f"\n  Misclassified ({len(wrong)}): {', '.join(wrong)}")

    return model, val_acc, val_f1


def main():
    t0 = time.time()

    # Load datasets
    synth_paths, y_synth = load_dataset(SYNTHETIC_DIR / "labels.jsonl", SYNTHETIC_DIR)
    real_paths, y_real = load_dataset(REAL_DIR / "sources.jsonl", REAL_DIR)

    # Split real data 70/30: mix 70% into training, hold 30% for validation
    from sklearn.model_selection import train_test_split
    if len(real_paths) > 10:
        real_train_paths, val_paths, y_real_train, y_val = train_test_split(
            real_paths, y_real, test_size=0.3, random_state=42, stratify=y_real
        )
        train_paths = synth_paths + list(real_train_paths)
        y_train = np.concatenate([y_synth, y_real_train])
    else:
        train_paths = synth_paths
        y_train = y_synth
        val_paths = real_paths
        y_val = y_real
    logger.info("Train: %d (%d synth + %d real, %d stamped), Val: %d (%d stamped)",
                len(train_paths), len(synth_paths), len(train_paths) - len(synth_paths),
                sum(y_train), len(val_paths), sum(y_val))

    # Load DINOv2
    dino_model, dino_device = load_model()

    # Extract features
    logger.info("Extracting training features...")
    train_cls, train_patches = extract_features_batch(dino_model, dino_device, train_paths)
    logger.info("Extracting validation features...")
    val_cls, val_patches = extract_features_batch(dino_model, dino_device, val_paths)

    # Free GPU memory from DINOv2
    del dino_model
    torch.cuda.empty_cache()

    # ===== Feature types to try =====
    feature_types = [
        "cls",
        "br_stats",
        "all_stats",
        "cls_br_stats",
        "cls_all_stats",
        "multi_region_stats",
        "cls_multi_region",
    ]

    best_model = None
    best_metrics = None
    best_feature_type = None
    best_scaler = None
    best_model_type = None  # "lr" or "mlp"

    # ===== Logistic Regression sweep =====
    print("\n" + "#" * 60)
    print("  LOGISTIC REGRESSION EXPERIMENTS")
    print("#" * 60)

    for ftype in feature_types:
        X_train = build_features(train_cls, train_patches, ftype)
        X_val = build_features(val_cls, val_patches, ftype)

        # Try with and without scaling
        for scale in [False, True]:
            scaler = None
            X_tr, X_va = X_train, X_val
            if scale:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train)
                X_va = scaler.transform(X_val)

            for C in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
                name = f"{ftype}{'_scaled' if scale else ''}"
                clf = LogisticRegression(
                    C=C, max_iter=2000, solver="lbfgs",
                    class_weight="balanced", random_state=42,
                )
                clf.fit(X_tr, y_train)

                val_pred = clf.predict(X_va)
                val_proba = clf.predict_proba(X_va)[:, 1]
                acc = accuracy_score(y_val, val_pred)
                f1 = f1_score(y_val, val_pred)

                # Only print promising results (F1 > 0.7)
                if f1 >= 0.7:
                    print(f"\n  {name} C={C}: acc={acc:.1%} F1={f1:.3f}")
                    wrong = [val_paths[i].name for i in range(len(y_val))
                             if val_pred[i] != y_val[i]]
                    if wrong:
                        print(f"    Wrong: {', '.join(wrong)}")

                if (best_metrics is None or f1 > best_metrics["val_f1"]
                    or (f1 == best_metrics["val_f1"] and acc > best_metrics["val_acc"])):
                    best_model = clf
                    best_metrics = {"feature_name": name, "val_acc": acc, "val_f1": f1, "C": C}
                    best_feature_type = ftype + ("_scaled" if scale else "")
                    best_scaler = scaler
                    best_model_type = "lr"

    # ===== MLP experiments =====
    print("\n" + "#" * 60)
    print("  MLP EXPERIMENTS")
    print("#" * 60)

    for ftype in ["br_stats", "all_stats", "cls_br_stats", "multi_region_stats"]:
        X_train = build_features(train_cls, train_patches, ftype)
        X_val = build_features(val_cls, val_patches, ftype)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_va_s = scaler.transform(X_val)

        for hidden, dropout, wd in [(64, 0.5, 0.1), (128, 0.3, 0.01), (32, 0.5, 0.1)]:
            mlp, acc, f1 = train_mlp(
                X_tr_s, y_train, X_va_s, y_val, val_paths,
                feature_name=f"{ftype}_scaled",
                hidden_dim=hidden, dropout=dropout, weight_decay=wd,
                epochs=300,
            )

            if (f1 > best_metrics["val_f1"]
                or (f1 == best_metrics["val_f1"] and acc > best_metrics["val_acc"])):
                best_model = mlp
                best_metrics = {
                    "feature_name": f"MLP_{ftype}_scaled",
                    "val_acc": acc, "val_f1": f1, "C": 0,
                    "hidden_dim": hidden, "dropout": dropout,
                }
                best_feature_type = ftype + "_scaled"
                best_scaler = scaler
                best_model_type = "mlp"

    # ===== Final report =====
    print(f"\n{'='*60}")
    print(f"  BEST MODEL: {best_metrics['feature_name']} (type={best_model_type})")
    print(f"  Val accuracy: {best_metrics['val_acc']:.1%}")
    print(f"  Val F1: {best_metrics['val_f1']:.3f}")
    print(f"  Feature type: {best_feature_type}")
    print(f"{'='*60}")

    # Re-evaluate best model for detailed report
    base_ftype = best_feature_type.replace("_scaled", "")
    X_val_final = build_features(val_cls, val_patches, base_ftype)
    if best_scaler is not None:
        X_val_final = best_scaler.transform(X_val_final)

    if best_model_type == "lr":
        evaluate(best_model, X_val_final, y_val, val_paths,
                 best_metrics["feature_name"], best_metrics.get("C", 0))
    else:
        # MLP evaluation already printed during training
        pass

    # Save
    save_obj = {
        "model": best_model,
        "feature_type": best_feature_type,
        "model_type": best_model_type,
        "metrics": best_metrics,
    }
    if best_scaler is not None:
        save_obj["scaler"] = best_scaler
    if best_model_type == "mlp":
        # Save state dict for portability
        save_obj["model_state"] = best_model.state_dict()
        save_obj["model_config"] = {
            "input_dim": X_val_final.shape[1],
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
