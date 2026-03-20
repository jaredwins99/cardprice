#!/usr/bin/env python3
"""Contrastive/triplet learning for stamp detection — explore the feature space.

Steps:
1. Load all stamp data (synthetic, reference photos, binder scans)
2. Extract DINOv2 CLS features for stamp-region crops
3. Visualize with t-SNE and UMAP, colored by stamped/clean, shaped by domain
4. Train a triplet-loss linear projection (768 -> 128)
5. Compare nearest-neighbor classification accuracy before/after projection
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
import umap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STAMPS_DIR = DATA / "condition_training" / "stamps"
REAL_DIR = DATA / "condition_training" / "stamps_real"
INBOX = DATA / "inbox"
PLOTS_DIR = ROOT / "data" / "stamp_analysis"

# DINOv2 preprocessing
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# Stamp region: bottom-right quadrant of the card (where EX stamps appear)
STAMP_CROP = (0.5, 0.5, 1.0, 1.0)  # (left_frac, top_frac, right_frac, bottom_frac)


def load_dino_model():
    """Load DINOv2 ViT-B/14."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading DINOv2 ViT-B/14 on %s", device)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    model = model.to(device)
    model.eval()
    return model, device


def crop_stamp_region(img: Image.Image) -> Image.Image:
    """Crop the bottom-right quadrant where stamps typically appear."""
    w, h = img.size
    left = int(w * STAMP_CROP[0])
    top = int(h * STAMP_CROP[1])
    right = int(w * STAMP_CROP[2])
    bottom = int(h * STAMP_CROP[3])
    return img.crop((left, top, right, bottom))


def extract_cls_features(model, device, images: list[Image.Image], batch_size=32) -> np.ndarray:
    """Extract DINOv2 CLS features from a list of PIL images."""
    all_features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        tensors = torch.stack([_transform(img) for img in batch]).to(device)
        with torch.no_grad():
            features = model(tensors)  # (B, 768)
        feats = features.cpu().numpy().astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats /= norms
        all_features.append(feats)
    return np.vstack(all_features)


# ── Data Loading ──────────────────────────────────────────────────────────

def load_synthetic_data():
    """Load synthetic stamp/clean pairs from labels.jsonl."""
    entries = []
    with open(STAMPS_DIR / "labels.jsonl") as f:
        for line in f:
            entries.append(json.loads(line))

    images, labels, domains = [], [], []
    loaded, skipped = 0, 0
    for e in entries:
        path = STAMPS_DIR / e["image"]
        if not path.exists():
            skipped += 1
            continue
        try:
            img = Image.open(path).convert("RGB")
            crop = crop_stamp_region(img)
            images.append(crop)
            labels.append(1 if e["stamped"] else 0)
            domains.append("synthetic")
            loaded += 1
        except Exception as ex:
            logger.warning("Failed to load %s: %s", path, ex)
            skipped += 1

    logger.info("Synthetic: loaded=%d, skipped=%d (stamped=%d, clean=%d)",
                loaded, skipped, sum(labels), loaded - sum(labels))
    return images, labels, domains


def load_reference_data():
    """Load real reference photos from sources.jsonl."""
    entries = []
    with open(REAL_DIR / "sources.jsonl") as f:
        for line in f:
            entries.append(json.loads(line))

    images, labels, domains = [], [], []
    loaded, skipped = 0, 0
    for e in entries:
        path = REAL_DIR / e["image"]
        if not path.exists():
            skipped += 1
            continue
        try:
            img = Image.open(path).convert("RGB")
            crop = crop_stamp_region(img)
            images.append(crop)
            labels.append(1 if e["stamped"] else 0)
            domains.append("reference")
            loaded += 1
        except Exception as ex:
            logger.warning("Failed to load %s: %s", path, ex)
            skipped += 1

    logger.info("Reference: loaded=%d, skipped=%d (stamped=%d, clean=%d)",
                loaded, skipped, sum(labels), loaded - sum(labels))
    return images, labels, domains


def load_binder_data():
    """Load binder scan ground truth from binder_ground_truth.jsonl."""
    entries = []
    with open(REAL_DIR / "binder_ground_truth.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    images, labels, domains = [], [], []
    loaded, skipped = 0, 0
    for e in entries:
        path = INBOX / e["image"]
        if not path.exists():
            skipped += 1
            logger.warning("Binder image not found: %s", path)
            continue
        try:
            img = Image.open(path).convert("RGB")
            crop = crop_stamp_region(img)
            images.append(crop)
            labels.append(1 if e["stamped"] else 0)
            domains.append("binder")
            loaded += 1
        except Exception as ex:
            logger.warning("Failed to load %s: %s", path, ex)
            skipped += 1

    logger.info("Binder: loaded=%d, skipped=%d (stamped=%d, clean=%d)",
                loaded, skipped, sum(labels), loaded - sum(labels))
    return images, labels, domains


# ── Visualization ─────────────────────────────────────────────────────────

def plot_embedding(coords_2d, labels, domains, title, filename):
    """Scatter plot: color=stamped/clean, shape=domain."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    domain_markers = {"synthetic": "o", "reference": "s", "binder": "^"}
    label_colors = {0: "#2196F3", 1: "#F44336"}  # blue=clean, red=stamped
    label_names = {0: "clean", 1: "stamped"}

    # Plot each domain/label combo
    for domain in ["synthetic", "reference", "binder"]:
        for label in [0, 1]:
            mask = np.array([(d == domain and l == label) for d, l in zip(domains, labels)])
            if not mask.any():
                continue
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                c=label_colors[label],
                marker=domain_markers[domain],
                s=40 if domain != "binder" else 100,
                alpha=0.6 if domain != "binder" else 1.0,
                edgecolors="black" if domain == "binder" else "none",
                linewidths=1.5 if domain == "binder" else 0,
                label=f"{domain} {label_names[label]}",
                zorder=3 if domain == "binder" else 1,
            )

    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    logger.info("Saved plot: %s", filename)


# ── Triplet Loss Training ────────────────────────────────────────────────

class LinearProjection(nn.Module):
    """Linear projection from 768 to out_dim."""
    def __init__(self, in_dim=768, out_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        out = self.proj(x)
        # L2 normalize
        return out / (out.norm(dim=1, keepdim=True) + 1e-8)


def mine_triplets(features, labels, domains, n_triplets=5000):
    """Mine triplets: anchor=stamped binder, positive=stamped any, negative=clean any.

    Also includes: anchor=stamped ref, positive=stamped synthetic, negative=clean.
    This bridges synthetic <-> real domains.
    """
    rng = np.random.RandomState(42)
    labels = np.array(labels)
    domains = np.array(domains)

    # Indices by category
    binder_stamped = np.where((labels == 1) & (domains == "binder"))[0]
    ref_stamped = np.where((labels == 1) & (domains == "reference"))[0]
    synth_stamped = np.where((labels == 1) & (domains == "synthetic"))[0]
    all_stamped = np.where(labels == 1)[0]
    all_clean = np.where(labels == 0)[0]

    logger.info("Triplet mining: binder_stamped=%d, ref_stamped=%d, synth_stamped=%d, all_clean=%d",
                len(binder_stamped), len(ref_stamped), len(synth_stamped), len(all_clean))

    anchors, positives, negatives = [], [], []

    if len(binder_stamped) == 0 or len(all_clean) == 0:
        logger.warning("Not enough data for triplet mining")
        return np.array([]), np.array([]), np.array([])

    # Strategy 1: binder stamped anchor, any stamped positive, any clean negative
    for _ in range(n_triplets // 2):
        a = rng.choice(binder_stamped)
        # Positive: any stamped that isn't the anchor
        pos_pool = all_stamped[all_stamped != a]
        if len(pos_pool) == 0:
            continue
        p = rng.choice(pos_pool)
        n = rng.choice(all_clean)
        anchors.append(a)
        positives.append(p)
        negatives.append(n)

    # Strategy 2: reference stamped anchor, synthetic stamped positive, any clean negative
    # Bridges the domain gap explicitly
    if len(ref_stamped) > 0 and len(synth_stamped) > 0:
        for _ in range(n_triplets // 4):
            a = rng.choice(ref_stamped)
            p = rng.choice(synth_stamped)
            n = rng.choice(all_clean)
            anchors.append(a)
            positives.append(p)
            negatives.append(n)

    # Strategy 3: synthetic stamped anchor, binder/ref stamped positive, any clean negative
    if len(synth_stamped) > 0 and len(binder_stamped) > 0:
        real_stamped = np.concatenate([binder_stamped, ref_stamped]) if len(ref_stamped) > 0 else binder_stamped
        for _ in range(n_triplets // 4):
            a = rng.choice(synth_stamped)
            p = rng.choice(real_stamped)
            n = rng.choice(all_clean)
            anchors.append(a)
            positives.append(p)
            negatives.append(n)

    return np.array(anchors), np.array(positives), np.array(negatives)


def train_projection(features, labels, domains, n_epochs=200, lr=1e-3, margin=0.3,
                     out_dim=128, n_triplets=10000):
    """Train a linear projection using triplet loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    anchors, positives, negatives = mine_triplets(features, labels, domains, n_triplets)
    if len(anchors) == 0:
        logger.error("No triplets mined, cannot train")
        return None

    logger.info("Training projection with %d triplets, margin=%.2f, out_dim=%d",
                len(anchors), margin, out_dim)

    model = LinearProjection(in_dim=features.shape[1], out_dim=out_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin)

    feat_tensor = torch.tensor(features, dtype=torch.float32).to(device)

    # Training loop
    batch_size = 512
    n_batches = max(1, len(anchors) // batch_size)

    losses = []
    for epoch in range(n_epochs):
        # Shuffle triplets
        perm = np.random.permutation(len(anchors))
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b*batch_size:(b+1)*batch_size]
            a_idx = anchors[idx]
            p_idx = positives[idx]
            n_idx = negatives[idx]

            a_feat = model(feat_tensor[a_idx])
            p_feat = model(feat_tensor[p_idx])
            n_feat = model(feat_tensor[n_idx])

            loss = triplet_loss_fn(a_feat, p_feat, n_feat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            logger.info("Epoch %d/%d, loss=%.4f", epoch + 1, n_epochs, avg_loss)

    # Project all features
    model.eval()
    with torch.no_grad():
        projected = model(feat_tensor).cpu().numpy()

    logger.info("Final triplet loss: %.4f", losses[-1])

    # Plot loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Triplet Loss")
    ax.set_title("Triplet Loss Training Curve")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "triplet_loss_curve.png", dpi=150)
    plt.close()

    # Save the projection model
    save_path = Path("data/stamp_triplet_projection.pkl")
    import pickle
    with open(save_path, "wb") as f:
        pickle.dump({
            "weight": model[0].weight.detach().cpu().numpy(),
            "bias": model[0].bias.detach().cpu().numpy(),
            "in_dim": model[0].in_features,
            "out_dim": model[0].out_features,
        }, f)
    logger.info("Saved triplet projection to %s", save_path)

    return projected, model, losses


def evaluate_knn(features, labels, domains, k=5, desc=""):
    """Evaluate k-NN classification, report per-domain and overall accuracy."""
    labels = np.array(labels)
    domains = np.array(domains)

    # Overall leave-one-out style with stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros_like(labels)

    for train_idx, test_idx in skf.split(features, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])
        knn = KNeighborsClassifier(n_neighbors=min(k, len(train_idx)), metric="cosine")
        knn.fit(X_train, labels[train_idx])
        preds[test_idx] = knn.predict(X_test)

    overall_acc = accuracy_score(labels, preds)
    logger.info("=== k-NN Evaluation %s ===", desc)
    logger.info("Overall accuracy: %.1f%% (%d/%d)", overall_acc * 100, int(overall_acc * len(labels)), len(labels))

    # Per-domain breakdown
    for domain in ["synthetic", "reference", "binder"]:
        mask = domains == domain
        if mask.sum() == 0:
            continue
        dom_acc = accuracy_score(labels[mask], preds[mask])
        dom_n = mask.sum()
        logger.info("  %s: %.1f%% (%d/%d)", domain, dom_acc * 100, int(dom_acc * dom_n), dom_n)

    # Per-domain stamped/clean breakdown
    for domain in ["synthetic", "reference", "binder"]:
        for label, lname in [(1, "stamped"), (0, "clean")]:
            mask = (domains == domain) & (labels == label)
            if mask.sum() == 0:
                continue
            acc = accuracy_score(labels[mask], preds[mask])
            logger.info("    %s %s: %.1f%% (%d/%d)", domain, lname, acc * 100, int(acc * mask.sum()), mask.sum())

    # Train on synthetic+reference, test on binder
    binder_mask = domains == "binder"
    train_mask = ~binder_mask
    if binder_mask.sum() > 0 and train_mask.sum() > 0:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_mask])
        X_test = scaler.transform(features[binder_mask])
        knn = KNeighborsClassifier(n_neighbors=min(k, train_mask.sum()), metric="cosine")
        knn.fit(X_train, labels[train_mask])
        binder_preds = knn.predict(X_test)
        binder_acc = accuracy_score(labels[binder_mask], binder_preds)
        logger.info("  Train(synth+ref)->Test(binder): %.1f%% (%d/%d)",
                    binder_acc * 100, int(binder_acc * binder_mask.sum()), binder_mask.sum())
        # Show per-sample predictions for binder
        binder_indices = np.where(binder_mask)[0]
        for i, (bi, pred, true) in enumerate(zip(binder_indices, binder_preds, labels[binder_mask])):
            status = "OK" if pred == true else "WRONG"
            logger.info("    binder[%d]: true=%s pred=%s %s", i,
                       "stamped" if true else "clean", "stamped" if pred else "clean", status)

    return overall_acc


def analyze_domain_distances(features, labels, domains):
    """Analyze inter/intra-domain distances to understand the gap."""
    labels = np.array(labels)
    domains = np.array(domains)

    logger.info("=== Domain Distance Analysis ===")

    categories = {}
    for domain in ["synthetic", "reference", "binder"]:
        for label in [0, 1]:
            mask = (domains == domain) & (labels == label)
            if mask.sum() > 0:
                key = f"{domain}_{'stamped' if label else 'clean'}"
                categories[key] = features[mask]

    # Compute pairwise centroid distances
    centroids = {k: v.mean(axis=0) for k, v in categories.items()}

    logger.info("Category sizes: %s", {k: len(v) for k, v in categories.items()})

    # Intra-class spread (mean distance from centroid)
    for k, v in categories.items():
        spread = np.mean(np.linalg.norm(v - centroids[k], axis=1))
        logger.info("  Spread(%s): %.4f", k, spread)

    # Inter-category centroid distances
    keys = sorted(centroids.keys())
    logger.info("Centroid distances:")
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            dist = np.linalg.norm(centroids[k1] - centroids[k2])
            cosine_sim = np.dot(centroids[k1], centroids[k2]) / (
                np.linalg.norm(centroids[k1]) * np.linalg.norm(centroids[k2]) + 1e-8)
            logger.info("  %s <-> %s: L2=%.4f, cos=%.4f", k1, k2, dist, cosine_sim)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load all data ──
    logger.info("Loading data from all three domains...")
    synth_imgs, synth_labels, synth_domains = load_synthetic_data()
    ref_imgs, ref_labels, ref_domains = load_reference_data()
    binder_imgs, binder_labels, binder_domains = load_binder_data()

    all_images = synth_imgs + ref_imgs + binder_imgs
    all_labels = synth_labels + ref_labels + binder_labels
    all_domains = synth_domains + ref_domains + binder_domains

    logger.info("Total: %d images (%d stamped, %d clean)",
                len(all_images), sum(all_labels), len(all_labels) - sum(all_labels))

    if len(all_images) == 0:
        logger.error("No images loaded!")
        return

    # ── Step 2: Extract DINOv2 CLS features ──
    logger.info("Extracting DINOv2 CLS features from stamp-region crops...")
    model, device = load_dino_model()
    features = extract_cls_features(model, device, all_images)
    logger.info("Feature matrix shape: %s", features.shape)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    labels_arr = np.array(all_labels)
    domains_arr = np.array(all_domains)

    # ── Step 3: Domain distance analysis ──
    analyze_domain_distances(features, all_labels, all_domains)

    # ── Step 4: t-SNE visualization (raw features) ──
    logger.info("Computing t-SNE (perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords_tsne = tsne.fit_transform(features)
    plot_embedding(coords_tsne, all_labels, all_domains,
                   "t-SNE of DINOv2 CLS (stamp-region crops) — Raw",
                   PLOTS_DIR / "tsne_raw.png")

    # ── Step 5: UMAP visualization (raw features) ──
    logger.info("Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords_umap = reducer.fit_transform(features)
    plot_embedding(coords_umap, all_labels, all_domains,
                   "UMAP of DINOv2 CLS (stamp-region crops) — Raw",
                   PLOTS_DIR / "umap_raw.png")

    # ── Step 6: k-NN accuracy on raw features ──
    raw_acc = evaluate_knn(features, all_labels, all_domains, k=5, desc="(raw DINOv2 CLS)")

    # ── Step 7: Triplet loss projection ──
    logger.info("Training triplet-loss linear projection...")
    result = train_projection(features, all_labels, all_domains,
                              n_epochs=300, lr=5e-4, margin=0.5, out_dim=128,
                              n_triplets=20000)

    if result is not None:
        projected, proj_model, losses = result

        # ── Step 8: Visualize projected features ──
        logger.info("Computing t-SNE on projected features...")
        tsne2 = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        coords_tsne_proj = tsne2.fit_transform(projected)
        plot_embedding(coords_tsne_proj, all_labels, all_domains,
                       "t-SNE of Projected Features (128-dim) — After Triplet Loss",
                       PLOTS_DIR / "tsne_projected.png")

        logger.info("Computing UMAP on projected features...")
        reducer2 = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords_umap_proj = reducer2.fit_transform(projected)
        plot_embedding(coords_umap_proj, all_labels, all_domains,
                       "UMAP of Projected Features (128-dim) — After Triplet Loss",
                       PLOTS_DIR / "umap_projected.png")

        # ── Step 9: k-NN accuracy on projected features ──
        proj_acc = evaluate_knn(projected, all_labels, all_domains, k=5,
                                desc="(projected 128-dim)")

        # Domain distance analysis after projection
        analyze_domain_distances(projected, all_labels, all_domains)

        # ── Step 10: Side-by-side comparison plot ──
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        domain_markers = {"synthetic": "o", "reference": "s", "binder": "^"}
        label_colors = {0: "#2196F3", 1: "#F44336"}
        label_names = {0: "clean", 1: "stamped"}

        for ax, coords, title in [
            (axes[0, 0], coords_tsne, "t-SNE Raw"),
            (axes[0, 1], coords_tsne_proj, "t-SNE Projected"),
            (axes[1, 0], coords_umap, "UMAP Raw"),
            (axes[1, 1], coords_umap_proj, "UMAP Projected"),
        ]:
            for domain in ["synthetic", "reference", "binder"]:
                for label in [0, 1]:
                    mask = np.array([(d == domain and l == label)
                                     for d, l in zip(all_domains, all_labels)])
                    if not mask.any():
                        continue
                    ax.scatter(
                        coords[mask, 0], coords[mask, 1],
                        c=label_colors[label],
                        marker=domain_markers[domain],
                        s=40 if domain != "binder" else 100,
                        alpha=0.6 if domain != "binder" else 1.0,
                        edgecolors="black" if domain == "binder" else "none",
                        linewidths=1.5 if domain == "binder" else 0,
                        label=f"{domain} {label_names[label]}",
                        zorder=3 if domain == "binder" else 1,
                    )
            ax.set_title(title, fontsize=13)
            ax.legend(loc="best", fontsize=8)

        fig.suptitle(f"Stamp Detection Feature Space Analysis\n"
                     f"Raw k-NN: {raw_acc*100:.1f}% | Projected k-NN: {proj_acc*100:.1f}%",
                     fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "comparison_all.png", dpi=150)
        plt.close()
        logger.info("Saved comparison plot: %s", PLOTS_DIR / "comparison_all.png")

    logger.info("Done! All plots saved to %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
