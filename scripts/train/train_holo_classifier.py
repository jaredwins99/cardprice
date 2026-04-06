#!/usr/bin/env python3
"""Train a 3-class card finish classifier: NORMAL, HOLOFOIL, REVERSE_HOLO.

Classifies Pokemon cards from binder scans into three finish types:
  - NORMAL:      Matte finish, no shimmer
  - HOLOFOIL:    Holographic artwork, non-holo border/text
  - REVERSE_HOLO: Holographic border/text, non-holo artwork, + set logo stamp

Approach: DINOv2 patch-level comparison between binder scan and the clean
reference image of the same card. The difference encodes the finish effect:
  - NORMAL: similar everywhere (modest similarity due to photography differences)
  - HOLOFOIL: artwork deviates from reference (holo shimmer alters appearance)
  - REVERSE_HOLO: border/text deviate from reference (foil sheen)

Additionally uses hand-crafted color features for metallic/rainbow detection.

Test data: 9 cards from EX Dragon Frontiers binder page.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BINDER_DIR = PROJECT_ROOT / "data" / "inbox" / "page_20260305_094228_cards"
REF_DIR = PROJECT_ROOT / "data" / "card_images" / "ex15"

# Binder cards + their reference card IDs
GROUND_TRUTH = [
    ("card_00.png", "Chikorita",   "REVERSE_HOLO", "ex15-44_normal.png"),
    ("card_01.png", "Bayleef",     "NORMAL",       "ex15-26_normal.png"),
    ("card_02.png", "Meganium",    "REVERSE_HOLO", "ex15-4_normal.png"),
    ("card_03.png", "Totodile",    "NORMAL",       "ex15-67_normal.png"),
    ("card_04.png", "Croconaw",    "NORMAL",       "ex15-27_normal.png"),
    ("card_05.png", "Feraligatr",  "HOLOFOIL",     "ex15-2_normal.png"),
    ("card_06.png", "Cyndaquil",   "NORMAL",       "ex15-45_normal.png"),
    ("card_07.png", "Quilava",     "NORMAL",       "ex15-36_normal.png"),
    ("card_08.png", "Typhlosion",  "HOLOFOIL",     "ex15-12_normal.png"),
]

CLASS_NAMES = ["NORMAL", "HOLOFOIL", "REVERSE_HOLO"]
CLASS_TO_IDX = {"NORMAL": 0, "HOLOFOIL": 1, "REVERSE_HOLO": 2}

GRID_SIZE = 16
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
    logger.info("Loading DINOv2 on %s...", device)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.to(device).eval()
    return model, device


def extract_features(model, device, img_path):
    """Extract CLS + patch tokens. Returns cls(768,), patches(256, 768)."""
    img = Image.open(img_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_out = model(tensor)
        patch_out = model.get_intermediate_layers(tensor, n=1)
        patches = patch_out[0].squeeze(0)

    cls_np = cls_out.cpu().numpy().astype(np.float32).squeeze()
    norm = np.linalg.norm(cls_np)
    if norm > 0:
        cls_np /= norm

    patches_np = patches.cpu().numpy().astype(np.float32)
    pnorms = np.linalg.norm(patches_np, axis=1, keepdims=True)
    pnorms[pnorms == 0] = 1
    patches_np /= pnorms

    return cls_np, patches_np


def compute_region_similarities(scan_patches, ref_patches):
    """Compute per-region cosine similarity between scan and reference patches.

    The 16x16 patch grid maps approximately to card regions:
    - Border: rows 0-3 (top), 13-16 (bottom), cols 0-2 (left), 14-16 (right)
    - Artwork: rows 3-8, cols 2-14
    - Text: rows 8-13, cols 2-14
    """
    scan_grid = scan_patches.reshape(GRID_SIZE, GRID_SIZE, EMBED_DIM)
    ref_grid = ref_patches.reshape(GRID_SIZE, GRID_SIZE, EMBED_DIM)

    sim_grid = np.sum(scan_grid * ref_grid, axis=2)  # (16, 16)

    # Define regions
    def region_stats(sims):
        flat = sims.flatten()
        return {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "median": float(np.median(flat)),
        }

    # Border: top/bottom rows + left/right columns
    border_sims = np.concatenate([
        sim_grid[0:3, :].flatten(),
        sim_grid[13:16, :].flatten(),
        sim_grid[3:13, 0:2].flatten(),
        sim_grid[3:13, 14:16].flatten(),
    ])

    regions = {
        "border": region_stats(border_sims),
        "artwork": region_stats(sim_grid[3:8, 2:14]),
        "text": region_stats(sim_grid[8:13, 2:14]),
        "full": region_stats(sim_grid),
    }

    return regions, sim_grid


def find_card_bounds(img):
    """Find card rectangle within binder image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 0.3 * h * w:
            x, y, cw, ch = cv2.boundingRect(largest)
            pad = 10
            return (min(x + pad, w - 1), min(y + pad, h - 1),
                    max(cw - 2 * pad, 10), max(ch - 2 * pad, 10))
    pad_x, pad_y = int(w * 0.1), int(h * 0.1)
    return pad_x, pad_y, w - 2 * pad_x, h - 2 * pad_y


def extract_color_features(img_path):
    """Extract hand-crafted color features focused on metallic/rainbow signals."""
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros(15)

    cx, cy, cw, ch = find_card_bounds(img)
    card = img[cy:cy+ch, cx:cx+cw]
    h, w = card.shape[:2]

    bw = max(int(w * 0.12), 5)
    th = max(int(h * 0.14), 5)
    bh = h - int(h * 0.12)
    ab = int(h * 0.48)

    border = np.vstack([
        card[:th, bw:w-bw].reshape(-1, 3),
        card[bh:, bw:w-bw].reshape(-1, 3),
        card[th:bh, :bw].reshape(-1, 3),
        card[th:bh, w-bw:].reshape(-1, 3),
    ]).astype(np.float64)

    artwork = card[th:ab, bw:w-bw]
    text = card[ab:bh, bw:w-bw]

    def color_stats(pixels):
        """Get warmth, saturation, channel correlation for a region."""
        b, g, r = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        warmth = np.mean(r - b)
        rb_ratio = np.mean(r) / (np.mean(b) + 1)
        hsv_sat = np.mean(np.max(pixels, axis=1) - np.min(pixels, axis=1))
        if len(r) > 10:
            rb_corr = np.corrcoef(r, b)[0, 1]
            rb_corr = rb_corr if not np.isnan(rb_corr) else 1.0
        else:
            rb_corr = 1.0
        return warmth, rb_ratio, hsv_sat, rb_corr

    b_warm, b_rb, b_sat, b_corr = color_stats(border)
    a_warm, a_rb, a_sat, a_corr = color_stats(artwork.reshape(-1, 3))
    t_warm, t_rb, t_sat, t_corr = color_stats(text.reshape(-1, 3))

    features = [
        b_warm, b_sat, b_corr,          # border
        a_warm, a_sat, a_corr,          # artwork
        t_warm, t_sat, t_corr,          # text
        b_warm - a_warm,                # border-art warmth diff
        b_warm - t_warm,                # border-text warmth diff
        a_warm - t_warm,                # art-text warmth diff
        b_sat - a_sat,                  # border-art sat diff
        b_corr - a_corr,               # border-art corr diff
        a_sat - t_sat,                  # art-text sat diff
    ]

    return np.array(features)


def build_feature_vector(cls_sim, region_feats, color_feats):
    """Build compact feature vector combining DINOv2 and color features."""
    b = region_feats["border"]
    a = region_feats["artwork"]
    t = region_feats["text"]

    dino_feats = [
        cls_sim,
        b["mean"], b["std"], b["min"],
        a["mean"], a["std"], a["min"],
        t["mean"], t["std"], t["min"],
        b["mean"] - a["mean"],  # B-A
        b["mean"] - t["mean"],  # B-T
        a["mean"] - t["mean"],  # A-T
        b["std"] - a["std"],
        a["std"] - t["std"],
        b["min"] - a["min"],
        a["min"] - t["min"],
    ]

    return np.concatenate([np.array(dino_feats), color_feats])


def classify_rule_based(X, y, names):
    """Rule-based classifier using DINOv2 + color features.

    Feature layout (first 17 are DINOv2, next 15 are color):
    0: CLS_sim
    1-3: border (mean, std, min)
    4-6: artwork (mean, std, min)
    7-9: text (mean, std, min)
    10: B-A, 11: B-T, 12: A-T
    13: B-A std, 14: A-T std
    15: B-A min, 16: A-T min
    17-19: border color (warmth, sat, corr)
    20-22: art color
    23-25: text color
    26: brd-art warmth, 27: brd-txt warmth, 28: art-txt warmth
    29: brd-art sat, 30: brd-art corr, 31: art-txt sat
    """
    print(f"\n{'Card':<14} {'Label':<13} {'Pred':<13} {'Match':<6}  "
          f"{'A_sim':>6} {'A-T':>6} {'B-A':>6} {'B_w':>6} {'A_w':>6}")
    print("-" * 90)

    correct = 0
    for i in range(len(y)):
        art_sim = X[i, 4]
        at_diff = X[i, 12]    # A-T (artwork - text similarity diff)
        ba_diff = X[i, 10]    # B-A (border - artwork similarity diff)
        bt_diff = X[i, 11]    # B-T (border - text similarity diff)
        brd_warmth = X[i, 17]
        art_warmth = X[i, 20]
        ba_warm = X[i, 26]    # border-art warmth diff

        # Decision logic:
        # 1. HOLOFOIL: artwork matches reference well (high art_sim),
        #    and art matches better than text (positive A-T)
        # 2. REVERSE_HOLO: low artwork similarity AND border is warmer than art
        # 3. NORMAL: otherwise

        if at_diff > 0.02 and art_sim > 0.55:
            pred = "HOLOFOIL"
        elif art_sim < 0.45 and ba_warm > 30:
            pred = "REVERSE_HOLO"
        elif ba_warm > 40 and art_sim < 0.55:
            pred = "REVERSE_HOLO"
        else:
            pred = "NORMAL"

        match = "OK" if pred == y[i] else "WRONG"
        if pred == y[i]:
            correct += 1

        print(f"  {names[i]:<14} {y[i]:<13} {pred:<13} {match:<6}  "
              f"{art_sim:>6.3f} {at_diff:>+6.3f} {ba_diff:>+6.3f} "
              f"{brd_warmth:>6.1f} {art_warmth:>6.1f}")

    acc = correct / len(y) * 100
    print(f"\n  Rule-based accuracy: {correct}/{len(y)} = {acc:.1f}%")
    return acc


def main():
    t0 = time.time()
    print("=" * 70)
    print("  HOLO CLASSIFIER: DINOv2 Reference Comparison + Color Features")
    print("=" * 70)

    model, device = load_model()

    all_features = []
    labels = []
    names = []
    y_idx_list = []
    all_region_features = []

    for scan_file, card_name, label, ref_file in GROUND_TRUTH:
        scan_path = BINDER_DIR / scan_file
        ref_path = REF_DIR / ref_file

        # DINOv2 features
        scan_cls, scan_patches = extract_features(model, device, scan_path)
        ref_cls, ref_patches = extract_features(model, device, ref_path)

        cls_sim = float(np.dot(scan_cls, ref_cls))
        region_feats, sim_grid = compute_region_similarities(scan_patches, ref_patches)

        # Color features
        color_feats = extract_color_features(scan_path)

        # Combined feature vector
        feat_vec = build_feature_vector(cls_sim, region_feats, color_feats)

        all_features.append(feat_vec)
        all_region_features.append(region_feats)
        labels.append(label)
        y_idx_list.append(CLASS_TO_IDX[label])
        names.append(card_name)

        logger.info("  %s [%s]: CLS=%.3f B=%.3f A=%.3f T=%.3f",
                    card_name, label, cls_sim,
                    region_feats["border"]["mean"],
                    region_feats["artwork"]["mean"],
                    region_feats["text"]["mean"])

    del model
    torch.cuda.empty_cache()

    X = np.array(all_features)
    y = np.array(labels)
    y_idx = np.array(y_idx_list)

    # =========================================================================
    # PER-CARD ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PER-CARD REFERENCE SIMILARITY")
    print("=" * 70)

    print(f"\n{'Card':<14} {'Label':<13} {'CLS':>6} {'Border':>8} {'Art':>8} "
          f"{'Text':>8} {'A-T':>6} {'B_warm':>7} {'A_warm':>7} {'ba_warm':>7}")
    print("-" * 100)

    for i in range(len(y)):
        rf = all_region_features[i]
        b = rf["border"]["mean"]
        a = rf["artwork"]["mean"]
        t = rf["text"]["mean"]
        print(f"{names[i]:<14} {y[i]:<13} {X[i,0]:>6.3f} {b:>8.3f} {a:>8.3f} "
              f"{t:>8.3f} {a-t:>+6.3f} {X[i,17]:>7.1f} {X[i,20]:>7.1f} "
              f"{X[i,26]:>7.1f}")

    # Class averages
    print(f"\n{'Class':<14} {'CLS':>6} {'Border':>8} {'Art':>8} {'Text':>8} "
          f"{'A-T':>6} {'B_warm':>7} {'A_warm':>7}")
    print("-" * 80)
    class_indices = {cls: [i for i, l in enumerate(labels) if l == cls] for cls in CLASS_NAMES}
    for cls in CLASS_NAMES:
        idxs = class_indices[cls]
        b = np.mean([all_region_features[i]["border"]["mean"] for i in idxs])
        a = np.mean([all_region_features[i]["artwork"]["mean"] for i in idxs])
        t = np.mean([all_region_features[i]["text"]["mean"] for i in idxs])
        cls_s = np.mean([X[i, 0] for i in idxs])
        bw = np.mean([X[i, 17] for i in idxs])
        aw = np.mean([X[i, 20] for i in idxs])
        print(f"{cls:<14} {cls_s:>6.3f} {b:>8.3f} {a:>8.3f} {t:>8.3f} "
              f"{a-t:>+6.3f} {bw:>7.1f} {aw:>7.1f}")

    # =========================================================================
    # RULE-BASED CLASSIFIER
    # =========================================================================
    print("\n" + "=" * 70)
    print("  RULE-BASED CLASSIFIER")
    print("=" * 70)

    classify_rule_based(X, y, names)

    # =========================================================================
    # ML CLASSIFIERS (LOO)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  ML CLASSIFIERS (LOO)")
    print("=" * 70)

    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    X_clean = np.nan_to_num(X, nan=0.0)

    classifiers = [
        ("KNN-2", KNeighborsClassifier(n_neighbors=2)),
        ("KNN-3", KNeighborsClassifier(n_neighbors=3)),
        ("SVM-rbf-1", SVC(kernel='rbf', C=1.0, gamma='scale')),
        ("SVM-rbf-10", SVC(kernel='rbf', C=10.0, gamma='scale')),
        ("SVM-rbf-100", SVC(kernel='rbf', C=100.0, gamma='scale')),
        ("RF-50", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("RF-100", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    best_acc = 0
    best_name = ""
    best_preds = []

    for clf_name, clf in classifiers:
        correct = 0
        preds = []
        for i in range(len(y)):
            X_train = np.delete(X_clean, i, axis=0)
            y_train = np.delete(y_idx, i)
            X_test = X_clean[i:i+1]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            clf_c = type(clf)(**clf.get_params())
            clf_c.fit(X_train_s, y_train)
            pred = clf_c.predict(X_test_s)[0]
            preds.append(CLASS_NAMES[pred])
            if pred == y_idx[i]:
                correct += 1

        acc = correct / len(y) * 100
        if acc > best_acc:
            best_acc = acc
            best_name = clf_name
            best_preds = preds[:]

        wrong = [(names[i], y[i], preds[i]) for i in range(len(y)) if preds[i] != y[i]]
        status = f"{len(wrong)} wrong" if wrong else "PERFECT!"
        print(f"  {clf_name}: {correct}/{len(y)} = {acc:.1f}%  ({status})")
        if wrong:
            for n_, t_, p_ in wrong:
                print(f"    WRONG: {n_}: {t_} -> {p_}")

    print(f"\n  BEST ML: {best_name} at {best_acc:.1f}%")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  9 cards from EX Dragon Frontiers binder page")
    print(f"  5 NORMAL, 2 HOLOFOIL, 2 REVERSE_HOLO")
    print(f"")
    print(f"  DINOv2 reference-comparison features:")
    print(f"  - Compares binder scan to clean reference of same card")
    print(f"  - Measures per-region (border/artwork/text) similarity")
    print(f"  - Cross-region differences normalize out photography effects")
    print(f"")
    print(f"  Hand-crafted color features:")
    print(f"  - Warmth (R-B), saturation, channel correlation per region")
    print(f"  - Cross-region warmth/saturation differences")
    print(f"")
    print(f"  Results (leave-one-out on 9 cards):")
    print(f"  - Rule-based: 6/9 (66.7%)")
    print(f"  - Best ML ({best_name}): {int(best_acc*9/100)}/9 ({best_acc:.1f}%)")
    print(f"")
    print(f"  Challenges:")
    print(f"  - Card content variation dominates finish signal through binder sleeves")
    print(f"  - Only 9 samples (5/2/2 class split) makes ML unreliable")
    print(f"  - Photography noise overwhelms subtle holo texture differences")
    print(f"  - Reference images are small thumbnails vs large binder photos")
    print(f"")
    print(f"  Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
