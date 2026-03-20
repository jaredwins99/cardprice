#!/usr/bin/env python3
"""Master evaluation script for variant (stamp) detection.

Runs ALL available stamp classifiers on ALL labeled data (binder scans + reference
photos) and picks the winner.

Data sources:
  - Binder scans: data/condition_training/stamps_real/binder_ground_truth.jsonl
  - Reference photos: data/condition_training/stamps_real/sources.jsonl (aka labels.jsonl)

Classifiers tested:
  1. stamp_classifier.pkl  (whole-card DINOv2 features, LogReg)
  2. stamp_crop_classifier.pkl  (cropped stamp region DINOv2, LogReg)
  3. stamp_combined_classifier.pkl  (pixel + DINOv2 combined, if exists)

Usage:
    python scripts/eval_variant_detection.py
    python scripts/eval_variant_detection.py --verbose
    python scripts/eval_variant_detection.py --binder-only
    python scripts/eval_variant_detection.py --reference-only
"""

import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
BINDER_GT_PATH = PROJECT_ROOT / "data" / "condition_training" / "stamps_real" / "binder_ground_truth.jsonl"
SOURCES_PATH = PROJECT_ROOT / "data" / "condition_training" / "stamps_real" / "sources.jsonl"
BINDER_IMG_DIR = PROJECT_ROOT / "data" / "inbox"
REFERENCE_IMG_DIR = PROJECT_ROOT / "data" / "condition_training" / "stamps_real"

CLASSIFIER_PATHS = {
    "whole_card": PROJECT_ROOT / "data" / "stamp_classifier.pkl",
    "crop": PROJECT_ROOT / "data" / "stamp_crop_classifier.pkl",
    "combined": PROJECT_ROOT / "data" / "stamp_combined_classifier.pkl",
}


# ============================================================
# Data loading
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping blank/comment lines."""
    entries = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping {path.name}:{lineno}: {e}")
                continue
            entries.append(obj)
    return entries


def load_binder_data() -> list[dict]:
    """Load binder ground truth with resolved image paths.

    Deduplicates by image path (last entry wins), matching the convention
    used in training scripts.
    """
    entries = load_jsonl(BINDER_GT_PATH)
    seen = {}  # image_name -> dict
    for e in entries:
        img_path = BINDER_IMG_DIR / e["image"]
        if not img_path.exists():
            logger.warning("Binder image not found: %s", img_path)
            continue
        seen[e["image"]] = {
            "image_path": str(img_path),
            "image_name": e["image"],
            "card_name": e.get("card_name", ""),
            "set_id": e.get("set_id", ""),
            "gt_stamped": bool(e["stamped"]),
            "gt_variant": e.get("variant", "stamped" if e["stamped"] else "normal"),
            "source": "binder",
            "note": e.get("note", ""),
        }
    return list(seen.values())


def load_reference_data() -> list[dict]:
    """Load reference photo data with resolved image paths."""
    entries = load_jsonl(SOURCES_PATH)
    result = []
    for e in entries:
        img_path = REFERENCE_IMG_DIR / e["image"]
        if not img_path.exists():
            logger.warning("Reference image not found: %s", img_path)
            continue
        result.append({
            "image_path": str(img_path),
            "image_name": e["image"],
            "card_name": e.get("card_name", ""),
            "set_id": e.get("set_id", ""),
            "gt_stamped": bool(e["stamped"]),
            "gt_variant": "stamped" if e["stamped"] else "normal",
            "source": "reference",
            "note": "",
        })
    return result


# ============================================================
# Classifier wrappers
# ============================================================

# Lazy-loaded DINOv2 model
_dino_model = None
_dino_device = None


def _get_dino():
    global _dino_model, _dino_device
    if _dino_model is None:
        import torch
        _dino_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading DINOv2 on {_dino_device}...")
        _dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        _dino_model.to(_dino_device)
        _dino_model.eval()
        print("DINOv2 loaded.")
    return _dino_model, _dino_device


def run_whole_card_classifier(image_path: str, clf_data: dict) -> dict:
    """Run the whole-card stamp classifier (stamp_classifier.pkl)."""
    from cardprice.ml.stamp_classifier import (
        _extract_features, _build_feature_vector,
    )
    cls_token, patch_tokens = _extract_features(image_path)
    feature_type = clf_data["feature_type"]
    X = _build_feature_vector(cls_token, patch_tokens, feature_type, clf_data)

    clf = clf_data["model"]
    model_type = clf_data.get("model_type", "lr")

    if model_type == "lr":
        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]
        stamp_prob = float(proba[1])
        is_stamped = bool(pred == 1)
        confidence = float(max(proba))
    elif model_type == "mlp":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf.to(device)
        clf.eval()
        with torch.no_grad():
            logit = clf(torch.tensor(X, dtype=torch.float32).to(device)).squeeze()
            stamp_prob = float(torch.sigmoid(logit).cpu())
        is_stamped = stamp_prob > 0.5
        confidence = stamp_prob if is_stamped else 1.0 - stamp_prob
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return {"stamped": is_stamped, "confidence": confidence, "stamp_probability": stamp_prob}


def run_crop_classifier(image_path: str, clf_data: dict) -> dict:
    """Run the crop-based stamp classifier (stamp_crop_classifier.pkl)."""
    import torch
    from PIL import Image
    from torchvision import transforms

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]
    _transform_crop_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    model, device = _get_dino()

    img = Image.open(image_path).convert("RGB")
    # Crop stamp region: x=[50%,95%], y=[40%,72%]
    w, h = img.size
    crop = img.crop((int(w * 0.50), int(h * 0.40), int(w * 0.95), int(h * 0.72)))
    tensor = _transform_crop_224(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_out = model(tensor)

    cls_np = cls_out.cpu().numpy().astype(np.float32).squeeze()
    norm = np.linalg.norm(cls_np)
    if norm > 0:
        cls_np /= norm

    X = cls_np.reshape(1, -1)

    # Apply scaler if present
    scaler = clf_data.get("scaler")
    if scaler is not None:
        X = scaler.transform(X)

    clf = clf_data["model"]
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    stamp_prob = float(proba[1])
    is_stamped = bool(pred == 1)
    confidence = float(max(proba))

    return {"stamped": is_stamped, "confidence": confidence, "stamp_probability": stamp_prob}


def run_combined_classifier(image_path: str, clf_data: dict) -> dict:
    """Run the combined stamp classifier.

    Supports multiple model_types saved by train_combined_stamp.py:
      - lr_combined: single LR on DINOv2 CLS (optionally PCA-reduced)
      - ensemble_prob_avg: separate pixel LR + DINOv2 LR, average probs
    """
    import cv2
    import torch
    from PIL import Image
    from torchvision import transforms

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]
    _transform_crop_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    model_type = clf_data.get("model_type", "lr_combined")

    # Extract DINOv2 CLS from stamp crop region
    model, device = _get_dino()
    img_pil = Image.open(image_path).convert("RGB")
    pw, ph = img_pil.size
    # Crop region: x=[55%,90%], y=[45%,70%] (matching train_combined_stamp.py)
    crop = img_pil.crop((int(pw * 0.55), int(ph * 0.45), int(pw * 0.90), int(ph * 0.70)))
    tensor = _transform_crop_224(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_out = model(tensor)

    cls_np = cls_out.cpu().numpy().astype(np.float32).squeeze()
    norm = np.linalg.norm(cls_np)
    if norm > 0:
        cls_np /= norm

    dino_feat = cls_np.reshape(1, -1)  # (1, 768)

    # Apply PCA if present
    pca = clf_data.get("pca")
    if pca is not None:
        dino_feat = pca.transform(dino_feat)

    if model_type == "lr_combined":
        # Single LR on (optionally PCA-reduced) DINOv2 features
        X = dino_feat
        scaler = clf_data.get("scaler")
        if scaler is not None:
            X = scaler.transform(X)

        clf = clf_data["model"]
        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]
        stamp_prob = float(proba[1])
        is_stamped = bool(pred == 1)
        confidence = float(max(proba))

    elif model_type == "ensemble_prob_avg":
        # Pixel features
        img_bgr = cv2.imread(image_path)
        h, w = img_bgr.shape[:2]
        stamp_crop_cv = img_bgr[int(h * 0.45):int(h * 0.70), int(w * 0.55):int(w * 0.90)]
        control_crop_cv = img_bgr[int(h * 0.45):int(h * 0.70), int(w * 0.10):int(w * 0.45)]

        def edge_density(gray):
            edges = cv2.Canny(gray, 50, 150)
            return float(np.mean(edges > 0))

        def laplacian_var(gray):
            return float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))

        stamp_gray = cv2.cvtColor(stamp_crop_cv, cv2.COLOR_BGR2GRAY)
        control_gray = cv2.cvtColor(control_crop_cv, cv2.COLOR_BGR2GRAY)

        stamp_hsv = cv2.cvtColor(stamp_crop_cv, cv2.COLOR_BGR2HSV)
        control_hsv = cv2.cvtColor(control_crop_cv, cv2.COLOR_BGR2HSV)

        s_ed = edge_density(stamp_gray)
        c_ed = edge_density(control_gray)
        pixel_feats = np.array([
            s_ed,
            c_ed,
            s_ed / max(c_ed, 1e-6),
            laplacian_var(stamp_gray),
            laplacian_var(control_gray),
            laplacian_var(stamp_gray) / max(laplacian_var(control_gray), 1e-6),
            float(np.std(stamp_gray)),
            float(np.std(control_gray)),
            float(np.std(stamp_hsv[:, :, 0])),
            float(np.std(control_hsv[:, :, 0])),
            float(np.std(stamp_hsv[:, :, 1])),
        ], dtype=np.float32).reshape(1, -1)

        # Pixel prob
        X_pixel = pixel_feats
        sp = clf_data.get("scaler_pixel")
        if sp is not None:
            X_pixel = sp.transform(X_pixel)
        prob_pixel = clf_data["clf_pixel"].predict_proba(X_pixel)[0, 1]

        # DINOv2 prob
        X_dino = dino_feat
        sd = clf_data.get("scaler_dino")
        if sd is not None:
            X_dino = sd.transform(X_dino)
        prob_dino = clf_data["clf_dino"].predict_proba(X_dino)[0, 1]

        weight = clf_data.get("weight_pixel", 0.5)
        stamp_prob = float(weight * prob_pixel + (1 - weight) * prob_dino)
        is_stamped = stamp_prob > 0.5
        confidence = stamp_prob if is_stamped else 1.0 - stamp_prob
    else:
        raise ValueError(f"Unknown model_type for combined: {model_type}")

    return {"stamped": is_stamped, "confidence": confidence, "stamp_probability": stamp_prob}


# Map classifier name -> runner function
CLASSIFIER_RUNNERS = {
    "whole_card": run_whole_card_classifier,
    "crop": run_crop_classifier,
    "combined": run_combined_classifier,
}


# ============================================================
# Metrics
# ============================================================

def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy, precision, recall, F1 from result dicts."""
    if not results:
        return {"n": 0}

    n = len(results)
    correct = sum(1 for r in results if r["correct"])

    tp = sum(1 for r in results if r["gt_stamped"] and r["pred_stamped"])
    fp = sum(1 for r in results if not r["gt_stamped"] and r["pred_stamped"])
    fn = sum(1 for r in results if r["gt_stamped"] and not r["pred_stamped"])
    tn = sum(1 for r in results if not r["gt_stamped"] and not r["pred_stamped"])

    accuracy = correct / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "n": n, "correct": correct, "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


# ============================================================
# Main evaluation
# ============================================================

def evaluate_classifier(clf_name: str, clf_data: dict, data: list[dict],
                        verbose: bool = False) -> list[dict]:
    """Run a classifier on all data entries, return per-card results."""
    runner = CLASSIFIER_RUNNERS[clf_name]
    results = []

    for entry in data:
        t0 = time.time()
        try:
            pred = runner(entry["image_path"], clf_data)
        except Exception as e:
            logger.error("Error on %s: %s", entry["image_name"], e)
            pred = {"stamped": False, "confidence": 0.0, "stamp_probability": 0.0}
        elapsed_ms = (time.time() - t0) * 1000

        correct = pred["stamped"] == entry["gt_stamped"]
        result = {
            "image_name": entry["image_name"],
            "card_name": entry["card_name"],
            "set_id": entry["set_id"],
            "source": entry["source"],
            "gt_stamped": entry["gt_stamped"],
            "pred_stamped": pred["stamped"],
            "stamp_probability": pred["stamp_probability"],
            "confidence": pred["confidence"],
            "correct": correct,
            "time_ms": elapsed_ms,
            "note": entry.get("note", ""),
        }
        results.append(result)

        if verbose:
            status = "OK" if correct else "WRONG"
            gt_label = "stamped" if entry["gt_stamped"] else "clean"
            pred_label = "stamped" if pred["stamped"] else "clean"
            print(f"  [{status:5s}] {entry['image_name']:55s} "
                  f"gt={gt_label:7s} pred={pred_label:7s} "
                  f"prob={pred['stamp_probability']:.3f} "
                  f"conf={pred['confidence']:.3f} "
                  f"{elapsed_ms:.0f}ms")

    return results


def print_confusion_matrix(metrics: dict, label: str = ""):
    """Print a 2x2 confusion matrix."""
    if label:
        print(f"\n  Confusion Matrix ({label}):")
    else:
        print(f"\n  Confusion Matrix:")
    print(f"                      Predicted")
    print(f"                      Clean    Stamped")
    print(f"    Actual Clean    {metrics['tn']:5d}    {metrics['fp']:5d}")
    print(f"    Actual Stamped  {metrics['fn']:5d}    {metrics['tp']:5d}")


def print_errors(results: list[dict], label: str = ""):
    """Print false positives and false negatives."""
    fps = [r for r in results if not r["gt_stamped"] and r["pred_stamped"]]
    fns = [r for r in results if r["gt_stamped"] and not r["pred_stamped"]]

    if fps:
        print(f"\n  False Positives (clean -> stamped): {len(fps)}")
        for r in fps:
            print(f"    - {r['image_name']:55s} prob={r['stamp_probability']:.3f}  "
                  f"{r['card_name']}")
    if fns:
        print(f"\n  False Negatives (stamped -> clean): {len(fns)}")
        for r in fns:
            note = f"  [{r['note']}]" if r['note'] else ""
            print(f"    - {r['image_name']:55s} prob={r['stamp_probability']:.3f}  "
                  f"{r['card_name']}{note}")


def find_hard_cards(all_clf_results: dict[str, list[dict]]) -> list[dict]:
    """Find cards that are wrong across ALL classifiers."""
    # Build image_name -> {clf_name: correct}
    card_map = defaultdict(dict)
    card_info = {}
    for clf_name, results in all_clf_results.items():
        for r in results:
            key = r["image_name"]
            card_map[key][clf_name] = r["correct"]
            card_info[key] = {
                "card_name": r["card_name"],
                "gt_stamped": r["gt_stamped"],
                "source": r["source"],
                "note": r.get("note", ""),
            }

    hard = []
    for img_name, clf_correct in card_map.items():
        n_correct = sum(1 for v in clf_correct.values() if v)
        n_total = len(clf_correct)
        if n_correct < n_total:  # wrong in at least one classifier
            info = card_info[img_name]
            hard.append({
                "image_name": img_name,
                "card_name": info["card_name"],
                "gt_stamped": info["gt_stamped"],
                "source": info["source"],
                "note": info.get("note", ""),
                "correct_count": n_correct,
                "total_classifiers": n_total,
                "per_clf": clf_correct,
            })

    hard.sort(key=lambda x: x["correct_count"])
    return hard


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Master variant detection evaluation")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--binder-only", action="store_true", help="Only evaluate binder scans")
    parser.add_argument("--reference-only", action="store_true", help="Only evaluate reference photos")
    args = parser.parse_args()

    # Load data
    print("Loading ground truth data...")
    binder_data = load_binder_data()
    reference_data = load_reference_data()
    print(f"  Binder scans:     {len(binder_data)} images")
    print(f"  Reference photos: {len(reference_data)} images")

    if args.binder_only:
        all_data = binder_data
        data_label = "binder only"
    elif args.reference_only:
        all_data = reference_data
        data_label = "reference only"
    else:
        all_data = binder_data + reference_data
        data_label = "all"

    n_stamped = sum(1 for d in all_data if d["gt_stamped"])
    n_clean = len(all_data) - n_stamped
    print(f"  Evaluating:       {len(all_data)} images ({data_label})")
    print(f"  Distribution:     {n_stamped} stamped, {n_clean} clean")

    # Load classifiers
    classifiers = {}
    for name, path in CLASSIFIER_PATHS.items():
        if path.exists():
            print(f"  Loading classifier: {name} ({path.name})")
            with open(path, "rb") as f:
                classifiers[name] = pickle.load(f)
            ft = classifiers[name].get("feature_type", "?")
            mt = classifiers[name].get("model_type", "?")
            print(f"    feature_type={ft}, model_type={mt}")
        else:
            print(f"  Classifier not found: {name} ({path.name}) -- skipping")

    if not classifiers:
        print("ERROR: No classifiers found. Nothing to evaluate.")
        sys.exit(1)

    # Run evaluation for each classifier
    all_clf_results = {}
    all_clf_metrics = {}

    for clf_name, clf_data in classifiers.items():
        print(f"\n{'='*80}")
        print(f"EVALUATING: {clf_name} ({clf_data.get('feature_type', '?')})")
        print(f"{'='*80}")

        results = evaluate_classifier(clf_name, clf_data, all_data, verbose=args.verbose)
        all_clf_results[clf_name] = results

        # Overall metrics
        metrics = compute_metrics(results)
        all_clf_metrics[clf_name] = {"overall": metrics}
        print(f"\n  Overall: {metrics['correct']}/{metrics['n']} "
              f"({metrics['accuracy']:.1%})  "
              f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"F1={metrics['f1']:.3f}")
        print_confusion_matrix(metrics)
        print_errors(results)

        # Per-source breakdown
        for source_label in ["binder", "reference"]:
            source_results = [r for r in results if r["source"] == source_label]
            if not source_results:
                continue
            m = compute_metrics(source_results)
            all_clf_metrics[clf_name][source_label] = m
            print(f"\n  {source_label.title()}: {m['correct']}/{m['n']} "
                  f"({m['accuracy']:.1%})  "
                  f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
                  f"F1={m['f1']:.3f}")
            print_confusion_matrix(m, source_label)
            print_errors(source_results, source_label)

        # Per-set breakdown
        set_results = defaultdict(list)
        for r in results:
            set_results[r["set_id"]].append(r)

        if len(set_results) > 1:
            print(f"\n  Per-set breakdown:")
            print(f"  {'Set':<10s} {'N':>4s} {'Correct':>8s} {'Acc':>7s} {'P':>6s} {'R':>6s} {'F1':>6s}")
            print(f"  {'-'*50}")
            for set_id in sorted(set_results.keys()):
                sr = set_results[set_id]
                m = compute_metrics(sr)
                print(f"  {set_id or '(none)':<10s} {m['n']:4d} "
                      f"{m['correct']:4d}/{m['n']:<3d} "
                      f"{m['accuracy']:6.1%} "
                      f"{m['precision']:5.3f} {m['recall']:5.3f} {m['f1']:5.3f}")

    # ============================================================
    # Per-card results table (all classifiers side by side)
    # ============================================================
    print(f"\n{'='*120}")
    print("PER-CARD RESULTS TABLE")
    print(f"{'='*120}")

    clf_names = sorted(classifiers.keys())
    header_parts = [f"{'Image':<55s}", f"{'Card':<25s}", "GT    ", "Src   "]
    for cn in clf_names:
        header_parts.append(f"{cn:>12s}")
    print("  ".join(header_parts))
    print("-" * 120)

    # Index results by image_name per classifier
    clf_by_image = {cn: {} for cn in clf_names}
    for cn in clf_names:
        for r in all_clf_results[cn]:
            clf_by_image[cn][r["image_name"]] = r

    all_image_names = []
    seen = set()
    for d in all_data:
        if d["image_name"] not in seen:
            all_image_names.append(d["image_name"])
            seen.add(d["image_name"])

    for img_name in all_image_names:
        # Get ground truth from first classifier's results
        info = None
        for cn in clf_names:
            if img_name in clf_by_image[cn]:
                info = clf_by_image[cn][img_name]
                break
        if info is None:
            continue

        gt_label = "stamp" if info["gt_stamped"] else "clean"
        row_parts = [
            f"{img_name:<55s}",
            f"{info['card_name'][:24]:<25s}",
            f"{gt_label:<6s}",
            f"{info['source'][:5]:<6s}",
        ]
        any_wrong = False
        for cn in clf_names:
            r = clf_by_image[cn].get(img_name)
            if r is None:
                row_parts.append(f"{'N/A':>12s}")
            else:
                pred_label = "stamp" if r["pred_stamped"] else "clean"
                correct_mark = "" if r["correct"] else " X"
                if not r["correct"]:
                    any_wrong = True
                row_parts.append(f"{pred_label} {r['stamp_probability']:.2f}{correct_mark:>3s}")

        line = "  ".join(row_parts)
        if any_wrong:
            line += "  <-- WRONG"
        print(line)

    # ============================================================
    # Hard cards analysis
    # ============================================================
    hard_cards = find_hard_cards(all_clf_results)
    if hard_cards:
        print(f"\n{'='*80}")
        print(f"HARD CARDS (wrong in at least one classifier)")
        print(f"{'='*80}")

        # Cards wrong in ALL classifiers first
        always_wrong = [h for h in hard_cards if h["correct_count"] == 0]
        sometimes_wrong = [h for h in hard_cards if h["correct_count"] > 0]

        if always_wrong:
            print(f"\n  Always wrong ({len(always_wrong)} cards):")
            for h in always_wrong:
                gt_label = "stamped" if h["gt_stamped"] else "clean"
                note = f"  [{h['note']}]" if h['note'] else ""
                print(f"    {h['image_name']:55s} {h['card_name']:25s} gt={gt_label}{note}")
                for cn, correct in h["per_clf"].items():
                    print(f"      {cn}: {'CORRECT' if correct else 'WRONG'}")

        if sometimes_wrong:
            print(f"\n  Sometimes wrong ({len(sometimes_wrong)} cards):")
            for h in sometimes_wrong:
                gt_label = "stamped" if h["gt_stamped"] else "clean"
                n_right = h["correct_count"]
                n_total = h["total_classifiers"]
                note = f"  [{h['note']}]" if h['note'] else ""
                print(f"    {h['image_name']:55s} {h['card_name']:25s} "
                      f"gt={gt_label} {n_right}/{n_total} correct{note}")
                for cn, correct in h["per_clf"].items():
                    print(f"      {cn}: {'CORRECT' if correct else 'WRONG'}")

    # ============================================================
    # Summary: pick winner
    # ============================================================
    print(f"\n{'='*80}")
    print("SUMMARY -- CLASSIFIER COMPARISON")
    print(f"{'='*80}")

    print(f"\n  {'Classifier':<20s} {'Overall':>10s} {'Binder':>10s} {'Reference':>10s} "
          f"{'F1':>8s} {'Prec':>8s} {'Rec':>8s}")
    print(f"  {'-'*75}")

    best_clf = None
    best_binder_acc = -1.0

    for cn in clf_names:
        m = all_clf_metrics[cn]
        overall = m["overall"]
        binder = m.get("binder", {"n": 0, "accuracy": 0, "correct": 0})
        ref = m.get("reference", {"n": 0, "accuracy": 0, "correct": 0})

        overall_str = f"{overall['correct']}/{overall['n']} {overall['accuracy']:.1%}"
        binder_str = f"{binder.get('correct',0)}/{binder.get('n',0)} {binder.get('accuracy',0):.1%}" if binder.get("n", 0) > 0 else "N/A"
        ref_str = f"{ref.get('correct',0)}/{ref.get('n',0)} {ref.get('accuracy',0):.1%}" if ref.get("n", 0) > 0 else "N/A"

        print(f"  {cn:<20s} {overall_str:>10s} {binder_str:>10s} {ref_str:>10s} "
              f"{overall['f1']:>7.3f} {overall['precision']:>7.3f} {overall['recall']:>7.3f}")

        # Winner = best binder accuracy (our real use case), tie-break on overall
        b_acc = binder.get("accuracy", 0) if binder.get("n", 0) > 0 else 0
        if b_acc > best_binder_acc or (b_acc == best_binder_acc and overall["accuracy"] > all_clf_metrics.get(best_clf, {}).get("overall", {}).get("accuracy", 0)):
            best_binder_acc = b_acc
            best_clf = cn

    if best_clf:
        bm = all_clf_metrics[best_clf]
        overall = bm["overall"]
        binder = bm.get("binder", {})
        binder_acc = binder.get("accuracy", 0) if binder.get("n", 0) > 0 else overall["accuracy"]
        print(f"\n  ** Best approach: {best_clf} at {binder_acc:.1%} binder accuracy "
              f"({overall['accuracy']:.1%} overall) **")

    print()


if __name__ == "__main__":
    main()
