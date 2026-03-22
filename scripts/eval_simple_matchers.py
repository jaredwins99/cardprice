#!/usr/bin/env python3
"""Evaluate simple matching methods on corrected card images vs DINOv2.

Tests: pHash, SSIM, Template Matching (NCC), Color Histogram, ORB features.
Runs on 10 ground truth cards against all ~20k reference images.
"""

import json
import os
import time
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DATA_DIR = Path("/home/godli/cardprice/data")
INBOX_DIR = DATA_DIR / "inbox"
REF_DIR = DATA_DIR / "card_images"
GT_PATH = DATA_DIR / "ground_truth.json"

# Standard size for comparisons
SIZE = (224, 224)


def load_ground_truth(n=10):
    """Load first n ground truth cards."""
    gt = json.load(open(GT_PATH))
    cards = []
    for page, data in gt["pages"].items():
        for key, val in data.items():
            if key.startswith("card_") and isinstance(val, dict) and "card_id" in val:
                img_path = INBOX_DIR / page / f"{key}.png"
                if img_path.exists():
                    cards.append({
                        "query_path": str(img_path),
                        "card_id": val["card_id"],
                        "name": val.get("name", ""),
                        "page": page,
                        "slot": key,
                    })
                if len(cards) >= n:
                    return cards
    return cards


def get_ref_image_path(card_id: str) -> str:
    """Convert card_id like 'ex15-26/normal' to file path."""
    # card_id format: "set-num/variant" -> "set/set-num_variant.png"
    parts = card_id.split("/")
    id_part = parts[0]  # e.g. "ex15-26"
    variant = parts[1] if len(parts) > 1 else "normal"
    set_name = id_part.rsplit("-", 1)[0]  # e.g. "ex15"
    filename = f"{id_part}_{variant}.png"
    return str(REF_DIR / set_name / filename)


def load_all_ref_paths():
    """Load all reference image paths with their card_ids."""
    refs = []
    for set_dir in sorted(REF_DIR.iterdir()):
        if not set_dir.is_dir():
            continue
        for img_file in sorted(set_dir.glob("*.png")):
            # Reconstruct card_id from filename
            stem = img_file.stem  # e.g. "ex15-26_normal"
            # Split on last underscore to get variant
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                card_id = f"{parts[0]}/{parts[1]}"
            else:
                card_id = f"{stem}/normal"
            refs.append({"path": str(img_file), "card_id": card_id})
    return refs


def load_and_resize(path, size=SIZE):
    """Load image and resize to standard size."""
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.resize(img, size)


# ─── Method 1: Perceptual Hash ───

def phash_compare(query_path, ref_paths_ids, top_k=5):
    """Compute pHash distance against all references."""
    import imagehash
    query_hash = imagehash.phash(Image.open(query_path), hash_size=16)

    results = []
    for ref in ref_paths_ids:
        try:
            ref_hash = imagehash.phash(Image.open(ref["path"]), hash_size=16)
            dist = query_hash - ref_hash  # Hamming distance
            results.append((dist, ref["card_id"]))
        except Exception:
            continue

    results.sort(key=lambda x: x[0])
    return results[:top_k]


# ─── Method 2: SSIM ───

def ssim_compare(query_path, ref_paths_ids, top_k=5):
    """Compute SSIM against all references."""
    from skimage.metrics import structural_similarity

    query = load_and_resize(query_path)
    if query is None:
        return []
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

    results = []
    for ref in ref_paths_ids:
        ref_img = load_and_resize(ref["path"])
        if ref_img is None:
            continue
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        score = structural_similarity(query_gray, ref_gray)
        results.append((-score, ref["card_id"]))  # Negate for sorting (higher=better)

    results.sort(key=lambda x: x[0])
    return [(-s, cid) for s, cid in results[:top_k]]


# ─── Method 3: Template Matching (NCC) ───

def ncc_compare(query_path, ref_paths_ids, top_k=5):
    """Normalized cross-correlation."""
    query = load_and_resize(query_path)
    if query is None:
        return []
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY).astype(np.float32)

    results = []
    for ref in ref_paths_ids:
        ref_img = load_and_resize(ref["path"])
        if ref_img is None:
            continue
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Since both are same size, just compute correlation coefficient
        ncc = cv2.matchTemplate(query_gray, ref_gray, cv2.TM_CCORR_NORMED)[0][0]
        results.append((-ncc, ref["card_id"]))

    results.sort(key=lambda x: x[0])
    return [(-s, cid) for s, cid in results[:top_k]]


# ─── Method 4: Color Histogram ───

def histogram_compare(query_path, ref_paths_ids, top_k=5):
    """HSV color histogram comparison."""
    query = load_and_resize(query_path)
    if query is None:
        return []
    query_hsv = cv2.cvtColor(query, cv2.COLOR_BGR2HSV)
    hist_q = cv2.calcHist([query_hsv], [0, 1], None, [30, 30], [0, 180, 0, 256])
    cv2.normalize(hist_q, hist_q)

    results = []
    for ref in ref_paths_ids:
        ref_img = load_and_resize(ref["path"])
        if ref_img is None:
            continue
        ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
        hist_r = cv2.calcHist([ref_hsv], [0, 1], None, [30, 30], [0, 180, 0, 256])
        cv2.normalize(hist_r, hist_r)
        score = cv2.compareHist(hist_q, hist_r, cv2.HISTCMP_CORREL)
        results.append((-score, ref["card_id"]))

    results.sort(key=lambda x: x[0])
    return [(-s, cid) for s, cid in results[:top_k]]


# ─── Method 5: ORB Feature Matching ───

def orb_compare(query_path, ref_paths_ids, top_k=5):
    """ORB keypoint matching."""
    query = load_and_resize(query_path)
    if query is None:
        return []
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=500)
    kp_q, des_q = orb.detectAndCompute(query_gray, None)
    if des_q is None:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    results = []
    for ref in ref_paths_ids:
        ref_img = load_and_resize(ref["path"])
        if ref_img is None:
            continue
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        kp_r, des_r = orb.detectAndCompute(ref_gray, None)
        if des_r is None:
            continue

        matches = bf.knnMatch(des_q, des_r, k=2)
        # Lowe's ratio test
        good = 0
        for m_list in matches:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < 0.75 * n.distance:
                    good += 1
        results.append((-good, ref["card_id"]))

    results.sort(key=lambda x: x[0])
    return [(-s, cid) for s, cid in results[:top_k]]


# ─── Main ───

def main():
    print("Loading ground truth...")
    test_cards = load_ground_truth(10)
    print(f"Loaded {len(test_cards)} test cards")
    for c in test_cards:
        print(f"  {c['name']:25s} -> {c['card_id']}")

    print("\nLoading reference images...")
    all_refs = load_all_ref_paths()
    print(f"Loaded {len(all_refs)} reference images")

    # Verify correct ref images exist for our test cards
    for c in test_cards:
        ref_path = get_ref_image_path(c["card_id"])
        if not os.path.exists(ref_path):
            print(f"  WARNING: missing ref for {c['card_id']}: {ref_path}")

    methods = {
        "pHash": phash_compare,
        "NCC": ncc_compare,
        "Histogram": histogram_compare,
        "ORB": orb_compare,
        # SSIM is very slow on 20k, run on subset
    }

    # For speed, first test on a manageable subset to estimate timing
    # Then run full if feasible
    print(f"\n{'='*80}")
    print("TIMING ESTIMATE (1 card vs 100 refs)")
    print(f"{'='*80}")
    subset_100 = all_refs[:100]

    for method_name, method_fn in methods.items():
        t0 = time.time()
        method_fn(test_cards[0]["query_path"], subset_100, top_k=1)
        elapsed = time.time() - t0
        est_full = elapsed / 100 * len(all_refs)
        est_10 = est_full * 10
        print(f"  {method_name:12s}: {elapsed:.3f}s/100refs, est {est_full:.1f}s/card, {est_10:.0f}s for 10 cards")

    # SSIM timing
    t0 = time.time()
    from skimage.metrics import structural_similarity
    q = load_and_resize(test_cards[0]["query_path"])
    q_gray = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)
    for ref in subset_100[:20]:
        r = load_and_resize(ref["path"])
        if r is not None:
            r_gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
            structural_similarity(q_gray, r_gray)
    elapsed_20 = time.time() - t0
    est_ssim = elapsed_20 / 20 * len(all_refs) * 10
    print(f"  {'SSIM':12s}: {elapsed_20:.3f}s/20refs, est {est_ssim:.0f}s for 10 cards")

    # Run methods that are feasible
    print(f"\n{'='*80}")
    print("FULL EVALUATION (10 cards vs all refs)")
    print(f"{'='*80}")

    # Decide which to run fully vs on a sample
    # If > 600s estimated, use a random sample of 2000 refs
    SAMPLE_SIZE = 2000
    np.random.seed(42)

    for method_name, method_fn in methods.items():
        # Estimate time
        t0 = time.time()
        method_fn(test_cards[0]["query_path"], subset_100, top_k=1)
        time_per_ref = (time.time() - t0) / 100
        full_time = time_per_ref * len(all_refs) * 10

        if full_time > 300:
            # Use sample but ensure correct ref is included
            print(f"\n--- {method_name} (sampled {SAMPLE_SIZE} refs, full would take {full_time:.0f}s) ---")
            use_refs = list(np.random.choice(len(all_refs), SAMPLE_SIZE, replace=False))
            # Make sure correct refs are included
            ref_id_to_idx = {r["card_id"]: i for i, r in enumerate(all_refs)}
            for c in test_cards:
                if c["card_id"] in ref_id_to_idx:
                    idx = ref_id_to_idx[c["card_id"]]
                    if idx not in use_refs:
                        use_refs.append(idx)
            sampled_refs = [all_refs[i] for i in use_refs]
            is_sampled = True
        else:
            print(f"\n--- {method_name} (full {len(all_refs)} refs, est {full_time:.0f}s) ---")
            sampled_refs = all_refs
            is_sampled = False

        correct = 0
        total_time = 0
        for c in test_cards:
            t0 = time.time()
            results = method_fn(c["query_path"], sampled_refs, top_k=5)
            elapsed = time.time() - t0
            total_time += elapsed

            top1_id = results[0][1] if results else "NONE"
            match = top1_id == c["card_id"]
            if match:
                correct += 1

            top1_score = results[0][0] if results else -1
            status = "OK" if match else "FAIL"
            print(f"  [{status}] {c['name']:25s} expected={c['card_id']:20s} got={top1_id:20s} score={top1_score:.4f} ({elapsed:.1f}s)")
            if not match and results:
                # Show where correct answer ranked
                for rank, (score, cid) in enumerate(results):
                    if cid == c["card_id"]:
                        print(f"         correct answer at rank {rank+1} score={score:.4f}")
                        break

        n_refs = len(sampled_refs)
        print(f"  Accuracy: {correct}/{len(test_cards)} ({100*correct/len(test_cards):.0f}%)")
        print(f"  Avg time: {total_time/len(test_cards):.2f}s/card ({n_refs} refs)")
        if is_sampled:
            print(f"  (Sampled — true accuracy on 20k may differ)")

    # SSIM on small sample only (too slow for 20k)
    print(f"\n--- SSIM (sampled 500 refs — too slow for full) ---")
    ssim_sample_idx = list(np.random.choice(len(all_refs), 500, replace=False))
    ref_id_to_idx = {r["card_id"]: i for i, r in enumerate(all_refs)}
    for c in test_cards:
        if c["card_id"] in ref_id_to_idx:
            idx = ref_id_to_idx[c["card_id"]]
            if idx not in ssim_sample_idx:
                ssim_sample_idx.append(idx)
    ssim_refs = [all_refs[i] for i in ssim_sample_idx]

    correct = 0
    total_time = 0
    for c in test_cards:
        t0 = time.time()
        results = ssim_compare(c["query_path"], ssim_refs, top_k=5)
        elapsed = time.time() - t0
        total_time += elapsed

        top1_id = results[0][1] if results else "NONE"
        match = top1_id == c["card_id"]
        if match:
            correct += 1

        top1_score = results[0][0] if results else -1
        status = "OK" if match else "FAIL"
        print(f"  [{status}] {c['name']:25s} expected={c['card_id']:20s} got={top1_id:20s} score={top1_score:.4f} ({elapsed:.1f}s)")

    print(f"  Accuracy: {correct}/{len(test_cards)} ({100*correct/len(test_cards):.0f}%)")
    print(f"  Avg time: {total_time/len(test_cards):.2f}s/card ({len(ssim_refs)} refs)")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("Can any method replace DINOv2 for clean images? See results above.")
    print("DINOv2 reference: 100% top-1 on name-path candidates (2-20 refs), <0.1s/card with FAISS")


if __name__ == "__main__":
    main()
