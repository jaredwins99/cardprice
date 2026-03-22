#!/usr/bin/env python3
"""Benchmark: can DINOv2-only matching work for slide-scan (clean card images)?

Tests the hypothesis that high-quality individual card images give DINOv2
similarity scores high enough (0.85+) to identify cards without OCR.

Approach:
1. Pick N reference card images (these are the "clean" images we'd get from slide-scan)
2. For each, compute DINOv2 embedding and search the FAISS index (all 20k cards)
3. Check if the correct card_id is rank-1 and what similarity score it gets
4. Also check top-100 FAISS results and compare scores

This simulates the best-case scenario for slide-scan: the query image IS the
reference image (or very similar). If DINOv2 can't match reference images to
themselves with high confidence, it definitely won't work on phone photos.
"""

import os
import sys
import time
import pickle
import random
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def main():
    import faiss
    from cardprice.ml.dino_matcher import extract_embedding, extract_embedding_batch

    # Load FAISS index
    print("Loading FAISS index...")
    index_path = str(PROJECT_ROOT / "data" / "dino_index.faiss")
    mapping_path = str(PROJECT_ROOT / "data" / "dino_card_ids.pkl")

    index = faiss.read_index(index_path)
    with open(mapping_path, "rb") as f:
        card_ids = pickle.load(f)

    print(f"  FAISS index: {index.ntotal} vectors, dim={index.d}")
    print(f"  Card IDs: {len(card_ids)}")

    # Pick random reference images to test
    card_images_dir = PROJECT_ROOT / "data" / "card_images"
    all_images = list(card_images_dir.rglob("*_normal.png"))
    random.seed(42)
    test_images = random.sample(all_images, min(50, len(all_images)))

    print(f"\nTesting {len(test_images)} reference card images...")
    print("=" * 70)

    # Extract embeddings in batch for speed
    t0 = time.time()
    embeddings = extract_embedding_batch([str(p) for p in test_images])
    t_embed = time.time() - t0
    print(f"Embedding extraction: {t_embed:.2f}s ({t_embed/len(test_images)*1000:.0f}ms/card)")

    # Search FAISS for each
    rank1_correct = 0
    top5_correct = 0
    top10_correct = 0
    top100_correct = 0
    similarities_correct = []
    similarities_rank1 = []
    similarities_rank1_wrong = []

    for i, (img_path, embedding) in enumerate(zip(test_images, embeddings)):
        # Extract expected card_id from filename: base1-4_normal.png -> base1-4
        # FAISS card_ids may include set prefix: "ex4/ex4-91/normal" or just "base1-4"
        fname = img_path.stem  # base1-4_normal
        expected_base = fname.replace("_normal", "")  # base1-4

        query = embedding.reshape(1, -1)
        k = 100
        distances, indices = index.search(query, k)

        results = [(card_ids[idx], float(dist)) for idx, dist in zip(indices[0], distances[0])]

        def _extract_base_id(cid):
            """Extract base card_id: 'ex4/ex4-91/normal' -> 'ex4-91', 'base1-4' -> 'base1-4'"""
            parts = cid.split("/")
            if len(parts) >= 2:
                return parts[1] if len(parts) >= 2 else parts[0]
            return cid

        # Find rank of correct card
        rank = None
        correct_sim = None
        for r, (cid, sim) in enumerate(results):
            base_cid = _extract_base_id(cid)
            if base_cid == expected_base:
                rank = r + 1
                correct_sim = sim
                break

        rank1_sim = results[0][1]
        rank1_id = results[0][0]
        rank1_base = _extract_base_id(rank1_id)

        similarities_rank1.append(rank1_sim)

        if rank1_base == expected_base:
            rank1_correct += 1
        else:
            similarities_rank1_wrong.append((expected_id, rank1_id, rank1_sim, correct_sim, rank))

        if rank is not None:
            if rank <= 5:
                top5_correct += 1
            if rank <= 10:
                top10_correct += 1
            if rank <= 100:
                top100_correct += 1
            similarities_correct.append(correct_sim)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(test_images)}...")

    n = len(test_images)
    print(f"\n{'='*70}")
    print(f"RESULTS ({n} cards)")
    print(f"{'='*70}")
    print(f"Rank-1 accuracy:   {rank1_correct}/{n} ({rank1_correct/n*100:.1f}%)")
    print(f"Top-5 accuracy:    {top5_correct}/{n} ({top5_correct/n*100:.1f}%)")
    print(f"Top-10 accuracy:   {top10_correct}/{n} ({top10_correct/n*100:.1f}%)")
    print(f"Top-100 accuracy:  {top100_correct}/{n} ({top100_correct/n*100:.1f}%)")

    if similarities_correct:
        sims = np.array(similarities_correct)
        print(f"\nCorrect-card similarity stats:")
        print(f"  Mean:   {sims.mean():.4f}")
        print(f"  Median: {np.median(sims):.4f}")
        print(f"  Min:    {sims.min():.4f}")
        print(f"  Max:    {sims.max():.4f}")
        print(f"  >0.85:  {(sims > 0.85).sum()}/{len(sims)}")
        print(f"  >0.90:  {(sims > 0.90).sum()}/{len(sims)}")
        print(f"  >0.95:  {(sims > 0.95).sum()}/{len(sims)}")

    r1_sims = np.array(similarities_rank1)
    print(f"\nRank-1 similarity stats (all queries):")
    print(f"  Mean:   {r1_sims.mean():.4f}")
    print(f"  Median: {np.median(r1_sims):.4f}")
    print(f"  Min:    {r1_sims.min():.4f}")

    if similarities_rank1_wrong:
        print(f"\nRank-1 WRONG matches ({len(similarities_rank1_wrong)}):")
        for expected, got, sim, correct_sim, rank in similarities_rank1_wrong[:10]:
            rank_str = f"rank {rank}" if rank else "not in top-100"
            csim_str = f"{correct_sim:.4f}" if correct_sim else "N/A"
            print(f"  Expected {expected}, got {got} (sim={sim:.4f}), correct {rank_str} (sim={csim_str})")

    # Timing estimate for slide-scan
    print(f"\n{'='*70}")
    print("TIMING ESTIMATE FOR SLIDE-SCAN")
    print(f"{'='*70}")
    print(f"DINOv2 embedding extraction: {t_embed/len(test_images)*1000:.0f}ms/card")
    print(f"FAISS top-100 search: ~1ms/card (negligible)")
    total_9 = t_embed / len(test_images) * 9
    print(f"9-card slide-scan (DINOv2-only): ~{total_9:.1f}s")
    print(f"vs full pipeline (OCR+DINOv2): ~13-38s per page")

    # Verdict
    print(f"\n{'='*70}")
    if rank1_correct / n >= 0.95:
        print("VERDICT: DINOv2-only is viable for slide-scan!")
        print("  Clean reference images match at high accuracy.")
        print("  Phone photos will be slightly worse but should still work.")
    elif rank1_correct / n >= 0.80:
        print("VERDICT: DINOv2-only is marginal for slide-scan.")
        print("  Works for most cards but needs OCR fallback for edge cases.")
    else:
        print("VERDICT: DINOv2-only is NOT viable for slide-scan.")
        print("  Even clean reference images don't match reliably.")
        print("  Full pipeline (OCR + DINOv2) is still needed.")


def benchmark_real_scans():
    """Test DINOv2-only matching on real binder scan segmented images.

    Uses ground truth data to check if FAISS rank-1 matches the correct card.
    These images are from phone binder page photos (segmented), which are lower
    quality than what slide-scan would produce but still realistic.
    """
    import json
    import faiss
    from cardprice.ml.dino_matcher import extract_embedding_batch

    print("\n" + "=" * 70)
    print("BENCHMARK: Real scanned card images vs FAISS index")
    print("=" * 70)

    # Load FAISS
    index_path = str(PROJECT_ROOT / "data" / "dino_index.faiss")
    mapping_path = str(PROJECT_ROOT / "data" / "dino_card_ids.pkl")
    index = faiss.read_index(index_path)
    with open(mapping_path, "rb") as f:
        card_ids = pickle.load(f)

    # Load ground truth
    with open(PROJECT_ROOT / "data" / "ground_truth.json") as f:
        gt = json.load(f)

    # Build list of (image_path, expected_card_id) pairs
    test_pairs = []
    inbox = PROJECT_ROOT / "data" / "inbox"
    for page_dir, page_data in gt["pages"].items():
        for key, card_info in page_data.items():
            if not key.startswith("card_"):
                continue
            if not isinstance(card_info, dict) or "card_id" not in card_info:
                continue
            img_path = inbox / page_dir / f"{key}.png"
            if img_path.is_file():
                # Strip variant suffix: "neo1-53/normal" -> "neo1-53"
                expected = card_info["card_id"].split("/")[0]
                test_pairs.append((str(img_path), expected))

    print(f"Found {len(test_pairs)} ground-truth card images")

    if not test_pairs:
        print("No images found!")
        return

    # Extract embeddings
    paths = [p for p, _ in test_pairs]
    t0 = time.time()
    embeddings = extract_embedding_batch(paths)
    t_embed = time.time() - t0
    print(f"Embedding extraction: {t_embed:.2f}s ({t_embed/len(paths)*1000:.0f}ms/card)")

    def _extract_base(cid):
        parts = cid.split("/")
        return parts[1] if len(parts) >= 2 else parts[0]

    rank1_correct = 0
    top5_correct = 0
    top10_correct = 0
    wrong_details = []
    correct_sims = []
    rank1_sims = []

    for (img_path, expected), emb in zip(test_pairs, embeddings):
        query = emb.reshape(1, -1)
        distances, indices = index.search(query, 100)
        results = [(_extract_base(card_ids[idx]), float(dist))
                   for idx, dist in zip(indices[0], distances[0])]

        rank1_sims.append(results[0][1])

        # Check rank of correct
        rank = None
        for r, (cid, sim) in enumerate(results):
            if cid == expected:
                rank = r + 1
                correct_sims.append(sim)
                break

        if results[0][0] == expected:
            rank1_correct += 1
        else:
            wrong_details.append((
                Path(img_path).parent.name + "/" + Path(img_path).name,
                expected,
                results[0][0], results[0][1],
                rank,
                correct_sims[-1] if rank else None,
            ))

        if rank and rank <= 5:
            top5_correct += 1
        if rank and rank <= 10:
            top10_correct += 1

    n = len(test_pairs)
    print(f"\nRESULTS ({n} real scanned cards)")
    print(f"Rank-1 accuracy:   {rank1_correct}/{n} ({rank1_correct/n*100:.1f}%)")
    print(f"Top-5 accuracy:    {top5_correct}/{n} ({top5_correct/n*100:.1f}%)")
    print(f"Top-10 accuracy:   {top10_correct}/{n} ({top10_correct/n*100:.1f}%)")

    if correct_sims:
        sims = np.array(correct_sims)
        print(f"\nCorrect-card similarity: mean={sims.mean():.4f}, median={np.median(sims):.4f}, min={sims.min():.4f}")

    r1 = np.array(rank1_sims)
    print(f"Rank-1 similarity: mean={r1.mean():.4f}, median={np.median(r1):.4f}, min={r1.min():.4f}")

    if wrong_details:
        print(f"\nWRONG rank-1 matches ({len(wrong_details)}):")
        for img, expected, got, sim, rank, csim in wrong_details[:15]:
            rank_str = f"rank {rank}" if rank else "not in top-100"
            csim_str = f"{csim:.4f}" if csim else "N/A"
            print(f"  {img}: expected {expected}, got {got} (sim={sim:.4f}), correct={rank_str} (sim={csim_str})")


if __name__ == "__main__":
    main()
    benchmark_real_scans()
