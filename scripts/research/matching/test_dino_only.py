"""Test DINOv2-only matching (no OCR) against ground truth.

Hypothesis: If card images are clean enough, DINOv2 global search against
all 20k references might be accurate enough to skip the OCR pipeline.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np

# ── Load ground truth ──────────────────────────────────────────────────────
gt = json.load(open("data/ground_truth.json"))
pages = gt["pages"]

# ── Load FAISS index + card ID mapping ─────────────────────────────────────
print("Loading FAISS index...")
index = faiss.read_index("data/dino_index.faiss")
card_ids_list = pickle.load(open("data/dino_card_ids.pkl", "rb"))
print(f"  {index.ntotal} vectors, {len(card_ids_list)} card IDs")

# ── Load DINOv2 model (lazy) ──────────────────────────────────────────────
print("Loading DINOv2 model...")
from cardprice.ml.dino_matcher import extract_embedding, extract_embedding_batch

# ── Collect all card images + expected card_ids ───────────────────────────
cards = []
for page_key, page_data in pages.items():
    for card_key in sorted(k for k in page_data if k.startswith("card_")):
        card_info = page_data[card_key]
        expected_id = card_info.get("card_id")
        if not expected_id:
            continue
        img_path = Path("data/inbox") / page_key / f"{card_key}.png"
        if not img_path.exists():
            print(f"  SKIP (missing): {img_path}")
            continue
        cards.append({
            "page": page_key,
            "card_key": card_key,
            "expected_id": expected_id,
            "name": card_info.get("name", "?"),
            "img_path": str(img_path),
        })

print(f"\nTesting {len(cards)} cards across {len(pages)} pages\n")

# ── Extract embeddings in batch ───────────────────────────────────────────
print("Extracting embeddings (batch)...")
t0 = time.time()
embeddings = extract_embedding_batch([c["img_path"] for c in cards])
embed_time = time.time() - t0
print(f"  Extracted {len(embeddings)} embeddings in {embed_time:.1f}s")

# ── FAISS search ──────────────────────────────────────────────────────────
K = 50  # top-K results
query_matrix = np.stack(embeddings).astype(np.float32)

print(f"Searching FAISS index (top-{K})...")
t0 = time.time()
D, I = index.search(query_matrix, K)
search_time = time.time() - t0
print(f"  Search completed in {search_time:.3f}s")

# ── Evaluate ──────────────────────────────────────────────────────────────
top1_correct = 0
top5_correct = 0
top10_correct = 0
top20_correct = 0
top50_correct = 0
correct_scores = []
wrong_top1_scores = []
correct_ranks = []

print(f"\n{'Card':<40} {'Expected':<30} {'Top-1 Match':<30} {'Score':>6} {'Rank':>5}")
print("-" * 115)

for i, card in enumerate(cards):
    expected = card["expected_id"]
    top_k_ids = [card_ids_list[I[i, j]] for j in range(K)]
    top_k_scores = D[i]

    # Normalize FAISS IDs: "base1/base1-26/normal" -> "base1-26/normal"
    def normalize_id(faiss_id):
        """Strip the leading set directory from FAISS card IDs."""
        parts = faiss_id.split("/")
        if len(parts) == 3:
            return f"{parts[1]}/{parts[2]}"
        return faiss_id

    # Find rank of correct answer (1-indexed, 0 = not found)
    rank = 0
    correct_score = None
    for j, cid in enumerate(top_k_ids):
        norm_cid = normalize_id(cid)
        if norm_cid == expected:
            rank = j + 1
            correct_score = float(top_k_scores[j])
            break

    # Also check by card portion (ignore variant: normal vs holofoil)
    expected_card = expected.split("/")[0]
    variant_rank = 0
    for j, cid in enumerate(top_k_ids):
        norm_cid = normalize_id(cid)
        if norm_cid.split("/")[0] == expected_card:
            if variant_rank == 0:
                variant_rank = j + 1
            break

    effective_rank = rank if rank > 0 else variant_rank

    if effective_rank == 1:
        top1_correct += 1
    if 0 < effective_rank <= 5:
        top5_correct += 1
    if 0 < effective_rank <= 10:
        top10_correct += 1
    if 0 < effective_rank <= 20:
        top20_correct += 1
    if 0 < effective_rank <= 50:
        top50_correct += 1

    if effective_rank > 0:
        correct_ranks.append(effective_rank)

    # Track scores
    if correct_score is not None:
        correct_scores.append(correct_score)

    if rank != 1 and variant_rank != 1:
        wrong_top1_scores.append(float(top_k_scores[0]))

    # Print result
    status = "OK" if effective_rank == 1 else (f"rank={effective_rank}" if effective_rank > 0 else "MISS")
    match_id = top_k_ids[0]
    score = float(top_k_scores[0])

    if status != "OK":
        print(f"{card['name']:<40} {expected:<30} {match_id:<30} {score:>6.3f} {status:>5}")

n = len(cards)
print(f"\n{'='*80}")
print(f"RESULTS ({n} cards)")
print(f"{'='*80}")
print(f"  Top-1  accuracy (exact):     {top1_correct:>3}/{n} ({100*top1_correct/n:.1f}%)")
print(f"  Top-5  accuracy:             {top5_correct:>3}/{n} ({100*top5_correct/n:.1f}%)")
print(f"  Top-10 accuracy:             {top10_correct:>3}/{n} ({100*top10_correct/n:.1f}%)")
print(f"  Top-20 accuracy:             {top20_correct:>3}/{n} ({100*top20_correct/n:.1f}%)")
print(f"  Top-50 accuracy:             {top50_correct:>3}/{n} ({100*top50_correct/n:.1f}%)")

if correct_scores:
    print(f"\n  Avg DINOv2 score (correct match in top-10): {np.mean(correct_scores):.4f}")
    print(f"  Min DINOv2 score (correct match in top-10): {np.min(correct_scores):.4f}")

if wrong_top1_scores:
    print(f"\n  Avg DINOv2 score (top-1 when WRONG):        {np.mean(wrong_top1_scores):.4f}")
    print(f"  Max DINOv2 score (top-1 when WRONG):        {np.max(wrong_top1_scores):.4f}")

print(f"\n  Embedding extraction: {embed_time:.1f}s ({embed_time/n*1000:.0f}ms/card)")
print(f"  FAISS search:         {search_time:.3f}s ({search_time/n*1000:.1f}ms/card)")

# ── Show rank distribution ────────────────────────────────────────────────
if correct_ranks:
    from collections import Counter
    rank_dist = Counter(correct_ranks)
    print(f"\n  Rank distribution (correct matches found in top-{K}):")
    for r in range(1, K + 1):
        cnt = rank_dist.get(r, 0)
        if cnt > 0:
            print(f"    Rank {r:>2}: {cnt:>3} cards {'█' * cnt}")
    missed = n - len(correct_ranks)
    if missed > 0:
        print(f"    Missed: {missed:>3} cards (not in top-{K})")
