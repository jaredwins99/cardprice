#!/usr/bin/env python3
"""Test Tip-Adapter on all binder page segments."""

import json
import os
import sys

import numpy as np
import torch
from PIL import Image

# Ground truth
gt = json.load(open("data/test_binder_pages/ground_truth.json"))


def gt_card_id(path):
    parts = path.split("/")
    fname = parts[-1].replace(".png", "")
    last_under = fname.rfind("_")
    return fname[:last_under] + "/" + fname[last_under + 1 :]


# Encode all test segments from ALL pages
from cardprice.ml.clip_matcher import _get_model_and_processor, _extract_image_features

model, processor = _get_model_and_processor()

all_embeddings = []
all_gt_ids = []
all_paths = []
all_pages = []

for page_name in sorted(gt.keys()):
    page_num = page_name.split("_")[2].split(".")[0]
    seg_dir = f"data/test_binder_pages/binder_page_{page_num}_cards"
    if not os.path.isdir(seg_dir):
        continue
    expected = [gt_card_id(p) for p in gt[page_name]["cards"]]
    for i in range(9):
        seg_path = f"{seg_dir}/card_{i:02d}.png"
        if not os.path.exists(seg_path):
            continue
        img = Image.open(seg_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            feats = _extract_image_features(model, **inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu().numpy().squeeze())
        all_gt_ids.append(expected[i])
        all_paths.append(seg_path)
        all_pages.append(page_name)

query_embs = np.array(all_embeddings)
print(f"Encoded {len(all_gt_ids)} test segments from {len(set(all_pages))} pages")

# --- Baseline: raw cosine similarity ---
import pickle

img_idx = pickle.load(open("data/clip_image_index.pkl", "rb"))
img_embs = img_idx["embeddings"]
raw_ids = img_idx["card_ids"]


def normalize_cid(cid):
    parts = cid.split("/")
    return "/".join(parts[1:]) if len(parts) >= 3 else cid


img_card_ids = [normalize_cid(c) for c in raw_ids]
img_norms = img_embs / (np.linalg.norm(img_embs, axis=1, keepdims=True) + 1e-8)

print("\n=== BASELINE: Raw CLIP cosine similarity ===")
baseline_correct = 0
baseline_top5 = 0
for i, (path, gt_id) in enumerate(zip(all_paths, all_gt_ids)):
    q = query_embs[i]
    scores = img_norms @ q
    top_indices = np.argsort(scores)[::-1][:5]
    top1_id = img_card_ids[top_indices[0]]
    top5_ids = [img_card_ids[j] for j in top_indices]
    if top1_id == gt_id:
        baseline_correct += 1
    if gt_id in top5_ids:
        baseline_top5 += 1

print(f"  Top-1: {baseline_correct}/{len(all_gt_ids)} = {baseline_correct/len(all_gt_ids):.1%}")
print(f"  Top-5: {baseline_top5}/{len(all_gt_ids)} = {baseline_top5/len(all_gt_ids):.1%}")

# --- Tip-Adapter: test multiple beta values ---
from cardprice.ml.tip_adapter import TipAdapter

adapter = TipAdapter(beta=5.5, alpha=1.0, text_weight=0.0)

print(f"\n=== TIP-ADAPTER: Beta sweep (cache-only, text_weight=0) ===")
print(f"{'beta':>6}  {'top1':>5}  {'top5':>5}  {'acc':>6}")
best_beta = 5.5
best_acc = 0
for beta_val in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 50.0, 100.0]:
    correct = 0
    top5_count = 0
    for i in range(len(all_gt_ids)):
        results = adapter.predict(query_embs[i], top_k=5, beta=beta_val)
        if results[0][0] == all_gt_ids[i]:
            correct += 1
        if all_gt_ids[i] in [r[0] for r in results]:
            top5_count += 1
    acc = correct / len(all_gt_ids)
    if acc > best_acc or (acc == best_acc and beta_val < best_beta):
        best_acc = acc
        best_beta = beta_val
    print(f"{beta_val:6.1f}  {correct:3d}/{len(all_gt_ids):2d}  {top5_count:3d}/{len(all_gt_ids):2d}  {acc:.1%}")

print(f"\nBest beta: {best_beta} (accuracy: {best_acc:.1%})")

# --- Tip-Adapter with text weight ---
print(f"\n=== TIP-ADAPTER: Text weight sweep (beta={best_beta}) ===")
adapter_txt = TipAdapter(beta=best_beta, alpha=1.0, text_weight=1.0)
print(f"{'tw':>6}  {'top1':>5}  {'acc':>6}")
for tw in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    correct = 0
    for i in range(len(all_gt_ids)):
        results = adapter_txt.predict(query_embs[i], top_k=1, text_weight=tw)
        if results[0][0] == all_gt_ids[i]:
            correct += 1
    acc = correct / len(all_gt_ids)
    print(f"{tw:6.3f}  {correct:3d}/{len(all_gt_ids):2d}  {acc:.1%}")

# --- Detailed results at best beta ---
print(f"\n=== DETAILED RESULTS (beta={best_beta}, tw=0) ===")
adapter_final = TipAdapter(beta=best_beta, alpha=1.0, text_weight=0.0)
tip_correct = 0
for i, (path, gt_id, page) in enumerate(zip(all_paths, all_gt_ids, all_pages)):
    results = adapter_final.predict(query_embs[i], top_k=5)
    top1_id, top1_score = results[0]
    match = "Y" if top1_id == gt_id else "N"
    if top1_id == gt_id:
        tip_correct += 1
    top5_ids = [r[0] for r in results]
    in_top5 = "Y" if gt_id in top5_ids else "N"
    page_num = page.split("_")[2].split(".")[0]
    print(
        f"  p{page_num} {os.path.basename(path)}: {match} GT={gt_id:30s} pred={top1_id:30s} ({top1_score:.4f}) top5={in_top5}"
    )

# --- Failure analysis ---
print(f"\n=== FAILURE ANALYSIS (beta={best_beta}) ===")
for i, (path, gt_id, page) in enumerate(zip(all_paths, all_gt_ids, all_pages)):
    results = adapter_final.predict(query_embs[i], top_k=10)
    top1_id = results[0][0]
    if top1_id != gt_id:
        page_num = page.split("_")[2].split(".")[0]
        print(f"\n  FAILURE: p{page_num}/{os.path.basename(path)} GT={gt_id}")
        for rank, (cid, score) in enumerate(results[:10]):
            marker = " <-- GT" if cid == gt_id else ""
            print(f"    #{rank+1}: {cid:30s} score={score:.4f}{marker}")

# --- Comparison: does Tip-Adapter change ranking of any borderline cases? ---
print(f"\n=== RANKING COMPARISON: Baseline vs Tip-Adapter ===")
for i, (path, gt_id, page) in enumerate(zip(all_paths, all_gt_ids, all_pages)):
    q = query_embs[i]

    # Baseline ranking
    scores_base = img_norms @ q
    top_base = np.argsort(scores_base)[::-1][:5]
    base_top5 = [(img_card_ids[j], float(scores_base[j])) for j in top_base]

    # Tip-Adapter ranking
    tip_top5 = adapter_final.predict(q, top_k=5)

    base_rank = next((r+1 for r, (c, _) in enumerate(base_top5) if c == gt_id), ">5")
    tip_rank = next((r+1 for r, (c, _) in enumerate(tip_top5) if c == gt_id), ">5")

    if base_rank != tip_rank:
        page_num = page.split("_")[2].split(".")[0]
        print(f"  p{page_num}/{os.path.basename(path)}: GT={gt_id}")
        print(f"    Baseline rank: {base_rank} ({base_top5[0][0]}, {base_top5[0][1]:.4f})")
        print(f"    TipAdap rank:  {tip_rank} ({tip_top5[0][0]}, {tip_top5[0][1]:.4f})")

print(f"\n=== FINAL SUMMARY ===")
print(f"Baseline (cosine sim):    {baseline_correct}/{len(all_gt_ids)} = {baseline_correct/len(all_gt_ids):.1%}")
print(f"Tip-Adapter (best beta):  {tip_correct}/{len(all_gt_ids)} = {tip_correct/len(all_gt_ids):.1%}")
print(f"Best beta: {best_beta}")
