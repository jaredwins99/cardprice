#!/usr/bin/env python3
"""Fine-grained Tip-Adapter hyperparameter sweep."""

import json
import os

import numpy as np
import torch
from PIL import Image

gt = json.load(open("data/test_binder_pages/ground_truth.json"))


def gt_card_id(path):
    parts = path.split("/")
    fname = parts[-1].replace(".png", "")
    last_under = fname.rfind("_")
    return fname[:last_under] + "/" + fname[last_under + 1 :]


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
n = len(all_gt_ids)
print(f"Encoded {n} test segments")

from cardprice.ml.tip_adapter import TipAdapter

# Fine-grained sweep
print(f"\n{'beta':>6} {'tw':>8}  {'top1':>5}  {'acc':>6}  failures")
best_overall = (0, 0, 0)

for beta in [0.5, 1.0, 2.0, 3.0, 5.0, 5.5]:
    adapter = TipAdapter(beta=beta, alpha=1.0, text_weight=1.0)
    for tw in [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005]:
        correct = 0
        failures = []
        for i in range(n):
            results = adapter.predict(query_embs[i], top_k=1, text_weight=tw)
            if results[0][0] == all_gt_ids[i]:
                correct += 1
            else:
                pn = all_pages[i].split("_")[2].split(".")[0]
                failures.append(f"p{pn}/{os.path.basename(all_paths[i])}")
        acc = correct / n
        if correct > best_overall[2]:
            best_overall = (beta, tw, correct)
        fail_str = ", ".join(failures) if failures else "-"
        print(f"{beta:6.1f} {tw:8.4f}  {correct:3d}/{n:2d}  {acc:.1%}  {fail_str}")

beta_best, tw_best, correct_best = best_overall
print(f"\nBest: beta={beta_best}, tw={tw_best}, acc={correct_best}/{n} = {correct_best/n:.1%}")

# Failure details at best config
print(f"\n=== Failures at beta={beta_best}, tw={tw_best} ===")
adapter = TipAdapter(beta=beta_best, alpha=1.0, text_weight=1.0)
for i in range(n):
    results = adapter.predict(query_embs[i], top_k=10, text_weight=tw_best)
    if results[0][0] != all_gt_ids[i]:
        pn = all_pages[i].split("_")[2].split(".")[0]
        print(f"\n  FAIL p{pn}/{os.path.basename(all_paths[i])}: GT={all_gt_ids[i]}")
        for rank, (cid, score) in enumerate(results[:10]):
            marker = " <-- GT" if cid == all_gt_ids[i] else ""
            print(f"    #{rank+1}: {cid:30s} score={score:.6f}{marker}")
