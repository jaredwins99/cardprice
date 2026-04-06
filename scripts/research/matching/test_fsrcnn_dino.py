#!/usr/bin/env python3
"""Test whether FSRCNN super-resolution upscaling improves DINOv2 matching scores.

Compares three approaches for preparing binder-scanned card images for DINOv2:
  A) Direct resize to 224x224 (baseline)
  B) Bicubic 2x upscale -> resize to 224x224
  C) FSRCNN 2x upscale -> resize to 224x224

For each approach, computes the DINOv2 cosine similarity against the correct
reference card embedding from the pre-computed embedding store.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Project root
ROOT = Path(__file__).resolve().parent.parent

def main():
    # ---- Load ground truth ----
    gt_path = ROOT / "data" / "ground_truth.json"
    with open(gt_path) as f:
        gt = json.load(f)

    # ---- Load pre-computed reference embeddings ----
    ref_emb_path = ROOT / "data" / "ref_embeddings.pkl"
    with open(ref_emb_path, "rb") as f:
        ref_embeddings = pickle.load(f)
    print(f"Loaded {len(ref_embeddings)} reference embeddings")

    # ---- Collect test cards (first 10 with valid card_ids in ref embeddings) ----
    test_cards = []
    for page_dir, page_data in gt["pages"].items():
        for key, card_info in page_data.items():
            if not key.startswith("card_"):
                continue
            card_id = card_info.get("card_id", "")
            if not card_id:
                continue
            # Check image exists
            img_path = ROOT / "data" / "inbox" / page_dir / f"{key}.png"
            if not img_path.exists():
                continue
            # Check ref embedding exists
            if card_id not in ref_embeddings:
                continue
            test_cards.append({
                "name": card_info.get("name", "?"),
                "card_id": card_id,
                "image_path": str(img_path),
                "ref_embedding": ref_embeddings[card_id],
            })
            if len(test_cards) >= 10:
                break
        if len(test_cards) >= 10:
            break

    print(f"Selected {len(test_cards)} test cards:")
    for tc in test_cards:
        print(f"  {tc['name']:20s}  {tc['card_id']:20s}  {Path(tc['image_path']).parent.name}/{Path(tc['image_path']).name}")

    # ---- Load DINOv2 ----
    import torch
    from torchvision import transforms
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading DINOv2 ViT-B/14 on {device}...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.to(device).eval()

    dino_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ---- Load FSRCNN ----
    from cardprice.ml.preprocess import upscale_for_ocr, _get_fsrcnn_model
    fsrcnn_model, fsrcnn_device = _get_fsrcnn_model()
    if fsrcnn_model is not None:
        print(f"FSRCNN loaded on {fsrcnn_device}")
    else:
        print("WARNING: FSRCNN not available, will compare bicubic only")

    # ---- Helper: BGR ndarray -> DINOv2 embedding ----
    def embed_bgr(bgr_img: np.ndarray) -> np.ndarray:
        """Convert BGR ndarray to 768-dim L2-normalized DINOv2 embedding."""
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = dino_transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(tensor)
        vec = emb.cpu().numpy().astype(np.float32).squeeze()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ---- Run comparison ----
    print(f"\n{'Card':<22s} {'Direct':<10s} {'Bicubic2x':<10s} {'FSRCNN2x':<10s} {'Best':<10s}")
    print("-" * 65)

    scores_direct = []
    scores_bicubic = []
    scores_fsrcnn = []

    for tc in test_cards:
        img = cv2.imread(tc["image_path"])
        if img is None:
            print(f"  ERROR: cannot read {tc['image_path']}")
            continue
        h, w = img.shape[:2]
        ref_emb = tc["ref_embedding"]

        # A) Direct resize to 224x224
        t0 = time.time()
        emb_direct = embed_bgr(img)  # transform handles resize
        t_direct = time.time() - t0
        sim_direct = float(np.dot(emb_direct, ref_emb))

        # B) Bicubic 2x -> resize to 224x224
        t0 = time.time()
        bicubic_2x = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        emb_bicubic = embed_bgr(bicubic_2x)  # transform resizes to 224x224
        t_bicubic = time.time() - t0
        sim_bicubic = float(np.dot(emb_bicubic, ref_emb))

        # C) FSRCNN 2x -> resize to 224x224
        if fsrcnn_model is not None:
            t0 = time.time()
            from cardprice.ml.preprocess import _fsrcnn_upscale
            fsrcnn_2x = _fsrcnn_upscale(img, fsrcnn_model, fsrcnn_device)
            emb_fsrcnn = embed_bgr(fsrcnn_2x)  # transform resizes to 224x224
            t_fsrcnn = time.time() - t0
            sim_fsrcnn = float(np.dot(emb_fsrcnn, ref_emb))
        else:
            sim_fsrcnn = float("nan")

        scores_direct.append(sim_direct)
        scores_bicubic.append(sim_bicubic)
        scores_fsrcnn.append(sim_fsrcnn)

        best = "Direct"
        best_val = sim_direct
        if sim_bicubic > best_val:
            best = "Bicubic"
            best_val = sim_bicubic
        if not np.isnan(sim_fsrcnn) and sim_fsrcnn > best_val:
            best = "FSRCNN"
            best_val = sim_fsrcnn

        delta_b = sim_bicubic - sim_direct
        delta_f = sim_fsrcnn - sim_direct if not np.isnan(sim_fsrcnn) else float("nan")

        print(f"  {tc['name']:<20s} {sim_direct:+.5f}  {sim_bicubic:+.5f} ({delta_b:+.4f})  "
              f"{sim_fsrcnn:+.5f} ({delta_f:+.4f})  {best}")

    # ---- Summary ----
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    arr_d = np.array(scores_direct)
    arr_b = np.array(scores_bicubic)
    arr_f = np.array(scores_fsrcnn)

    print(f"  Direct:   mean={arr_d.mean():.5f}  std={arr_d.std():.5f}  min={arr_d.min():.5f}  max={arr_d.max():.5f}")
    print(f"  Bicubic:  mean={arr_b.mean():.5f}  std={arr_b.std():.5f}  min={arr_b.min():.5f}  max={arr_b.max():.5f}")
    if not np.any(np.isnan(arr_f)):
        print(f"  FSRCNN:   mean={arr_f.mean():.5f}  std={arr_f.std():.5f}  min={arr_f.min():.5f}  max={arr_f.max():.5f}")

    print(f"\n  Bicubic vs Direct:  mean delta = {(arr_b - arr_d).mean():+.5f}")
    if not np.any(np.isnan(arr_f)):
        print(f"  FSRCNN  vs Direct:  mean delta = {(arr_f - arr_d).mean():+.5f}")
        print(f"  FSRCNN  vs Bicubic: mean delta = {(arr_f - arr_b).mean():+.5f}")

    # Which method wins most often?
    direct_wins = sum(1 for d, b, f in zip(scores_direct, scores_bicubic, scores_fsrcnn)
                      if d >= b and d >= f)
    bicubic_wins = sum(1 for d, b, f in zip(scores_direct, scores_bicubic, scores_fsrcnn)
                       if b > d and b >= f)
    fsrcnn_wins = sum(1 for d, b, f in zip(scores_direct, scores_bicubic, scores_fsrcnn)
                      if f > d and f > b)
    print(f"\n  Win count: Direct={direct_wins}  Bicubic={bicubic_wins}  FSRCNN={fsrcnn_wins}")


if __name__ == "__main__":
    main()
