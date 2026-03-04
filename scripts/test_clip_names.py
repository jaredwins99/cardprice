#!/usr/bin/env python3
"""Test CLIP zero-shot Pokemon name classification on binder eval cards.

Approach:
1. Query all unique Pokemon names from dim_cards
2. Encode text prompts "a Pokemon trading card of {name}" with CLIP text encoder
3. For each eval card segment, encode with CLIP image encoder
4. Find closest name by cosine similarity
5. Report accuracy (name match, ignoring set/variant)

Results saved to data/eval/clip_name_results.json
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cardprice.db.session import SessionLocal
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "openai/clip-vit-large-patch14"
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "binder_eval.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "eval" / "clip_name_results.json"

# Prompt templates to test
PROMPT_TEMPLATES = [
    "a Pokemon trading card of {}",
    "{}",
    "a photo of a {} Pokemon card",
]


def _extract_text_features(model, **inputs) -> torch.Tensor:
    out = model.get_text_features(**inputs)
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


def _extract_image_features(model, **inputs) -> torch.Tensor:
    out = model.get_image_features(**inputs)
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


def get_unique_names() -> list[str]:
    """Get all unique Pokemon names from dim_cards."""
    session = SessionLocal()
    try:
        rows = session.execute(
            text("SELECT DISTINCT name FROM dim_cards WHERE name IS NOT NULL ORDER BY name")
        ).fetchall()
        names = [r[0] for r in rows]
        logger.info("Found %d unique Pokemon names in DB", len(names))
        return names
    finally:
        session.close()


def build_name_embeddings(
    model, processor, names: list[str], template: str, batch_size: int = 64
) -> np.ndarray:
    """Encode name prompts with CLIP text encoder. Returns (N, D) normalized."""
    prompts = [template.format(n) for n in names]
    all_embs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            feats = _extract_text_features(model, **inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embs.append(feats.cpu().numpy())
    return np.vstack(all_embs).astype(np.float32)


def encode_image(model, processor, image_path: str) -> np.ndarray:
    """Encode a single image. Returns (D,) normalized."""
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        feats = _extract_image_features(model, **inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().squeeze()


def normalize_name(name: str) -> str:
    """Normalize name for comparison (lowercase, strip delta/ex suffixes, etc.)."""
    # Strip delta symbol and "ex" suffix for matching
    n = name.lower().strip()
    # Remove delta symbol variants
    for delta in ["δ", "delta", " δ"]:
        n = n.replace(delta, "")
    n = n.strip()
    return n


def main():
    t0 = time.time()

    # Load eval data
    with open(EVAL_PATH) as f:
        eval_data = json.load(f)

    # Get unique names from DB
    names = get_unique_names()

    # Load CLIP model
    logger.info("Loading CLIP model: %s", MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    logger.info("Model loaded in %.1fs", time.time() - t0)

    # Build text embeddings for each template
    template_embeddings = {}
    for tmpl in PROMPT_TEMPLATES:
        logger.info("Encoding names with template: %r", tmpl)
        t1 = time.time()
        embs = build_name_embeddings(model, processor, names, tmpl)
        template_embeddings[tmpl] = embs
        logger.info("  Encoded %d names in %.1fs, shape %s", len(names), time.time() - t1, embs.shape)

    # Build normalized name lookup for accuracy
    name_to_normalized = {n: normalize_name(n) for n in names}

    # Process each eval card
    results = []
    total_cards = 0
    correct_by_template = {tmpl: 0 for tmpl in PROMPT_TEMPLATES}
    correct_top5_by_template = {tmpl: 0 for tmpl in PROMPT_TEMPLATES}

    for page in eval_data["pages"]:
        seg_dir = PROJECT_ROOT / page["segments_dir"]
        for card in page["cards"]:
            # Skip empty slots
            if card["card_id"] is None:
                continue

            total_cards += 1
            gt_name = card["name"]
            gt_normalized = normalize_name(gt_name)
            seg_path = seg_dir / card["segment"]

            if not seg_path.exists():
                logger.warning("Segment not found: %s", seg_path)
                results.append({
                    "card_id": card["card_id"],
                    "gt_name": gt_name,
                    "segment": str(seg_path),
                    "error": "segment not found",
                })
                continue

            # Encode image
            img_emb = encode_image(model, processor, str(seg_path))

            card_result = {
                "card_id": card["card_id"],
                "gt_name": gt_name,
                "segment": card["segment"],
                "templates": {},
            }

            for tmpl in PROMPT_TEMPLATES:
                embs = template_embeddings[tmpl]
                # Cosine similarity (both already normalized)
                scores = embs @ img_emb
                top_indices = np.argsort(scores)[::-1][:10]

                top_matches = [
                    {"name": names[i], "score": float(scores[i])}
                    for i in top_indices
                ]

                pred_name = names[top_indices[0]]
                pred_normalized = normalize_name(pred_name)

                is_correct = pred_normalized == gt_normalized
                is_top5 = any(
                    normalize_name(names[top_indices[j]]) == gt_normalized
                    for j in range(min(5, len(top_indices)))
                )

                if is_correct:
                    correct_by_template[tmpl] += 1
                if is_top5:
                    correct_top5_by_template[tmpl] += 1

                card_result["templates"][tmpl] = {
                    "predicted": pred_name,
                    "correct": is_correct,
                    "top5_correct": is_top5,
                    "top10": top_matches,
                }

            results.append(card_result)
            logger.info(
                "Card %d/%d: %s -> %s (%s)",
                total_cards,
                27,
                gt_name,
                card_result["templates"][PROMPT_TEMPLATES[0]]["predicted"],
                "CORRECT" if card_result["templates"][PROMPT_TEMPLATES[0]]["correct"] else "WRONG",
            )

    # Summary
    elapsed = time.time() - t0
    summary = {
        "total_cards": total_cards,
        "elapsed_seconds": round(elapsed, 1),
        "model": MODEL_NAME,
        "num_unique_names": len(names),
        "templates": {},
    }
    for tmpl in PROMPT_TEMPLATES:
        acc = correct_by_template[tmpl] / total_cards if total_cards else 0
        top5_acc = correct_top5_by_template[tmpl] / total_cards if total_cards else 0
        summary["templates"][tmpl] = {
            "top1_correct": correct_by_template[tmpl],
            "top1_accuracy": round(acc, 4),
            "top5_correct": correct_top5_by_template[tmpl],
            "top5_accuracy": round(top5_acc, 4),
        }
        logger.info(
            "Template %r: top-1 %d/%d (%.1f%%), top-5 %d/%d (%.1f%%)",
            tmpl,
            correct_by_template[tmpl], total_cards, acc * 100,
            correct_top5_by_template[tmpl], total_cards, top5_acc * 100,
        )

    output = {
        "summary": summary,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", OUTPUT_PATH)
    logger.info("Total time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
