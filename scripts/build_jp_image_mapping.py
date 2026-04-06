"""Expand data/jp_en_card_mapping.json by visually matching JP card images
against the 20k English DINOv2 reference embeddings.

Idempotent: skips JP images already mapped. Uses batched GPU inference.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_jp_map")

ROOT = Path(__file__).resolve().parent.parent
JP_DIR = ROOT / "data" / "card_images_jp"
MAPPING_PATH = ROOT / "data" / "jp_en_card_mapping.json"
REF_EMB_PATH = ROOT / "data" / "ref_embeddings.pkl"
CARD_NAMES_PATH = ROOT / "data" / "card_names.json"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_card_names() -> dict[str, str]:
    if not CARD_NAMES_PATH.exists():
        return {}
    with open(CARD_NAMES_PATH) as f:
        entries = json.load(f)
    out = {}
    for e in entries:
        cid, name = e[0], e[1]
        out[cid] = name
    return out


def find_jp_images() -> list[Path]:
    imgs = []
    for p in JP_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            imgs.append(p)
    return sorted(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="Process at most N new images (0=all)")
    ap.add_argument("--dry-run", action="store_true", help="Don't write mapping file")
    args = ap.parse_args()

    # Load existing mapping
    if MAPPING_PATH.exists():
        with open(MAPPING_PATH) as f:
            mapping = json.load(f)
    else:
        mapping = {}
    log.info("Existing mapping has %d entries", len(mapping))

    # Load ref embeddings -> matrix
    log.info("Loading reference embeddings from %s", REF_EMB_PATH)
    with open(REF_EMB_PATH, "rb") as f:
        ref = pickle.load(f)
    ref_ids = list(ref.keys())
    ref_mat = np.stack([ref[k] for k in ref_ids]).astype(np.float32)  # (N, 768)
    # Already L2-normalized
    log.info("Loaded %d reference embeddings, shape %s", len(ref_ids), ref_mat.shape)

    card_names = load_card_names()

    # Find JP images
    all_jp = find_jp_images()
    log.info("Found %d JP images total", len(all_jp))

    # Use posix-style relative paths matching the existing mapping format
    def rel_key(p: Path) -> str:
        return p.relative_to(ROOT).as_posix()

    pending = [p for p in all_jp if rel_key(p) not in mapping]
    log.info("%d already mapped, %d pending", len(all_jp) - len(pending), len(pending))

    if args.limit > 0:
        pending = pending[: args.limit]
        log.info("Limiting to %d pending images", len(pending))

    if not pending:
        log.info("Nothing to do")
        return

    # Lazy import torch / DINO
    from cardprice.ml.dino_matcher import extract_embedding_batch

    new_matches = 0
    sims_recorded: list[float] = []
    samples: list[tuple[str, str, str, float]] = []  # (jp_path, en_id, en_name, sim)
    rejected_low: list[tuple[str, str, float]] = []

    bs = args.batch_size
    for i in range(0, len(pending), bs):
        batch = pending[i : i + bs]
        try:
            embs = extract_embedding_batch(batch)
        except Exception as e:
            log.warning("Batch %d failed: %s — falling back per-image", i, e)
            from cardprice.ml.dino_matcher import extract_embedding
            embs = []
            for p in batch:
                try:
                    embs.append(extract_embedding(p))
                except Exception as ee:
                    log.warning("Skip %s: %s", p, ee)
                    embs.append(None)

        # Stack valid embeddings
        valid_idx = [j for j, e in enumerate(embs) if e is not None]
        if not valid_idx:
            continue
        Q = np.stack([embs[j] for j in valid_idx]).astype(np.float32)  # (B,768)
        # Cosine similarity = Q @ ref_mat.T (both L2-normalized)
        sims = Q @ ref_mat.T  # (B, N)
        best_idx = sims.argmax(axis=1)
        best_sim = sims[np.arange(len(valid_idx)), best_idx]

        for k, j in enumerate(valid_idx):
            jp_path = batch[j]
            sim = float(best_sim[k])
            sims_recorded.append(sim)
            best_card_id = ref_ids[int(best_idx[k])]
            key = rel_key(jp_path)
            if sim >= args.threshold:
                mapping[key] = best_card_id
                new_matches += 1
                if len(samples) < 25:
                    samples.append((key, best_card_id, card_names.get(best_card_id.split("/")[0] + "-" + best_card_id.split("-",1)[1].split("/")[0], card_names.get(best_card_id, "?")), sim))
            else:
                if len(rejected_low) < 10:
                    rejected_low.append((key, best_card_id, sim))

        log.info("Processed %d/%d  matched_so_far=%d", min(i + bs, len(pending)), len(pending), new_matches)

    # Save
    if not args.dry_run:
        # atomic write
        tmp = MAPPING_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False, sort_keys=True)
        tmp.replace(MAPPING_PATH)
        log.info("Wrote %d total entries to %s", len(mapping), MAPPING_PATH)
    else:
        log.info("[dry-run] Would write %d total entries", len(mapping))

    # Report
    arr = np.array(sims_recorded) if sims_recorded else np.array([0.0])
    print("\n===== REPORT =====")
    print(f"Pending processed:  {len(sims_recorded)}")
    print(f"New matches (>= {args.threshold}): {new_matches}")
    print(f"Total mapping size: {len(mapping)}")
    print(f"Similarity distribution: min={arr.min():.3f} p25={np.percentile(arr,25):.3f} "
          f"med={np.median(arr):.3f} p75={np.percentile(arr,75):.3f} max={arr.max():.3f}")
    bins = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
    hist, _ = np.histogram(arr, bins=bins)
    for lo, hi, c in zip(bins[:-1], bins[1:], hist):
        print(f"  [{lo:.2f}, {hi:.2f}): {c}")

    print("\nSample matches:")
    for jp, cid, name, s in samples[:15]:
        print(f"  {s:.3f}  {jp}  ->  {cid}  ({name})")

    if rejected_low:
        print("\nSample rejections (best match below threshold):")
        for jp, cid, s in rejected_low[:5]:
            print(f"  {s:.3f}  {jp}  ->  {cid}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
