#!/usr/bin/env python3
"""Profile the card identification pipeline step by step."""

import time
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    stream=sys.stderr,
)

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

from pathlib import Path

CARD_DIR = Path('data/inbox/page_20260320_223702_cards')
PAGE_IMG = 'data/inbox/page_20260320_223702.jpg'
card_paths = sorted(CARD_DIR.glob('card_*.png'))

print(f"=== PROFILING PIPELINE ({len(card_paths)} cards) ===\n", flush=True)

# 1. Segmentation
print("--- SEGMENTATION ---", flush=True)
t0 = time.time()
from cardprice.ml.card_segmenter import segment_cards
seg = segment_cards(PAGE_IMG)
t_seg = time.time() - t0
print(f"  Segmentation: {t_seg:.1f}s ({len(seg)} cards)\n", flush=True)

# 2. Color detection (single card)
print("--- COLOR DETECTION (single card) ---", flush=True)
from cardprice.ml import _run_color_detect
t0 = time.time()
ct, cc = _run_color_detect(str(card_paths[0]))
t_color_one = time.time() - t0
print(f"  Color detect (1 card): {t_color_one:.2f}s => {ct} ({cc:.2f})", flush=True)

# Color detection all 9
t0 = time.time()
for p in card_paths:
    _run_color_detect(str(p))
t_color_all = time.time() - t0
print(f"  Color detect (9 cards serial): {t_color_all:.1f}s\n", flush=True)

# 3. Name OCR (single card)
print("--- NAME OCR (single card) ---", flush=True)
from cardprice.ml import _run_name_and_hp
t0 = time.time()
name, conf, raw, hp = _run_name_and_hp(str(card_paths[0]))
t_ocr_one = time.time() - t0
print(f"  Name OCR (1 card): {t_ocr_one:.2f}s => {name} ({conf:.2f}) hp={hp}", flush=True)

# Name OCR all 9 serial
t0 = time.time()
ocr_results = []
for i, p in enumerate(card_paths):
    tc = time.time()
    n, c, r, h = _run_name_and_hp(str(p))
    ocr_results.append((n, c, r, h))
    print(f"    card_{i:02d}: {time.time()-tc:.2f}s => {n} ({c:.2f}) hp={h}", flush=True)
t_ocr_all = time.time() - t0
print(f"  Name OCR (9 cards serial): {t_ocr_all:.1f}s\n", flush=True)

# 4. DINOv2 embeddings
print("--- DINOv2 EMBEDDINGS ---", flush=True)
from cardprice.ml.preprocess import preprocess_for_matching
from cardprice.ml.dino_matcher import extract_embedding_batch

# Preprocessing
t0 = time.time()
preproc_paths = []
for p in card_paths:
    preproc_paths.append(preprocess_for_matching(str(p)))
t_preproc = time.time() - t0
print(f"  Preprocessing (9 cards): {t_preproc:.1f}s", flush=True)

# DINOv2 batch
t0 = time.time()
dino_embs = extract_embedding_batch(preproc_paths)
t_dino_batch = time.time() - t0
print(f"  DINOv2 batch (9 cards): {t_dino_batch:.1f}s", flush=True)

# Cleanup preprocessed
for p in preproc_paths:
    try:
        os.unlink(p)
    except:
        pass
print(flush=True)

# 5. Attack OCR (single card)
print("--- ATTACK OCR ---", flush=True)
from cardprice.ml.attack_ocr import extract_attack_names_paddle
for i in range(min(3, len(card_paths))):
    t0 = time.time()
    a = extract_attack_names_paddle(str(card_paths[i]))
    print(f"  card_{i:02d}: {time.time()-t0:.2f}s => {a}", flush=True)
t_attack_one = time.time() - t0  # last one
print(flush=True)

# 6. DB candidate lookup
print("--- DB CANDIDATE LOOKUP ---", flush=True)
from cardprice.ml import _get_candidates_from_db
from cardprice.db.session import SessionLocal
with SessionLocal() as session:
    for i, (n, c, r, h) in enumerate(ocr_results):
        if n:
            t0 = time.time()
            cands = _get_candidates_from_db(name=n, hp=h, session=session)
            t_db = time.time() - t0
            print(f"  card_{i:02d}: {t_db:.3f}s => {len(cands)} candidates for {n} hp={h}", flush=True)
print(flush=True)

# 7. DINOv2 dot product matching (against candidates)
print("--- DINOv2 DOT PRODUCT (against refs) ---", flush=True)
from cardprice.ml import _dino_dot_product_against_refs
from cardprice.ml.ref_matcher import load_ref_embeddings
t0 = time.time()
ref_embs = load_ref_embeddings()
t_load_refs = time.time() - t0
print(f"  Load ref embeddings: {t_load_refs:.2f}s ({len(ref_embs)} entries)", flush=True)

with SessionLocal() as session:
    for i in range(min(3, len(card_paths))):
        n, c, r, h = ocr_results[i]
        if n:
            cands = _get_candidates_from_db(name=n, hp=h, session=session)
            t0 = time.time()
            matches = _dino_dot_product_against_refs(
                str(card_paths[i]), cands,
                query_embedding=dino_embs[i],
            )
            t_dot = time.time() - t0
            best = matches[0] if matches else ('?', 0)
            print(f"  card_{i:02d}: {t_dot:.3f}s => best={str(best[0])[:30]} ({best[1]:.3f})", flush=True)
print(flush=True)

# 8. Variant detection (single card)
print("--- VARIANT DETECTION ---", flush=True)
from cardprice.ml.variant_detector import detect_variants
with SessionLocal() as session:
    for i in range(min(3, len(card_paths))):
        n, c, r, h = ocr_results[i]
        if n:
            cands = _get_candidates_from_db(name=n, hp=h, session=session)
            matches = _dino_dot_product_against_refs(
                str(card_paths[i]), cands, query_embedding=dino_embs[i],
            )
            if matches:
                t0 = time.time()
                vr = detect_variants(str(card_paths[i]), matches[0][0], session=session)
                t_var = time.time() - t0
                print(f"  card_{i:02d}: {t_var:.2f}s => {vr}", flush=True)
print(flush=True)

# 9. Full identify_card_v2 (single card, with precomputed data)
print("--- FULL identify_card_v2 (single, precomputed) ---", flush=True)
from cardprice.ml import identify_card_v2, _scan_cache
for i in range(min(3, len(card_paths))):
    _scan_cache.clear()
    precomp = {
        'ocr_name': ocr_results[i][0],
        'ocr_conf': ocr_results[i][1],
        'ocr_raw': ocr_results[i][2],
        'hp_value': ocr_results[i][3],
    }
    t0 = time.time()
    r = identify_card_v2(
        str(card_paths[i]),
        _precomputed_ocr=precomp,
        _precomputed_dino_embedding=dino_embs[i],
    )
    t_card = time.time() - t0
    print(f"  card_{i:02d}: {t_card:.2f}s => {r.get('card_id','?')} conf={r.get('confidence',0):.2f} method={r.get('method','?')}", flush=True)
print(flush=True)

# 10. Full identify_card_v2 (single card, cold - no precomputed)
print("--- FULL identify_card_v2 (single, COLD) ---", flush=True)
_scan_cache.clear()
t0 = time.time()
r = identify_card_v2(str(card_paths[0]))
t_cold = time.time() - t0
print(f"  card_00 COLD: {t_cold:.2f}s => {r.get('card_id','?')} method={r.get('method','?')}\n", flush=True)

# 11. Full identify_page_v2
print("--- FULL identify_page_v2 ---", flush=True)
_scan_cache.clear()
t0 = time.time()
results = identify_page_v2(card_paths)
t_page = time.time() - t0
print(f"  identify_page_v2: {t_page:.1f}s", flush=True)
for i, r in enumerate(results):
    print(f"    card_{i:02d}: {str(r.get('card_id','?'))[:30]:30s} conf={r.get('confidence',0):.2f} method={r.get('method','?')}", flush=True)
print(flush=True)

# Summary
print("=" * 60, flush=True)
print("TIMING SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  Segmentation:            {t_seg:6.1f}s", flush=True)
print(f"  Color detect (1 card):   {t_color_one:6.2f}s", flush=True)
print(f"  Color detect (9 cards):  {t_color_all:6.1f}s", flush=True)
print(f"  Name OCR (1 card):       {t_ocr_one:6.2f}s", flush=True)
print(f"  Name OCR (9 cards):      {t_ocr_all:6.1f}s", flush=True)
print(f"  Preprocess (9 cards):    {t_preproc:6.1f}s", flush=True)
print(f"  DINOv2 batch (9 cards):  {t_dino_batch:6.1f}s", flush=True)
print(f"  Attack OCR (1 card):     {t_attack_one:6.2f}s", flush=True)
print(f"  Load ref embeddings:     {t_load_refs:6.2f}s", flush=True)
print(f"  identify_card_v2 COLD:   {t_cold:6.2f}s", flush=True)
print(f"  identify_page_v2 TOTAL:  {t_page:6.1f}s", flush=True)
