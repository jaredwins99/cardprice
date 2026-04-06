#!/usr/bin/env python3
"""Timing test for card identification pipeline. Run one page at a time."""
import time, json, os, sys, gc

page_dir = sys.argv[1]
label = sys.argv[2]

img = f'data/inbox/{page_dir}.jpg'
if not os.path.exists(img):
    print(f'{label}: FILE NOT FOUND {img}')
    sys.exit(1)

from cardprice.ml.card_segmenter import segment_cards
from cardprice.ml import identify_page_v2
from cardprice.db.session import SessionLocal

with SessionLocal() as s:
    t0 = time.time()

    t_seg = time.time()
    card_images = segment_cards(img)
    dt_seg = time.time() - t_seg

    t_id = time.time()
    results = identify_page_v2(card_images, session=s)
    dt_id = time.time() - t_id

    dt_total = time.time() - t0

    identified = sum(1 for r in results if r.get('card_id'))
    high_conf = sum(1 for r in results if r.get('confidence', 0) > 0.5)
    names = [r.get('name', '?') for r in results]
    confs = [round(r.get('confidence', 0), 2) for r in results]
    methods = [r.get('method', '?') for r in results]

    print(f'{label}: {dt_total:.1f}s total ({dt_seg:.1f}s seg + {dt_id:.1f}s id), {len(card_images)} cards, {identified} identified, {high_conf} high-conf')
    print(f'  Names: {names}')
    print(f'  Confs: {confs}')
    print(f'  Methods: {methods}')
