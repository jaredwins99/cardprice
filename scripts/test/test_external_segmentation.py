#!/usr/bin/env python3
"""Test segmentation pipeline on external binder images."""
import sys
import os
import glob
import signal
import time

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

from cardprice.ml.card_segmenter import segment_cards

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Timed out")

base = 'data/external_test'
dirs = ['reddit_binders', 'ebay_binder', 'elkortya', 'nolanamblard', 'roboflow_binder']
images = []
for d in dirs:
    full = os.path.join(base, d)
    if os.path.isdir(full):
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            images.extend(glob.glob(os.path.join(full, ext)))

print(f'Found {len(images)} total binder images across {len(dirs)} dirs')

reddit = sorted([i for i in images if 'reddit' in i])[:20]
ebay = sorted([i for i in images if 'ebay' in i])[:20]
other = sorted([i for i in images if 'reddit' not in i and 'ebay' not in i])
sample = reddit + ebay + other
print(f'Testing {len(sample)} images: {len(reddit)} reddit, {len(ebay)} ebay, {len(other)} other')
print()

results = {'9_cards': [], '7-8_cards': [], '4-6_cards': [], '1-3_cards': [], '0_cards': [], 'error': []}
for i, img_path in enumerate(sample):
    fname = os.path.relpath(img_path, base)
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout per image
        t0 = time.time()
        cards = segment_cards(img_path)
        signal.alarm(0)
        elapsed = time.time() - t0
        n = len(cards)
        if n == 9:
            results['9_cards'].append((fname, n))
        elif n >= 7:
            results['7-8_cards'].append((fname, n))
        elif n >= 4:
            results['4-6_cards'].append((fname, n))
        elif n >= 1:
            results['1-3_cards'].append((fname, n))
        else:
            results['0_cards'].append((fname, n))
        print(f'[{i+1}/{len(sample)}] {fname}: {n} cards ({elapsed:.1f}s)')
    except TimeoutError:
        signal.alarm(0)
        results['error'].append((fname, 'TIMEOUT (>30s)'))
        print(f'[{i+1}/{len(sample)}] {fname}: TIMEOUT')
    except Exception as e:
        signal.alarm(0)
        results['error'].append((fname, str(e)[:80]))
        print(f'[{i+1}/{len(sample)}] {fname}: ERROR - {str(e)[:80]}')

print()
print('=== SUMMARY ===')
total = sum(len(v) for v in results.values())
for k, v in results.items():
    print(f'  {k}: {len(v)} ({len(v)/total*100:.0f}%)')

good = len(results['9_cards']) + len(results['7-8_cards'])
print()
print(f'Success rate (7+ cards): {good}/{total} = {good/total*100:.1f}%')
print(f'Perfect (9 cards): {len(results["9_cards"])}/{total} = {len(results["9_cards"])/total*100:.1f}%')

# Breakdown by source
for source in ['reddit', 'ebay', 'elkortya', 'nolanamblard', 'roboflow']:
    src_items = [(k, f, n) for k, items in results.items() for f, n in items if source in f]
    if src_items:
        src_9 = sum(1 for k, _, _ in src_items if k == '9_cards')
        src_good = sum(1 for k, _, _ in src_items if k in ('9_cards', '7-8_cards'))
        print(f'  {source}: {src_9}/{len(src_items)} perfect, {src_good}/{len(src_items)} 7+ cards')

if results['7-8_cards'] or results['4-6_cards'] or results['1-3_cards'] or results['0_cards']:
    print()
    print('=== NON-9 DETAILS ===')
    for k in ['7-8_cards', '4-6_cards', '1-3_cards', '0_cards']:
        for fname, n in results[k]:
            print(f'  {n} cards: {fname}')
if results['error']:
    print()
    print('=== ERRORS ===')
    for fname, err in results['error']:
        print(f'  {fname}: {err}')
