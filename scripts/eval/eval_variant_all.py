#!/usr/bin/env python3
"""Run detect_all_variants evaluation on all binder ground truth cards."""
import json, subprocess, sys, os
os.chdir('/home/godli/cardprice')

gt_lines = open('data/condition_training/stamps_real/binder_ground_truth.jsonl').readlines()
gt_cards = [json.loads(l) for l in gt_lines]

results = []
for i, g in enumerate(gt_cards):
    name = g.get('card_name', '?')
    print(f'[{i+1}/{len(gt_cards)}] {name}...', end=' ', flush=True)

    proc = subprocess.run(
        [sys.executable, 'scripts/eval/eval_variant_one.py', json.dumps(g)],
        capture_output=True, text=True, timeout=300,
    )

    if proc.returncode != 0:
        print(f'ERROR: {proc.stderr[:200]}')
        results.append({
            'name': name,
            'expected': g.get('variant', 'unknown'),
            'expected_stamped': g.get('stamped', False),
            'stamps': [],
            'flags': {},
            'checked': [],
            'time': 0,
            'details': {},
            'error': proc.stderr[:200],
        })
        continue

    # Parse last line of stdout as JSON (earlier lines may be warnings)
    out_lines = proc.stdout.strip().split('\n')
    r = json.loads(out_lines[-1])
    print(f'stamps={r["stamps_detected"]} flags={list(r.get("variant_flags",{}).keys())} ({r["time"]:.1f}s)')

    results.append({
        'name': name,
        'expected': g.get('variant', 'unknown'),
        'expected_stamped': g.get('stamped', False),
        'stamps': r['stamps_detected'],
        'flags': r.get('variant_flags', {}),
        'checked': r.get('stamps_checked', []),
        'time': r['time'],
        'details': r.get('stamp_details', {}),
    })

# Print per-card results table
print('\n' + '='*100)
print('=== Per-Card Results ===')
print(f'{"STATUS":6s} {"CARD NAME":25s} {"EXPECTED":20s} {"STAMPS DETECTED":40s} {"FLAGS":30s} {"TIME":>5s}')
print('-'*100)
for r in results:
    detected_any_stamp = bool(r['stamps'])
    if r['expected_stamped'] and detected_any_stamp:
        status = 'TP'
    elif r['expected_stamped'] and not detected_any_stamp:
        status = 'FN'
    elif not r['expected_stamped'] and detected_any_stamp:
        status = 'FP'
    else:
        status = 'TN'

    print(f'{status:6s} {r["name"]:25s} {r["expected"]:20s} {str(r["stamps"]):40s} {str(list(r["flags"].keys())):30s} {r["time"]:5.1f}s')
    if r['details']:
        for sname, sinfo in r['details'].items():
            conf = sinfo.get('confidence', 0)
            evidence = sinfo.get('evidence', '')
            ocr = sinfo.get('ocr_text', '')
            extra = f' ocr="{ocr}"' if ocr else ''
            print(f'       -> {sname}: conf={conf:.2f} evidence="{evidence}"{extra}')

# Summary stats
total = len(results)
stamped_cards = [r for r in results if r['expected_stamped']]
normal_cards = [r for r in results if not r['expected_stamped']]
true_pos = sum(1 for r in stamped_cards if r['stamps'])
false_neg = sum(1 for r in stamped_cards if not r['stamps'])
false_pos = sum(1 for r in normal_cards if r['stamps'])
true_neg = sum(1 for r in normal_cards if not r['stamps'])

print(f'\n{"="*60}')
print(f'=== Summary ===')
print(f'Total cards evaluated:    {total}')
print(f'Expected stamped:         {len(stamped_cards)}')
print(f'Expected normal:          {len(normal_cards)}')
print(f'')
print(f'True positives  (stamped & detected):   {true_pos}/{len(stamped_cards)}')
print(f'False negatives (stamped & missed):     {false_neg}/{len(stamped_cards)}')
print(f'True negatives  (normal & clean):       {true_neg}/{len(normal_cards)}')
print(f'False positives (normal & wrongly det): {false_pos}/{len(normal_cards)}')
print(f'')
if true_pos + false_pos > 0:
    prec = true_pos / (true_pos + false_pos)
    print(f'Precision: {prec:.1%}  ({true_pos}/{true_pos+false_pos})')
if true_pos + false_neg > 0:
    recall = true_pos / (true_pos + false_neg)
    print(f'Recall:    {recall:.1%}  ({true_pos}/{true_pos+false_neg})')
if total > 0:
    accuracy = (true_pos + true_neg) / total
    print(f'Accuracy:  {accuracy:.1%}  ({true_pos+true_neg}/{total})')
print(f'')
avg_t = sum(r['time'] for r in results) / max(total, 1)
print(f'Avg time per card: {avg_t:.1f}s')
print(f'Total time:        {sum(r["time"] for r in results):.1f}s')

# Detail sections for errors
if false_neg > 0:
    print(f'\n=== False Negatives (stamped but missed) ===')
    for r in stamped_cards:
        if not r['stamps']:
            print(f'  {r["name"]:25s} expected={r["expected"]:15s} checked={r["checked"]}')

if false_pos > 0:
    print(f'\n=== False Positives (normal but wrongly detected) ===')
    for r in normal_cards:
        if r['stamps']:
            print(f'  {r["name"]:25s} stamps={r["stamps"]}')
            for sname, sinfo in r['details'].items():
                print(f'    -> {sname}: conf={sinfo.get("confidence",0):.2f} evidence="{sinfo.get("evidence","")}"')

# Checks coverage
print(f'\n=== Checks Run (sample from first card) ===')
if results:
    print(f'  {results[0]["name"]}: {results[0]["checked"]}')
