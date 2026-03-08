#!/usr/bin/env python3
"""Run identify_card_v2 on all binder page cards, one page per process to avoid OOM."""
import subprocess, json, sys, re
from pathlib import Path

inbox = Path('/home/godli/cardprice/data/inbox')
out_file = Path('/home/godli/cardprice/data/eval/all_pages_v2_results.json')

# Load any existing partial results
if out_file.exists():
    all_results = json.loads(out_file.read_text())
    print(f"Loaded {len(all_results)} existing pages from {out_file}")
else:
    all_results = {}

page_dirs = sorted([d for d in inbox.glob('page_*_cards') if re.match(r'page_\d+_\d+_cards$', d.name)])
print(f"Found {len(page_dirs)} page directories")

for page_dir in page_dirs:
    page_name = page_dir.name.replace('_cards', '')

    if page_name in all_results:
        print(f"\nSkipping {page_name} (already done)")
        continue

    print(f"\n=== Processing {page_name} ===")

    # Run single page in subprocess
    code = f"""
import sys, json, time
sys.path.insert(0, '/home/godli/cardprice')
from pathlib import Path
from cardprice.ml import identify_card_v2

page_dir = Path('{page_dir}')
results = []

for card_file in sorted(page_dir.glob('card_*.png')):
    t0 = time.time()
    result = identify_card_v2(str(card_file))
    dt = time.time() - t0

    entry = {{
        'file': card_file.name,
        'card_id': result.get('card_id', 'NONE'),
        'method': result.get('method', ''),
        'confidence': round(result.get('confidence', 0), 3),
        'explanation': result.get('explanation', ''),
        'time': round(dt, 2),
    }}
    results.append(entry)
    print(f'{{card_file.name}}: {{entry["card_id"]}} ({{entry["method"]}}, {{entry["confidence"]:.3f}}, {{dt:.1f}}s)', flush=True)

# Output JSON on last line after separator
print('===JSON===')
print(json.dumps(results))
"""

    proc = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True, timeout=600
    )

    # Print stdout for visibility
    print(proc.stdout)
    if proc.stderr:
        # Filter out known warnings
        for line in proc.stderr.split('\n'):
            if line and not any(w in line for w in ['xFormers', 'CUDAExecutionProvider', 'UserWarning', 'warnings.warn']):
                print(f"  STDERR: {line}", file=sys.stderr)

    # Parse JSON from output
    if '===JSON===' in proc.stdout:
        json_str = proc.stdout.split('===JSON===\n')[-1].strip()
        page_results = json.loads(json_str)
        all_results[page_name] = page_results

        # Save after each page
        out_file.write_text(json.dumps(all_results, indent=2))
        print(f"  Saved {len(page_results)} cards for {page_name} (total pages: {len(all_results)})")
    else:
        print(f"  FAILED to get results for {page_name}")
        if proc.returncode != 0:
            print(f"  Exit code: {proc.returncode}")

# Summary
total_cards = sum(len(v) for v in all_results.values())
print(f"\n{'='*60}")
print(f"Done. {total_cards} cards across {len(all_results)} pages")
print(f"Saved to {out_file}")
