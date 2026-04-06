"""Test prerelease stamp detection accuracy - full pipeline."""
import warnings
warnings.filterwarnings('ignore')
import sys
import time

sys.path.insert(0, '/home/godli/cardprice')

print("Importing...", flush=True)
from cardprice.ml.stamp_detection import detect_all_variants
print("Import done.", flush=True)

# Cards that SHOULD detect as prerelease/stamped
positives = [
    ('data/inbox/page_20260307_014406_cards/card_02.png', 'gym1-9/normal', "Misty's Seadra", True),
    ('data/inbox/page_20260307_020047_cards/card_08.png', 'base3-1/normal', 'Aerodactyl', True),
]

# Cards that should NOT
negatives = [
    ('data/inbox/page_20260307_014406_cards/card_00.png', 'gym2-87/normal', "Misty's Horsea", False),
    ('data/inbox/page_20260307_014406_cards/card_03.png', 'gym1-89/normal', "Misty's Shellder", False),
    ('data/inbox/page_20260307_014406_cards/card_05.png', 'gym1-55/normal', "Misty's Seaking", False),
    ('data/inbox/page_20260320_223702_cards/card_00.png', 'ex1-64/normal', 'Poochyena', False),
]

all_tests = positives + negatives
correct = 0
for path, card_id, name, expected in all_tests:
    t0 = time.time()
    print(f'Processing {name}...', flush=True)
    result = detect_all_variants(path, card_id)
    dt = time.time() - t0
    stamps = result.get('stamps_detected', [])
    is_prerelease = 'prerelease' in stamps
    ok = is_prerelease == expected
    correct += ok
    print(f'{"OK" if ok else "WRONG":5s} {name:25s} expected={expected} got={is_prerelease} stamps={stamps} ({dt:.1f}s)', flush=True)

print(f'\nPrerelease accuracy: {correct}/{len(all_tests)}', flush=True)
