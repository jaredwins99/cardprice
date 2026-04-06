#!/usr/bin/env python3
"""Test promo star detection on known promo and non-promo cards."""
import time
from cardprice.ml.stamp_detection import detect_all_variants

PROMO_STAMPS = {'black_star_promo', 'modern_promo', 'promo_stamp'}

promos = [
    ('data/inbox/page_20260307_015320_cards/card_05.png', 'basep-5/normal', 'Dragonite promo', True),
    ('data/inbox/page_20260307_020047_cards/card_05.png', 'basep-34/normal', 'Entei promo', True),
    ('data/inbox/page_20260307_132359_cards/card_00.png', 'dpp-DP09/normal', 'Torterra LV.X promo', True),
    ('data/inbox/page_20260307_132359_cards/card_02.png', 'dpp-DP29/normal', 'Rhyperior LV.X promo', True),
    ('data/inbox/page_20260307_132359_cards/card_08.png', 'dpp-DP56/normal', 'Arceus LV.X promo', True),
    ('data/inbox/page_20260305_094228_cards/card_00.png', 'ex15-10/normal', 'Chikorita NOT promo', False),
    ('data/inbox/page_20260320_223702_cards/card_00.png', 'ex1-64/normal', 'Poochyena NOT promo', False),
    ('data/inbox/page_20260307_014406_cards/card_00.png', 'gym2-87/normal', 'Misty Horsea NOT promo', False),
]

correct = 0
for path, card_id, desc, expected_promo in promos:
    t0 = time.time()
    result = detect_all_variants(path, card_id)
    dt = time.time() - t0
    detected = set(result.get('stamps_detected', []))
    detected_promo = bool(detected & PROMO_STAMPS)
    ok = detected_promo == expected_promo
    correct += ok
    status = 'OK' if ok else 'WRONG'
    print(f'{status:5s} {desc:30s} expected={str(expected_promo):5s} got={str(detected_promo):5s}  '
          f'stamps={result["stamps_detected"]}  checked={result["stamps_checked"]}  {dt:.1f}s')
    if not ok:
        details = result.get('stamp_details', {})
        for k, v in details.items():
            if k in PROMO_STAMPS:
                print(f'       detail: {k} -> {v}')

print(f'\nPromo accuracy: {correct}/{len(promos)}')
