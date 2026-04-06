#!/usr/bin/env python3
"""Evaluate detect_all_variants on a single card. Called as subprocess."""
import json, sys, time, os
sys.path.insert(0, '/home/godli/cardprice')
os.chdir('/home/godli/cardprice')

card_json = json.loads(sys.argv[1])
path = f'data/inbox/{card_json["image"]}'

from cardprice.ml.stamp_detection import detect_all_variants

set_id = card_json.get('set_id', '')
card_id = card_json.get('card_id', f'{set_id}-1/normal')

t0 = time.time()
try:
    result = detect_all_variants(path, card_id)
    dt = time.time() - t0
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)

print(json.dumps({
    "stamps_detected": result.get("stamps_detected", []),
    "stamp_details": {k: {kk: vv for kk, vv in v.items() if kk != "position"}
                      for k, v in result.get("stamp_details", {}).items()},
    "stamps_checked": result.get("stamps_checked", []),
    "variant_flags": {k: v for k, v in result.get("variant_flags", {}).items() if v},
    "time": round(dt, 2),
}))
