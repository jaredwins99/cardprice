#!/usr/bin/env python3
"""Download reverse holofoil card images from pokemontcg.io for e-Card era sets."""

import os
import time
import requests

BASE_DIR = "/home/godli/cardprice/data/card_images"
SETS = ["ecard1", "ecard2", "ecard3"]
API_URL = "https://api.pokemontcg.io/v2/cards"

for set_id in SETS:
    out_dir = os.path.join(BASE_DIR, set_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Fetching card list for {set_id} ===")
    resp = requests.get(API_URL, params={"q": f"set.id:{set_id}", "pageSize": 250})
    resp.raise_for_status()
    cards = resp.json()["data"]
    print(f"Found {len(cards)} cards in {set_id}")

    for card in cards:
        card_id = card["id"]
        img_url = card.get("images", {}).get("large")
        if not img_url:
            print(f"  SKIP {card_id}: no large image")
            continue

        out_path = os.path.join(out_dir, f"{card_id}_reverse_holofoil.png")
        if os.path.exists(out_path):
            print(f"  EXISTS {card_id}")
            continue

        print(f"  Downloading {card_id} ...", end=" ", flush=True)
        img_resp = requests.get(img_url, timeout=30)
        img_resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(img_resp.content)
        print(f"OK ({len(img_resp.content)//1024} KB)")
        time.sleep(1)

    print(f"Done with {set_id}")

print("\nAll done.")
