#!/usr/bin/env python3
"""Download reverse holofoil card images from pokemontcg.io for sets ex4, ex5, ex6."""

import os
import time
import requests

BASE_DIR = "/home/godli/cardprice/data/card_images"
SETS = ["ex4", "ex5", "ex6"]
API_URL = "https://api.pokemontcg.io/v2/cards"

session = requests.Session()

for set_id in SETS:
    print(f"\n=== Fetching card list for {set_id} ===")
    url = f"{API_URL}?q=set.id:{set_id}&pageSize=250"
    resp = session.get(url)
    resp.raise_for_status()
    cards = resp.json()["data"]
    print(f"Found {len(cards)} cards in {set_id}")
    time.sleep(1)  # rate limit API calls

    out_dir = os.path.join(BASE_DIR, set_id)
    os.makedirs(out_dir, exist_ok=True)

    skipped = 0
    downloaded = 0
    errors = 0

    for card in cards:
        card_id = card["id"]
        filename = f"{card_id}_reverse_holofoil.png"
        filepath = os.path.join(out_dir, filename)

        if os.path.exists(filepath):
            skipped += 1
            continue

        image_url = card.get("images", {}).get("large")
        if not image_url:
            print(f"  WARNING: No large image for {card_id}")
            errors += 1
            continue

        try:
            img_resp = session.get(image_url, timeout=30)
            img_resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(img_resp.content)
            downloaded += 1
            if downloaded % 20 == 0:
                print(f"  Downloaded {downloaded} so far...")
        except Exception as e:
            print(f"  ERROR downloading {card_id}: {e}")
            errors += 1

    print(f"  {set_id}: downloaded={downloaded}, skipped={skipped}, errors={errors}")

print("\nDone!")
