#!/usr/bin/env python3
"""Download holofoil card images from pokemontcg.io API for Base Set era sets."""

import os
import time
import requests

SETS = ["base1", "base2", "base3", "base4", "base5", "base6"]
BASE_DIR = "/home/godli/cardprice/data/card_images"
API_URL = "https://api.pokemontcg.io/v2/cards"


def download_set(set_id: str):
    out_dir = os.path.join(BASE_DIR, set_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Fetching card list for {set_id} ===")
    resp = requests.get(API_URL, params={"q": f"set.id:{set_id}", "pageSize": 250})
    resp.raise_for_status()
    cards = resp.json()["data"]
    print(f"  Found {len(cards)} cards")
    time.sleep(1)

    downloaded = 0
    skipped = 0
    for card in cards:
        card_id = card["id"]
        filename = f"{card_id}_holofoil.png"
        filepath = os.path.join(out_dir, filename)

        if os.path.exists(filepath):
            skipped += 1
            continue

        image_url = card.get("images", {}).get("large")
        if not image_url:
            print(f"  WARNING: No large image for {card_id}")
            continue

        try:
            img_resp = requests.get(image_url, timeout=30)
            img_resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(img_resp.content)
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"  Downloaded {downloaded} images...")
            time.sleep(1)
        except Exception as e:
            print(f"  ERROR downloading {card_id}: {e}")

    print(f"  {set_id} done: {downloaded} downloaded, {skipped} skipped (already existed)")


def main():
    for set_id in SETS:
        download_set(set_id)
    print("\nAll done!")


if __name__ == "__main__":
    main()
