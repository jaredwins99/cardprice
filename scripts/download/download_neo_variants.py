#!/usr/bin/env python3
"""Download 1st Edition and Holofoil card images for Neo era sets from pokemontcg.io."""

import os
import time
import requests

BASE_DIR = "/home/godli/cardprice/data/card_images"
SETS = ["neo1", "neo2", "neo3", "neo4"]
API_URL = "https://api.pokemontcg.io/v2/cards"

session = requests.Session()
session.headers.update({"User-Agent": "cardprice-downloader/1.0"})

def download_with_retry(url, filepath, max_retries=3):
    """Download a file with retries."""
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(resp.content)
            return True
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return False

def download_set(set_id):
    """Query API for all cards in a set and download large images."""
    url = f"{API_URL}?q=set.id:{set_id}&pageSize=250"
    print(f"Fetching card list for {set_id}...")
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    cards = data.get("data", [])
    print(f"  Found {len(cards)} cards in {set_id}")

    out_dir = os.path.join(BASE_DIR, set_id)
    os.makedirs(out_dir, exist_ok=True)

    downloaded = 0
    skipped = 0
    errors = 0

    for card in cards:
        card_id = card["id"]
        image_url = card.get("images", {}).get("large")
        if not image_url:
            print(f"  No large image for {card_id}, skipping")
            continue

        for suffix in ["1st_edition", "holofoil"]:
            filename = f"{card_id}_{suffix}.png"
            filepath = os.path.join(out_dir, filename)

            if os.path.exists(filepath):
                skipped += 1
                continue

            print(f"  Downloading {filename}...")
            if download_with_retry(image_url, filepath):
                downloaded += 1
            else:
                print(f"  FAILED: {filename}")
                errors += 1
            time.sleep(1)  # Rate limit

    print(f"  {set_id}: downloaded={downloaded}, skipped={skipped}, errors={errors}")
    return downloaded, skipped

def main():
    total_dl = 0
    total_skip = 0
    for set_id in SETS:
        dl, skip = download_set(set_id)
        total_dl += dl
        total_skip += skip
        time.sleep(1)
    print(f"\nDone. Total downloaded={total_dl}, skipped={total_skip}")

if __name__ == "__main__":
    main()
