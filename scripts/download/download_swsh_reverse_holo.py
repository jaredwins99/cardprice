#!/usr/bin/env python3
"""Download reverse holofoil images for Sword & Shield sets from pokemontcg.io API."""

import os
import shutil
import time
import requests

STAGING_DIR = "/home/godli/cardprice/data/card_images_reverse_staging"
FINAL_DIR = "/home/godli/cardprice/data/card_images"
API_URL = "https://api.pokemontcg.io/v2/cards"

SETS = [f"swsh{i}" for i in range(1, 13)]

def download_set(set_id):
    out_dir = os.path.join(STAGING_DIR, set_id)
    os.makedirs(out_dir, exist_ok=True)

    page = 1
    total_downloaded = 0
    total_skipped = 0

    while True:
        url = f"{API_URL}?q=set.id:{set_id}&pageSize=250&page={page}"
        print(f"[{set_id}] Fetching page {page}: {url}")

        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        cards = data.get("data", [])
        if not cards:
            break

        time.sleep(1)  # rate limit after API query

        for card in cards:
            card_id = card["id"]
            image_url = card.get("images", {}).get("large")
            if not image_url:
                print(f"  [{card_id}] No large image, skipping")
                continue

            filename = f"{card_id}_reverse_holofoil.png"
            filepath = os.path.join(out_dir, filename)
            final_path = os.path.join(FINAL_DIR, set_id, filename)

            if os.path.exists(filepath) or os.path.exists(final_path):
                total_skipped += 1
                continue

            print(f"  Downloading {card_id} -> {filename}")
            try:
                img_resp = requests.get(image_url, timeout=30)
                img_resp.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(img_resp.content)
                total_downloaded += 1
                time.sleep(1)  # rate limit between image downloads
            except Exception as e:
                print(f"  ERROR downloading {card_id}: {e}")

        total_count = data.get("totalCount", 0)
        fetched = page * 250
        if fetched >= total_count:
            break
        page += 1

    print(f"[{set_id}] Done: {total_downloaded} downloaded, {total_skipped} skipped (already existed)")
    return total_downloaded, total_skipped


def main():
    grand_downloaded = 0
    grand_skipped = 0

    for set_id in SETS:
        downloaded, skipped = download_set(set_id)
        grand_downloaded += downloaded
        grand_skipped += skipped

    # Move all staged files into final card_images directory
    print(f"\n=== Moving files to {FINAL_DIR} ===")
    moved = 0
    for set_id in SETS:
        staging_set = os.path.join(STAGING_DIR, set_id)
        final_set = os.path.join(FINAL_DIR, set_id)
        os.makedirs(final_set, exist_ok=True)
        if not os.path.isdir(staging_set):
            continue
        for fname in os.listdir(staging_set):
            src = os.path.join(staging_set, fname)
            dst = os.path.join(final_set, fname)
            shutil.move(src, dst)
            moved += 1
        os.rmdir(staging_set)
    print(f"Moved {moved} files")

    print(f"\n=== COMPLETE ===")
    print(f"Total downloaded: {grand_downloaded}")
    print(f"Total skipped: {grand_skipped}")


if __name__ == "__main__":
    main()
