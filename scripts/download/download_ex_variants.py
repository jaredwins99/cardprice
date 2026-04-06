#!/usr/bin/env python3
"""Download Pokemon TCG card images from pokemontcg.io API for EX-era sets ex7-ex11."""

import os
import time
import requests
from pathlib import Path

BASE_DIR = Path("/home/godli/cardprice/data/card_images")
SETS = ["ex7", "ex8", "ex9", "ex10", "ex11"]
API_URL = "https://api.pokemontcg.io/v2/cards"
SUMMARY_FILE = "/tmp/variant_download_ex7-11.txt"

def fetch_set_cards(set_id):
    """Fetch all cards for a set, handling pagination."""
    cards = []
    page = 1
    page_size = 250
    while True:
        url = f"{API_URL}?q=set.id:{set_id}&select=id,name,images&pageSize={page_size}&page={page}"
        print(f"  Fetching {set_id} page {page}...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        cards.extend(data.get("data", []))
        total = data.get("totalCount", 0)
        if len(cards) >= total:
            break
        page += 1
        time.sleep(1)
    return cards

def download_image(url, dest_path):
    """Download an image to dest_path."""
    resp = requests.get(url, timeout=30, stream=True)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)

def main():
    summary_lines = []
    total_downloaded = 0
    total_skipped = 0
    total_errors = 0

    for set_id in SETS:
        set_dir = BASE_DIR / set_id
        set_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Set: {set_id} ===")
        cards = fetch_set_cards(set_id)
        print(f"  Found {len(cards)} cards")
        time.sleep(1)  # Rate limit after API call

        set_downloaded = 0
        set_skipped = 0
        set_errors = 0

        for card in cards:
            card_id = card["id"]
            name = card.get("name", "unknown")
            images = card.get("images", {})
            large_url = images.get("large")

            if not large_url:
                print(f"  SKIP {card_id} ({name}): no large image URL")
                set_skipped += 1
                continue

            dest = set_dir / f"{card_id}_reverse_holofoil.png"
            if dest.exists():
                set_skipped += 1
                continue

            try:
                download_image(large_url, dest)
                print(f"  OK {card_id} ({name})")
                set_downloaded += 1
                time.sleep(1)  # Rate limit
            except Exception as e:
                print(f"  ERROR {card_id} ({name}): {e}")
                set_errors += 1
                # Clean up partial download
                if dest.exists():
                    dest.unlink()

        line = f"{set_id}: {len(cards)} cards, {set_downloaded} downloaded, {set_skipped} skipped, {set_errors} errors"
        print(f"  {line}")
        summary_lines.append(line)
        total_downloaded += set_downloaded
        total_skipped += set_skipped
        total_errors += set_errors

    summary_lines.append("")
    summary_lines.append(f"TOTAL: {total_downloaded} downloaded, {total_skipped} skipped, {total_errors} errors")

    summary = "\n".join(summary_lines)
    print(f"\n{summary}")
    with open(SUMMARY_FILE, "w") as f:
        f.write(summary + "\n")
    print(f"\nSummary saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
